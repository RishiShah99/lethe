"""HF causal-LM + LoRA policy behind ``PolicyInterface``.

Design constraints this module encodes:

- ``transformers``/``peft`` live in the ``rl`` extra and are absent from
  some dev environments, so every heavy import is deferred into
  :meth:`HFPolicy.from_pretrained`. The class body is duck-typed against
  the small model/tokenizer surface it actually uses (``apply_chat_template``,
  ``generate``, forward logits, ``device``), which lets CPU tests drive the
  full logic with stubs.
- ``apply_chat_template(..., return_dict=True)`` returns a ``BatchEncoding``
  on the box's transformers — ``input_ids`` and ``attention_mask`` are
  extracted and passed to ``generate`` explicitly.
- GRPO needs new/old/ref per-token log-probs over the *same* token
  sequence. Completions round-trip through strings (the verifier consumes
  source text), so all three streams retokenize identically via
  :meth:`HFPolicy.completion_log_probs` — at the first optimizer step
  new == old exactly and the importance ratio starts at 1. Retokenization
  may drift from the sampled token ids (non-canonical BPE splits) —
  accepted by convention since all streams drift together.
- Log-probs are computed under the *sampling* distribution: logits are
  divided by the sampling temperature before the softmax (the behaviour
  policy is the tempered one; scoring the untempered distribution would
  make the policy gradient a biased estimator). ``top_p`` truncation is
  ignored by the usual GRPO convention.
- EOS is appended to a completion's token sequence only when generation
  terminated naturally (the policy actually chose to stop); for
  length-truncated samples training on a fabricated EOS would push the
  policy toward stopping at the truncation point.
- The KL reference policy is the base model with the LoRA adapter
  disabled (``peft``'s ``disable_adapter``), exposed as
  :class:`ReferencePolicyView` — one set of base weights in memory, two
  ``PolicyInterface`` objects.
- The differentiable log-prob pass stores activations for the whole
  K x (P + T) batch; at 32B that exceeds a single B200 without gradient
  checkpointing (measured: OOM at 177 GiB). ``gradient_checkpointing=True``
  enables HF non-reentrant checkpointing — which only engages in train
  mode, so :meth:`completion_log_probs` toggles ``train()`` around the
  grad-enabled forward only. Every dropout in the stack is 0.0 (LoRA
  dropout off by policy default, Qwen2.5 attention dropout 0.0), so the
  train-mode forward stays deterministic and the step-0 ratio identity
  holds. Scoring forwards always pass ``use_cache=False`` — a KV cache
  is pure waste on a full-sequence pass and incompatible with
  checkpointing.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class SamplingSettings:
    """Generation hyperparameters (defaults match the Phase D bakeoff)."""

    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 4096
    batch_size: int = 4


class HFPolicy:
    """A trainable HF causal LM (optionally LoRA-wrapped) kernel-generation policy.

    Satisfies ``PolicyInterface`` (``generate`` / ``log_probs``) and adds the
    batched, differentiable :meth:`completion_log_probs` the GRPO update
    consumes, plus adapter checkpointing hooks.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        sampling: SamplingSettings | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.sampling = sampling if sampling is not None else SamplingSettings()
        self.last_terminated: list[bool] = []
        self._gradient_checkpointing = gradient_checkpointing

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        # Dropout breaks GRPO's step-0 ratio identity (sampled vs scored
        # log-probs diverge under different dropout masks) — default off.
        lora_dropout: float = 0.0,
        lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES,
        adapter_path: str | None = None,
        torch_dtype: str = "bfloat16",
        device_map: str | dict[str, str] = "auto",
        sampling: SamplingSettings | None = None,
        gradient_checkpointing: bool = False,
    ) -> HFPolicy:
        """Load a real HF model (+ fresh or checkpointed LoRA adapter).

        Requires the ``rl`` extra (transformers, peft). ``adapter_path``
        resumes a saved adapter; otherwise a fresh adapter is attached when
        ``lora=True``. ``gradient_checkpointing`` trades ~30% step time for
        the activation memory of the differentiable log-prob pass —
        required for 32B-class models on a single device.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, torch_dtype),
            device_map=device_map,
        )
        if gradient_checkpointing:
            # Non-reentrant variant recomputes unconditionally; the input
            # require-grads hook covers peft's frozen-embedding edge anyway.
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
        if lora:
            from peft import LoraConfig, PeftModel, get_peft_model

            if adapter_path is not None:
                model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
            else:
                config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=list(lora_target_modules),
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, config)
        return cls(
            model, tokenizer, sampling=sampling, gradient_checkpointing=gradient_checkpointing
        )

    # ------------------------------------------------------------------
    # PolicyInterface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, n: int) -> list[str]:
        """Sample ``n`` completions of ``prompt`` (chat-templated, batched).

        Sets ``self.last_terminated`` (one bool per completion): True iff
        the sequence emitted EOS before ``max_new_tokens`` — early-finished
        sequences are padded with EOS by ``generate``, so any EOS in the
        generated tail means natural termination.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if n == 0:
            self.last_terminated = []
            return []
        input_ids, attention_mask = self._encode_prompt(prompt)
        prompt_len = input_ids.shape[1]
        s = self.sampling
        eos = self._tokenizer.eos_token_id
        completions: list[str] = []
        terminated: list[bool] = []
        remaining = n
        while remaining > 0:
            k = min(s.batch_size, remaining)
            with torch.no_grad():
                out = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=s.temperature,
                    top_p=s.top_p,
                    max_new_tokens=s.max_new_tokens,
                    num_return_sequences=k,
                    pad_token_id=eos,
                )
            for seq in out:
                tail = seq[prompt_len:]
                completions.append(self._tokenizer.decode(tail, skip_special_tokens=True))
                terminated.append(bool((tail == eos).any().item()))
            remaining -= k
        self.last_terminated = terminated
        return completions

    def log_probs(self, prompt: str, completion: str) -> list[float]:
        """Per-token log-probs of ``completion`` (model-tokenised, + EOS)."""
        with torch.no_grad():
            lp, mask = self.completion_log_probs(prompt, [completion])
        return [float(x) for x in lp[0][mask[0]].tolist()]

    # ------------------------------------------------------------------
    # Trainer surface
    # ------------------------------------------------------------------

    def completion_log_probs(
        self,
        prompt: str,
        completions: list[str],
        *,
        use_adapter: bool = True,
        append_eos: bool | Sequence[bool] = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched per-token log-probs of each completion given ``prompt``.

        Returns ``(log_probs, mask)``, both ``(K, T_max)``. ``mask`` is True
        at valid completion positions. Logits are divided by the sampling
        temperature before the softmax — these are the *behaviour* policy's
        log-probs. Differentiable: wrap in ``torch.no_grad()`` for old/ref
        streams. ``use_adapter=False`` computes under the frozen base model
        via peft's ``disable_adapter``. ``append_eos`` (scalar or one bool
        per completion, e.g. ``last_terminated`` from ``generate``) controls
        whether the stop decision is part of the scored trajectory — pass
        the same value to every stream.
        """
        if not completions:
            raise ValueError("completions must be non-empty")
        if isinstance(append_eos, bool):
            append_eos = [append_eos] * len(completions)
        if len(append_eos) != len(completions):
            raise ValueError("append_eos length must match completions")
        prompt_ids, _ = self._encode_prompt(prompt)
        prompt_len = prompt_ids.shape[1]
        device = prompt_ids.device
        eos = self._tokenizer.eos_token_id

        comp_token_lists: list[list[int]] = []
        for completion, add_eos in zip(completions, append_eos, strict=True):
            ids = list(self._tokenizer(completion, add_special_tokens=False)["input_ids"])
            comp_token_lists.append([*ids, eos] if add_eos else ids)
        t_max = max(len(ids) for ids in comp_token_lists)

        k = len(completions)
        full = torch.full((k, prompt_len + t_max), eos, dtype=torch.long, device=device)
        attention = torch.zeros((k, prompt_len + t_max), dtype=torch.long, device=device)
        mask = torch.zeros((k, t_max), dtype=torch.bool, device=device)
        full[:, :prompt_len] = prompt_ids
        attention[:, :prompt_len] = 1
        for i, ids in enumerate(comp_token_lists):
            full[i, prompt_len : prompt_len + len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=device
            )
            attention[i, prompt_len : prompt_len + len(ids)] = 1
            mask[i, : len(ids)] = True

        # Checkpointing only engages in train mode; the toggle is scoped to
        # the grad-enabled pass (no_grad streams store no activations, and
        # eval elsewhere keeps the policy's documented sampling behaviour).
        toggle_train = self._gradient_checkpointing and use_adapter and torch.is_grad_enabled()
        if toggle_train:
            self._model.train()
        try:
            if use_adapter:
                logits = self._model(
                    input_ids=full, attention_mask=attention, use_cache=False
                ).logits
            else:
                with self._disabled_adapter():
                    logits = self._model(
                        input_ids=full, attention_mask=attention, use_cache=False
                    ).logits
        finally:
            if toggle_train:
                self._model.eval()

        # Logits at position j predict token j+1: completion tokens occupy
        # absolute positions [prompt_len, prompt_len + t_max).
        pred = logits[:, prompt_len - 1 : prompt_len + t_max - 1, :]
        targets = full[:, prompt_len : prompt_len + t_max]
        log_probs = torch.log_softmax(pred.float() / self.sampling.temperature, dim=-1)
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return gathered * mask, mask

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        params: Iterator[torch.nn.Parameter] = (
            p for p in self._model.parameters() if p.requires_grad
        )
        return params

    def save_adapter(self, path: str) -> None:
        """Persist the LoRA adapter weights (peft ``save_pretrained``)."""
        self._model.save_pretrained(path)

    @property
    def device(self) -> torch.device:
        dev: torch.device = self._model.device
        return dev

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """The current LoRA tensors (detached, CPU) for broadcast to replicas.

        CPU-resident so a generation-pool replica on another device loads
        them without a cross-device peer copy from the trainer.
        """
        from peft import get_peft_model_state_dict

        sd = get_peft_model_state_dict(self._model)
        return {k: v.detach().to("cpu") for k, v in sd.items()}

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Overwrite this policy's LoRA tensors (generation-replica refresh)."""
        from peft import set_peft_model_state_dict

        device = self._model.device
        moved = {k: v.to(device) for k, v in state.items()}
        set_peft_model_state_dict(self._model, moved)

    def train_mode(self) -> None:
        self._model.train()

    def eval_mode(self) -> None:
        self._model.eval()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        device = self._model.device
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def _disabled_adapter(self) -> Any:
        disable = getattr(self._model, "disable_adapter", None)
        if disable is None:
            raise RuntimeError(
                "model has no disable_adapter — reference log-probs require a peft-wrapped model"
            )
        return disable()


class ReferencePolicyView:
    """``PolicyInterface`` view of an :class:`HFPolicy` with the adapter off.

    The KL reference for LoRA training is the base model itself; this view
    shares the policy's weights instead of loading a second copy.
    """

    def __init__(self, policy: HFPolicy) -> None:
        self._policy = policy

    def generate(self, prompt: str, n: int) -> list[str]:
        raise NotImplementedError("the reference policy only scores, never samples")

    def log_probs(self, prompt: str, completion: str) -> list[float]:
        with torch.no_grad():
            lp, mask = self._policy.completion_log_probs(prompt, [completion], use_adapter=False)
        return [float(x) for x in lp[0][mask[0]].tolist()]

    def completion_log_probs(
        self,
        prompt: str,
        completions: list[str],
        *,
        append_eos: bool | Sequence[bool] = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self._policy.completion_log_probs(
                prompt, completions, use_adapter=False, append_eos=append_eos
            )
