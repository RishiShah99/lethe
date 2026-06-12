"""CPU tests for HFPolicy via stub model/tokenizer.

transformers/peft are not required: the stubs implement exactly the
duck-typed surface HFPolicy consumes, so the chat-template plumbing,
batched log-prob gather/mask alignment, gradient flow, and the
adapter-disabled reference path are all pinned without a real model.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from flash_mamba_rl.rl.hf_policy import HFPolicy, ReferencePolicyView, SamplingSettings

VOCAB = 256
EOS = 0


class StubTokenizer:
    eos_token_id = EOS

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool,
        return_tensors: str,
        return_dict: bool,
    ) -> dict[str, torch.Tensor]:
        assert add_generation_prompt and return_tensors == "pt" and return_dict
        text = "<u>" + messages[0]["content"] + "<a>"
        ids = [ord(c) for c in text]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids if int(i) != EOS)


class StubModel(torch.nn.Module):
    """Tiny deterministic LM: embedding -> linear head over a char vocab."""

    def __init__(self, completion: str = "ok!") -> None:
        super().__init__()
        torch.manual_seed(7)
        self.emb = torch.nn.Embedding(VOCAB, 8)
        self.head = torch.nn.Linear(8, VOCAB)
        self.completion_ids = [ord(c) for c in completion]
        self.generate_calls: list[int] = []
        self.ref_logit_shift = 0.0
        self.emit_eos = True

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool | None = None,
    ) -> Any:
        logits = self.head(self.emb(input_ids)) + self.ref_logit_shift
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        num_return_sequences: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.generate_calls.append(num_return_sequences)
        ids = [*self.completion_ids, EOS] if self.emit_eos else self.completion_ids
        comp = torch.tensor([ids], dtype=torch.long)
        full = torch.cat([input_ids, comp], dim=1)
        return full.repeat(num_return_sequences, 1)


class StubPeftModel(StubModel):
    def __init__(self, completion: str = "ok!") -> None:
        super().__init__(completion)
        self.adapter_disabled_calls = 0

    @contextlib.contextmanager
    def disable_adapter(self) -> Any:
        self.adapter_disabled_calls += 1
        self.ref_logit_shift = 3.0
        try:
            yield
        finally:
            self.ref_logit_shift = 0.0


def make_policy(model: StubModel | None = None, **sampling: Any) -> HFPolicy:
    sampling.setdefault("temperature", 1.0)
    return HFPolicy(
        model if model is not None else StubModel(),
        StubTokenizer(),
        sampling=SamplingSettings(**sampling),
    )


def manual_log_probs(
    model: StubModel, prompt: str, completion: str, *, temperature: float = 1.0
) -> torch.Tensor:
    tok = StubTokenizer()
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"]
    comp_ids = [*[ord(c) for c in completion], EOS]
    full = torch.cat([prompt_ids, torch.tensor([comp_ids], dtype=torch.long)], dim=1)
    with torch.no_grad():
        logits = model(input_ids=full, attention_mask=torch.ones_like(full)).logits
    p = prompt_ids.shape[1]
    pred = logits[0, p - 1 : p + len(comp_ids) - 1, :]
    return torch.log_softmax(pred.float() / temperature, dim=-1)[
        torch.arange(len(comp_ids)), torch.tensor(comp_ids)
    ]


class TestGenerate:
    def test_returns_n_completions_batched(self) -> None:
        model = StubModel(completion="ok!")
        policy = make_policy(model, batch_size=2, max_new_tokens=16)
        out = policy.generate("write a kernel", 5)
        assert out == ["ok!"] * 5
        assert model.generate_calls == [2, 2, 1]

    def test_zero_n(self) -> None:
        assert make_policy().generate("p", 0) == []

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            make_policy().generate("p", -1)

    def test_last_terminated_tracks_eos_presence(self) -> None:
        model = StubModel(completion="xy")
        policy = make_policy(model)
        policy.generate("p", 2)
        assert policy.last_terminated == [True, True]
        model.emit_eos = False
        policy.generate("p", 2)
        assert policy.last_terminated == [False, False]

    def test_decode_strips_prompt(self) -> None:
        # Completion must not contain any prompt text even though the
        # stub returns prompt+completion concatenated.
        out = make_policy(StubModel(completion="xyz")).generate("PROMPT", 1)
        assert out == ["xyz"]


class TestLogProbs:
    def test_matches_manual_computation(self) -> None:
        model = StubModel()
        policy = make_policy(model)
        completion = "abc"
        got = policy.log_probs("the prompt", completion)
        want = manual_log_probs(model, "the prompt", completion)
        assert len(got) == len(completion) + 1  # +1 appended EOS
        torch.testing.assert_close(torch.tensor(got), want)

    def test_batched_matches_single(self) -> None:
        model = StubModel()
        policy = make_policy(model)
        completions = ["short", "a much longer completion!", ""]
        lp, mask = policy.completion_log_probs("p", completions)
        assert lp.shape == mask.shape
        assert mask.sum(dim=1).tolist() == [6, 26, 1]
        for i, completion in enumerate(completions):
            single = manual_log_probs(model, "p", completion)
            torch.testing.assert_close(lp[i][mask[i]], single)

    def test_padding_positions_zeroed(self) -> None:
        policy = make_policy()
        lp, mask = policy.completion_log_probs("p", ["ab", "abcdef"])
        assert (lp[0][~mask[0]] == 0.0).all()

    def test_empty_completions_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            make_policy().completion_log_probs("p", [])

    def test_gradient_flows_to_model(self) -> None:
        model = StubModel()
        policy = make_policy(model)
        lp, mask = policy.completion_log_probs("p", ["abc", "de"])
        loss = -(lp * mask).sum()
        loss.backward()
        assert model.head.weight.grad is not None
        assert torch.isfinite(model.head.weight.grad).all()

    def test_log_probs_under_sampling_temperature(self) -> None:
        # The behaviour policy is the tempered one — log-probs must divide
        # logits by the sampling temperature (biased gradient otherwise).
        model = StubModel()
        policy = make_policy(model, temperature=2.0)
        got = policy.log_probs("p", "abc")
        want = manual_log_probs(model, "p", "abc", temperature=2.0)
        torch.testing.assert_close(torch.tensor(got), want)
        raw = manual_log_probs(model, "p", "abc", temperature=1.0)
        assert not torch.allclose(want, raw)

    def test_append_eos_false_drops_stop_token(self) -> None:
        model = StubModel()
        policy = make_policy(model)
        lp, mask = policy.completion_log_probs("p", ["abc", "de"], append_eos=[False, True])
        assert mask.sum(dim=1).tolist() == [3, 3]  # no EOS vs +EOS
        single = manual_log_probs(model, "p", "abc")[:-1]  # manual minus EOS slot
        torch.testing.assert_close(lp[0][mask[0]], single)

    def test_append_eos_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="append_eos"):
            make_policy().completion_log_probs("p", ["a", "b"], append_eos=[True])

    def test_no_grad_streams_identical_to_grad_stream(self) -> None:
        # new == old at step 0: same retokenization path must give the
        # exact same values with and without grad.
        model = StubModel()
        policy = make_policy(model)
        lp_grad, _ = policy.completion_log_probs("p", ["abc"])
        with torch.no_grad():
            lp_nograd, _ = policy.completion_log_probs("p", ["abc"])
        assert torch.equal(lp_grad.detach(), lp_nograd)


class TestReferenceView:
    def test_disable_adapter_used_and_restored(self) -> None:
        model = StubPeftModel()
        policy = make_policy(model)
        ref = ReferencePolicyView(policy)
        ref_lp = ref.log_probs("p", "abc")
        assert model.adapter_disabled_calls == 1
        assert model.ref_logit_shift == 0.0
        pol_lp = policy.log_probs("p", "abc")
        # Uniform logit shift cancels in softmax — values match, but the
        # disable path must have been exercised; use a head-bias change
        # to verify a real difference is observable.
        assert len(ref_lp) == len(pol_lp)

    def test_reference_differs_when_adapter_changes_logits(self) -> None:
        class ShiftedPeft(StubPeftModel):
            @contextlib.contextmanager
            def disable_adapter(self) -> Any:
                self.adapter_disabled_calls += 1
                old = self.head.bias.data.clone()
                self.head.bias.data += torch.linspace(-1, 1, VOCAB)
                try:
                    yield
                finally:
                    self.head.bias.data = old

        model = ShiftedPeft()
        policy = make_policy(model)
        ref = ReferencePolicyView(policy)
        assert ref.log_probs("p", "abc") != policy.log_probs("p", "abc")

    def test_reference_never_samples(self) -> None:
        ref = ReferencePolicyView(make_policy(StubPeftModel()))
        with pytest.raises(NotImplementedError):
            ref.generate("p", 1)

    def test_plain_model_without_peft_raises(self) -> None:
        policy = make_policy(StubModel())
        with pytest.raises(RuntimeError, match="disable_adapter"):
            policy.completion_log_probs("p", ["abc"], use_adapter=False)


class TestTrainerSurface:
    def test_trainable_parameters_filters_frozen(self) -> None:
        model = StubModel()
        model.emb.weight.requires_grad_(False)
        policy = make_policy(model)
        params = list(policy.trainable_parameters())
        assert len(params) == 2  # head weight + bias
        assert all(p.requires_grad for p in params)

    def test_train_eval_mode(self) -> None:
        model = StubModel()
        policy = make_policy(model)
        policy.eval_mode()
        assert not model.training
        policy.train_mode()
        assert model.training
