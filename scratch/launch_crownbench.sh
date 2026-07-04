#!/bin/bash
# Detach-launch the crown (dv128) graph bench so plink can drop without killing it.
cd ~
export PYTHONPATH=src:.
rm -f /tmp/crownbench.done
nohup ~/cuteenv/bin/python scratch/gdn2_graph_bench.py \
  --dv 128 --shapes 2x2048x8 \
  --out results/gdn2_graph_bench_dv128_k2fntiled.json \
  --trials 20 --eager-trials 2 > /tmp/crownbench.log 2>&1 &
echo $! > /tmp/crownbench.pid
