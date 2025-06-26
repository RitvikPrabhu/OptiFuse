# FFN Forge

Goal: Given a list of GPU tensor ops, automatically decide
*whether* to fuse them and *which* kernel back-end (CUTLASS or Triton)
is fastest on the current GPU, then JIT-compile that kernel.
