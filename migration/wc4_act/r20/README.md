# R20 complete-bypass support control

Only class2's anchor support changes relative to R19: all original compact_bypass frames in each of72 expert episodes instead of the +/-1s bypass-to-rendezvous window. Class1/3/4 pools are copied exactly from R19 sealed windows, and the exact same seed1000, numpy choice-with-replacement, perbatch8+2+2+2+2 and within-batch permutation algorithm is retained. Rejection/selection RNG coupling is measured rather than claimed without proof.

Source is ORIGINAL R15 true050000 full checkpoint, NOT R19 75k. Exactly25000 new updates to75000, then ONLY16 endpoint initial states62200–62215 at trueh25,180s maximum. Existing R19/R18 75k andR15 50k are controls, never rerun. Preserve chunk100, episode padding, original statistics/loss/network/inputs/physics/R11/R10. Author DataLoader cursor is not serialized.

Require at least80GiB free before preparation and launch; ongoing40GiB cooperative stop margin. Preserve all history. Existing ACT/Gym environments and shared GPU UUID lock only. CPU fixtures build no real policy or scene. No smoke loops or additional diagnostic forwards.
