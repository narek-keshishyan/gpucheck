# Spec-table calibration notes (2026-11-06)

Web verification of the `specs.py` pins (NVIDIA product pages/datasheets,
whitepapers, community benchmarks; full source list in the research session).

## Verified

- **All six `peak_bf16_tflops` pins are on a DENSE basis** (BF16 tensor,
  FP32 accumulate) and match the datasheets: 4090 165.2 (330.4 sparse),
  3090 71.0 (142), A100 312 (624), H100-SXM 989 (1979 sparse),
  H100-PCIe 756 (1513 sparse).
- TDP and boost clocks: all six correct.
- Max temp: 4090 90 °C and 3090 93 °C verified from NVIDIA pages.

## Soft (verify on real hardware via `nvidia-smi -q -d TEMPERATURE`)

- A100 85 °C is the *max operating* figure, not the slowdown threshold
  (community data: slowdown ≈ 86–89 °C depending on SKU/VBIOS).
- H100 83 °C is unverified; community tables suggest slowdown ≈ 87 °C for
  SXM5. The watchdog margin (+5 °C) keeps this safe either way.

## Achieved GEMM vs dense peak (why the eligibility gate moved to 0.50)

Real healthy cards sustain well below dense datasheet peak:

| Card | Measured/estimated sustained BF16 GEMM | Fraction of dense peak |
|---|---|---|
| RTX 4090 | 130–160 TFLOPS (est.) | 80–95% |
| RTX 3090 | 55–68 TFLOPS (est.) | 75–95% |
| A100 SXM | 230–290 TFLOPS (measured) | 74–93% |
| A100 PCIe | 200–260 TFLOPS (est., 300 W cap) | 65–83% |
| H100 SXM | 600–800 TFLOPS (measured, power-limited) | 60–81% |
| H100 PCIe | 450–620 TFLOPS (est., 350 W cap) | 60–82% |

The original server-side eligibility gate (ratio ≥ 0.93 of dense peak)
would therefore have rejected **every healthy real card**. It is now 0.50 —
a floor that only filters clearly broken or spoofed runs. After the first
~30–50 seeded real runs, re-tune per model: the right long-term shape is a
per-model *expected sustained* value learned from the reference set itself,
with the datasheet peak kept only as the spoof ceiling (quarantine > 1.15).

## Open items for the smoke test

- Record `nvidia-smi -q -d TEMPERATURE` slowdown/shutdown values per card.
- Record one tick's wall time and the achieved TFLOPS for the rented card;
  compare against the table above.
