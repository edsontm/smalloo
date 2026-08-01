# VISO Paper Table: V1 vs V5 vs V6 (2026-07-31)

- Dataset profile: viso
- Seeds: 101, 202, 303, 404, 505
- Steps per run: 8
- Evaluation mode: mmb_proxy

## Main Metrics

| version | strategy | precision | recall | f1 | ap | mAP | tp | fp | fn |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1_mmb_baseline_reproduction | baseline_proxy | 0.003142 | 0.005152 | 0.003904 | 0.000043 | 0.000044 | 2.20 | 700.80 | 424.80 |
| v5_mmb_v4_plus_v2 | v5_v4_plus_v2 | 0.004209 | 0.007026 | 0.005264 | 0.000058 | 0.000058 | 3.00 | 710.00 | 424.00 |
| v6_mmb_v5_plus_light_v3_calibration | v6_v5_plus_light_v3 | 0.004244 | 0.005152 | 0.004629 | 0.000047 | 0.000048 | 2.20 | 534.40 | 424.80 |

## Decision

- Primary criterion: maximize F1 on viso with recall drop <= 0.02 versus v5.
- Winner: v5_mmb_v4_plus_v2
- Recall delta (v6 - v5): -0.001874; equivalent recall drop vs v5 = 0.001874 (constraint satisfied)

## Notes

- Metrics were recomputed after leakage corrections in the proxy pipeline.
- Proxy predictions no longer condition on test labels, and v6 threshold selection now uses train split only.
- Under leakage-safe evaluation, v6 reduces FP versus v5 but does not improve F1.
- Results remain proxy-only and must be revalidated after full MMB integration.