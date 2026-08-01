# VISO Analysis: V3 Calibration Metadata + V5/V6 Combination (2026-07-31)

- Leakage correction: applied (no test-label conditioning; train-only calibration).
- Dataset profile: `viso`
- Seeds: [101, 202, 303, 404, 505]
- Steps per run: 8

## V3 Threshold Sweep (Global Metrics)

| threshold | precision | recall | f1 | ap | mAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.45 | 0.002883 | 0.003747 | 0.003258 | 0.000035 | 0.000036 |
| 0.50 | 0.002442 | 0.002810 | 0.002613 | 0.000030 | 0.000031 |
| 0.55 | 0.002305 | 0.002342 | 0.002323 | 0.000027 | 0.000027 |

## V3 Metadata Table: Which Characteristics Suggest Each Threshold

| threshold | area_bucket | precision | recall | f1 | tp | fp | fn |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.45 | small | 0.002077 | 0.003333 | 0.002559 | 2 | 961 | 598 |
| 0.45 | medium | 0.004197 | 0.006452 | 0.005086 | 4 | 949 | 616 |
| 0.45 | large | 0.002315 | 0.002186 | 0.002248 | 2 | 862 | 913 |
| 0.50 | small | 0.001172 | 0.001667 | 0.001376 | 1 | 852 | 599 |
| 0.50 | medium | 0.003505 | 0.004839 | 0.004065 | 3 | 853 | 617 |
| 0.50 | large | 0.002632 | 0.002186 | 0.002388 | 2 | 758 | 913 |
| 0.55 | small | 0.000000 | 0.000000 | 0.000000 | 0 | 749 | 600 |
| 0.55 | medium | 0.004000 | 0.004839 | 0.004380 | 3 | 747 | 617 |
| 0.55 | large | 0.003035 | 0.002186 | 0.002541 | 2 | 657 | 913 |

## Combined Versions (V1..V6, VISO Only)

| slug | strategy | precision | recall | f1 | ap | mAP | tp | fp | fn | selected_threshold_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1_mmb_baseline_reproduction | baseline_proxy | 0.003142 | 0.005152 | 0.003904 | 0.000043 | 0.000044 | 2.20 | 700.80 | 424.80 |  |
| v2_mmb_tiling_overlap | tiling_overlap | 0.004883 | 0.008899 | 0.006305 | 0.000064 | 0.000067 | 3.80 | 776.40 | 423.20 |  |
| v3_mmb_threshold_nms_calibration | threshold_nms_calibration | 0.002442 | 0.002810 | 0.002613 | 0.000030 | 0.000031 | 1.20 | 492.60 | 425.80 |  |
| v4_mmb_hard_negative_mining | hard_negative_mining | 0.005509 | 0.007963 | 0.006512 | 0.000097 | 0.000098 | 3.40 | 619.00 | 423.60 |  |
| v5_mmb_v4_plus_v2 | v5_v4_plus_v2 | 0.004209 | 0.007026 | 0.005264 | 0.000058 | 0.000058 | 3.00 | 710.00 | 424.00 |  |
| v6_mmb_v5_plus_light_v3_calibration | v6_v5_plus_light_v3 | 0.004244 | 0.005152 | 0.004629 | 0.000047 | 0.000048 | 2.20 | 534.40 | 424.80 | 0.47 |

## Decision by Requested Criterion

- Criterion: maximize F1 on viso, with recall drop <= 0.02 versus v5.
- Winner: `v4_mmb_hard_negative_mining`
- Winner F1: 0.006512
- Winner Recall: 0.007963
- v6 vs v5 recall drop: 0.001874
- Constraint satisfied: True