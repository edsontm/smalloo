# VISO Error Analysis - 2026-07-31

- Experiment: v1_mmb_baseline_reproduction
- Dataset profile: viso
- Evaluation mode: mmb_proxy
- Images: 351
- Annotations: 427

## Aggregate Error Signature (5 seeds)

- Images with any FP: 250
- Images with any FN: 154

## Top FP Images

| image_id | fp_total | file_name |
| --- | --- | --- |
| 19000168 | 4 | 000168.jpg |
| 19000085 | 4 | 000085.jpg |
| 19000118 | 4 | 000118.jpg |
| 19000300 | 4 | 000300.jpg |
| 19000018 | 4 | 000018.jpg |
| 19000275 | 3 | 000275.jpg |
| 19000231 | 3 | 000231.jpg |
| 19000250 | 3 | 000250.jpg |
| 19000132 | 3 | 000132.jpg |
| 19000062 | 3 | 000062.jpg |

## Top FN Images

| image_id | fn_total | file_name |
| --- | --- | --- |
| 19000342 | 4 | 000342.jpg |
| 19000049 | 3 | 000049.jpg |
| 19000145 | 3 | 000145.jpg |
| 19000284 | 3 | 000284.jpg |
| 19000325 | 3 | 000325.jpg |
| 19000046 | 3 | 000046.jpg |
| 19000328 | 3 | 000328.jpg |
| 19000333 | 3 | 000333.jpg |
| 19000340 | 3 | 000340.jpg |
| 19000338 | 3 | 000338.jpg |

## Top Total Error Images (FP + FN)

| image_id | fp_total | fn_total | error_total | file_name |
| --- | --- | --- | --- | --- |
| 19000325 | 3 | 3 | 6 | 000325.jpg |
| 19000342 | 1 | 4 | 5 | 000342.jpg |
| 19000046 | 2 | 3 | 5 | 000046.jpg |
| 19000338 | 2 | 3 | 5 | 000338.jpg |
| 19000118 | 4 | 1 | 5 | 000118.jpg |
| 19000168 | 4 | 1 | 5 | 000168.jpg |
| 19000333 | 1 | 3 | 4 | 000333.jpg |
| 19000286 | 2 | 2 | 4 | 000286.jpg |
| 19000335 | 2 | 2 | 4 | 000335.jpg |
| 19000336 | 2 | 2 | 4 | 000336.jpg |
| 19000349 | 2 | 2 | 4 | 000349.jpg |
| 19000009 | 3 | 1 | 4 | 000009.jpg |
| 19000061 | 3 | 1 | 4 | 000061.jpg |
| 19000081 | 3 | 1 | 4 | 000081.jpg |
| 19000107 | 3 | 1 | 4 | 000107.jpg |

## Notes

- This error analysis is based on the current mmb_proxy evaluation path.
- Counts aggregate FP/FN over five deterministic seeds using VISO overlap protocol.
