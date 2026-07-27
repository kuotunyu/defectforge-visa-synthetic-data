# M13 synthetic quality filter

<!-- filter-summary-sha256: 831ec0936a7e3068c30f46f5fac1240634ae0de23aa8dc3e8ad9db87dd88a8a0 -->
<!-- filter-summary-json: {"accepted":1770,"first_reason_counts":{"AREA_OUT_OF_RANGE":365,"ASPECT_OUT_OF_RANGE":56,"EMBEDDING_OUTLIER":75,"NN_TOO_LOW":697,"PHASH_DUPLICATE":37},"funnel":[{"rule":"roi","survivors":3000},{"rule":"area","survivors":2635},{"rule":"aspect","survivors":2579},{"rule":"phash","survivors":2542},{"rule":"dinov2","survivors":1770},{"rule":"seam","survivors":1770}],"reason_counts":{"AREA_OUT_OF_RANGE":365,"ASPECT_OUT_OF_RANGE":103,"EMBEDDING_OUTLIER":897,"NN_TOO_LOW":979,"PHASH_DUPLICATE":37},"rejected":1230,"rows":[{"AREA_OUT_OF_RANGE":33,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":83,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":89,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":276,"after_area":342,"after_aspect":342,"after_dinov2":276,"after_phash":342,"after_roi":375,"after_seam":276,"defect_type":"type0","generator":"stageA_copypaste","input":"stageA_copypaste","object":"capsules","total":375},{"AREA_OUT_OF_RANGE":3,"ASPECT_OUT_OF_RANGE":49,"EMBEDDING_OUTLIER":0,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":1,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":74,"after_area":122,"after_aspect":75,"after_dinov2":74,"after_phash":75,"after_roi":125,"after_seam":74,"defect_type":"type1","generator":"stageA_copypaste","input":"stageA_copypaste","object":"capsules","total":125},{"AREA_OUT_OF_RANGE":177,"ASPECT_OUT_OF_RANGE":48,"EMBEDDING_OUTLIER":163,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":169,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":84,"after_area":171,"after_aspect":162,"after_dinov2":84,"after_phash":162,"after_roi":348,"after_seam":84,"defect_type":"type0","generator":"stageA_copypaste","input":"stageA_copypaste","object":"pcb1","total":348},{"AREA_OUT_OF_RANGE":152,"ASPECT_OUT_OF_RANGE":6,"EMBEDDING_OUTLIER":150,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":151,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":0,"after_area":0,"after_aspect":0,"after_dinov2":0,"after_phash":0,"after_roi":152,"after_seam":0,"defect_type":"type1","generator":"stageA_copypaste","input":"stageA_copypaste","object":"pcb1","total":152},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":1,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":6,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":119,"after_area":125,"after_aspect":125,"after_dinov2":119,"after_phash":125,"after_roi":125,"after_seam":119,"defect_type":"crack","generator":"stageA_procedural","input":"stageA_procedural","object":"capsules","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":27,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":43,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":82,"after_area":125,"after_aspect":125,"after_dinov2":82,"after_phash":125,"after_roi":125,"after_seam":82,"defect_type":"perlin","generator":"stageA_procedural","input":"stageA_procedural","object":"capsules","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":32,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":44,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":79,"after_area":125,"after_aspect":125,"after_dinov2":79,"after_phash":125,"after_roi":125,"after_seam":79,"defect_type":"scratch","generator":"stageA_procedural","input":"stageA_procedural","object":"capsules","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":36,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":46,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":74,"after_area":125,"after_aspect":125,"after_dinov2":74,"after_phash":125,"after_roi":125,"after_seam":74,"defect_type":"spot","generator":"stageA_procedural","input":"stageA_procedural","object":"capsules","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":2,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":1,"PHASH_DUPLICATE":15,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":108,"after_area":125,"after_aspect":125,"after_dinov2":108,"after_phash":110,"after_roi":125,"after_seam":108,"defect_type":"crack","generator":"stageA_procedural","input":"stageA_procedural","object":"pcb1","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":32,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":31,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":90,"after_area":125,"after_aspect":125,"after_dinov2":90,"after_phash":125,"after_roi":125,"after_seam":90,"defect_type":"perlin","generator":"stageA_procedural","input":"stageA_procedural","object":"pcb1","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":27,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":27,"PHASH_DUPLICATE":6,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":89,"after_area":125,"after_aspect":125,"after_dinov2":89,"after_phash":119,"after_roi":125,"after_seam":89,"defect_type":"scratch","generator":"stageA_procedural","input":"stageA_procedural","object":"pcb1","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":41,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":36,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":80,"after_area":125,"after_aspect":125,"after_dinov2":80,"after_phash":125,"after_roi":125,"after_seam":80,"defect_type":"spot","generator":"stageA_procedural","input":"stageA_procedural","object":"pcb1","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":131,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":173,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":201,"after_area":375,"after_aspect":375,"after_dinov2":201,"after_phash":375,"after_roi":375,"after_seam":201,"defect_type":"type0","generator":"stageB_sd2","input":"stageB_sd2/searched","object":"capsules","total":375},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":12,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":20,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":105,"after_area":125,"after_aspect":125,"after_dinov2":105,"after_phash":125,"after_roi":125,"after_seam":105,"defect_type":"type1","generator":"stageB_sd2","input":"stageB_sd2/searched","object":"capsules","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":77,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":63,"PHASH_DUPLICATE":15,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":247,"after_area":348,"after_aspect":348,"after_dinov2":247,"after_phash":333,"after_roi":348,"after_seam":247,"defect_type":"type0","generator":"stageB_sd2","input":"stageB_sd2/searched","object":"pcb1","total":348},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":83,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":79,"PHASH_DUPLICATE":1,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":62,"after_area":152,"after_aspect":152,"after_dinov2":62,"after_phash":151,"after_roi":152,"after_seam":62,"defect_type":"type1","generator":"stageB_sd2","input":"stageB_sd2/searched","object":"pcb1","total":152}],"schema_version":1,"thresholds":{"capsules":{"area_p05":0.00010116666666666667,"area_p95":0.015780066666666665,"maximum_area_abs_zscore":2.5,"maximum_aspect_abs_zscore":2.5,"minimum_containment":1.0,"minimum_phash_distance":5.0,"minimum_seam_score":0.85,"tau_copy":0.98,"tau_low":0.6304827928543091,"tau_outlier":0.44221726059913635},"pcb1":{"area_p05":0.0012674068748835103,"area_p95":0.037032144473733195,"maximum_area_abs_zscore":2.5,"maximum_aspect_abs_zscore":2.5,"minimum_containment":1.0,"minimum_phash_distance":5.0,"minimum_seam_score":0.85,"tau_copy":0.98,"tau_low":0.7391234636306763,"tau_outlier":0.395549476146698}},"total":3000} -->

The six rules run in the locked order: ROI, area, aspect, pHash, DINOv2, then seam. Counts below are reconstructed from published `unfiltered/metadata.jsonl`; accepted samples are hardlinked into `filtered/`.

## Outcome

- Total: 3000
- Accepted: 1770
- Rejected: 1230
- Summary SHA-256: `831ec0936a7e3068c30f46f5fac1240634ae0de23aa8dc3e8ad9db87dd88a8a0`

## Funnel

| Rule | Survivors |
|---|---:|
| roi | 3000 |
| area | 2635 |
| aspect | 2579 |
| phash | 2542 |
| dinov2 | 1770 |
| seam | 1770 |

## Reject reasons

A sample may trigger multiple rules. `First rejects` classifies each rejected sample once by the locked funnel order; `All triggers` counts every triggered rule and can therefore exceed the rejected total.

| Reason | First rejects | All triggers |
|---|---:|---:|
| ROI_OVERFLOW | 0 | 0 |
| AREA_OUT_OF_RANGE | 365 | 365 |
| ASPECT_OUT_OF_RANGE | 56 | 103 |
| PHASH_DUPLICATE | 37 | 37 |
| NN_TOO_LOW | 697 | 979 |
| NN_TOO_HIGH_COPY | 0 | 0 |
| EMBEDDING_OUTLIER | 75 | 897 |
| SEAM_POOR | 0 | 0 |

## Generator and defect-type detail

| Input | Object | Generator | Type | Generated | ROI | Area | Aspect | pHash | DINOv2 | Seam | Final | Pass rate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stageA_copypaste | capsules | stageA_copypaste | type0 | 375 | 375 | 342 | 342 | 342 | 276 | 276 | 276 | 73.60% |
| stageA_copypaste | capsules | stageA_copypaste | type1 | 125 | 125 | 122 | 75 | 75 | 74 | 74 | 74 | 59.20% |
| stageA_copypaste | pcb1 | stageA_copypaste | type0 | 348 | 348 | 171 | 162 | 162 | 84 | 84 | 84 | 24.14% |
| stageA_copypaste | pcb1 | stageA_copypaste | type1 | 152 | 152 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00% |
| stageA_procedural | capsules | stageA_procedural | crack | 125 | 125 | 125 | 125 | 125 | 119 | 119 | 119 | 95.20% |
| stageA_procedural | capsules | stageA_procedural | perlin | 125 | 125 | 125 | 125 | 125 | 82 | 82 | 82 | 65.60% |
| stageA_procedural | capsules | stageA_procedural | scratch | 125 | 125 | 125 | 125 | 125 | 79 | 79 | 79 | 63.20% |
| stageA_procedural | capsules | stageA_procedural | spot | 125 | 125 | 125 | 125 | 125 | 74 | 74 | 74 | 59.20% |
| stageA_procedural | pcb1 | stageA_procedural | crack | 125 | 125 | 125 | 125 | 110 | 108 | 108 | 108 | 86.40% |
| stageA_procedural | pcb1 | stageA_procedural | perlin | 125 | 125 | 125 | 125 | 125 | 90 | 90 | 90 | 72.00% |
| stageA_procedural | pcb1 | stageA_procedural | scratch | 125 | 125 | 125 | 125 | 119 | 89 | 89 | 89 | 71.20% |
| stageA_procedural | pcb1 | stageA_procedural | spot | 125 | 125 | 125 | 125 | 125 | 80 | 80 | 80 | 64.00% |
| stageB_sd2/searched | capsules | stageB_sd2 | type0 | 375 | 375 | 375 | 375 | 375 | 201 | 201 | 201 | 53.60% |
| stageB_sd2/searched | capsules | stageB_sd2 | type1 | 125 | 125 | 125 | 125 | 125 | 105 | 105 | 105 | 84.00% |
| stageB_sd2/searched | pcb1 | stageB_sd2 | type0 | 348 | 348 | 348 | 348 | 333 | 247 | 247 | 247 | 70.98% |
| stageB_sd2/searched | pcb1 | stageB_sd2 | type1 | 152 | 152 | 152 | 152 | 151 | 62 | 62 | 62 | 40.79% |

The table above contains survivors after each rule. Exact per-group all-trigger counts remain embedded in the machine-readable summary and are checked by `scripts/verify_filter_report.py`.

## Locked thresholds

### capsules

- `area_p05`: 0.00010116666666666667
- `area_p95`: 0.015780066666666665
- `maximum_area_abs_zscore`: 2.5
- `maximum_aspect_abs_zscore`: 2.5
- `minimum_containment`: 1.0
- `minimum_phash_distance`: 5.0
- `minimum_seam_score`: 0.85
- `tau_copy`: 0.98
- `tau_low`: 0.6304827928543091
- `tau_outlier`: 0.44221726059913635

### pcb1

- `area_p05`: 0.0012674068748835103
- `area_p95`: 0.037032144473733195
- `maximum_area_abs_zscore`: 2.5
- `maximum_aspect_abs_zscore`: 2.5
- `minimum_containment`: 1.0
- `minimum_phash_distance`: 5.0
- `minimum_seam_score`: 0.85
- `tau_copy`: 0.98
- `tau_low`: 0.7391234636306763
- `tau_outlier`: 0.395549476146698

## Visual audit

The accepted and rejected sheets are deterministic, evenly spaced samples from their respective populations. They are audit views, not hand-picked examples.

- `reports/figures/filter_accepted.png`
- `reports/figures/filter_rejected.png`
