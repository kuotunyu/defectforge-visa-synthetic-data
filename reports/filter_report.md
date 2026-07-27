# M13 synthetic quality filter

<!-- filter-summary-sha256: d306207ef5186bfff29aae0222ee1acd4eba17d7b3c48511c356ad52c920efb1 -->
<!-- filter-summary-json: {"accepted":1770,"funnel":[{"rule":"roi","survivors":3000},{"rule":"area","survivors":2635},{"rule":"aspect","survivors":2579},{"rule":"phash","survivors":2542},{"rule":"dinov2","survivors":1770},{"rule":"seam","survivors":1770}],"reason_counts":{"AREA_OUT_OF_RANGE":365,"ASPECT_OUT_OF_RANGE":103,"EMBEDDING_OUTLIER":897,"NN_TOO_LOW":979,"PHASH_DUPLICATE":37},"rejected":1230,"rows":[{"AREA_OUT_OF_RANGE":33,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":83,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":89,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":276,"defect_type":"type0","generator":"stageA_copypaste","input":"stageA_copypaste","object":"capsules","total":375},{"AREA_OUT_OF_RANGE":3,"ASPECT_OUT_OF_RANGE":49,"EMBEDDING_OUTLIER":0,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":1,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":74,"defect_type":"type1","generator":"stageA_copypaste","input":"stageA_copypaste","object":"capsules","total":125},{"AREA_OUT_OF_RANGE":177,"ASPECT_OUT_OF_RANGE":48,"EMBEDDING_OUTLIER":163,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":169,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":84,"defect_type":"type0","generator":"stageA_copypaste","input":"stageA_copypaste","object":"pcb1","total":348},{"AREA_OUT_OF_RANGE":152,"ASPECT_OUT_OF_RANGE":6,"EMBEDDING_OUTLIER":150,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":151,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":0,"defect_type":"type1","generator":"stageA_copypaste","input":"stageA_copypaste","object":"pcb1","total":152},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":1,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":6,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":119,"defect_type":"crack","generator":"stageA_procedural","input":"stageA_procedural","object":"capsules","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":27,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":43,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":82,"defect_type":"perlin","generator":"stageA_procedural","input":"stageA_procedural","object":"capsules","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":32,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":44,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":79,"defect_type":"scratch","generator":"stageA_procedural","input":"stageA_procedural","object":"capsules","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":36,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":46,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":74,"defect_type":"spot","generator":"stageA_procedural","input":"stageA_procedural","object":"capsules","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":2,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":1,"PHASH_DUPLICATE":15,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":108,"defect_type":"crack","generator":"stageA_procedural","input":"stageA_procedural","object":"pcb1","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":32,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":31,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":90,"defect_type":"perlin","generator":"stageA_procedural","input":"stageA_procedural","object":"pcb1","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":27,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":27,"PHASH_DUPLICATE":6,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":89,"defect_type":"scratch","generator":"stageA_procedural","input":"stageA_procedural","object":"pcb1","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":41,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":36,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":80,"defect_type":"spot","generator":"stageA_procedural","input":"stageA_procedural","object":"pcb1","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":131,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":173,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":201,"defect_type":"type0","generator":"stageB_sd2","input":"stageB_sd2/searched","object":"capsules","total":375},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":12,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":20,"PHASH_DUPLICATE":0,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":105,"defect_type":"type1","generator":"stageB_sd2","input":"stageB_sd2/searched","object":"capsules","total":125},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":77,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":63,"PHASH_DUPLICATE":15,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":247,"defect_type":"type0","generator":"stageB_sd2","input":"stageB_sd2/searched","object":"pcb1","total":348},{"AREA_OUT_OF_RANGE":0,"ASPECT_OUT_OF_RANGE":0,"EMBEDDING_OUTLIER":83,"NN_TOO_HIGH_COPY":0,"NN_TOO_LOW":79,"PHASH_DUPLICATE":1,"ROI_OVERFLOW":0,"SEAM_POOR":0,"accepted":62,"defect_type":"type1","generator":"stageB_sd2","input":"stageB_sd2/searched","object":"pcb1","total":152}],"schema_version":1,"thresholds":{"capsules":{"area_p05":0.00010116666666666667,"area_p95":0.015780066666666665,"maximum_area_abs_zscore":2.5,"maximum_aspect_abs_zscore":2.5,"minimum_containment":1.0,"minimum_phash_distance":5.0,"minimum_seam_score":0.85,"tau_copy":0.98,"tau_low":0.6304827928543091,"tau_outlier":0.44221726059913635},"pcb1":{"area_p05":0.0012674068748835103,"area_p95":0.037032144473733195,"maximum_area_abs_zscore":2.5,"maximum_aspect_abs_zscore":2.5,"minimum_containment":1.0,"minimum_phash_distance":5.0,"minimum_seam_score":0.85,"tau_copy":0.98,"tau_low":0.7391234636306763,"tau_outlier":0.395549476146698}},"total":3000} -->

The six rules run in the locked order: ROI, area, aspect, pHash, DINOv2, then seam. Counts below are reconstructed from published `unfiltered/metadata.jsonl`; accepted samples are hardlinked into `filtered/`.

## Outcome

- Total: 3000
- Accepted: 1770
- Rejected: 1230
- Summary SHA-256: `d306207ef5186bfff29aae0222ee1acd4eba17d7b3c48511c356ad52c920efb1`

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

| Reason | Count |
|---|---:|
| ROI_OVERFLOW | 0 |
| AREA_OUT_OF_RANGE | 365 |
| ASPECT_OUT_OF_RANGE | 103 |
| PHASH_DUPLICATE | 37 |
| NN_TOO_LOW | 979 |
| NN_TOO_HIGH_COPY | 0 |
| EMBEDDING_OUTLIER | 897 |
| SEAM_POOR | 0 |

## Generator and defect-type detail

| Input | Object | Generator | Type | Total | Accepted | ROI | Area | Aspect | pHash | NN low | NN copy | Outlier | Seam |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stageA_copypaste | capsules | stageA_copypaste | type0 | 375 | 276 | 0 | 33 | 0 | 0 | 89 | 0 | 83 | 0 |
| stageA_copypaste | capsules | stageA_copypaste | type1 | 125 | 74 | 0 | 3 | 49 | 0 | 1 | 0 | 0 | 0 |
| stageA_copypaste | pcb1 | stageA_copypaste | type0 | 348 | 84 | 0 | 177 | 48 | 0 | 169 | 0 | 163 | 0 |
| stageA_copypaste | pcb1 | stageA_copypaste | type1 | 152 | 0 | 0 | 152 | 6 | 0 | 151 | 0 | 150 | 0 |
| stageA_procedural | capsules | stageA_procedural | crack | 125 | 119 | 0 | 0 | 0 | 0 | 6 | 0 | 1 | 0 |
| stageA_procedural | capsules | stageA_procedural | perlin | 125 | 82 | 0 | 0 | 0 | 0 | 43 | 0 | 27 | 0 |
| stageA_procedural | capsules | stageA_procedural | scratch | 125 | 79 | 0 | 0 | 0 | 0 | 44 | 0 | 32 | 0 |
| stageA_procedural | capsules | stageA_procedural | spot | 125 | 74 | 0 | 0 | 0 | 0 | 46 | 0 | 36 | 0 |
| stageA_procedural | pcb1 | stageA_procedural | crack | 125 | 108 | 0 | 0 | 0 | 15 | 1 | 0 | 2 | 0 |
| stageA_procedural | pcb1 | stageA_procedural | perlin | 125 | 90 | 0 | 0 | 0 | 0 | 31 | 0 | 32 | 0 |
| stageA_procedural | pcb1 | stageA_procedural | scratch | 125 | 89 | 0 | 0 | 0 | 6 | 27 | 0 | 27 | 0 |
| stageA_procedural | pcb1 | stageA_procedural | spot | 125 | 80 | 0 | 0 | 0 | 0 | 36 | 0 | 41 | 0 |
| stageB_sd2/searched | capsules | stageB_sd2 | type0 | 375 | 201 | 0 | 0 | 0 | 0 | 173 | 0 | 131 | 0 |
| stageB_sd2/searched | capsules | stageB_sd2 | type1 | 125 | 105 | 0 | 0 | 0 | 0 | 20 | 0 | 12 | 0 |
| stageB_sd2/searched | pcb1 | stageB_sd2 | type0 | 348 | 247 | 0 | 0 | 0 | 15 | 63 | 0 | 77 | 0 |
| stageB_sd2/searched | pcb1 | stageB_sd2 | type1 | 152 | 62 | 0 | 0 | 0 | 1 | 79 | 0 | 83 | 0 |

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
