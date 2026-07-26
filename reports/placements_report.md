# M9 Mask Placement Report

- ROI method: `intersect`
- DINOv2 revision: `f9e44c814b77203eaa57a6bdbbd535f21ede1415`
- Every mask is binary, non-empty, inside legal ROI, within frozen real-stat bounds,
  test-blocklist clean, and non-overlapping with sibling variants on its background.

| object | records | backgrounds | type quotas | failed transform/place attempts |
|---|---:|---:|---|---:|
| pcb1 | 1806 | 602 | type0=1256, type1=550 | 1912 |
| capsules | 1083 | 361 | type0=812, type1=271 | 1323 |
