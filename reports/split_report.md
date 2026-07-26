# M4 Split Freeze Report

- pHash: `imagehash.phash(hash_size=16)`
- Hamming threshold: `6`
- Cross-partition groups resolved to test: `0`
- Test images: `723`
- Test bad masks: `80`
- Unique blocked SHA256: `803`

## pHash calibration

| object | images | groups | largest | pairs ≤ threshold | nearest min / p05 / median / p95 / max |
|---|---:|---:|---:|---:|---|
| capsules | 702 | 702 | 1 | 0 | 34 / 84 / 92 / 96 / 102 |
| pcb1 | 1104 | 1103 | 2 | 1 | 6 / 14 / 20 / 30 / 68 |

Threshold 6 was retained: it is far below the median nearest-neighbour distance,
so it targets only near-identical captures. Connected components apply transitive closure.

## Frozen counts

| object | train good | train bad | test good | test bad | moved normal / anomaly |
|---|---:|---:|---:|---:|---|
| capsules | 361 | 60 | 241 | 40 | 0 / 0 |
| pcb1 | 602 | 60 | 402 | 40 | 0 / 0 |

## Assertions

- Every pHash group belongs to exactly one final set: **passed**
- Every final test image SHA256 is blocklisted: **passed**
- Every final test bad-mask SHA256 is blocklisted: **passed**
- Manifest checksum written after serialization: **passed**
