# DefectForge Public Repository Acceptance

> Read-only local audit. This command does not publish, upload, or change visibility.

## Passed

- [x] `closeout_metadata_valid`
- [x] `demo_gif_valid`
- [x] `final_evidence_hashes_current`
- [x] `final_evidence_reports_passed`
- [x] `final_figures_valid`
- [x] `github_social_preview_valid`
- [x] `hf_upload_plan_safe_dry_run`
- [x] `model_license_verification_fresh`
- [x] `no_coauthor_trailers`
- [x] `no_oversized_tracked_files`
- [x] `no_secret_or_personal_path`
- [x] `no_tracked_env`
- [x] `owner_local_files_untracked`
- [x] `readme_final`
- [x] `single_git_identity`
- [x] Classification CSV SHA256 `aa083aa7524cea80d211efa911616ea71b4888049deced5678441ee8f05cbce6`
- [x] Segmentation CSV SHA256 `fbe4f97379e2597713c1c7255bfd6b5df7ac49c948b396469f598d34b03615aa`

## Fixed

- [x] Frozen evidence bytes survive Windows, Linux, clone, and GitHub Source ZIP.
- [x] M18 source packaging works without `.git` and excludes outputs and secrets.
- [x] Owner-only workflow files are excluded from the public Git tree.
- [x] Final reports, figures, README, demo, licenses, and HF dry-run remain hash-bound.

## Residual risks

- [x] Scope is limited to VisA `pcb1` and `capsules` with ten defect seeds each.
- [x] Pseudo-types are not official VisA per-image defect labels.
- [x] Negative or null synthetic-data outcomes are retained in README Limitations.
- [x] Public Demo remains a research/evaluation tool rather than production AOI.
