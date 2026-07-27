# SDXL searched visual review

This report records human visual inspection separately from the automatic
integrity validator. A passed validator establishes provenance, uniqueness,
mask alignment, frozen-input hashes, blocklist exclusion, and non-regression
under the preregistered refine score. It does not establish semantic realism.

## PCB (`pcb1`)

- Evidence: [`figures/sdxl_searched_pcb1.png`](figures/sdxl_searched_pcb1.png)
- Automatic validation:
  [`sdxl_searched_pcb1_validation.json`](sdxl_searched_pcb1_validation.json)
- Review status: mixed quality; preserve as a formal negative result.

The contact sheet is legible, the masks align with the edited regions, and
several outputs resemble plausible local burns, stains, scratches, or
discolouration. However, multiple samples hallucinate beetles, insects, or
small integrated-circuit-like components. The first type-0 sample is an
especially clear insect-shaped failure. These objects are locally blended but
are not faithful reconstructions of the underlying VisA defect semantics.

The run is intentionally retained without post-hoc prompt, ranker, or sample
selection changes. M16 evaluates the locked 250-image bucket as registered;
M17 then tests whether its automatic quality scores predict downstream
classifier utility. The semantic failures must also remain visible in the
README limitations and publication cards.

## Capsules (`capsules`)

- Evidence:
  [`figures/sdxl_searched_capsules.png`](figures/sdxl_searched_capsules.png)
- Automatic validation:
  [`sdxl_searched_capsules_validation.json`](sdxl_searched_capsules_validation.json)
- Review status: mixed quality with systematic type-0 semantic failure; preserve
  as a formal negative result.

Several type-1 samples resemble plausible corrosion, surface roughness,
discolouration, or coating loss. In contrast, type-0 repeatedly produces
purple jewellery-, button-, lens-, or mechanical-ring-like objects. The
pattern occurs across multiple source images, mask sizes, crop ratios, and
guidance values in the contact sheet, so it is not an isolated outlier.

The automatic refine objective improves boundary and local-change scores but
does not distinguish an industrial surface defect from a well-blended,
semantically unrelated circular object. The full 250-image bucket therefore
remains unchanged for the registered SDXL ablation and downstream utility
measurement.

## Cross-object conclusion

Both objects pass provenance, uniqueness, blocklist, mask-alignment, and
monotonic-refine checks. Both also expose systematic semantic hallucinations
that those checks do not measure. SDXL searched data must be described as a
technically valid but visually unreliable baseline unless downstream M16
results demonstrate utility; even a positive downstream result would not
remove the documented semantic limitation.
