# Contact-audio simulation experiment

**Run date:** 2026-08-11  
**Design:** ticket 04 plus Amendment 1, executed without post-hoc design changes  
**Outcome:** the full probe fails the pre-registered 181D sanity and mid-cell tracking criteria; the negative environment and ablation findings are still usable.

## Executive finding

The synthetic probe is not validated as a four-family, environment-robust instrument. In the clean reference, the frozen 181D endpoint classifier separated L1 from L5 perfectly for every family, but sharpness had no individual feature with \(|\rho| \ge 0.9\). At the 10 m/15 dB mid cell, only intensity retained such a feature; location, decay, and sharpness did not. M2D passed clean-reference sanity for all four families, but it also retained strong mid-cell tracking only for intensity.

The locked quantitative environment finding is:

> **At 10 m/15 dB, the location-family time-domain group shifted by 322.06 within-class reference SD and its 181D endpoint balanced accuracy collapsed from 1.00 to 0.51.**

The literal pre-registered drift statistic is ill-conditioned for some groups: `spectral_stats`, `frame_energy`, and `local_transient` have infinite group drift in every non-reference family/cell row because at least one constituent feature has zero reference SD but a non-zero environment shift. No epsilon or post-hoc feature exclusion was introduced.

## Pre-registered criterion verdicts

| Criterion | Verdict | Result |
|---|---:|---|
| 1. Reference sanity: endpoint BA ≥ 0.85 **and** at least one 181D feature with \(|\rho| \ge 0.9\) per family | **FAIL** | BA was 1.00 for all four. Intensity, location, and decay had 132, 3, and 139 strong features; sharpness had 0 (best \(\rho=-0.816\)). |
| 2. Mid-cell tracking: at least 3/4 families retain \(|\rho| \ge 0.9\) | **FAIL** | 1/4 retained tracking. Intensity had 57 strong features; location, decay, and sharpness had 0. |
| 3. At least one locked-form quantitative environment finding | **PASS** | At 10 m/15 dB, location time-domain drift was 322.06 reference SD and endpoint BA fell 1.00→0.51. |
| 4. Name contact-carrying versus environment-driven groups from ablation | **PASS** | `logmel` was the strongest contact-carrying 181D group. `spectral_stats`, `frame_energy`, and `local_transient` were the most environment-driven under the literal statistic (infinite); among finite groups, `time_domain` was largest. |

Per the pre-registered failure path, the full four-family probe should be narrowed or stopped rather than presented as validated. The robust part is intensity; the transfer failure for the other families is the main reportable result.

## Experiment execution

The run produced 1,600 mono PCM16 WAVs at 16 kHz:

- 100 deterministic bases: 4 families × 5 levels × 5 random-realization variants;
- 800 environment WAVs: 100 bases × the locked 8-cell grid;
- 700 degradation WAVs: 100 mid-cell inputs × all 7 listed variants, including a separately materialized `none` variant.

The 181D extractor processed all 1,600 files. M2D processed the 900-file locked subset: reference, the environment mid cell, and all seven materialized mid-cell degradation variants. Truncated files use distinct `window_name` values and fixed lengths within each M2D batch group.

Endpoint BA used the repository's frozen `build_model` (`StandardScaler` plus balanced L2 logistic regression, `C=0.1`, `max_iter=5000`) with repeated stratified 5-fold CV, 10 repeats, `random_state=42`. Each endpoint problem had five L1 and five L5 variants. Tracking used all 25 family/cell samples.

The 181D allocation was verified as 80 logmel, 44 spectral statistics, 26 MFCC, 11 time-domain, 8 frame-energy, 8 band-ratio, and 4 local-transient features. The locked counts uniquely place 5/10 ms spectral statistics in `spectral_stats`, the unqualified plus 10 ms energy statistics in `frame_energy`, and 5 ms energy statistics in `local_transient`.

For a zero reference SD, the literal drift ratio was evaluated as zero for zero shift and infinity for a non-zero shift. This retains all 181 locked features and avoids adding an unregistered epsilon.

## Tracking and endpoint results

### Frozen 181D

| Cell | Family | Best \(\rho\) | Count \(|\rho|\ge0.9\) | Endpoint BA |
|---|---|---:|---:|---:|
| reference | intensity | 0.981 | 132 | 1.00 |
| reference | location | 0.910 | 3 | 1.00 |
| reference | decay | 0.984 | 139 | 1.00 |
| reference | sharpness | -0.816 | 0 | 1.00 |
| 10 m/15 dB | intensity | 0.961 | 57 | 1.00 |
| 10 m/15 dB | location | -0.616 | 0 | 0.51 |
| 10 m/15 dB | decay | 0.647 | 0 | 0.67 |
| 10 m/15 dB | sharpness | 0.655 | 0 | 0.44 |

The clean sharpness result is a useful distinction: its endpoints are multivariately separable, but no frozen scalar feature monotonically tracks all five physical levels strongly enough to satisfy the criterion.

Across the environment grid, intensity endpoint BA stayed at 0.98–1.00. The other families dropped below the 0.85 threshold in every noisy cell. Their curves were not monotonic across distance/SNR cells: independent, fixed noise realizations and the ten-sample endpoint problems produce cell-level variation, but none restores pre-registered tracking.

### M2D

M2D clean-reference endpoint BA was 1.00 for every family. Its best reference \(\rho\) values were 0.981 (intensity), 0.957 (location), -0.981 (decay), and 0.973 (sharpness), with 404, 59, 463, and 68 strong dimensions respectively.

At 10 m/15 dB, M2D endpoint BA was 1.00, 0.35, 0.84, and 0.59 for intensity, location, decay, and sharpness. Best \(|\rho|\) was 0.961, 0.726, 0.718, and 0.722 respectively, so only intensity retained strong monotonic tracking. M2D therefore fixes the clean sharpness sanity failure but not the mid-cell transfer failure.

## Environment fingerprint

The finite group means, averaged across the 28 non-reference family/cell combinations, were:

| Feature group | Mean drift in reference SD units |
|---|---:|
| time_domain | 250.55 |
| logmel | 71.71 |
| mfcc | 12.30 |
| band_ratios | 1.03 |
| spectral_stats | infinite |
| frame_energy | infinite |
| local_transient | infinite |

The infinite values are direct outcomes of the locked formula, not overflow. For example, reference p10/ p90 statistics can be identical across all 25 variants, while added noise shifts them. Consequently, no finite rank should be claimed among those three groups. Among groups with a finite mean, `time_domain` is the clearest environment fingerprint.

At the 10 m/15 dB mid cell, time-domain drift was 305.16, 322.06, 281.54, and 187.32 SD for intensity, location, decay, and sharpness. Corresponding endpoint BA changed from the universal reference value of 1.00 to 1.00, 0.51, 0.67, and 0.44.

## LOGO/SGI and M2D pooling blocks

Across reference plus the environment mid cell, `logmel` was the strongest contact-carrying 181D group:

- logmel-only mean endpoint BA: **0.848**, the best SGI result;
- mean absolute best-feature \(\rho\): **0.781**;
- removing logmel caused the largest mean BA loss: **-0.029**.

Frame energy (SGI BA 0.794) and local transient (0.791) were the next-best single groups, but both are also maximally environment-driven under the locked drift statistic. This is a confound, not clean evidence of contact specificity. `time_domain` is likewise highly environment-driven despite carrying some endpoint signal. The ablation therefore names logmel as the leading contact carrier while showing that none of the useful groups is environment-invariant.

For M2D pooling-block SGI over its full subset, mean endpoint BA was 0.770 for `std`, 0.765 for `max`, and 0.694 for `mean`. Every block alone achieved 1.00 reference BA for every family. At the mid cell, the `std` block was strongest for decay (0.86) and the `max` block for location/sharpness (0.54/0.66), but none recovered their monotonic tracking criterion.

## Collection-degradation findings

The two hard-clipping variants were exact no-ops: all mid-cell WAV peaks were at most 0.1597, below both absolute ceilings, so the 0.5 and 0.25 outputs were byte-identical to `none` for all 100 signals. Their 181D and M2D results are therefore identical to the mid baseline. This is a legitimate negative result of applying the locked ceilings after physical distance attenuation.

MP3-128k, AAC-64k, and truncation changed endpoint BA but did not change the central tracking verdict: for both instruments, intensity was the only family with any \(|\rho|\ge0.9\) feature in every mid degradation variant. The full per-condition values are in the tracking table.

## Handoff-ready confound hypotheses

- Intensity survives because it is encoded in broad absolute-level and spectral-energy changes; the other three physical controls rely on subtler shape or tail cues that the locked pink-noise floor masks.
- Logmel carries the most endpoint information but also moves strongly with environment, so a real-clip classifier using this group could learn collection SNR or distance rather than contact physics.
- Frame/local energy and spectral summary groups cannot be given a stable finite environment rank under the registered SD normalization; their zero reference variance itself is a collection-condition diagnostic.

These are hypotheses for the separate audio-research-recovery effort, not decisions for that map.

## Artifacts

### Three analysis tables

1. [Tracking and endpoint table](../../outputs/contact_synth_probe_full/tracking_endpoint_table.csv) — 181D all 15 conditions; M2D its 9-condition subset.
2. [Environment fingerprint table](../../outputs/contact_synth_probe_full/environment_fingerprint_table.csv) — family × 8 environment cells × 7 feature groups, with BA curve values.
3. [Ablation table](../../outputs/contact_synth_probe_full/ablation_table.csv) — 181D LOGO/SGI plus M2D mean/std/max pooling LOGO/SGI.

### Two figures

- [Best-feature rho heatmap](../../outputs/contact_synth_probe_full/tracking_rho_heatmap.png)
- [BA-vs-severity curves](../../outputs/contact_synth_probe_full/ba_vs_severity.png)

### Reproducibility data

- [Full manifest](../../outputs/contact_synth_probe_full/manifest.csv)
- [M2D subset manifest](../../outputs/contact_synth_probe_full/manifest_m2d.csv)
- [181D extracted features](../../outputs/contact_synth_probe_full/traditional_features.csv)
- [M2D extracted features](../../outputs/contact_synth_probe_full/m2d_features.csv)
- [Machine-readable summary](../../outputs/contact_synth_probe_full/analysis_summary.json)
- [Generation metadata](../../outputs/contact_synth_probe_full/generation_metadata.json)

Generation and analysis code:

- `scripts/audio_pipeline/prototypes/contact_synth_probe/synthesize_probe.py`
- `scripts/audio_pipeline/prototypes/contact_synth_probe/analyze_full_probe.py`

## Iteration 2 (protocol-fixed)

**Run date:** 2026-08-11  
**Design:** ticket 04 Amendment 2; every non-amended synthesis, grid, degradation, extraction, and analysis setting is unchanged  
**Outcome:** the sharpness re-parameterization repairs reference sanity, but environment transfer still fails because only intensity retains strong tracking at 10 m/15 dB.

Iteration 2 regenerated the complete 1,600-WAV probe in a new directory: 100 bases, 800 environment signals, and 700 mid-cell degradation signals. The frozen 181D extractor again processed all 1,600 files, and M2D processed the locked 900-file subset on CPU. Iteration-1 artifacts were not modified. For the sharpness excitation, the implementation applies the one-pole magnitude response \(f/\sqrt{f^2+f_c^2}\), with \(f_c=1/\tau\), to the finite click-noise realization before its decay envelope; the 10 kHz corner at \(\tau=0.1\) ms is retained rather than clamped to the 8 kHz Nyquist frequency.

### Iteration-2 criterion verdicts

| Criterion | Verdict | Iteration-2 result |
|---|---:|---|
| 1. Reference sanity: endpoint BA ≥ 0.85 **and** at least one 181D feature with \(|\rho| \ge 0.9\) per family | **PASS** | BA was 1.00 for all four families. Intensity, location, decay, and sharpness had 132, 3, 139, and 19 strong features. Sharpness's best \(\rho\) improved from -0.816 to **-0.969**. |
| 2. Mid-cell tracking: at least 3/4 families retain \(|\rho| \ge 0.9\) | **FAIL** | Still **1/4**: intensity had 57 strong features; location, decay, and sharpness had 0. Their 181D endpoint BAs were 1.00, 0.51, 0.67, and 0.68 respectively. |
| 3. At least one locked-form quantitative environment finding | **PASS** | At 10 m/15 dB, location-family `time_domain` drift was **322.05** under the amended denominator and endpoint BA collapsed **1.00→0.51**. |
| 4. Name contact-carrying versus environment-driven groups | **PASS** | `logmel` remains the leading contact-carrying group (SGI BA 0.883; removing it changes BA by -0.035). `spectral_stats` remains infinite because three all-zero reference features shift; `frame_energy` and `local_transient` have the largest finite epsilon-dominated drift, while `time_domain` is the largest well-conditioned fingerprint. |

The pre-registered failure path is therefore now “sanity passes but transfer collapses”: the result is a legitimate negative environment-transfer finding, not validation of a four-family robust probe.

### Sharpness and M2D

The revised sharpness family now has a clean scalar monotonic signature. Its strongest 181D feature is `feat_band_0_1000_ratio` (\(\rho=-0.969\)); logmel, spectral-statistic, MFCC, and band-ratio SGI sets each contain at least one strong clean-reference feature. At the mid cell, however, its best 181D \(|\rho|\) is only 0.667 despite endpoint BA improving from iteration 1's 0.44 to 0.68.

M2D tells the same transfer story. Sharpness has 204 strong reference dimensions, best \(\rho=-0.981\), and BA 1.00. At 10 m/15 dB, its BA improves from iteration 1's 0.59 to 0.90, but best \(|\rho|\) is 0.832 and no dimension reaches 0.9. Across all four families, M2D also retains strong mid-cell tracking only for intensity.

### Relative clipping

Both relative ceilings engaged for every one of the 100 mid-cell signals; all 200 clipping WAVs differ from their `none` inputs. Their output peaks are, up to PCM16 quantization, exactly 0.5× and 0.25× each input peak.

Clipping changes endpoint separation but not the tracking verdict:

| Instrument | Family | `none` BA | 0.5× peak BA | 0.25× peak BA |
|---|---|---:|---:|---:|
| 181D | intensity | 1.00 | 1.00 | 0.90 |
| 181D | location | 0.51 | 0.52 | 0.54 |
| 181D | decay | 0.67 | 0.68 | 0.70 |
| 181D | sharpness | 0.68 | 0.78 | 0.70 |
| M2D | intensity | 1.00 | 1.00 | 1.00 |
| M2D | location | 0.35 | 0.29 | 0.19 |
| M2D | decay | 0.84 | 0.67 | 0.60 |
| M2D | sharpness | 0.90 | 0.92 | 0.64 |

For both instruments and both clipping strengths, intensity remains the only family with any \(|\rho|\ge0.9\) feature. Relative clipping is therefore a material collection degradation, unlike the iteration-1 no-op, but it does not rescue or spuriously broaden monotonic tracking.

### Epsilon-stabilized environment fingerprint

The amended denominator makes `frame_energy` and `local_transient` finite, but it does not make every locked feature finite. `feat_spectral_flux_p10`, `feat_5ms_spectral_flux_p10`, and `feat_10ms_spectral_flux_p10` are exactly zero over each family's 25 references, so both their reference SD and \(\epsilon_x\) are zero; added noise shifts them, producing infinite drift without a post-hoc exclusion. Consequently, all 28 non-reference `spectral_stats` group rows remain infinite.

The mean non-reference group drifts are:

| Feature group | Mean amended drift |
|---|---:|
| spectral_stats | infinite |
| frame_energy | \(6.21\times10^{13}\) |
| local_transient | \(1.25\times10^{13}\) |
| time_domain | 228.44 |
| logmel | 57.20 |
| mfcc | 10.28 |
| band_ratios | 0.91 |

The very large finite energy-group values arise when a feature has zero sample SD but a non-zero reference scale, leaving only \(10^{-6}\max|x|\) in the denominator. They are protocol outcomes, not physically meaningful effect-size magnitudes. `time_domain` remains the clearest finite, non-degenerate environment fingerprint.

### Changed handoff hypotheses

- Sharpness can be made monotonic in clean frozen features by coupling click duration to spectral tilt, but stadium-style noise still masks that cue. This strengthens the hypothesis that shape-based contact parameters require explicit noise robustness rather than a wider clean synthesis range alone.
- Relative clipping is now a real nonlinear degradation. It preserves intensity tracking but can materially alter endpoint separation, especially M2D location/decay/sharpness at the 0.25× ceiling; real-clip studies should treat clipping severity as a collection property.
- `logmel` remains the strongest contact-carrying 181D group and also drifts substantially with environment. The collection-SNR/distance confound hypothesis is unchanged and strengthened by surviving the protocol fixes.
- Near-constant energy and flux statistics can dominate SD-normalized fingerprints even with a scale-relative epsilon. Their huge or infinite values should be handed off as a variance-floor diagnostic, not as evidence that those groups carry the most physical contact information.

These remain hypotheses for the separate audio-research-recovery effort, not decisions for that map.

### Iteration-2 artifacts

Three analysis tables:

1. [Tracking and endpoint table](../../outputs/contact_synth_probe_iter2/tracking_endpoint_table.csv)
2. [Environment fingerprint table](../../outputs/contact_synth_probe_iter2/environment_fingerprint_table.csv)
3. [Ablation table](../../outputs/contact_synth_probe_iter2/ablation_table.csv)

Two figures:

- [Best-feature rho heatmap](../../outputs/contact_synth_probe_iter2/tracking_rho_heatmap.png)
- [BA-vs-severity curves](../../outputs/contact_synth_probe_iter2/ba_vs_severity.png)

Reproducibility data:

- [Full manifest](../../outputs/contact_synth_probe_iter2/manifest.csv)
- [M2D subset manifest](../../outputs/contact_synth_probe_iter2/manifest_m2d.csv)
- [181D extracted features](../../outputs/contact_synth_probe_iter2/traditional_features.csv)
- [M2D extracted features](../../outputs/contact_synth_probe_iter2/m2d_features.csv)
- [Machine-readable summary](../../outputs/contact_synth_probe_iter2/analysis_summary.json)
- [Generation metadata](../../outputs/contact_synth_probe_iter2/generation_metadata.json)

The same generation and analysis scripts now accept `--protocol iteration2`; their default remains `iteration1` so the first run stays reproducible.
