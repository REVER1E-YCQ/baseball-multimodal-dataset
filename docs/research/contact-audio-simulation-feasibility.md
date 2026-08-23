# Feasibility of low-cost sound simulation for bat–ball contact audio

Date: 2026-08-05
Scope: feasibility research for the `contact-audio-simulation` map — synthesis approaches, free room-acoustics tooling, auralization fidelity, precedents, a minimal credible experiment, and a repo-level check of local measuring instruments.
Method: primary-source research (official docs, papers, source code, package metadata) with live verification of every cited claim; local verification of the repo's extraction scripts and frozen feature artifacts. No purchases, hardware, players, or git workflow are assumed.
Ticket: [01-feasibility-of-low-cost-sound-simulation](../.scratch/contact-audio-simulation/issues/01-feasibility-of-low-cost-sound-simulation.md)

## Executive conclusion

A low-cost, engineering-flavored sound-simulation contribution is **feasible within every stated constraint**, and the credible minimal experiment is a **synthetic contact probe with controlled environment transfer**: generate short bat–ball-like impact transients by modal synthesis in numpy, auralize them through a small room/environment grid with `pyroomacoustics` (pip-installable; ISM + ray tracing; no GPU, no purchase), apply collection-style degradations (distance, reverb, clipping, MP3/AAC via the `ffmpeg` already in the frozen env), and measure the effect with the repo's **frozen 181D traditional feature extractor, which is pure numpy/scipy and runs headlessly today**. The frozen M2D 40 ms extractor is also headless but additionally requires a torch-equipped environment (the README's `vector` env) and fixed-length WAV batches.

The three decisions the map was waiting on:

1. **Synthesis**: modal synthesis (a bank of damped sinusoids plus a short noise/click excitation) is the standard, well-precedented, numpy-trivial approach for rigid-body impacts.[^vdp98][^adrien91][^cook02][^krotkov96] Physically motivated parameter ranges exist for baseball bats (modal analyses, crack spectra, sub-1 ms collision).[^collier90][^collier01][^russell04][^jaramillo03][^zhang22] Non-linear contact-force physical modeling is a documented but optional refinement.[^avanzini01] Wave-based (finite-difference) synthesis is overkill and beyond a beginner's reach.[^bilbao09]
2. **Environment simulation**: `pyroomacoustics` (0.10.1, deps Cython/numpy/scipy) is the right default: shoebox/polyhedral rooms, Sabine-derived RT60, octave-band and air-absorption options, microphone distance and directivity, direct-sound-only free field via `max_order=0`, and ~10-line canonical usage.[^pra-docs][^pra-paper] `gpuRIR` is a faster ISM-only alternative but builds from source and needs the NVIDIA CUDA toolkit (AGPL-3.0), so it is not the beginner default.[^gpurir-repo][^gpurir-paper] No mainstream free wave-based room-acoustics library exists at beginner level; wave-based is unnecessary at feature level.[^vorlander]
3. **Fidelity**: the experiment needs *feature-level plausibility*, not perceptual auralization quality. Direct-path 1/r attenuation, air absorption, RT60, and direct-to-reverberant ratio are the effects that move the features the project measures (onset sharpness, decay, band ratios, level); ISM with a few reflection orders captures these.[^allen79][^borish84][^traer16] Compression and clipping are applied as real transforms (`ffmpeg` codecs; hard clipping), following the degradation-robustness precedent.[^adt-paper][^adt-py]

The minimal credible experiment and its success criterion are specified in §5.

## 1. Impact/contact sound synthesis

### 1.1 The standard approaches and their cost with numpy/scipy

| Approach | Idea | Primary sources | Difficulty with numpy/scipy | Verdict for this project |
| --- | --- | --- | --- | --- |
| Sample-based | Play/concatenate/transform recorded impact samples | Standard sampling practice (textbook treatment in Cook)[^cook02] | Trivial, but requires recordings | Not usable as the primary method: zero-budget constraint forbids hardware capture, and there is no licensed corpus of bat–ball impacts in the repo. |
| Impulse-based | A noise burst or click excitation passed through one or more resonant filters (the "infinite impulse response of the object") | Classic digital synthesis treatment[^cook02] | Trivial (scipy.signal filters or explicit exponential-decay resonators) | Good as the excitation stage of modal synthesis; weak alone because the mode structure is implicit. |
| **Modal synthesis** | Sum of exponentially decaying sinusoids, one per vibration mode: `s(t) = Σ Aₖ exp(−αₖ t) sin(2π fₖ t + φₖ)`, plus an excitation transient | The canonical method for impact sounds of rigid objects[^adrien91][^vdp98][^cook02]; material-dependent decay shown for impacts[^krotkov96][^klatzky00]; used as the sound engine of interactive animation[^foley01] | A few vectorized numpy lines; mode parameters (frequency, damping, amplitude) are directly controllable | **The recommended method.** Parameters map one-to-one onto features the project's 181D set measures (attack/decay, band ratios, modal statistics). |
| Physical modeling (non-linear contact force) | Collision excitation modeled as a non-linear (Hertz-like) contact force driving a resonator bank; identification of excitation parameters | Avanzini & Rocchesso[^avanzini01]; modal-excitation separation in interactive engines[^foley01] | Moderate: an ODE/RK step for the contact force plus the resonator bank; more code, more tuning | Documented optional refinement — a nicer physical story, but the feature-level experiments in §5 do not need it. |
| Wave-based (finite-difference, FEM) | Solve the wave equation numerically for the object's geometry | Numerical synthesis monograph[^bilbao09]; precomputed wave simulation for scenes[^raghuvanshi10] | High: meshes, stability conditions, memory; no beginner-level maintained free Python package for this purpose | Overkill: the questions here are about environment/collection transforms, not about sub-millisecond collision mechanics. |

### 1.2 What a bat–ball impact actually is (parameter ranges for synthesis)

Primary sources give concrete, citable numbers to ground the synthetic transients:

- The collision itself lasts well under 1 ms for baseball; the radiated "crack" is the impulsive excitation plus the bat's vibrational response and its decay after separation.[^russell04]
- The bat's sound is dominated by its vibration modes; modal analyses of wooden bats and the "sweet spot / sweet zone" literature quantify mode frequencies and node locations.[^jaramillo03][^brody86] Vibration-based work reports that time-domain peak amplitude and modal-frequency peaks change with impact intensity while ratios among modal peaks are comparatively stable.[^zhang22]
- Early acoustical studies describe the crack as a broad shock followed by a resonant, decaying tail whose character depends on material properties and impact location.[^collier90][^collier01]
- General impact-sound research shows listeners and classifiers rely on the frequency dependence of damping and the decay envelope — exactly the quantities modal synthesis exposes as parameters.[^krotkov96][^klatzky00]

A beginner can therefore parameterize a credible transient as: a 0.2–1 ms broadband click/noise excitation; 5–20 modes in the roughly 100 Hz–8 kHz band with physically plausible damping (higher modes decay faster); amplitude ratios chosen so low-frequency modes dominate the tail; and a light "ball" component if desired. This matches what the existing representation note already identified as the physically relevant structure (onset, broadband energy, modes, decay).[^repr-note]

## 2. Room/environment acoustics simulation

### 2.1 pyroomacoustics (recommended default)

Verified facts from the official documentation and package metadata:

- Current release **0.10.1** on PyPI; install `pip install pyroomacoustics`; dependencies are only Cython, numpy, scipy — no GPU, no torch.[^pra-pypi]
- Core: a room impulse response (RIR) generator based on the **image source model (ISM)** plus **ray tracing** for general polyhedral rooms (convex and non-convex, 2D and 3D), with a fast C++ core.[^pra-docs]
- Shoebox rooms are set up from a desired RT60 by inverting Sabine's formula: `e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)` — so the student never has to hand-tune absorption coefficients.[^pra-room]
- The room simulator supports: wall **scattering coefficients**, **multi-band simulation** in octave bands from 125 Hz to half the sampling frequency, frequency-dependent **air absorption** (`air_absorption=True`, energy decays as exp(−α(f)·d) with distance), microphone arrays, and source/microphone **directivity** (cardioid family).[^pra-room]
- Microphone signals are produced by convolving source audio with the appropriate RIR at a chosen sampling frequency — i.e., auralization is one method call once the room is built.[^pra-room]
- A free-field / open-field condition (direct sound only, no reflections) is the same API with `max_order=0` (only the direct path), so "room vs open field" is a parameter, not a second tool.[^pra-room]
- Canonical usage is ~10 lines (docs example). The package is cited by its ICASSP 2018 paper.[^pra-paper]

Learning curve: low. This is the tool the minimal experiment should use.

### 2.2 gpuRIR (faster ISM, heavier install)

- Free and open-source (AGPL-3.0) Python library for RIR simulation using ISM with CUDA GPU acceleration; the README claims ~100× faster than CPU implementations.[^gpurir-repo]
- Install is from source: needs the NVIDIA CUDA toolkit, a C++11 compiler, and CMake (≥3.23); it is not published on PyPI (PyPI returns 404 for both `gpuRIR` and `gpurir`).[^gpurir-repo][^gpurir-pypi]
- Features: shoebox rooms, per-wall reflection coefficients (with a Sabine-based estimator `beta_SabineEstimation`), multiple sources/receivers, source and receiver polar patterns (omni, cardioid, bidirectional, …), optional transition `Tdiff` to a diffuse reverberation model, speed-of-sound parameter.[^gpurir-repo]
- Paper: Díaz-Guerra, Miguel, Beltrán, Multimedia Tools and Applications, 2021.[^gpurir-paper]

Verdict: a legitimate faster alternative, but the CUDA + build-from-source requirement makes it the wrong default for a solo beginner; it adds nothing at feature level that ISM does not already provide.

### 2.3 Wave-based methods

Free wave-based room-acoustics simulation in Python at beginner level does not exist as a maintained package: the referenced wave-simulation systems are research code with significant meshing/precomputation demands,[^raghuvanshi10] and the numerical-synthesis monograph is a methods text, not a tool.[^bilbao09] For this project they are out of scope (§3).

### 2.4 What these models can and cannot do (for our questions)

Geometric acoustics (ISM/ray tracing) models specular reflections, direct path, distance attenuation, air absorption, and wall absorption/scattering as energy effects. It does **not** model diffraction around obstacles or full wave phenomena; late reverb is approximated (ISM order limit, or a diffuse tail as in gpuRIR's `Tdiff`). This is adequate for every feature-level question in §3 because those effects act on short transients mainly through the direct path and early reflections — exactly the region ISM is known to model well.[^allen79][^borish84]

## 3. Auralization fidelity: what is needed vs overkill

The study target is **feature-level effects on short transients** — clipping/truncation, reverberation, microphone distance, compression — as measured by the repo's frozen instruments, not human listening quality.

- The features in play (181D: peak/attack/decay, band ratios, spectral shape, fine-frame statistics; M2D 40 ms tokens) are moved by: overall level and clipping (peak statistics, crest factor, kurtosis); direct-path attenuation and air absorption (band ratios, high-frequency content with distance); early reflections and RT60 (decay-to-20 %, frame-energy dynamics, spectral flux); codec artifacts (band edges, noise floor). ISM with 1–3 reflection orders plus Sabine-set absorption reproduces all of these at physically plausible values.
- Perceptual auralization quality (anechoic rendering, binaural cues, measured material data) is **overkill**. The relevant standard is *measurement plausibility*: the RIR must have the right direct delay/attenuation, RT60, and direct-to-reverberant ratio, not a psychoacoustically exact tail. The auralization literature itself distinguishes physical modelling, simulation, and rendering as separate stages with independent fidelity needs.[^vorlander]
- Compression and clipping should not be "modeled" analytically: apply the real transforms. `ffmpeg` (already in the frozen `baseball-audio` env) does real MP3/AAC/Opus encode–decode; hard clipping and truncation are one numpy line each. The Audio Degradation Toolbox is the established precedent for exactly this strategy (a catalog of degradations — mp3, clipping, dynamic-range compression, impulse-response convolution, noise, resampling — applied to features/robustness evaluation), and a Python port exists.[^adt-paper][^adt-py]
- A subtlety worth keeping: reverberation changes the *statistics* of natural sounds (longer tails, spectral smoothing), and those statistics are what generic features/embeddings pick up; this is precisely why environment transfer can masquerade as outcome signal in the real collection — the synthetic probe is the controlled way to quantify that.[^traer16] But no perceptual model is required to measure it.

## 4. Precedents worth citing

Baseball/impact audio (physics):
- Collier & Dresens, "The crack of the bat: material properties, dynamic response, and sound radiation of baseball bats," JASA 1990.[^collier90]
- Collier, "The sounds of baseball: the bat–ball collision and the crack of the bat," JASA 2001.[^collier01]
- Russell, "The sweet spot of a hollow baseball or softball bat," ASA 2004 (sub-1 ms collision; bat modes).[^russell04]
- Brody, "The sweet spot of a baseball bat," Am. J. Phys. 1986.[^brody86]
- Jaramillo, Manarky, Adrezin, "Sweet spot or sweet zone? Modal analysis of a wooden baseball bat for design optimization," ASME IMECE 2003.[^jaramillo03]
- Zhang et al., "Impact position estimation for baseball batting with a force-irrelevant vibration feature," Sensors 2022 (modal ratios stable across intensity).[^zhang22]
- Tennis analogue: "Radiated sound and transmitted vibration following the ball/racket impact of a tennis serve," Vibration 2024 (open access) — same impact-sound measurement pattern in another racket sport.[^tennis24]

Impact-sound synthesis and perception:
- Adrien, "The missing link: modal synthesis," 1991 (the origin of modal synthesis for sound).[^adrien91]
- van den Doel & Pai, "The sounds of physical shapes," Presence 1998 (modal synthesis of struck objects; parameter estimation from recordings).[^vdp98]
- van den Doel, Kry & Pai, "FoleyAutomatic," SIGGRAPH 2001 (modal-synthesis engine for interactive scenes — the open-code-era precedent for cheap physical impact audio).[^foley01]
- Krotkov, Klatzky & Zumel, ICPR 1996 (analysis **and synthesis** of impact sounds; material-dependent damping).[^krotkov96]
- Klatzky, Pai & Krotkov, Presence 2000 (material perception from contact sounds; decay especially influential).[^klatzky00]
- Avanzini & Rocchesso, DAFx-01 (non-linear contact-force model of collision sounds).[^avanzini01]
- Cook, *Real Sound Synthesis for Interactive Applications*, 2002 (textbook covering sample-based through modal synthesis).[^cook02]
- Bilbao, *Numerical Sound Synthesis*, 2009 (wave-based; cited as the contrast case).[^bilbao09]
- Raghuvanshi et al., SIGGRAPH 2010 (precomputed wave simulation; cited as the contrast case for scene acoustics).[^raghuvanshi10]

Room acoustics and auralization:
- Allen & Berkley, "Image method for efficiently simulating small-room acoustics," JASA 1979 (the ISM reference).[^allen79]
- Borish, "Extension of the image model to arbitrary polyhedra," JASA 1984.[^borish84]
- Scheibler, Bezzam & Dokmanić, "Pyroomacoustics," ICASSP 2018; official repo LCAV/pyroomacoustics; official docs.[^pra-paper][^pra-docs]
- Díaz-Guerra, Miguel & Beltrán, "gpuRIR," Multimedia Tools and Applications 2021; official repo DavidDiazGuerra/gpuRIR.[^gpurir-paper][^gpurir-repo]
- Vorländer, *Auralization*, 2nd ed., Springer 2020 (the standard auralization reference; physical modelling vs simulation vs rendering).[^vorlander]
- Traer & McDermott, "Statistics of natural reverberation enable perceptual separation of sound and space," PNAS 2016 (reverb statistics shape what listeners — and by extension feature extractors — receive).[^traer16]

Degradation/robustness methodology:
- Mauch & Ewert, "The Audio Degradation Toolbox and its application to robustness evaluation," ISMIR 2013; Python port (GPL) with the same degradation catalog.[^adt-paper][^adt-py]

The repo's own representation note already cites the baseball physics core of this list and should remain the anchor for the *representation* half of the story.[^repr-note]

## 5. Minimal credible experiment

**One experiment a beginner can actually run, satisfying constraint (a) and (b) of the ticket, with a concrete success criterion.**

*Title*: **Synthetic contact probe with controlled environment transfer** — how recording environment and collection processing transform contact audio, measured by the repo's frozen instruments.

*Pipeline* (single Python script + one manifest; all free, all headless):

1. **Synthesize** `N` contact transients (e.g., 400) by modal synthesis in numpy at 16 kHz, with two physically motivated "classes" designed after §1.2 (e.g., *A*: sharp, high-frequency-dominant, fast-decay crack; *B*: lower-modal, slower-decay thwack), each with known ground-truth parameters (mode frequencies, decays, excitation sharpness) saved to a manifest. Emit fixed-length 2.0 s WAVs in the exact layout `make_audio_windows.py` produces (`impact_200`-style, 16 kHz, WAV), so the frozen extractors run unchanged.
2. **Auralize** each transient through a controlled grid using pyroomacoustics: free field (`max_order=0`) vs shoebox rooms with RT60 ∈ {0.2, 0.8, 2.0} s (Sabine setup), microphone distance ∈ {1, 10, 30} m, air absorption on/off. A small grid (≈10 cells) keeps CPU runtime in minutes.
3. **Degrade** like collection processing: 16-bit quantization, hard clipping at a few peak ceilings, MP3/AAC encode–decode via `ffmpeg` at a few bitrates, and truncation (which the repo's windowing already does by construction).
4. **Measure** with the repo's frozen instruments: run `extract_traditional_features.py` (guaranteed headless; §6) on every cell, and — if a torch environment is available — `extract_m2d_v3.py` on a subset.
5. **Analyze** three pre-registered outputs:
   - *Instrument sanity*: can the 181D set separate classes A/B under free-field, close-mic conditions?
   - *Transfer sensitivity*: per cell, which 181D features drift most and in which direction (feature-level "environment fingerprint"), and at what severity (distance/RT60/codec/clip) class separability collapses;
   - *Recoverability*: how well do the 181D features recover injected ground truth (e.g., modal decay rate, class) as severity increases.

*Success criterion* (concrete, pre-registered):

> The experiment succeeds if (i) under clean free-field close-mic conditions the frozen 181D instrument separates the two synthetic classes at ≥ 0.85 balanced accuracy (the instrument chain demonstrably carries contact-structure information), and (ii) the transfer grid yields at least one quantitative, reproducible finding of the form "environment/collection condition *X* at severity *S* shifts feature group *F* by more than *k* within-class standard deviations / collapses class separability from *p₁* to *p₂*" — i.e., a number the `audio-research-recovery` map can adopt as a confound hypothesis, not a demo.

*Why this is the minimal credible choice*: it needs no purchases, no hardware, no players; it reuses the repo's measuring instruments verbatim (no new feature code to validate); every environmental effect is a controlled parameter with known ground truth (satisfying ticket point 5b), and the transfer-sensitivity analysis mechanically demonstrates how environment/collection processing transforms contact audio (point 5a). It strengthens the SURF outcome by turning the fly/ground screening task's confound worries into measured quantities (e.g., "at broadcast distances with reverb, band-ratio features drift more than decay features"), which is a report-section-grade deliverable under the unsettled SURF deliverable. The experiment deliberately surfaces — but does not decide — the handoff to `audio-research-recovery`, per the map's relationship rules.

*Risks to note*: synthetic classes are not real fly/ground outcomes; the probe validates the measuring chain and quantifies transfer, it does not label real events. The M2D leg is optional because it depends on a torch environment. The grid must be small enough to finish on CPU (pyroomacoustics ISM on 2 s RIRs is fast; M2D CPU inference is the slow leg — batch sizes and a subset keep it bounded).

## 6. Local measuring instruments (repo-level check)

Verified against the actual scripts and frozen artifacts:

- **Traditional 181D — headless today, no new dependencies.** `scripts/audio_pipeline/extract_traditional_features.py` imports only argparse/numpy/pandas/scipy; CLI `--windows-manifest/--out/--lowpass-hz`; CPU-only; no model weights; computes exactly the 181 `feat_*` columns (verified: the frozen V4/V5 evaluation CSVs contain 181 feature columns each).[^v4][^v5] It reads WAV paths from a manifest CSV, so a synthetic manifest with `window_path` pointing at the generated WAVs runs it unchanged.
- **M2D 40 ms — headless but environment-conditional.** `scripts/audio_pipeline/extract_m2d_v3.py` is a CLI with no GUI; `--device cpu` is supported (the CUDA guard only triggers when CUDA is requested and absent); the 1.6 GB frozen checkpoint exists at `data/models/m2d_40ms/m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth`, and the portable runtime exists at `external/m2d/examples/portable_m2d.py`.[^m2d-repo] Two caveats verified in code: (1) the runtime requires torch/timm/einops/nnAudio — the README documents the `vector` env for the pipeline, but the frozen `environment-audio.yml` (numpy/scipy/librosa/soundfile/ffmpeg) contains **no torch**, so a fresh `baseball-audio` env cannot run M2D; (2) batches are asserted to contain equal-length waveforms, so synthetic WAVs must be fixed-length (2.0 s at 16 kHz works). CPU inference is feasible but is the slow leg.
- **No sound-simulation tooling exists in the workspace** (map fact re-confirmed): `environment-audio.yml` has no acoustics library; adding `pyroomacoustics` is the single new dependency the experiment needs.
- The frozen window pipeline (`make_audio_windows.py`) and feature CSVs (v3/v4/v5) provide the conventions (window names like `impact_200` = 2.0 s, 16 kHz WAV, manifest CSV) that a synthetic dataset should mirror so the measuring instruments and any future comparison are drop-in.

## 7. Limitations

- No synthetic probe can substitute for real recordings; the experiment's claims are about the measuring chain and transfer mechanics, not about real fly/ground audio.
- Statements about pyroomacoustics/gpuRIR behavior come from official docs and package metadata verified 2026-08-05, not from a local install (none exists in the workspace yet).
- Geometric acoustics is approximate at high frequencies and in large spaces; the experiment's conclusions are relative (severity grids), which is robust to these biases as long as all cells share the same simulator.
- The gpuRIR speed claim (~100×) is the authors' own README claim, not independently benchmarked here.
- The `vector` env's torch availability was not verified on this machine (no conda found on PATH during the check); the report only records what the repo's README and frozen yml say.

## Primary sources

[^adrien91]: J.-M. Adrien, ["The Missing Link: Modal Synthesis"](https://mitpress.mit.edu/9780262041308/representations-of-musical-signals/), in *Representations of Musical Signals*, MIT Press, 1991, pp. 269–298.

[^vdp98]: K. van den Doel and D. K. Pai, [“The Sounds of Physical Shapes”](https://doi.org/10.1162/105474698565794), *Presence: Teleoperators and Virtual Environments* 7(4), 1998.

[^foley01]: K. van den Doel, P. G. Kry, and D. K. Pai, [“FoleyAutomatic: Physically-based sound effects for interactive simulation and animation”](https://doi.org/10.1145/383259.383322), SIGGRAPH 2001.

[^cook02]: P. R. Cook, [*Real Sound Synthesis for Interactive Applications*](https://doi.org/10.1201/b19597), A K Peters / CRC Press, 2002 (see esp. ch. “Modal Synthesis”, [10.1201/b19597-6](https://doi.org/10.1201/b19597-6)).

[^krotkov96]: E. Krotkov, R. Klatzky, and N. Zumel, [“Analysis and synthesis of the sounds of impact based on shape-invariant properties of materials”](https://publications.ri.cmu.edu/analysis-and-synthesis-of-the-sounds-of-impact-based-on-shape-invariant-properties-of-materials), ICPR, 1996.

[^klatzky00]: R. L. Klatzky, D. K. Pai, and E. P. Krotkov, [“Perception of Material from Contact Sounds”](https://doi.org/10.1162/105474600566907), *Presence*, 2000.

[^avanzini01]: F. Avanzini and D. Rocchesso, [“Modeling collision sounds: non-linear contact force”](https://openalex.org/W29529489), Proc. DAFx-01, Limerick, 2001.

[^bilbao09]: S. Bilbao, [*Numerical Sound Synthesis*](https://doi.org/10.1002/9780470749012), Wiley, 2009.

[^raghuvanshi10]: N. Raghuvanshi et al., [“Precomputed wave simulation for real-time sound propagation of dynamic sources in complex scenes”](https://doi.org/10.1145/1833349.1778805), ACM SIGGRAPH/TOG, 2010.

[^collier90]: R. D. Collier and P. Dresens, [“The crack of the bat: material properties, dynamic response, and sound radiation of baseball bats”](https://doi.org/10.1121/1.2028570), *JASA* 88(S1), 1990.

[^collier01]: R. D. Collier, [“The sounds of baseball: The bat–ball collision and the crack of the bat”](https://doi.org/10.1121/1.4744893), *JASA* 109, 2001.

[^russell04]: D. A. Russell, [“The sweet spot of a hollow baseball or softball bat”](https://acoustics.org/pressroom/httpdocs/148th/russell.html), Acoustical Society of America, 2004.

[^brody86]: H. Brody, [“The sweet spot of a baseball bat”](https://doi.org/10.1119/1.14854), *American Journal of Physics* 54, 1986.

[^jaramillo03]: P. Jaramillo, K. S. Manarky, and R. S. Adrezin, [“‘Sweet Spot’ or ‘Sweet Zone’? Modal analysis of a wooden baseball bat for design optimization”](https://doi.org/10.1115/imece2003-41924), ASME IMECE 2003.

[^zhang22]: S. Zhang et al., [“Impact position estimation for baseball batting with a force-irrelevant vibration feature”](https://pmc.ncbi.nlm.nih.gov/articles/PMC8878515/), *Sensors*, 2022.

[^tennis24]: [“Radiated sound and transmitted vibration following the ball/racket impact of a tennis serve”](https://doi.org/10.3390/vibration7040047), *Vibration* 7(4), MDPI, 2024.

[^allen79]: J. B. Allen and D. A. Berkley, [“Image method for efficiently simulating small-room acoustics”](https://doi.org/10.1121/1.382599), *JASA* 65(4), 1979.

[^borish84]: J. Borish, [“Extension of the image model to arbitrary polyhedra”](https://doi.org/10.1121/1.390983), *JASA* 75(6), 1984.

[^pra-docs]: Pyroomacoustics official documentation, [Summary](https://pyroomacoustics.readthedocs.io/en/pypi-release/index.html) and [Room API](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html), EPFL LCAV, accessed 2026-08-05; [official repository](https://github.com/LCAV/pyroomacoustics).

[^pra-paper]: R. Scheibler, E. Bezzam, and I. Dokmanić, [“Pyroomacoustics: A Python Package for Audio Room Simulation and Array Processing Algorithms”](https://doi.org/10.1109/icassp.2018.8461310), ICASSP 2018.

[^pra-pypi]: [pyroomacoustics 0.10.1 on PyPI](https://pypi.org/project/pyroomacoustics/) — requires Cython, numpy≥1.13, scipy≥0.18 (verified 2026-08-05).

[^gpurir-repo]: D. Díaz-Guerra, A. Miguel, J. R. Beltrán, [official gpuRIR repository README](https://github.com/DavidDiazGuerra/gpuRIR) (installation from source; CUDA toolkit, C++11, CMake; AGPL-3.0; ~100× speed claim; `beta_SabineEstimation`, `Tdiff`, polar patterns), accessed 2026-08-05.

[^gpurir-paper]: D. Díaz-Guerra, A. Miguel, and J. R. Beltrán, [“gpuRIR: A python library for room impulse response simulation with GPU acceleration”](https://doi.org/10.1007/s11042-020-09905-3), *Multimedia Tools and Applications* 80, 2021.

[^gpurir-pypi]: [PyPI API responses for `gpuRIR` and `gpurir` are 404 Not Found](https://pypi.org/pypi/gpuRIR/json) (verified 2026-08-05) — the package is not published on PyPI.

[^vorlander]: M. Vorländer, [*Auralization: Fundamentals of Acoustics, Modelling, Simulation, Algorithms and Acoustic Virtual Reality*](https://doi.org/10.1007/978-3-030-51202-6), 2nd ed., Springer (RWTHedition), 2020.

[^traer16]: J. Traer and J. H. McDermott, [“Statistics of natural reverberation enable perceptual separation of sound and space”](https://doi.org/10.1073/pnas.1612524113), *PNAS* 113(48), 2016.

[^adt-paper]: M. Mauch and S. Ewert, [“The Audio Degradation Toolbox and its application to robustness evaluation”](http://www.eecs.qmul.ac.uk/~ewerts/publications/2013_MauchEwert_AudioDegradationToolbox_ISMIR.pdf), ISMIR 2013.

[^adt-py]: [Python implementation of the Audio Degradation Toolbox](https://github.com/sevagh/audio-degradation-toolbox) (GPL; degradations incl. `mp3`, `clipping`, `dynamic_range_compression`, `impulse_response`, `noise`, `low_pass`, `resample`), accessed 2026-08-05.

[^m2d-repo]: NTT Communication Science Laboratories, [official M2D source repository](https://github.com/nttcslab/m2d) (portable runtime requirements: `pip install timm einops nnAudio transformers`); local copy at `external/m2d`.

## Local evidence

[^repr-note]: Project research note, [“Representation strategies for sub-200 ms bat–ball impact audio”](short-impact-audio-representations.md), 2026-07-21.

[^v4]: Project report, [“V4 unified frozen audio benchmark”](../v4_unified_frozen_audio_benchmark_zh.md), 2026-07-16 (Traditional 181D composition).

[^v5]: Project report, “V5 M2D and traditional-feature rerun” (frozen evaluation CSVs at `data/processed/v5/features/*.csv` contain 181 `feat_*` columns; M2D 40 ms artifacts at `data/processed/v4/features/short/m2d_40ms_last_stats.csv`).

Repo artifacts verified 2026-08-05: `scripts/audio_pipeline/extract_traditional_features.py` (pure numpy/scipy/pandas; headless CLI); `scripts/audio_pipeline/extract_m2d_v3.py` (torch; headless CLI; equal-length batch assertion; `--device cpu` supported; default `--device cuda` guarded); `external/m2d/examples/portable_m2d.py`; `data/models/m2d_40ms/m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth` (1.6 GB); `scripts/audio_pipeline/make_audio_windows.py` (window conventions); `environment-audio.yml` (numpy/scipy/librosa/soundfile/ffmpeg — no torch, no acoustics library); `scripts/audio_pipeline/README.md` (pipeline runs in the `vector` conda environment).
