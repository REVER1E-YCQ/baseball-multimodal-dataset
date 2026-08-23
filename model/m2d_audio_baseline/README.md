# M2D frozen-audio baseline for fly ball vs ground ball

This folder contains the smallest reproducible version of the current best
audio baseline. It intentionally excludes raw data, derived windows, embeddings,
third-party source code, and model weights.

## What the baseline does

1. Use the manual `[event_start, event_end]` interval only as a search range.
2. Find the absolute waveform-amplitude peak inside that interval.
3. Extract a 200 ms event window centred on that peak.
4. Extract a matched 200 ms strict-pre window ending 50 ms before
   `event_start`; no waveform padding is allowed.
5. Run the frozen M2D 40 ms encoder and pool timestamp tokens with
   mean, population standard deviation, and maximum, producing 2304 features.
6. Train only `StandardScaler + L2 logistic regression`; the M2D encoder has
   zero training epochs.
7. Evaluate with repeated nested stratified group CV and report the strict-pre
   negative control.

The event endpoint is treated as `event_end`, not as the peak.

## High-level short-contact benchmark interface

New protocol-driven work enters through `run_short_contact_benchmark`. A caller
provides one frozen `BenchmarkProtocol`, an immutable `DatasetSnapshot`, and one
or more encoder adapters; the module returns an `ArtifactBundle` after snapshot
audit, contact-window preparation, frozen feature extraction, grouped
out-of-fold evaluation, and artifact publication.

The first tracer supports only the exact 200 ms event condition and a fixed
balanced L2 logistic-regression probe. Its integration tests use synthetic
multi-game audio and a fake encoder, so the normal correctness suite does not
need model weights, external model source, or a GPU. Real M2D/BEATs adapters,
matched controls, and the locked nested evaluation are added through later
benchmark slices rather than coordinated by callers.

The bundle records a portable content-derived identity and emits the frozen
protocol, snapshot audit, window and fold manifests, encoder features,
out-of-fold predictions, metrics, and a checksummed artifact manifest. Absolute
local paths do not participate in the artifact identity.

## Primary 200 ms M2D benchmark

The locked primary run audits the verified-dataset snapshot, prepares exact
200 ms peak-centred contact windows (rows that cannot provide an exact window
receive explicit exclusion reasons), extracts frozen mean-pooled M2D
embeddings, and evaluates a balanced L2 logistic regression with `C` selected
inside each outer training set from `0.001`, `0.01`, and `0.1` using inner
lineage-grouped folds. Run it with:

```powershell
python -m scripts.run_m2d_primary
```

The runner defaults to the pinned dataset worktree
(`data/branch_datasets_20260804/baseball-multimodal-dataset`), the pinned M2D
upstream revision and checkpoint hash, and writes a content-addressed artifact
bundle under `outputs/m2d_primary_benchmark/`. Completed feature files are
reused when the snapshot, detector, window, normalization, pooling, checkpoint,
precision, or upstream revision are unchanged. The real-adapter smoke test runs
only when `M2D_SMOKE=1`.

On the 822-sample verified snapshot, the current primary run reports balanced
accuracy 0.595 with 817 eligible samples (5 excluded with reasons).

BEATs runs the identical protocol through `python -m scripts.run_beats_primary`
(with FP32 forced and non-finite outputs rejected) and reports balanced accuracy
0.599 on the same eligible set.

## Paired common 200 ms comparison

`python -m scripts.run_common_200ms` runs both encoders with matched controls
and validates that they share the same snapshot revision, fold policy, window
membership, prediction cardinalities, and control conditions before emitting
a paired table. On the verified snapshot (803 paired samples):

| Condition | M2D | BEATs | M2D - BEATs |
|---|---|---|---:|
| Event-trained on event | 0.619 | 0.610 | +0.009 |
| Same event probe on strict pre | 0.520 | 0.496 | +0.024 |
| Independent strict-pre probe | 0.544 | 0.524 | +0.020 |
| Event probe after transient removal | 0.507 | 0.500 | +0.007 |
| Independent removed probe | 0.556 | 0.558 | -0.003 |
| Contact-specific increment | 0.099 | 0.114 | -0.015 |

Both encoders show a positive paired event-minus-pre increment; BEATs starts
from a lower strict-pre baseline, so its increment is slightly larger.

## Group-aware statistical evidence

`run_common_200ms` also computes group-aware statistics. Ninety-five-percent
intervals resample whole lineage groups (never clips), paired intervals cover
each encoder's event-minus-pre increment and the M2D-minus-BEATs event
difference, and a 999-permutation test stratifies labels inside each locked
outer fold (preserving per-fold class totals and mixed-label games) with
max-stat multiplicity correction across the two encoders. A screening-positive
interpretation requires both corrected evidence against the null and a
positive increment interval.

On the verified snapshot (803 samples, 620 groups):

- M2D increment 0.099 [0.063, 0.138]; BEATs increment 0.114 [0.073, 0.156].
- M2D-minus-BEATs event difference 0.009 [-0.028, 0.048].
- Family-corrected permutation p = 0.001 for both encoders; both are
  screening-positive.
- Source-transfer conclusiveness is marked false because 516 of 620 groups
  are singletons: grouped evaluation reduces same-game leakage but does not
  by itself prove transfer to a new collection workflow.

## Locked sensitivity conditions

`python -m scripts.run_m2d_sensitivity` runs the M2D sensitivity family with
matched controls. Duration sensitivity uses exact 50/100/200 ms peak-centred
windows with same-duration strict-pre controls (200 ms keeps the
transient-removal condition); BEATs stays limited to 200 ms and a run that
requests shorter windows fails visibly. On the verified snapshot:

| Window | Event BA | Strict-pre BA | Increment |
|---|---:|---:|---:|
| 50 ms | 0.636 | 0.498 | +0.137 |
| 100 ms | 0.612 | 0.501 | +0.111 |
| 200 ms | 0.619 | 0.520 | +0.099 |

The shortest window shows the largest contact-specific increment with a
near-chance strict-pre baseline. Two further sensitivity runs are available:
`--rms-normalized` (gain-normalized to 0.1 RMS; increment 0.075, still
positive) and `--legacy-pooling` (mean/std/max tokens, 2304 features; increment
0.129). Each has a distinct artifact identity, and one-token degeneracy is
recorded per feature row.

## Secondary development evidence

`python -m scripts.run_secondary_evidence` reproduces the branch's existing
525/132/165 fixed benchmark split exactly (membership is validated against the
snapshot; 59 MLB games are recorded as crossing partitions) and evaluates the
test partition once after tuning only on the training and validation
partitions. It also runs the predeclared balanced RBF SVM probe (C in
`0.3/1.0/3.0`, gamma `scale/0.001`) inside game-grouped folds. On the verified
snapshot (523/131/163 eligible):

- Fixed split test balanced accuracy: BEATs 0.642, M2D 0.588.
- RBF SVM OOF balanced accuracy: M2D 0.621, BEATs 0.614 (no gain over the
  linear probe).

These outputs are explicitly labelled development evidence, never source-
transfer evidence, and never alter the primary ranking or the frozen protocol.

## Validation and reporting

`python -m scripts.run_validate_and_report` validates the complete run with one
command — snapshot and window manifests, feature dimensions and finite values,
fold isolation, prediction uniqueness, eligible-set cardinalities, selections,
metrics condition sets, exclusion reasons, checksums, statistical evidence,
and secondary outputs — and writes a Chinese technical report
(`report_zh.md`) plus a group-meeting summary (`summary_zh.md`) under
`outputs/validated_run/`. The report visibly separates primary, negative-
control, sensitivity, exploratory, fixed-benchmark, contact-specific, and
source-transfer claims, uses Balanced Accuracy as the headline metric with
Accuracy, ROC-AUC, Macro-F1 and confusion counts, and states that game-grouped
evaluation does not by itself prove transfer to a new collection workflow.

## Matched negative controls

`run_m2d_primary --controls` adds the strict-pre and transient-removal
conditions. The strict-pre window has the same duration, ends exactly 50 ms
before the preserved event interval, and uses no padding. The transient-removal
window replaces the central 40 ms around the detected peak with an equal-length
background segment from the same sample's strict-pre region using a
deterministic 5 ms crossfade; source and destination coordinates are recorded
in the window manifest. Samples without enough pre-contact audio are excluded
with `strict_pre_unavailable` and remain in the event-only analysis.

Evaluation reports five paired conditions per encoder — event-trained on event,
the same event probe on strict pre and on transient-removed audio, and
independently trained strict-pre and transient-removed probes — plus a
`contact_specific_increment` row equal to the paired event-minus-pre balanced
accuracy. On the verified snapshot, M2D scores 0.619 on the event condition
versus 0.520 for the same probe on strict pre (increment 0.099), with the
event-trained probe falling to 0.507 after transient removal, over 803 paired
samples.

## Why this is the selected baseline

On the frozen V5 data snapshot (`main@5b38414`), the paired 200 ms experiment
used 1,851 Codex-collected samples:

| Condition | Balanced accuracy |
|---|---:|
| Event-trained model on event | `0.606 +/- 0.006` |
| Same event-trained model on strict pre | `0.584 +/- 0.003` |
| Independently trained strict-pre model | `0.587 +/- 0.008` |

M2D remains stronger than the 181D traditional baseline inside the Codex
`primary_dev` data. However, strict-pre performance increased relative to V4 and
the event-minus-pre increment fell from about 0.055 to 0.022. On 87 disjoint
human-collected samples, the Codex-trained M2D event model reached only 0.490
balanced accuracy. These results support M2D as an in-domain baseline, not as a
verified cross-collector model or proof of bat-ball contact physics.

## Current headline on the verified snapshot

The numbers above use mean pooling. The current best controlled result uses a
different pooling and must not be compared to them directly:

| Item | Value |
|---|---|
| Balanced accuracy | `0.667` (ROC-AUC 0.693) |
| Contact-specific increment | `+0.168` [0.132, 0.203], permutation p = 0.001 |
| Strict-pre negative control | `0.499` (at chance) |
| Pooling | frozen M2D attention pooling, layer 11, fixed 0.5 threshold |
| Data | 803 eligible samples of the 822-sample human-verified snapshot |

Two reasons explain the gap from the mean-pooling rows: the verified snapshot is
a smaller human-checked subset (label noise removed), and attention pooling
replaces mean pooling. Treat 0.667 as the verified-data headline and the tables
above as the full-Codex-set baseline; neither number transfers automatically to
a new collection workflow. The diagnostics that closed every representation
side (pooling family, layer scan, alignment sensitivity, encoder fusion,
LoRA fine-tuning, augmentation) are summarised in
`docs/experiments/EXPERIMENTS_INDEX.md`, with per-experiment reports beside it.

## Important grouping limitation

The V5 paired `primary_dev` set contains 1,851 samples and 1,851 distinct
`source_id` values. Every group is a singleton. Grouped CV therefore behaves
like stratified CV inside the Codex collection workflow. The evaluator prints a
warning and records this fact in `protocol.json`; use the disjoint external-role
evaluator to test a different collection workflow.

## Files

```text
baselines/m2d_audio/
|-- scripts/
|   |-- prepare_windows.py
|   |-- audit_verified_snapshot.py
|   |-- extract_m2d_embeddings.py
|   |-- m2d_encoder.py
|   |-- beats_encoder.py
|   |-- short_contact_benchmark.py
|   |-- compare_common_200ms.py
|   |-- statistical_evidence.py
|   |-- run_m2d_primary.py
|   |-- run_beats_primary.py
|   |-- run_common_200ms.py
|   |-- run_m2d_sensitivity.py
|   |-- run_secondary_evidence.py
|   |-- run_validate_and_report.py
|   |-- evaluate_linear_probe.py
|   `-- evaluate_external_holdout.py
|-- tests/
|-- examples/input_manifest.example.csv
|-- results/reference_metrics.csv
|-- environment.yml
|-- requirements.txt
|-- protocol.json
`-- NOTICE.md
```

## Environment

The tested local stack was:

- Python 3.12.12
- PyTorch 2.10.0+cu128
- CUDA 12.8
- NumPy 2.2.6
- pandas 2.2.3
- SciPy 1.17.0
- scikit-learn 1.8.0
- timm 1.0.24
- einops 0.8.1
- nnAudio 0.3.4
- transformers 4.57.6

Create the non-PyTorch dependencies:

```powershell
conda env create -f environment.yml
conda activate baseball-m2d-baseline
```

Install a PyTorch build appropriate for the local CPU/CUDA environment first,
then install the remaining M2D dependencies:

```powershell
# Follow https://pytorch.org/get-started/locally/ for this machine, then:
pip install -r requirements.txt
```

PyTorch is deliberately not pinned because CPU and CUDA wheel indexes differ.
Installing it before `requirements.txt` prevents pip from silently selecting an
incompatible default build through `timm`. Record `python -c "import torch;
print(torch.__version__)"` in the experiment log.

## Obtain M2D without vendoring it

Clone the upstream source outside the tracked package and use the tested commit:

```powershell
git clone https://github.com/nttcslab/m2d.git third_party/m2d
git -C third_party/m2d checkout 3d0c4de9447c404a8d3f9f37e04f53bc902e09b3
```

Download `m2d_vit_base-80x200p16x4-230529` from the official M2D release linked
in the upstream README. Do not commit the approximately 1.68 GB checkpoint.

Expected checkpoint:

```text
filename: checkpoint-300.pth
sha256: 63578974bc004ef57a8e5456bac8c684f62c9285537a7b2ddef13b442386786f
```

Only load a checkpoint obtained from the trusted upstream release.

## Input manifest

Create a CSV matching `examples/input_manifest.example.csv`:

| Column | Meaning |
|---|---|
| `uid` | Stable sample identifier |
| `label` | `fly_ball` or `ground_ball` |
| `source_id` | Recording session/source group; related samples must share it |
| `protocol_role` | For example `primary_dev` or `primary_locked_test` |
| `audio_path` | Absolute path or path relative to the manifest |
| `event_start` | Manual event start in seconds |
| `event_end` | Manual event end in seconds |

Do not use collector identity, file duration, source URL, or paths as classifier
features.

## Run

The examples below keep generated files under `artifacts/`, which is ignored.

### 1. Prepare event and strict-pre windows

```powershell
python scripts/prepare_windows.py `
  --manifest examples/input_manifest.csv `
  --out-root artifacts/windows `
  --window-ms 200 `
  --pre-gap-ms 50
```

### 2. Extract frozen M2D embeddings

CUDA AMP (`--amp`) is the reference precision used for the reported V4 M2D features. CUDA is selected automatically when available.

```powershell
python scripts/extract_m2d_embeddings.py `
  --windows-manifest artifacts/windows/windows_manifest.csv `
  --out artifacts/features/m2d_200ms.csv `
  --m2d-root third_party/m2d `
  --checkpoint D:/path/to/m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth `
  --expected-checkpoint-sha256 63578974bc004ef57a8e5456bac8c684f62c9285537a7b2ddef13b442386786f `
  --device auto `
  --amp `
  --batch-size 16
```

GPU kernels and mixed precision can cause very small floating-point differences
between machines. Record the output provenance fields and compare evaluation metrics,
not byte-for-byte embedding equality.

The output contains identifiers, protocol metadata, checkpoint/source
provenance, and `feat_*` columns. It does not contain absolute audio paths.

### 3. Run the nested linear probe and negative control

```powershell
python scripts/evaluate_linear_probe.py `
  --features artifacts/features/m2d_200ms.csv `
  --out-dir outputs/m2d_200ms `
  --event-window event_200ms `
  --pre-window pre_200ms `
  --protocol-role primary_dev `
  --outer-splits 5 `
  --inner-splits 3 `
  --repeats 5 `
  --c-grid 0.001 0.01 0.1 `
  --seed 20260716
```

Outputs:

- `summary.csv`: mean and standard deviation across repeats.
- `repeat_metrics.csv`: one row per repeat and condition.
- `outer_predictions.csv`: out-of-fold predictions only.
- `selections.csv`: selected `C`, inner scores, and actual solver iterations.
- `protocol.json`: sample counts, grouping diagnostics, and frozen settings.

### 4. Evaluate a disjoint external role without tuning on it

Use a `C` value selected only inside the development data. In V5, `C=0.001`
was selected in 23 of 25 M2D event outer folds.

```powershell
python scripts/evaluate_external_holdout.py `
  --features artifacts/features/m2d_200ms.csv `
  --out-dir outputs/m2d_external `
  --train-role primary_dev `
  --test-role external_test `
  --event-window event_200ms `
  --pre-window pre_200ms `
  --c-value 0.001 `
  --seed 20260716
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests verify:

- peak-centred event slicing;
- a strict 50 ms gap before the annotated event;
- exact sample counts and no waveform padding;
- grouped train/test isolation;
- nested selection on training folds only;
- the high-level synthetic benchmark interface and grouped OOF artifacts;
- immutable snapshot audit and lineage grouping with synthetic git repos;
- portable, protocol-sensitive artifact identity;
- protocol and output creation.

## What must not be uploaded

- M2D or any other model checkpoint;
- cloned `third_party/m2d` source;
- raw or recut audio/video outside the dataset's existing policy;
- generated windows or embedding CSVs;
- local Conda environments and caches;
- files containing absolute local paths;
- experimental outputs with sample-level predictions unless explicitly approved.

See `NOTICE.md` before publishing. The parent repository currently needs an
explicit project-license decision from its owner.
