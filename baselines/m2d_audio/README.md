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
|   |-- extract_m2d_embeddings.py
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

