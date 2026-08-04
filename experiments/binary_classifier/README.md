# Binary Fly-Ball vs Ground-Ball Experiments

The first experiment is an audio-only, source-grouped baseline. It uses the same deterministic split for every feature set and keeps the test set untouched while choosing logistic-regression regularization on the validation set.

Audio controls:

- `contact_logmel`: one second centered on the verified bat-ball contact.
- `contact_waveform`: the same window represented as a fixed waveform.
- `masked_contact_logmel`: the contact-centered window with the central 240 ms removed.
- `background_logmel`: a one-second window taken from the end farthest from contact.

The masked and background controls test whether accuracy comes from the contact sound or from broadcast, commentary, and collection-source leakage.

Run:

```powershell
.\.venv\Scripts\python.exe experiments\binary_classifier\audio_baseline.py
```

Then run the video-only and early-fusion comparison on the identical split:

```powershell
.\.venv\Scripts\python.exe experiments\binary_classifier\multimodal_baseline.py
```

Run the source-grouped late-fusion and early-decision experiments:

```powershell
.\.venv\Scripts\python.exe experiments\binary_classifier\late_fusion_experiment.py
.\.venv\Scripts\python.exe experiments\binary_classifier\crossval_late_fusion.py
.\.venv\Scripts\python.exe experiments\binary_classifier\temporal_ablation_experiment.py
.\.venv\Scripts\python.exe experiments\binary_classifier\early_multimodal_architecture.py
```

Run the pretrained raw-waveform control:

```powershell
.\.venv\Scripts\python.exe experiments\binary_classifier\pretrained_wav2vec_experiment.py
```
