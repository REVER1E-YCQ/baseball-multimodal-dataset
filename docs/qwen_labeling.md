# Qwen Omni Labeling

Scripts read the API key from `QWEN_API_KEY` first, then `DASHSCOPE_API_KEY`.
They also load those variables from the repository-local `.env` file when process variables are absent. `.env` is ignored by Git and must never be force-added.

Do not commit real keys. For PowerShell:

```powershell
$env:QWEN_API_KEY = "..."
```

Use a Model Studio key that starts with `sk-` or `sk-ws`. If the script prints `authentication failed`, fix the key before rerunning. The script aborts immediately on authentication errors so a whole batch is not marked as model failures.

The labeling script uses OpenAI-compatible chat completions over standard-library HTTP/SSE and falls back across models configured in `config/qwen_models.json` or `QWEN_MODEL_FALLBACKS`.

Default models:

```text
qwen3-omni-flash-2025-12-01
qwen3-omni-flash-2025-09-15
qwen-omni-turbo
qwen3.5-omni-flash-2026-03-15
qwen3.5-omni-plus-2026-03-15
qwen3-omni-flash
qwen-omni-turbo-latest
qwen3.5-omni-flash
qwen3.5-omni-plus
```

The default order is quota-conscious: dated flash/turbo snapshots are tried before aliases that may already be near the local cap. Override with `QWEN_MODEL_FALLBACKS` only when you intentionally want a different cost/quality tradeoff.

The labeling script also enforces a local per-model token cap before and during each run. By default, any model with `local_usage + 10000 >= 800000` total tokens in `reports/qwen_labels.jsonl` is skipped. Override with `QWEN_MODEL_TOKEN_CAP` and `QWEN_MODEL_TOKEN_RESERVE`; set the cap to `0` only when you intentionally want to disable the guard.

If DashScope returns `AllocationQuota.FreeTierOnly`, the script treats that model as quota-blocked for the current run and falls through to the next configured model.

Clips whose `source_id` already exists in `dataset/*/*/*/source.txt` are skipped before model calls. This avoids spending tokens on alternate cuts from a source that cannot be materialized again under the source de-duplication rule.

For local clips, the script sends Base64 data URLs when the encoded file is under the model/API limit. If local video is too large, recut or downscale the clip before retrying.

The script writes raw model responses to `reports/qwen_labels.jsonl`; accepted samples are materialized only after QA gates.

For a staged audit of already materialized samples, use:

```powershell
python scripts/qwen_review_dataset.py --dry-run
python scripts/qwen_review_dataset.py
```

The review is resumable and writes each timing, semantics, and optional adjudication call to `reports/qwen_dataset_review.jsonl`. Its token guard combines the original labeling log with the review log, so review calls also stop before a model reaches the configured 800,000-token cap. The generated summary does not alter formal sample CSV files.

Use `--sample-id F_001` for a targeted rerun and add `--force` after repairing its media or labels. Review defaults come from `review_fallback_models` in `config/qwen_models.json`, with stronger Qwen3.5 Omni snapshots first. Set `QWEN_REVIEW_MODEL_FALLBACKS` to override that order without changing production labeling.

When one model version systematically misinterprets a stage, use `--retry-model MODEL --force-stage semantics --force-stage adjudication` to reset and rerun only affected stages while preserving valid timing calls and their token accounting.

After tightening the exact stage allowlist, use `--retry-disallowed-stage-models --force-stage semantics --force-stage adjudication` to replace only historical stage records produced by models no longer permitted for those stages.

After the offline HTTP pool reaches its local cap or server quota, dataset review falls through to `review_realtime_models`. Realtime review sends 16 kHz mono PCM and timestamped 512-pixel JPEG frames through the official manual-mode WebSocket protocol. It uses the documented 1 fps recommendation for semantic review and 2 fps plus local audio-transient hints for timing. Capacity-limit errors receive bounded backoff retries. Each call uses a fresh session to avoid historical-context token accumulation. Override the endpoint with `QWEN_REALTIME_BASE_URL` only when the account requires a workspace-specific domain.

`review_realtime_stage_models` is an evidence-based allowlist. Models that pass connectivity but fail to retain the full play may assist timing but are excluded from region, bounce, and final adjudication. Do not widen the semantic/adjudication lists until a full-play visual probe passes.

When available, `qwen3.5-omni-plus-realtime` is preferred for semantic and adjudication stages. The generic Qwen3.5 realtime aliases and dated Qwen3 realtime snapshots are independent quota pools; every exact model name still obeys the local 800,000-token cap and reserve.

`review_stage_models` is the strict exact-model allowlist. Semantic and adjudication stages require Qwen3.5 realtime models; older turbo snapshots may assist timing but cannot certify region, bounce, or final fields. Evidence strings must describe concrete frame/audio observations; placeholders such as `short evidence` and `ball-path evidence` fail the automatic gate.

The `qwen3.5-omni-flash` alias may temporarily remain in semantic/adjudication stages when the Model Studio console still reports unused free quota. Server-side free-tier exhaustion blocks it and falls through to the Qwen3.5 realtime models; paid overage must remain disabled.

Before spending model tokens, run local billing and quality guards:

```powershell
python scripts/summarize_qwen_usage.py
python scripts/prefilter_pending_clips.py
```

`prefilter_pending_clips.py` keeps clips with a clear local audio transient as `pending` and marks weak clips as `prefilter_reject`, so the default Qwen labeling command skips them. When cost or quota is uncertain, test with a very small batch first:

```powershell
python scripts/qwen_omni_label.py --limit 5
```

To retry clips after a fixed key:

```powershell
python scripts/reset_clip_status.py --from-status label_failed --to-status pending --clear-notes
python scripts/qwen_omni_label.py --limit 20
```
