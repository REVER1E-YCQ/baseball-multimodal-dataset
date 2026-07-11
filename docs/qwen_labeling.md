# Qwen Omni Labeling

Scripts read the API key from `QWEN_API_KEY` first, then `DASHSCOPE_API_KEY`.

Do not commit real keys. For PowerShell:

```powershell
$env:QWEN_API_KEY = "..."
```

Use a Model Studio key that starts with `sk-` or `sk-ws`. If the script prints `authentication failed`, fix the key before rerunning. The script aborts immediately on authentication errors so a whole batch is not marked as model failures.

The labeling script uses OpenAI-compatible chat completions over standard-library HTTP/SSE and falls back across models configured in `config/qwen_models.json` or `QWEN_MODEL_FALLBACKS`.

Default models:

```text
qwen3-omni-flash
qwen-omni-turbo-latest
qwen3.5-omni-flash
qwen3.5-omni-plus
```

The default order is quota-conscious: newer flash/turbo models are tried before the nearly capped 3.5 models. Override with `QWEN_MODEL_FALLBACKS` only when you intentionally want a different cost/quality tradeoff.

The labeling script also enforces a local per-model token cap before and during each run. By default, any model with `local_usage + 10000 >= 800000` total tokens in `reports/qwen_labels.jsonl` is skipped. Override with `QWEN_MODEL_TOKEN_CAP` and `QWEN_MODEL_TOKEN_RESERVE`; set the cap to `0` only when you intentionally want to disable the guard.

Clips whose `source_id` already exists in `dataset/*/*/*/source.txt` are skipped before model calls. This avoids spending tokens on alternate cuts from a source that cannot be materialized again under the source de-duplication rule.

For local clips, the script sends Base64 data URLs when the encoded file is under the model/API limit. If local video is too large, recut or downscale the clip before retrying.

The script writes raw model responses to `reports/qwen_labels.jsonl`; accepted samples are materialized only after QA gates.

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
