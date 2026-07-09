# Qwen Omni Labeling

Scripts read the API key from `QWEN_API_KEY` first, then `DASHSCOPE_API_KEY`.

Do not commit real keys. For PowerShell:

```powershell
$env:QWEN_API_KEY = "..."
```

The labeling script uses OpenAI-compatible chat completions over standard-library HTTP/SSE and falls back across models configured in `config/qwen_models.json` or `QWEN_MODEL_FALLBACKS`.

Default models:

```text
qwen3.5-omni-plus
qwen3.5-omni-flash
qwen3-omni-flash
qwen-omni-turbo-latest
```

For local clips, the script sends Base64 data URLs when the encoded file is under the model/API limit. If local video is too large, recut or downscale the clip before retrying.

The script writes raw model responses to `reports/qwen_labels.jsonl`; accepted samples are materialized only after QA gates.
