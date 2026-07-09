# Baseball Hit Dataset QC Checklist

Use these categories in manual review reports:

- `pass`: Label and cut are usable.
- `wrong_label`: Ground/fly class or subclass fields are wrong.
- `bad_audio`: Contact sound is missing, masked, clipped, or too noisy.
- `bad_cut`: Clip misses pre-contact context, contact, or early ball movement.
- `source_issue`: Source URL, title, rights note, or traceability is missing or suspicious.

Recommended report fields:

```csv
sample_path,label,event_start,event_end,manual_result,reviewer,notes
```

Escalate a batch when sampled error rate is high enough that isolated fixes are less reliable than recutting or relabeling the whole batch.

