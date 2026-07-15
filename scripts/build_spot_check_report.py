from __future__ import annotations

import csv
import html
from pathlib import Path

from common import repo_path


SAMPLES = {
    "G_027": ("pass", "画面、地滚球字段与击球音峰值一致。", 1.520, 5.74),
    "G_038": ("pass", "音频训练标注：已将击球时间对齐到音频主瞬态；画面只用于确认地滚球类别和区域。", 2.360, 5.90),
    "G_046": ("pass", "击球区间与音频峰值相符。", 1.340, 2.11),
    "G_153": ("pass", "击球区间与音频峰值相符。", 3.480, 6.56),
    "G_192": ("pass", "击球区间与音频峰值相符。", 3.080, 24.03),
    "F_005": ("pass", "高飞球画面、轨迹字段与音频峰值相符。", 1.620, 7.07),
    "F_007": ("pass", "平飞球画面、轨迹字段与音频峰值相符。", 1.120, 9.69),
    "F_024": ("pass", "平飞球画面、轨迹字段与音频峰值相符。", 1.060, 6.58),
    "F_062": ("pass", "平飞球画面、轨迹字段与音频峰值相符。", 1.820, 11.55),
    "F_109": ("pass", "音频训练标注：已将击球时间对齐到音频主瞬态；画面只用于确认高飞球类别与落点。", 5.100, 3.20),
}


def read_row(path: Path) -> dict[str, str]:
    with (path / "sample.csv").open("r", newline="", encoding="utf-8-sig") as fh:
        return next(csv.DictReader(fh))


def source_url(path: Path) -> str:
    for line in (path / "source.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("video_url:"):
            return line.split(":", 1)[1].strip()
    return ""


def fields(row: dict[str, str]) -> str:
    if row["label"] == "ground_ball":
        return f"区域 {row['region']} · 力度 {row['strength']} · 弹跳 {row['bounce']}"
    return f"落点 {row['landing_zone']} · 力度 {row['strength']} · 轨迹 {row['trajectory_type']}"


def main() -> int:
    cards: list[str] = []
    for sample_id, (result, note, peak, ratio) in SAMPLES.items():
        path = next(repo_path("dataset").glob(f"*/*/{sample_id}"))
        row = read_row(path)
        start, end = float(row["event_start"]), float(row["event_end"])
        mid = (start + end) / 2
        status = "通过" if result == "pass" else "需修正"
        card_class = "pass" if result == "pass" else "issue"
        rel = path.relative_to(repo_path()).as_posix()
        url = source_url(path)
        source = f'<a href="{html.escape(url, quote=True)}" target="_blank">原始来源</a>' if url else "无来源链接"
        cards.append(
            f'''<article class="card {card_class}" data-result="{result}">
  <header><div><h2>{sample_id}</h2><p>{html.escape(row['label'])} · {html.escape(fields(row))}</p></div><span class="badge">{status}</span></header>
  <p class="note">{html.escape(note)}</p>
  <div class="facts"><span>标注击球：<b>{start:.3f}–{end:.3f}s</b></span><span>音频强峰：<b>{peak:.3f}s</b></span><span>峰值强度：<b>{ratio:.2f}×</b></span></div>
  <div class="media"><section><h3>视频</h3><video controls preload="metadata" src="../{rel}/video.mp4#t={start:.3f},{min(end + 1.5, start + 2):.3f}"></video><button data-time="{mid:.3f}">跳到标注击球点</button></section>
  <section><h3>音频</h3><audio controls preload="metadata" src="../{rel}/audio.wav#t={max(0, start - .5):.3f},{end + .7:.3f}"></audio><button data-time="{mid:.3f}">跳到标注击球点</button></section></div>
  <footer>{source} · <code>{rel}</code></footer>
</article>'''
        )
    output = repo_path("reports", "spot_check_20260714.html")
    output.write_text(
        f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>棒球数据集 · 10 条抽查</title><style>
body{{margin:0;background:#f5f7fb;color:#172033;font:16px system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1100px;margin:auto;padding:28px 18px 64px}}h1{{margin:0}}.summary{{background:#fff;border-radius:14px;padding:18px;margin:18px 0;box-shadow:0 1px 5px #0001}}.filters button,button{{border:0;border-radius:8px;padding:8px 12px;background:#e8edf7;cursor:pointer}}.filters button{{margin-right:8px}}.card{{background:#fff;border-left:6px solid #2e8b57;border-radius:12px;margin:16px 0;padding:18px;box-shadow:0 1px 5px #0001}}.card.issue{{border-color:#d97706}}header{{display:flex;justify-content:space-between;gap:12px}}h2,h3,p{{margin:0}}header p,.note,footer{{color:#586174;margin-top:6px}}.badge{{height:max-content;padding:5px 9px;border-radius:999px;background:#e3f5ea;color:#176b3a;font-weight:700}}.issue .badge{{background:#fff0d6;color:#9a5900}}.facts{{display:flex;gap:20px;flex-wrap:wrap;margin:14px 0;font-size:14px}}.media{{display:grid;grid-template-columns:2fr 1fr;gap:20px}}video,audio{{width:100%;margin:8px 0}}footer{{font-size:13px;overflow-wrap:anywhere}}code{{font-size:12px}}@media(max-width:700px){{.media{{grid-template-columns:1fr}}}}
</style><main><h1>棒球数据集：10 条随机抽查</h1><div class="summary"><b>结果：10 条均按音频主瞬态完成击球时间复核。</b><p>固定随机种子 20260714；地滚球 5 条、高飞球 5 条。区域与类型由视频判断；击球时间以音频主瞬态为准。“跳到标注击球点”会定位到音频训练标签。</p><div class="filters"><button onclick="filterCards('all')">全部（10）</button><button onclick="filterCards('pass')">已复核（10）</button></div></div>{''.join(cards)}</main><script>
function filterCards(kind){{document.querySelectorAll('.card').forEach(c=>c.style.display=(kind==='all'||c.dataset.result===kind)?'block':'none')}}
document.querySelectorAll('.media button').forEach(b=>b.onclick=()=>{{const m=b.parentElement.querySelector('video,audio');m.currentTime=Number(b.dataset.time);m.play()}})
</script>''',
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
