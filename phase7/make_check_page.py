#!/usr/bin/env python3
"""
Build the human check page for candidate floods: before photo, after photo,
after photo with the computer's flood marks, and Right / Wrong / Not sure
buttons whose answers are saved in the page's database (collection given by
--collection), where Claude can read them back.

    ./.venv/bin/python -u phase7/make_check_page.py --labels data/trace_v3/labels/candidates_v1

Output: handoff/flood_check/index.html and one image per flood
"""
from __future__ import annotations
import argparse, html, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "handoff/flood_check"


def money(s):
    s = str(s)
    return "no damage listed" if s in ("0.00K", "nan", "None") else "$" + s.replace(".00", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--collection", default="checks_v1")
    args = ap.parse_args()

    lab = ROOT / args.labels
    dec = pd.read_csv(lab / "decisions.csv", dtype={"event": str})
    cand = dec[dec.skip.isna()].copy()
    ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    ev["EVENT_ID"] = ev.EVENT_ID.astype(str)
    place = ev.set_index("EVENT_ID")[["CZ_NAME", "STATE"]]
    cand = cand.sort_values(["flood_date"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)

    cards = []
    for k, r in cand.iterrows():
        eid = r.event
        z = np.load(ROOT / "data/trace_v3" / args.stage / "arrays" / f"{eid}.npz")
        with __import__("rasterio").open(lab / f"{eid}.tif") as s:
            m = s.read(1) > 0.5
        over = z["rgb_after"].copy()
        over[m] = (0.25 * over[m] + 0.75 * np.array([235, 40, 20], "float32")).astype("uint8")
        S = 500
        row = Image.new("RGB", (3 * S + 20, S), "white")
        for j, a in enumerate((z["rgb_before"], z["rgb_after"], over)):
            row.paste(Image.fromarray(a).resize((S, S)), (j * (S + 10), 0))
        row.save(OUT / f"f{eid}.jpg", quality=84)
        pl = place.loc[eid] if eid in place.index else None
        where = f"{str(pl.CZ_NAME).title()}, {str(pl.STATE).title()}" if pl is not None else ""
        d = pd.Timestamp(r.flood_date).strftime("%-d %b %Y")
        cards.append(f"""
    <article class="flood" data-event="{eid}">
      <a class="shot" href="f{eid}.jpg" target="_blank" rel="noopener">
        <img loading="lazy" src="f{eid}.jpg" alt="Before photo, after photo, and after photo with red flood marks, flood {k + 1}">
      </a>
      <div class="meta">
        <div class="facts">
          <h2>{k + 1}. {html.escape(where)}</h2>
          <p class="mono">{d} &middot; {money(r.damage)} &middot; photo {int(r.days_after)} days after &middot; {r.marked_pct:.1f}% marked &middot; event {eid}</p>
        </div>
        <div class="vote" role="group" aria-label="Your check for flood {k + 1}">
          <button type="button" id="v{eid}-right" data-v="right" disabled>Right</button>
          <button type="button" id="v{eid}-wrong" data-v="wrong" disabled>Wrong</button>
          <button type="button" id="v{eid}-unsure" data-v="unsure" disabled>Not sure</button>
        </div>
      </div>
    </article>""")

    page = f"""<title>Flood mark check</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
  :root {{ --ground:#f2f4f5; --ink:#17212b; --muted:#56636e; --rule:#d3d9dd; --accent:#2e6e73; --flood:#d23a24; --frame:#ffffff; --bar:#e6eaec; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --ground:#10161b; --ink:#e4e9ec; --muted:#9aa6af; --rule:#2a343c; --accent:#72b5b8; --flood:#ff6a4d; --frame:#1a2229; --bar:#1d252c; }} }}
  :root[data-theme="dark"] {{ --ground:#10161b; --ink:#e4e9ec; --muted:#9aa6af; --rule:#2a343c; --accent:#72b5b8; --flood:#ff6a4d; --frame:#1a2229; --bar:#1d252c; }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--ground); color: var(--ink); font: 400 16px/1.6 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; padding-inline: 20px; margin: 0; }}
  main {{ max-width: 1120px; margin: 0 auto; padding-block: 36px 80px; display: grid; gap: 36px; }}
  h1, h2 {{ font-family: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif; font-weight: 600; margin: 0; line-height: 1.15; text-wrap: balance; }}
  h1 {{ font-size: clamp(30px, 4.2vw, 42px); }}
  h2 {{ font-size: 20px; }}
  p {{ margin: 0; max-width: 68ch; }}
  .mono {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 13px; color: var(--muted); letter-spacing: 0.02em; font-variant-numeric: tabular-nums; }}
  header {{ display: grid; gap: 14px; }}
  .how {{ display: grid; gap: 8px; padding: 0; margin: 0; list-style: none; max-width: 70ch; }}
  .how li {{ padding-left: 14px; border-left: 3px solid var(--rule); }}
  .how li.r {{ border-left-color: var(--accent); }}
  .how li.w {{ border-left-color: var(--flood); }}
  .counter {{ position: sticky; top: 0; z-index: 2; background: var(--ground); padding-block: 10px; border-bottom: 1px solid var(--rule); display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: center; }}
  .track {{ flex: 1 1 200px; height: 8px; background: var(--bar); border-radius: 4px; overflow: hidden; min-width: 160px; }}
  .fill {{ height: 100%; width: 0; background: var(--accent); transition: width .3s; }}
  @media (prefers-reduced-motion: reduce) {{ .fill {{ transition: none; }} }}
  .floods {{ display: grid; gap: 34px; }}
  .flood {{ display: grid; gap: 12px; }}
  .shot {{ display: block; border-radius: 3px; }}
  .shot:focus-visible {{ outline: 3px solid var(--accent); outline-offset: 3px; }}
  .shot img {{ display: block; width: 100%; max-width: 100%; height: auto; aspect-ratio: 1520 / 500; background: var(--frame); border: 1px solid var(--rule); border-radius: 3px; }}
  .meta {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px 24px; }}
  .facts {{ display: grid; gap: 2px; }}
  .vote {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .vote button {{ font: 500 15px/1 "IBM Plex Sans", system-ui, sans-serif; color: var(--ink); background: transparent; border: 1px solid var(--rule); border-radius: 4px; padding: 10px 16px; cursor: pointer; }}
  .vote button:hover:not(:disabled) {{ border-color: var(--muted); }}
  .vote button:focus-visible {{ outline: 3px solid var(--accent); outline-offset: 2px; }}
  .vote button:disabled {{ opacity: .55; cursor: default; }}
  .vote button[data-v="right"][aria-pressed="true"] {{ background: var(--accent); border-color: var(--accent); color: var(--ground); }}
  .vote button[data-v="wrong"][aria-pressed="true"] {{ background: var(--flood); border-color: var(--flood); color: var(--ground); }}
  .vote button[data-v="unsure"][aria-pressed="true"] {{ background: var(--ink); border-color: var(--ink); color: var(--ground); }}
  .labels {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
  .note[hidden] {{ display: none; }}
</style>

<main>
  <header>
    <p class="mono">Arid flash floods &middot; Arizona, California, Nevada &middot; {len(cand)} candidate floods</p>
    <h1>Flood mark check</h1>
    <p>Each row shows one 10 km square: the photo before the flood, the photo after it, and the after photo with the computer's flood marks in red. Only floods you mark Right become training labels.</p>
    <ul class="how">
      <li class="r"><b>Right</b>: the red follows fresh pale sand or mud along washes or channels, ground that looks different after the flood.</li>
      <li class="w"><b>Wrong</b>: the red sits on towns, house lots, fields, cloud edges, mountain shadows, or on ground that looks the same in both photos.</li>
      <li><b>Not sure</b>: you can't tell. Tap a row to open it full size first.</li>
    </ul>
    <div class="labels mono" aria-hidden="true"><span>before</span><span>after</span><span>after + red marks</span></div>
  </header>

  <div class="counter" aria-live="polite">
    <span class="mono" id="count">0 of {len(cand)} checked</span>
    <div class="track"><div class="fill" id="fill"></div></div>
    <span class="mono" id="tally"></span>
    <span class="mono note" id="note" hidden>Saving is not available in this view. Tell me your answers in the chat.</span>
  </div>

  <section class="floods">{''.join(cards)}
  </section>
</main>

<script>
(async () => {{
  const cards = [...document.querySelectorAll(".flood")];
  const total = cards.length, state = {{}};
  const render = () => {{
    const vals = Object.values(state);
    const n = vals.length;
    document.getElementById("count").textContent = n + " of " + total + " checked";
    document.getElementById("fill").style.width = (100 * n / Math.max(total, 1)) + "%";
    const c = v => vals.filter(x => x === v).length;
    document.getElementById("tally").textContent = n ? "right " + c("right") + " · wrong " + c("wrong") + " · not sure " + c("unsure") : "";
    cards.forEach(card => card.querySelectorAll("button").forEach(b =>
      b.setAttribute("aria-pressed", String(state[card.dataset.event] === b.dataset.v))));
  }};
  const db = window.claude && window.claude.use ? await window.claude.use("db") : null;
  if (!db) {{ document.getElementById("note").hidden = false; return; }}
  const col = db.collection("{args.collection}");
  cards.forEach(card => card.querySelectorAll("button").forEach(b => {{
    b.disabled = false;
    b.addEventListener("click", async () => {{
      const id = card.dataset.event;
      state[id] = b.dataset.v; render();
      try {{ await col.doc(id).set({{ verdict: b.dataset.v, at: new Date().toISOString() }}); }}
      catch (e) {{ document.getElementById("note").hidden = false; }}
    }});
  }}));
  col.onSnapshot(snap => {{ snap.docs.forEach(d => {{ const v = (d.data() || {{}}).verdict; if (v) state[d.id] = v; }}); render(); }}, () => {{}});
}})();
</script>
"""
    (OUT / "index.html").write_text(page)
    (OUT / "floods.json").write_text(json.dumps(cand.event.tolist()))
    print(f"wrote {OUT / 'index.html'} with {len(cand)} floods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
