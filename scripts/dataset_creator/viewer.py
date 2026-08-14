#!/usr/bin/env python3

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config

BASE = None

data: dict[str, dict[str, list[dict]]] = {}


def load() -> None:
    for model_dir in sorted(BASE.iterdir()):
        if not model_dir.is_dir():
            continue
        files = {}
        for pq in sorted(model_dir.glob("*.parquet")):
            df = pd.read_parquet(pq)
            files[pq.name] = [
                {"sample_id": r.sample_id, "messages": [dict(m) for m in r.messages]}
                for r in df.itertuples()
            ]
        if files:
            data[model_dir.name] = files
    n = sum(len(rows) for files in data.values() for rows in files.values())
    print(
        f"loaded {n} samples from {sum(len(f) for f in data.values())} parquets "
        f"({len(data)} models)"
    )


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reference Trajectory Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #10151f; color: #dbe2ee; font: 15px/1.55 system-ui, sans-serif;
         display: flex; height: 100vh; overflow: hidden; }
  #sidebar { width: 330px; min-width: 330px; background: #171e2b; border-right: 1px solid #232c3d;
             display: flex; flex-direction: column; }
  #sidebar .controls { padding: 12px; border-bottom: 1px solid #232c3d; display: flex;
                       flex-direction: column; gap: 8px; }
  select, input[type=text] { background: #0f1420; color: #dbe2ee; border: 1px solid #2a3548;
             border-radius: 6px; padding: 7px 9px; font-size: 14px; width: 100%; }
  #samples { flex: 1; overflow-y: auto; }
  .sample-item { padding: 9px 12px; border-bottom: 1px solid #1d2534; cursor: pointer;
                 font-size: 12.5px; color: #9fb0c8; word-break: break-all; }
  .sample-item:hover { background: #1d2637; }
  .sample-item.active { background: #1d3a6e; color: #e8eefb; }
  .sample-item .meta { color: #64758f; font-size: 11px; margin-top: 2px; }
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #topbar { padding: 12px 20px; background: #171e2b; border-bottom: 1px solid #232c3d;
            display: flex; align-items: center; gap: 12px; }
  #topbar h1 { font-size: 15px; font-weight: 600; color: #e8eefb; }
  #topbar .sid { font-size: 12px; color: #7f92ad; font-family: ui-monospace, monospace;
                 word-break: break-all; flex: 1; }
  button.nav { background: #232d40; color: #dbe2ee; border: 1px solid #2f3b52; border-radius: 6px;
               padding: 6px 14px; cursor: pointer; font-size: 13px; }
  button.nav:hover { background: #2b3750; }
  #chat { flex: 1; overflow-y: auto; padding: 24px 28px 60px; }
  .msg { max-width: 900px; margin: 0 auto 18px; border-radius: 12px; padding: 14px 18px;
         white-space: pre-wrap; word-break: break-word; }
  .msg .role-tag { display: block; font-size: 10.5px; font-weight: 700; letter-spacing: .08em;
                   text-transform: uppercase; margin-bottom: 6px; opacity: .65; white-space: normal; }
  .msg.system { background: #1652d8; color: #f0f5ff; }
  .msg.user   { background: #1a66f5; color: #f0f5ff; }
  .msg.user.observation { background: #1a66f5; margin-left: 120px; }
  .msg.assistant { background: #1b2333; border: 1px solid #242f44; margin-right: 120px; }
  .codeblock { background: #0d1220; border: 1px solid #222c40; border-radius: 8px;
               margin: 10px 0; overflow: hidden; white-space: normal; }
  .codeblock .cb-head { display: flex; justify-content: space-between; align-items: center;
               background: #182034; padding: 5px 12px; font-size: 12px; color: #93a5c0; }
  .codeblock .cb-head button { background: none; border: none; color: #93a5c0; cursor: pointer;
               font-size: 12px; }
  .codeblock .cb-head button:hover { color: #e8eefb; }
  .codeblock pre { padding: 12px 14px; overflow-x: auto; font: 13px/1.5 ui-monospace, monospace;
                   color: #cdd9ec; white-space: pre; }
  .msg.system .codeblock, .msg.user .codeblock { background: #0e2e7a; border-color: #2c55c0; }
  .msg.system .codeblock .cb-head, .msg.user .codeblock .cb-head { background: #10389a; color: #b9ccf5; }
  .msg.system .codeblock pre, .msg.user .codeblock pre { color: #e3ecff; }
  #empty { color: #64758f; text-align: center; margin-top: 120px; }
</style>
</head>
<body>
<div id="sidebar">
  <div class="controls">
    <select id="model"></select>
    <select id="file"></select>
    <input id="filter" type="text" placeholder="filter by sample_id...">
  </div>
  <div id="samples"></div>
</div>
<div id="main">
  <div id="topbar">
    <h1>Conversational View</h1>
    <span class="sid" id="sid"></span>
    <button class="nav" id="prev">&larr; Prev</button>
    <button class="nav" id="next">Next &rarr;</button>
  </div>
  <div id="chat"><div id="empty">Select a sample</div></div>
</div>
<script>
let index = {};        // model -> {file: n_rows}
let list = [];         // current file's [{idx, sample_id, n_messages}]
let cur = -1;

const $ = id => document.getElementById(id);

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderContent(text) {
  const re = /```(\w*)\n?([\s\S]*?)```/g;
  let out = '', last = 0, m;
  while ((m = re.exec(text)) !== null) {
    out += esc(text.slice(last, m.index));
    out += '<div class="codeblock"><div class="cb-head"><span>' + esc(m[1] || 'code') +
           '</span><button onclick="copyCode(this)">Copy</button></div><pre>' +
           esc(m[2].replace(/\n$/, '')) + '</pre></div>';
    last = re.lastIndex;
  }
  out += esc(text.slice(last));
  return out;
}

function copyCode(btn) {
  const pre = btn.closest('.codeblock').querySelector('pre');
  navigator.clipboard.writeText(pre.textContent).then(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 1200);
  });
}

async function loadIndex() {
  index = await (await fetch('/api/index')).json();
  const sel = $('model');
  sel.innerHTML = Object.keys(index).map(m => `<option>${esc(m)}</option>`).join('');
  onModel();
}

function onModel() {
  const files = Object.keys(index[$('model').value]);
  $('file').innerHTML = files.map(f => `<option>${esc(f)}</option>`).join('');
  onFile();
}

async function onFile() {
  const q = `model=${encodeURIComponent($('model').value)}&file=${encodeURIComponent($('file').value)}`;
  list = await (await fetch('/api/list?' + q)).json();
  cur = -1;
  renderList();
  if (list.length) show(0);
}

function renderList() {
  const f = $('filter').value.toLowerCase();
  $('samples').innerHTML = list
    .filter(s => !f || s.sample_id.toLowerCase().includes(f))
    .map(s => `<div class="sample-item${s.idx === cur ? ' active' : ''}" onclick="show(${s.idx})">` +
              `${esc(s.sample_id)}<div class="meta">${s.n_messages} messages</div></div>`)
    .join('');
}

async function show(idx) {
  cur = idx;
  renderList();
  const q = `model=${encodeURIComponent($('model').value)}&file=${encodeURIComponent($('file').value)}&idx=${idx}`;
  const s = await (await fetch('/api/sample?' + q)).json();
  $('sid').textContent = s.sample_id;
  $('chat').innerHTML = s.messages.map((m, i) => {
    const obs = m.role === 'user' && i > 1 ? ' observation' : '';
    const tag = obs ? 'observation' : m.role;
    return `<div class="msg ${m.role}${obs}"><span class="role-tag">${tag}</span>${renderContent(m.content)}</div>`;
  }).join('');
  $('chat').scrollTop = 0;
}

$('model').onchange = onModel;
$('file').onchange = onFile;
$('filter').oninput = renderList;
$('prev').onclick = () => { if (cur > 0) show(cur - 1); };
$('next').onclick = () => { if (cur >= 0 && cur < list.length - 1) show(cur + 1); };
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') $('prev').click();
  if (e.key === 'ArrowRight') $('next').click();
});

loadIndex();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/index":
            self._json(
                {m: {f: len(rows) for f, rows in files.items()} for m, files in data.items()}
            )
        elif url.path == "/api/list":
            rows = data.get(q.get("model", ""), {}).get(q.get("file", ""))
            if rows is None:
                return self._json({"error": "not found"}, 404)
            self._json(
                [
                    {"idx": i, "sample_id": r["sample_id"], "n_messages": len(r["messages"])}
                    for i, r in enumerate(rows)
                ]
            )
        elif url.path == "/api/sample":
            rows = data.get(q.get("model", ""), {}).get(q.get("file", ""))
            try:
                row = rows[int(q["idx"])]
            except (TypeError, KeyError, ValueError, IndexError):
                return self._json({"error": "not found"}, 404)
            self._json(row)
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *args):
        pass


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="dataset dir (default: <data_dir>/hf_out from config)",
    )
    args = ap.parse_args()
    BASE = args.dir if args.dir else Config().out_dir
    load()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"viewer at http://localhost:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
