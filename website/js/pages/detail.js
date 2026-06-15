import { ARTIFACT_TYPES } from "../config.js";
import { fetchDashboard } from "../fetch.js";
import { verdictInfo, faultCategory, faultCodeLabel } from "../data.js";
import { el, mount, link } from "../dom.js";
import { pct, fmtDateTime, shortHotkey, shortDigest } from "../format.js";
import { judgeMeta, hubRepoUrl, modelRepo, modelName, taoMinerUrl, kingTitleName } from "../model.js";

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const evalRunId = params.get("eval_run_id");
const submissionId = params.get("submission_id");

function setHead(name, sub, modelUri) {
  const repoUrl = modelUri && hubRepoUrl(modelUri);
  mount($("d-title"), repoUrl ? link(repoUrl, name, { title: modelRepo(modelUri) }) : name);
  $("d-sub").textContent = sub || "";
}

const kv = (k, v, cls) => el("div", { class: "kv" }, el("span", { class: "k" }, k), el("span", { class: cls ? "v " + cls : "v" }, v));

async function downloadZip(btn, entries, map, zipName) {
  if (typeof JSZip === "undefined") return;
  btn.disabled = true;
  const label = btn.textContent;
  btn.textContent = "zipping…";
  try {
    const zip = new JSZip();
    await Promise.all(entries.map(async a => {
      try { const r = await fetch(map[a.key]); if (r.ok) zip.file(a.label, await r.blob()); } catch {}
    }));
    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `${zipName || "artifacts"}.zip`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

function renderArtifacts(map, zipName) {
  const section = $("d-artifacts");
  const entries = ARTIFACT_TYPES.filter(a => map && map[a.key]);
  if (!entries.length) {
    mount(section, el("h2", {}, "artifacts"), el("div", { class: "note" }, "No artifacts recorded for this eval."));
    return;
  }
  const zipBtn = el("button", { class: "btn-zip" }, "↓ download all");
  zipBtn.addEventListener("click", () => downloadZip(zipBtn, entries, map, zipName));

  const rows = entries.map(a => el("div", { class: "artifact-row" },
    el("span", { class: "a-name" }, a.label),
    el("span", { class: "a-type" }, a.type),
    el("a", { href: map[a.key], download: a.label, class: "a-open" }, "download ↓")));

  mount(section,
    el("h2", {}, "artifacts"),
    el("div", { class: "artifacts-actions" }, zipBtn),
    el("div", { class: "artifact-list" }, rows));
}

function renderEval(r, netuid) {
  $("d-eyebrow").textContent = "eval result";
  setHead(modelName(r), `${r.eval_run_id} · uid ${r.uid ?? "—"} · ${shortHotkey(r.hotkey)}`, r.model_uri);

  const v = verdictInfo(r);
  const grid = el("div", { class: "kv-grid" },
    kv("result", v.badge, v.won ? "gold" : "bad"),
    kv("challenger", pct(v.chalMean), "gold"),
    kv("king", pct(v.kingMean)),
    kv("margin", v.winMargin != null ? `${Number(v.winMargin).toFixed(2)} pp` : "—"),
    kv("turns", r.total_turns != null ? `${r.valid_turns ?? r.total_turns}/${r.total_turns}` : "—"),
    kv("vllm errors", `${r.chal_vllm_errors ?? 0}c / ${r.king_vllm_errors ?? 0}k`),
    kv("finished", fmtDateTime(r.finished_at)));

  const byJudge = Object.entries(r.score_breakdown?.by_judge || {});
  const judgeTable = byJudge.length
    ? el("table", { class: "data-table judges-table" },
        el("thead", {}, el("tr", {}, el("th", {}, "judge"), el("th", { class: "r" }, "challenger"), el("th", { class: "r" }, "king"), el("th", { class: "center" }, "outcome"))),
        el("tbody", {}, byJudge.map(([model, chal]) => {
          const o = chal > 0.5 ? "win" : chal < 0.5 ? "lose" : "tie";
          return el("tr", {},
            el("td", { class: "judge", title: model }, judgeMeta(model).label),
            el("td", { class: "r" }, pct(chal)),
            el("td", { class: "r" }, pct(1 - chal)),
            el("td", { class: "center " + o }, o));
        })))
    : el("div", { class: "empty" }, "no judge breakdown.");

  const byMetric = r.score_breakdown?.by_metric || {};
  const metrics = Object.keys(byMetric).length
    ? el("div", { class: "kv-grid" }, Object.entries(byMetric).map(([k, val]) => kv(k, pct(val))))
    : null;

  const king = r.king || {};
  const tao = taoMinerUrl(netuid, king.hotkey);

  mount($("d-body"),
    grid,
    el("div", { class: "detail-section" }, el("h2", {}, "judges"), judgeTable),
    metrics ? el("div", { class: "detail-section" }, el("h2", {}, "metrics"), metrics) : false,
    el("div", { class: "detail-section" }, el("h2", {}, "king it faced"),
      el("div", { class: "kv-grid" },
        kv("king era", kingTitleName(king.king_version)),
        kv("king model", modelName(king)),
        kv("king uid", tao ? link(tao, String(king.uid ?? "—")) : (king.uid ?? "—")))));

  renderArtifacts(r.artifacts, `eval-${r.eval_run_id}`);
}

function renderFail(f) {
  $("d-eyebrow").textContent = "failed submission";
  setHead(modelName(f), `${f.submission_id} · uid ${f.uid ?? "—"} · ${shortHotkey(f.hotkey)}`, f.model_uri);

  const cat = faultCategory(f);
  const raw = (f.fault_message || f.fault_code || "").toString();
  const grid = el("div", { class: "kv-grid" },
    kv("category", cat.label, "bad"),
    kv("state", f.state || "—"),
    kv("fault class", f.fault_class || "—"),
    kv("fault code", f.fault_code || "—"),
    kv("model digest", shortDigest(f.model_hash)),
    kv("when", fmtDateTime(f.updated_at)));

  mount($("d-body"),
    grid,
    el("div", { class: "detail-section" }, el("h2", {}, "failure detail"),
      el("div", { class: "fail-panel" },
        el("div", { class: "fp-code" }, faultCodeLabel(f)),
        el("div", { class: "fp-detail" }, raw || "(no detail)"))));

  renderArtifacts(f.artifacts, `submission-${f.submission_id}`);
}

async function load() {
  const raw = await fetchDashboard();
  if (!raw) { setHead("unavailable", "could not load dashboard data"); return; }
  const netuid = raw.chain?.netuid;

  if (evalRunId) {
    const r = (raw.eval_runs || []).find(e => e.eval_run_id === evalRunId)
      || (raw.current_eval?.eval_run_id === evalRunId ? raw.current_eval : null);
    if (r) return renderEval(r, netuid);
    const f = (raw.fails || []).find(e => e.eval_run_id === evalRunId);
    if (f) return renderFail(f);
    setHead(evalRunId, "eval run not found");
    return;
  }

  if (submissionId) {
    const f = (raw.fails || []).find(e => e.submission_id === submissionId)
      || (raw.queue || []).find(e => e.submission_id === submissionId);
    if (f) return renderFail(f);
    setHead(submissionId, "submission not found");
    return;
  }

  setHead("missing id", "pass ?eval_run_id=… or ?submission_id=…");
}

load();
