import { toRoman } from "./format.js";
import { EVALS_BASE } from "./config.js";

export const JUDGE_META = {
  "deepseek-ai/DeepSeek-V3.2-TEE": { letter: "D", label: "DEEPSEEK" },
  "Qwen/Qwen3-235B-A22B-Thinking-2507": { letter: "Q", label: "QWEN" },
  "moonshotai/Kimi-K2.6-TEE": { letter: "K", label: "KIMI" },
};

export function judgeShortName(model) {
  if (!model) return "—";
  const parts = model.split("/");
  let name = parts[parts.length - 1];
  if (name.length > 18) name = name.slice(0, 16) + "…";
  return name;
}

export function judgeMeta(model) {
  if (JUDGE_META[model]) return JUDGE_META[model];
  const short = judgeShortName(model);
  return { letter: short.charAt(0).toUpperCase(), label: short.toUpperCase().slice(0, 12) };
}

export function kingTitleName(reignNumber) {
  const n = Number(reignNumber);
  if (reignNumber == null || !Number.isFinite(n) || n <= 0) return "base model";
  const roman = toRoman(n);
  return roman ? `ALBEDO-${roman}` : "base model";
}

export function challengerDisplayName(hotkey) {
  if (!hotkey) return "—";
  return `ALBEDO-${hotkey.slice(0, 5).toUpperCase()}`;
}

export function hubRepoUrl(repo) {
  if (!repo) return null;
  const parts = repo.split("/");
  if (parts.length < 2) return "https://hub.hippius.com/models";
  return `https://hub.hippius.com/models/${parts[0]}/${parts.slice(1).join("/")}`;
}

export function modelLinkHtml(repo, digest, label) {
  const text = label || "—";
  const url = hubRepoUrl(repo);
  if (!url || text === "—") return text;
  const title = digest ? `${repo}@${digest}` : (repo || text);
  return `<a href="${url}" target="_blank" rel="noopener" title="${title}">${text}</a>`;
}

export function taoMinerUrl(netuid, hotkey) {
  if (netuid == null || !hotkey) return null;
  return `https://taomarketcap.com/subnets/${netuid}/miners?query=${encodeURIComponent(hotkey)}`;
}

// Eval-artifact directory URL for a history entry. The backend doesn't emit
// eval_dir_url/evals_url, so derive it from eval_id + completed_at, which map
// 1:1 to the S3 layout: <EVALS_BASE>/<YYYY-MM-DD>/<NNN>/{scores.json,...}.
// Failures never produce scores/rollouts, so they have no artifact dir — return
// null so the detail page renders the fail record instead of 404ing on scores.json.
export function evalDirUrl(h) {
  if (h?.eval_dir_url) return h.eval_dir_url;
  if (h?.evals_url)    return h.evals_url;
  if (h?.error_code || h?.code) return null;
  const m    = String(h?.eval_id || "").match(/(\d+)\s*$/);
  const date = String(h?.completed_at || "").slice(0, 10);
  if (!m || !/^\d{4}-\d{2}-\d{2}$/.test(date)) return null;
  const dir = String(parseInt(m[1], 10)).padStart(3, "0");
  return `${EVALS_BASE}${date}/${dir}/`;
}
