import { toRoman } from "./format.js";

export const JUDGE_META = {
  "z-ai/glm-5.1":            { letter: "G", label: "GLM" },
  "qwen/qwen3.5-397b-a17b":  { letter: "Q", label: "QWEN" },
  "deepseek/deepseek-v3.2":  { letter: "D", label: "DEEPSEEK" },
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

// king_version is renumbered server-side (monitor.py) within the 35b lineage: the genesis
// seed is 0, the first miner-crowned king is 1. Show 0 as GENESIS, n>=1 as ALBEDO-<roman>,
// and anything else (null/pre-35b) as BASE MODEL.
export function kingTitleName(kingVersion) {
  const n = Number(kingVersion);
  if (kingVersion == null || !Number.isFinite(n) || n < 0) return "BASE MODEL";
  if (n === 0) return "GENESIS";
  const roman = toRoman(n);
  return roman ? `ALBEDO-${roman}` : "BASE MODEL";
}

// Friendly model identity: a crowned model is named by its reign (ALBEDO-II), everyone else by
// the first 6 chars of their hotkey (ALBEDO-5GcD3P). The real repo stays available as a tooltip.
export function modelName(item) {
  const v = Number(item?.king_version);
  if (item?.king_version != null && Number.isFinite(v) && v > 0) return kingTitleName(v);
  const hk = item?.hotkey;
  return hk ? `ALBEDO-${hk.slice(0, 6)}` : "—";
}

export function modelRepo(uri) {
  if (!uri) return "—";
  let s = uri.replace(/^[a-z][a-z0-9+.-]*:\/\//i, ""); // strip scheme:// (oci://, https://…)
  s = s.replace(/@[^/]*$/, "");                         // strip @sha256:… digest suffix
  const i = s.indexOf("/");
  if (i > 0 && s.slice(0, i).includes(".")) s = s.slice(i + 1); // strip registry host (e.g. registry.hippius.com/)
  return s || "—";
}

// hf:// URIs and bare 40/64-hex git revisions are HuggingFace models; sha256: digests are Hippius.
function isHfUri(uri) {
  if (!uri) return false;
  if (uri.startsWith("hf://")) return true;
  const tail = uri.split("@").pop();
  return /^[0-9a-f]{40}$/.test(tail) || /^[0-9a-f]{64}$/.test(tail);
}

export function hubRepoUrl(uri) {
  const repo = modelRepo(uri);
  if (!repo || repo === "—") return null;
  if (isHfUri(uri)) return `https://huggingface.co/${repo}`;
  const parts = repo.split("/");
  if (parts.length < 2) return "https://hub.hippius.com/models";
  return `https://hub.hippius.com/models/${parts[0]}/${parts.slice(1).join("/")}`;
}

export function taoMinerUrl(netuid, hotkey) {
  if (netuid == null || !hotkey) return null;
  return `https://taomarketcap.com/subnets/${netuid}/miners?query=${encodeURIComponent(hotkey)}`;
}
