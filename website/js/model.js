import { toRoman } from "./format.js";
import { GENESIS_KING_VERSION } from "./config.js";

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

// Roman reign names are reserved for the 35b lineage. The 35b genesis seed
// (king_version === GENESIS_KING_VERSION) shows as GENESIS; the first miner-crowned
// 35b king (genesis + 1) is ALBEDO-I; pre-35b (4b era) kings are BASE MODEL.
export function kingTitleName(kingVersion) {
  const n = Number(kingVersion);
  if (kingVersion == null || !Number.isFinite(n)) return "BASE MODEL";
  if (n === GENESIS_KING_VERSION) return "GENESIS";
  const reign = n - GENESIS_KING_VERSION; // kv 41 -> 1 -> ALBEDO-I
  if (reign < 1) return "BASE MODEL";
  const roman = toRoman(reign);
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

export function hubRepoUrl(uri) {
  const repo = modelRepo(uri);
  if (!repo || repo === "—") return null;
  const parts = repo.split("/");
  if (parts.length < 2) return "https://hub.hippius.com/models";
  return `https://hub.hippius.com/models/${parts[0]}/${parts.slice(1).join("/")}`;
}

export function taoMinerUrl(netuid, hotkey) {
  if (netuid == null || !hotkey) return null;
  return `https://taomarketcap.com/subnets/${netuid}/miners?query=${encodeURIComponent(hotkey)}`;
}
