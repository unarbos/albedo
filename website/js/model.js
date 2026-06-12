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

export function kingTitleName(reignNumber) {
  const n = Number(reignNumber);
  if (reignNumber == null || !Number.isFinite(n) || n <= 0) return "BASE MODEL";
  const roman = toRoman(n);
  return roman ? `ALBEDO-${roman}` : "BASE MODEL";
}

export function modelRepo(uri) {
  if (!uri) return "—";
  return uri.replace(/^[a-z0-9_-]+:/i, "");
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
