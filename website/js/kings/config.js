export const POLL_MS = 8000;
export const NETUID  = 97;
export const BITTENSOR_BLOCK_TIME_S = 12;

export const DATA_ENDPOINTS = [
  "../dashboard.json",
  "https://us-east-1.hippius.com/albedo/dashboard.json",
  "https://eu-central-1.hippius.com/albedo/dashboard.json",
  "https://s3.hippius.com/albedo/dashboard.json",
];

export const EVALS_BASE = "https://us-east-1.hippius.com/albedo/evals/";

export const JUDGE_META = {
  "minimax/minimax-m3":         { letter: "M" },
  "deepseek/deepseek-v4-flash": { letter: "D" },
  "z-ai/glm-5.1":               { letter: "G" },
};
