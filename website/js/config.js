export const POLL_MS = 4000;

export const DATA_ENDPOINTS = [
  "https://s3.hippius.com/albedo/data/dashboard.json",
  "https://us-east-1.hippius.com/albedo/data/dashboard.json",
  "https://eu-central-1.hippius.com/albedo/data/dashboard.json",
  "./data/dashboard.json",
];
export const STATE_ENDPOINTS = [
  "https://s3.hippius.com/albedo/data/state.json",
  "https://us-east-1.hippius.com/albedo/data/state.json",
  "https://eu-central-1.hippius.com/albedo/data/state.json",
  "./data/state.json",
];

export const LLMS_URLS = [
  "./llms.txt",
  "https://s3.hippius.com/albedo/llms.txt",
  "https://us-east-1.hippius.com/albedo/llms.txt",
  "https://eu-central-1.hippius.com/albedo/llms.txt",
];

export const ENDPOINT_CACHE_KEY = "albedo.endpoint.v3";
export const SUBNET_NETUID = 97;

export const ARTIFACT_TYPES = [
  { key: "EVAL_VERDICT", label: "verdict.json", type: "json" },
  { key: "GENERATED_SAMPLES", label: "generated-samples.jsonl", type: "jsonl" },
  { key: "SCORING_RESULTS", label: "scoring-results.jsonl", type: "jsonl" },
  { key: "JUDGE_RESULTS", label: "judge-results.jsonl", type: "jsonl" },
  { key: "EVAL_TRANSCRIPT", label: "duel-transcript.jsonl", type: "jsonl" },
  { key: "REMOTE_PROGRESS", label: "progress.jsonl", type: "jsonl" },
  { key: "REMOTE_LOGS", label: "remote-logs.txt", type: "text" },
  { key: "SANITY_RESULT", label: "sanity-result.json", type: "json" },
];
