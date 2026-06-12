export const POLL_MS = 4000;

export const DATA_ENDPOINTS = ["./data/dashboard.json"];

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
];
