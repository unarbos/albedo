export const POLL_MS = 4000;
export const BUILD_ID = document.querySelector('meta[name="build"]').content;

export const DATA_ENDPOINTS = [
  "../dashboard.json",
  "https://us-east-1.hippius.com/albedo/dashboard.json",
  "https://eu-central-1.hippius.com/albedo/dashboard.json",
  "https://s3.hippius.com/albedo/dashboard.json",
];
export const HTML_ENDPOINTS = [
  "https://us-east-1.hippius.com/albedo/index.html",
  "https://eu-central-1.hippius.com/albedo/index.html",
  "https://s3.hippius.com/albedo/index.html",
];
export const LLMS_URLS = [
  "../llms.txt",
  "https://us-east-1.hippius.com/albedo/llms.txt",
  "https://eu-central-1.hippius.com/albedo/llms.txt",
  "https://s3.hippius.com/albedo/llms.txt",
];
export const SWEBENCH_LITE_URLS = [
  "../swebench-lite/index.json",
  "https://us-east-1.hippius.com/albedo/swebench-lite/index.json",
  "https://s3.hippius.com/albedo/swebench-lite/index.json",
];

export const ENDPOINT_CACHE_KEY = "albedo.endpoint.v1";
export const EVALS_BASE = "https://us-east-1.hippius.com/albedo/evals/";
export const SUBNET_ALPHA_DAY = 2960;
export const BITTENSOR_BLOCK_TIME_S = 12;
export const EVO_SCALE_MAX = 1.0;
