// PM2 process file for chain_reader.
// Start:  pm2 start chain_reader/ecosystem.config.js
// Logs:   pm2 logs chain_reader
//
// Runs `python -m chain_reader` from the albedo root so the chain_reader package
// imports cleanly. Uses the albedo-old venv (bittensor 10.4 + asyncpg). Env values
// are read from albedo/.env by config.py.
module.exports = {
  apps: [
    {
      name: "chain_reader",
      cwd: "",
      script: "",
      args: "-m chain_reader",
      interpreter: "none",
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      time: true,
    },
    {
      name: "hippius_validation",
      cwd: "",
      script: "",
      args: "-m hippius_validation",
      interpreter: "none",
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      time: true,
    },
  ],
};
