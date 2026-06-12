# Reign/Weight PM2 Dev Notes

## Env

Both use `ALBEDO_EVAL_DATABASE_URL` from `.env`.

Weight setter also needs:

```bash
ALBEDO_WEIGHT_COLDKEY=
ALBEDO_WEIGHT_HOTKEY=
ALBEDO_WEIGHT_WALLET_PATH=
ALBEDO_WEIGHT_NETWORK=finney
ALBEDO_WEIGHT_NETUID=97
ALBEDO_WEIGHT_SET_RATE_BLOCKS=100
ALBEDO_WEIGHT_BURN_UID=0
```

## Start

```bash
pm2 start pm2/ecosystem.set-reign-worker.config.js
pm2 start pm2/ecosystem.weight-setter.config.js
```

## Stop

```bash
pm2 stop albedo-set-reign-worker
pm2 stop albedo-weight-setter
```

Remove from PM2:

```bash
pm2 delete albedo-set-reign-worker
pm2 delete albedo-weight-setter
```

## Logs

```bash
pm2 logs albedo-set-reign-worker --lines 100
pm2 logs albedo-weight-setter --lines 100
```

## Notes

- Set-reign consumes `EVAL_WIN` / `SET_REIGN_RETRYABLE`, writes active reign, then creates `weight_epochs`.
- Weight setter consumes `weight_epochs`, calls `set_weights`, and creates `PERIODIC_REFRESH` every `ALBEDO_WEIGHT_SET_RATE_BLOCKS` blocks.
- With genesis only, weights are `[uid 0] -> [1.0]`.
