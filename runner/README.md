# Runner

Run the current fake-env smoke experiment:

```bash
python3 -m runner.run --config runner/configs/fake_smoke.yaml
```

Run through the current `env.gym.Env` adapter:

```bash
python3 -m runner.run --config runner/configs/real_smoke.yaml
```

Aggregate saved `EpisodeResult` files into `reports/`:

```bash
python3 -m runner.aggregate \
  --input runs/fake_smoke \
  --output reports/fake_smoke_summary
```

Run the end-to-end smoke check in a temporary directory:

```bash
python3 -m runner.smoke_check
```

The runner supports `env: real` through `env.runner_adapter.build_real_env` and
keeps `env: fake` for deterministic runner tests. Keep the `EpisodeResult` JSON
format unchanged; the aggregator only depends on that frozen contract.
