# Runner

Run the current fake-env smoke experiment:

```bash
python3 -m runner.run --config runner/configs/fake_smoke.yaml
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

The runner currently supports `env: fake`. Replacing it with the real
environment should happen in `runner.env_factory.build_env`; the aggregator only
depends on the frozen `EpisodeResult` JSON format.
