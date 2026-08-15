# Contributing

Thanks for taking an interest in Free Model Pulse.

## Ground rules

1. Keep commits under your own GitHub identity only.
2. Do not add Co authored by trailers for bots or AI agents.
3. Prefer small, focused pull requests.
4. Run tests before opening a PR: `pytest tests/ -v`

## Local setup

```bash
git clone https://github.com/mohabdelkarim/free-model-pulse.git
cd free-model-pulse
pip install -r requirements.txt
cp .env.example .env
```

Add your `OPENROUTER_API_KEY` to `.env`, then:

```bash
python watch_models.py
python benchmark.py
python analyze.py --min-runs 1 --update-readme --write-pages
```

## What to change

1. Discovery logic lives in `watch_models.py`.
2. Benchmarks live in `benchmark.py`.
3. Aggregation and published tables live in `analyze.py`.
4. Shared helpers live in `common.py`.

## Pull requests

1. Describe the why in a few sentences.
2. Link related issues when they exist.
3. Keep secrets out of the repo.

## Issues

Use the bug report template for failures. Include workflow run links when CI is involved.
