# Signal

PV/MA buy/sell daily screener.

`screener.py` scans a watchlist, fits a Student-t tail parameter (nu) to
recent returns to gauge signal reliability, and writes the result to
`index.html` — no notebook or display needed.

## Running locally

```
pip install -r requirements.txt
python screener.py
```

Open the generated `index.html` in a browser.

## Editing the watchlist / thresholds

Edit the `WATCHLIST` list and the config constants near the top of
`screener.py`.

## Daily automation

`.github/workflows/daily-screener.yml` runs the screener every day at
18:00 Singapore Time (10:00 UTC), regenerates `index.html`, and commits it
back to the branch the workflow runs on. It also supports manual runs via
the "Run workflow" button (workflow_dispatch).

Scheduled workflows only fire from the repository's **default branch**, so
merge this workflow there for the daily cron to run.

## Viewing the report as a webpage (optional)

To browse `index.html` at a URL instead of opening the file locally, enable
GitHub Pages: repo Settings → Pages → Source: "Deploy from a branch" →
select the default branch and `/ (root)`.
