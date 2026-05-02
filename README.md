# NIFTY F&O Breadth — Streamlit dashboard

Free, broker-free, live-ish breadth view across the NSE F&O cash universe.
Polls Yahoo Finance every 60 seconds, computes breadth in memory, renders
a heatmap. **~15 min delayed**, no API key required.

## Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in a browser.

## Streamlit Cloud deploy (free)

Streamlit Cloud free tier supports **GitHub only** — your main repo on
GitLab is mirrored here as a small public GitHub repo containing just
this folder.

**One-time setup:**

1. Create a public GitHub repo, e.g. `<your-username>/nest-breadth-web`.
2. Copy the contents of this `streamlit_app/` folder into the GitHub repo's
   root (or wherever you point Streamlit Cloud at).
3. Sign in to https://streamlit.io/cloud → "Deploy app".
4. Pick the GitHub repo + branch + the `streamlit_app.py` file path.
5. Click Deploy. Streamlit gives you a URL like
   `https://nest-breadth-web.streamlit.app/`.

**Updating after changes:**

```bash
# Inside the parent green-field repo:
bash scripts/mirror_streamlit_to_github.sh /path/to/local/github/clone
cd /path/to/local/github/clone
git add . && git commit -m "Sync from green-field" && git push
# Streamlit Cloud auto-redeploys on push.
```

## Bundled files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Main dashboard script |
| `breadth_compute.py` | Pure scoring math (mirror of `python/src/nest_breadth_lite/breadth_compute.py`) |
| `universe.py` | F&O cash-universe CSV loader (mirror) |
| `universe.csv` | ~205 NSE symbols (mirror of `data/fno_cash_universe.csv`) |
| `requirements.txt` | Streamlit Cloud deps (no broker libs) |
| `.streamlit/config.toml` | Theme + server settings |

The mirror keeps the Streamlit app **completely self-contained** — no
imports from the parent `nest` package, no broker libraries. Streamlit
Cloud's slim image stays small.

## Caveats

- **Delay:** Yahoo Finance for NSE is delayed ~15 minutes. Useful for
  pattern-watching, NOT for live trading triggers.
- **Cold start:** Streamlit Cloud free tier sleeps the app after ~30 min
  of inactivity. First visit after sleep takes ~30 sec to wake.
- **Coverage:** Yahoo occasionally lags on newly-added F&O constituents.
  The dashboard's "missing symbols" expander shows which symbols Yahoo
  didn't return on the latest poll.
- **Rate limits:** at one batch call per 60 sec, well below Yahoo's
  throttling threshold. Don't reduce the poll interval below 30 sec.
