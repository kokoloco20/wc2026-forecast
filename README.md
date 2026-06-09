# World Cup 2026 Forecast

Elo + Poisson Monte Carlo forecast of the 2026 World Cup, blended 50/50 with
bookmaker outright odds. Dashboard with per-round probabilities, championship
odds, odds history over time, per-match xG / score predictions, and the
expected bracket.

**Live dashboard:** served via GitHub Pages from this repo (`index.html` + `forecast.json`).

## How it updates

- **Automatic:** a GitHub Action runs `update.py` every day at 06:00 UTC,
  picking up the previous matchday's results, and commits a fresh
  `forecast.json`.
- **Manual:** Actions tab → *Update forecast* → *Run workflow*.
- **Local:** `pip install -r requirements.txt`, then `python update.py` and
  open `index.html`.

## How it works

1. Downloads all international results (1872–present) from
   [martj42/international_results](https://github.com/martj42/international_results)
2. Computes World-Football-style Elo ratings (K by importance, margin
   multiplier, home advantage)
3. Fits a Poisson goal model on competitive matches since 2010:
   `λ = exp(a + b · Δelo/400)`
4. Converts bookmaker outright odds (margin-removed) to an implied Elo and
   blends 50/50 with computed Elo (`MARKET_WEIGHT` in `update.py`)
5. Treats already-played WC 2026 matches (auto-detected from the data) as
   fixed facts
6. Simulates the remaining tournament 20,000 times with the official 48-team
   format: 12 groups, 8 best third-placed teams, official bracket constraints

## Knobs (top of `update.py`)

| Setting | Default | Meaning |
|---|---|---|
| `MARKET_WEIGHT` | 0.5 | 0 = pure Elo, 1 = pure market |
| `MARKET_ODDS` | bet365 2026-06-09 | paste fresh outright odds anytime |
| `HOST_ADV_GROUP` / `HOST_ADV_KO` | 50 / 25 | Elo bonus for USA/Mexico/Canada |
| `EXTRA_PLAYED` | empty | results the source CSV doesn't have yet |
