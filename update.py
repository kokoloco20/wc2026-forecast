"""
update.py  –  run once before/after each match day
  1. Downloads latest results.csv from martj42/international_results
  2. Recomputes Elo
  3. Treats already-played WC 2026 matches as fixed facts
  4. Simulates the remaining 20 000 tournaments
  5. Writes forecast.json (consumed by index.html)
"""
import json, urllib.request, os, sys
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# ── paths ───────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
CSV    = os.path.join(BASE, "results.csv")
OUT    = os.path.join(BASE, "forecast.json")
HIST   = os.path.join(BASE, "odds_history.json")   # rolling append
URL    = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

N_SIMS = 20_000
rng = np.random.default_rng()                       # fresh seed each run

# ── Elo engine ───────────────────────────────────────────────────────────────
HOME_ADV = 100.0

def k_factor(tournament: str) -> float:
    t = tournament.lower()
    if t == "fifa world cup":                        return 60
    if any(s in t for s in ["euro", "copa américa", "copa america",
                             "africa cup", "asian cup", "gold cup",
                             "confederations"]):
        return 50 if "qualification" not in t else 40
    if "qualification" in t:                         return 40
    if "friendly" in t:                              return 20
    return 30

def margin_mult(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    if gd == 3: return 1.75
    return 1.75 + (gd - 3) / 8.0

def compute_elo(df: pd.DataFrame):
    elo = defaultdict(lambda: 1500.0)
    pre_h, pre_a = [], []
    for row in df.itertuples(index=False):
        rh, ra = elo[row.home_team], elo[row.away_team]
        pre_h.append(rh); pre_a.append(ra)
        adv = 0.0 if row.neutral else HOME_ADV
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + adv) - ra) / 400.0))
        res_h = (0.5 if row.home_score == row.away_score
                 else float(row.home_score > row.away_score))
        delta = (k_factor(row.tournament)
                 * margin_mult(row.home_score - row.away_score)
                 * (res_h - exp_h))
        elo[row.home_team] = rh + delta
        elo[row.away_team] = ra - delta
    df = df.copy()
    df["elo_h"], df["elo_a"] = pre_h, pre_a
    return dict(elo), df

def fit_poisson(df_elo: pd.DataFrame):
    d = df_elo[(df_elo.date >= "2010-01-01")
               & (~df_elo.tournament.str.contains("Friendly", case=False))]
    adv = np.where(d.neutral, 0.0, HOME_ADV)
    x_h = ((d.elo_h + adv) - d.elo_a) / 400.0
    x_a = -x_h
    x = np.concatenate([x_h, x_a])
    g = np.concatenate([d.home_score.values, d.away_score.values]).astype(float)
    def nll(p):
        lam = np.exp(p[0] + p[1] * x)
        return -(g * np.log(lam) - lam).sum()
    res = minimize(nll, x0=[0.2, 0.8], method="Nelder-Mead")
    return res.x[0], res.x[1]

# ── 2026 tournament structure ────────────────────────────────────────────────
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
ALL_TEAMS = [t for g in GROUPS.values() for t in g]
HOSTS = {"United States", "Mexico", "Canada"}
HOST_ADV_GROUP, HOST_ADV_KO = 50.0, 25.0

# ── market anchor ────────────────────────────────────────────────────────────
# Outright winner odds (decimal) from a bookmaker. The market prices in squad
# quality, injuries and manager situations that Elo can't see. The model
# converts these to an implied team rating and blends it with Elo.
# Update these whenever you like (they're from bet365, 2026-06-09).
# Set MARKET_WEIGHT = 0 to run on pure Elo.
MARKET_WEIGHT = 0.5
MARKET_ODDS = {
    "Spain": 5.5, "France": 6, "England": 7.5, "Brazil": 9, "Portugal": 9,
    "Argentina": 10, "Germany": 15, "Netherlands": 21, "Norway": 26,
    "Belgium": 34, "Colombia": 34, "Japan": 51, "Morocco": 51,
    "United States": 67, "Uruguay": 67, "Mexico": 67,
    "Switzerland": 81, "Croatia": 81, "Turkey": 81, "Ecuador": 101,
    "Senegal": 126, "Sweden": 126, "Canada": 126,
    "Austria": 151, "Paraguay": 151, "Scotland": 251,
    "Ivory Coast": 301, "Egypt": 301, "Czech Republic": 301,
    "Bosnia and Herzegovina": 351,
    "Ghana": 401, "Algeria": 401, "South Korea": 401,
    "Tunisia": 501, "Australia": 501, "Iran": 501,
    "DR Congo": 751, "South Africa": 1001, "Saudi Arabia": 1001,
    "Panama": 1501, "Iraq": 1501, "Uzbekistan": 1501,
    "Qatar": 2001, "Cape Verde": 2001, "New Zealand": 2501,
    "Jordan": 2501, "Haiti": 2501, "Curaçao": 3501,
}

BRACKET_R32 = [
    ("1E", "3rd:ABCDF"), ("1I", "3rd:CDFGH"),
    ("2A", "2B"),        ("1F", "2C"),
    ("2K", "2L"),        ("1H", "2J"),
    ("1D", "3rd:BEFIJ"), ("1G", "3rd:AEHIJ"),
    ("1C", "2F"),        ("2E", "2I"),
    ("1A", "3rd:CEFHI"), ("1L", "3rd:EHIJK"),
    ("1J", "2H"),        ("2D", "2G"),
    ("1B", "3rd:EFGIJ"), ("1K", "3rd:DEIJL"),
]
THIRD_SLOTS = [(i, j, set(s.split(":")[1]))
               for i, pair in enumerate(BRACKET_R32)
               for j, s in enumerate(pair) if s.startswith("3rd")]

# ── known WC 2026 results ────────────────────────────────────────────────────
# Played WC 2026 matches are auto-detected from the downloaded results.csv
# (tournament == "FIFA World Cup", date >= 2026-06-01). Use this list only to
# add results the CSV doesn't have yet (e.g. a match that finished an hour ago).
# Format: (home_team, away_team, home_goals, away_goals)
EXTRA_PLAYED: list[tuple[str, str, int, int]] = [
    # ("Mexico", "South Africa", 2, 0),
]

def wc2026_played(df: pd.DataFrame):
    wc = df[(df.tournament == "FIFA World Cup") & (df.date >= "2026-06-01")]
    return [(r.home_team, r.away_team, int(r.home_score), int(r.away_score))
            for r in wc.itertuples(index=False)]

# ── helpers ──────────────────────────────────────────────────────────────────
MAX_G = 10

def sim_goals(elo1, elo2, a, b, adv1=0.0, adv2=0.0, factor=1.0):
    x = ((elo1 + adv1) - (elo2 + adv2)) / 400.0
    return (rng.poisson(np.exp(a + b * x) * factor),
            rng.poisson(np.exp(a - b * x) * factor))

def host_adv(team, stage):
    if team not in HOSTS: return 0.0
    return HOST_ADV_GROUP if stage == "group" else HOST_ADV_KO

def play_ko(t1, t2, elo, a, b, fixed_facts=None):
    # already played in reality -> fixed result (draw = went to pens, 50/50)
    if fixed_facts and (t1, t2) in fixed_facts:
        g1, g2 = fixed_facts[(t1, t2)]
        if g1 != g2: return t1 if g1 > g2 else t2
    a1, a2 = host_adv(t1, "ko"), host_adv(t2, "ko")
    g1, g2 = sim_goals(elo[t1], elo[t2], a, b, a1, a2)
    if g1 != g2: return t1 if g1 > g2 else t2
    e1, e2 = sim_goals(elo[t1], elo[t2], a, b, a1, a2, factor=1/3)
    if e1 != e2: return t1 if e1 > e2 else t2
    return t1 if rng.random() < 0.5 else t2

def assign_thirds(qualified):
    slots = sorted(THIRD_SLOTS, key=lambda s: len(s[2] & set(qualified)))
    assignment, used = {}, set()
    def bt(k):
        if k == len(slots): return True
        i, j, allowed = slots[k]
        for g in qualified:
            if g not in used and g in allowed:
                used.add(g); assignment[(i, j)] = g
                if bt(k + 1): return True
                used.discard(g); del assignment[(i, j)]
        return False
    if not bt(0):
        assignment.clear()
        for (i, j, _), g in zip(slots, list(qualified)):
            assignment[(i, j)] = g
    return assignment

# ── simulation ───────────────────────────────────────────────────────────────
def build_fixed_facts(played):
    """Return dict (team_a, team_b) -> (g1, g2) for played matches, normalised."""
    facts = {}
    for h, a, gh, ga in played:
        facts[(h, a)] = (gh, ga)
        facts[(a, h)] = (ga, gh)      # both orderings for easy lookup
    return facts

def simulate_once(elo, a, b, fixed_facts):
    winners, runners, thirds = {}, {}, []
    for grp, teams in GROUPS.items():
        pts = defaultdict(int); gf = defaultdict(int); ga = defaultdict(int)
        for i in range(4):
            for j in range(i + 1, 4):
                t1, t2 = teams[i], teams[j]
                if (t1, t2) in fixed_facts:
                    g1, g2 = fixed_facts[(t1, t2)]
                else:
                    g1, g2 = sim_goals(elo[t1], elo[t2], a, b,
                                       host_adv(t1, "group"), host_adv(t2, "group"))
                gf[t1] += g1; ga[t1] += g2; gf[t2] += g2; ga[t2] += g1
                if g1 > g2:   pts[t1] += 3
                elif g2 > g1: pts[t2] += 3
                else:         pts[t1] += 1; pts[t2] += 1
        order = sorted(teams,
                       key=lambda t: (pts[t], gf[t]-ga[t], gf[t], rng.random()),
                       reverse=True)
        winners[grp], runners[grp] = order[0], order[1]
        thirds.append((pts[order[2]], gf[order[2]]-ga[order[2]], gf[order[2]],
                       rng.random(), grp, order[2]))

    ranked = sorted(thirds, reverse=True)[:8]
    third_team = {t[4]: t[5] for t in ranked}
    slot_map = assign_thirds(list(third_team.keys()))

    def resolve(i, j, code):
        if code.startswith("3rd"): return third_team[slot_map[(i, j)]]
        return winners[code[1]] if code[0] == "1" else runners[code[1]]

    r32 = [(resolve(i, 0, p[0]), resolve(i, 1, p[1]))
           for i, p in enumerate(BRACKET_R32)]
    rnd = [play_ko(t1, t2, elo, a, b, fixed_facts) for t1, t2 in r32]
    results = {t: "R32" for pair in r32 for t in pair}
    for stage in ["R16", "QF", "SF", "F"]:
        for t in rnd: results[t] = stage
        rnd = [play_ko(rnd[i], rnd[i+1], elo, a, b, fixed_facts)
               for i in range(0, len(rnd), 2)]
    results[rnd[0]] = "W"
    return results

STAGES = ["R32", "R16", "QF", "SF", "F", "W"]

def run_simulation(elo, a, b, played, n_sims=N_SIMS):
    fixed = build_fixed_facts(played)
    counts = {t: defaultdict(int) for t in ALL_TEAMS}
    for _ in range(n_sims):
        res = simulate_once(elo, a, b, fixed)
        for team in ALL_TEAMS:
            reached = res.get(team)
            if reached is None: continue
            for s in STAGES[:STAGES.index(reached) + 1]:
                counts[team][s] += 1
    probs = {}
    for t, c in counts.items():
        probs[t] = {s: round(c[s] / n_sims, 6) for s in STAGES}
    return probs

# ── market blend ─────────────────────────────────────────────────────────────
def blend_market(elo, a, b, played):
    """Convert bookmaker outright odds into an implied Elo and blend.

    Uses the model's own Elo <-> ln(champion prob) relationship (fitted on a
    quick 5k-sim run) to translate margin-free market probabilities into the
    same Elo scale, then takes a weighted average per team."""
    if MARKET_WEIGHT <= 0 or not MARKET_ODDS:
        return dict(elo)
    missing = [t for t in ALL_TEAMS if t not in MARKET_ODDS]
    if missing:
        print(f"  WARNING: no market odds for {missing}; skipping blend")
        return dict(elo)
    print("Calibrating market blend (5,000 sims) ...")
    n_cal = 5000
    base = run_simulation(elo, a, b, played, n_sims=n_cal)
    p_model = np.array([max(base[t]["W"], 0.5 / n_cal) for t in ALL_TEAMS])
    inv = np.array([1.0 / MARKET_ODDS[t] for t in ALL_TEAMS])
    p_mkt = inv / inv.sum()                      # strips the overround
    elos = np.array([elo[t] for t in ALL_TEAMS])
    beta, alpha = np.polyfit(np.log(p_model), elos, 1)
    market_elo = alpha + beta * np.log(p_mkt)
    blended = dict(elo)
    for t, e_raw, e_mkt in zip(ALL_TEAMS, elos, market_elo):
        blended[t] = (1 - MARKET_WEIGHT) * e_raw + MARKET_WEIGHT * e_mkt
    moved = sorted(ALL_TEAMS, key=lambda t: abs(blended[t] - elo[t]))[-5:]
    print("  largest rating shifts: " + ", ".join(
        f"{t} {blended[t]-elo[t]:+.0f}" for t in reversed(moved)))
    return blended

# ── per-match xG / score prediction ──────────────────────────────────────────
def match_prediction(t1, t2, elo, a, b, stage="group"):
    """xG, result probs and most likely scorelines for one match (90 min)."""
    adv1, adv2 = host_adv(t1, stage), host_adv(t2, stage)
    x = ((elo[t1] + adv1) - (elo[t2] + adv2)) / 400.0
    lam1, lam2 = float(np.exp(a + b * x)), float(np.exp(a - b * x))
    p1 = poisson.pmf(np.arange(MAX_G + 1), lam1)
    p2 = poisson.pmf(np.arange(MAX_G + 1), lam2)
    grid = np.outer(p1, p2)
    grid /= grid.sum()
    pw = float(np.tril(grid, -1).sum())
    pd_ = float(np.trace(grid))
    pl = float(np.triu(grid, 1).sum())
    flat = [(f"{i}-{j}", float(grid[i, j]))
            for i in range(MAX_G + 1) for j in range(MAX_G + 1)]
    top = sorted(flat, key=lambda kv: -kv[1])[:5]
    return {"xg1": round(lam1, 2), "xg2": round(lam2, 2),
            "p1": round(pw, 4), "pd": round(pd_, 4), "p2": round(pl, 4),
            "top_scores": [[s, round(p, 4)] for s, p in top]}

def build_fixtures(elo, a, b, facts):
    """All 72 group matches with xG + score prediction; played ones flagged."""
    fixtures = []
    for grp, teams in GROUPS.items():
        for i in range(4):
            for j in range(i + 1, 4):
                t1, t2 = teams[i], teams[j]
                fx = {"group": grp, "t1": t1, "t2": t2}
                fx.update(match_prediction(t1, t2, elo, a, b, "group"))
                if (t1, t2) in facts:
                    fx["played"] = list(facts[(t1, t2)])
                fixtures.append(fx)
    return fixtures

# ── odds history (rolling append) ────────────────────────────────────────────
def load_history():
    if os.path.exists(HIST):
        with open(HIST) as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HIST, "w") as f:
        json.dump(history, f, separators=(",", ":"))

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. download
    print("Downloading results.csv …")
    urllib.request.urlretrieve(URL, CSV)
    print("  done.")

    # 2. Elo
    df = (pd.read_csv(CSV, parse_dates=["date"])
            .dropna(subset=["home_score", "away_score"])
            .sort_values("date")
            .reset_index(drop=True))
    print(f"Loaded {len(df):,} matches ({df.date.min().date()} to {df.date.max().date()})")
    elo, df_elo = compute_elo(df)

    # 3. Poisson fit
    a, b = fit_poisson(df_elo)
    print(f"Poisson: lam=exp({a:.4f}+{b:.4f}*dElo/400)")

    # 4. Fixed facts: WC 2026 matches already played (CSV + manual extras)
    played = wc2026_played(df) + EXTRA_PLAYED
    print(f"WC 2026 matches locked in as facts: {len(played)}")
    facts = build_fixed_facts(played)

    # 5. Blend market odds into ratings (squads/injuries/manager info)
    elo_blend = blend_market(elo, a, b, played)

    # 6. Simulate
    print(f"Simulating {N_SIMS:,} tournaments ...")
    probs = run_simulation(elo_blend, a, b, played)
    print("  done.")

    # 7. Per-match xG predictions for the group stage
    fixtures = build_fixtures(elo_blend, a, b, facts)

    # 8. Odds history
    ts = datetime.now(timezone.utc).isoformat()
    history = load_history()
    history.append({"ts": ts, "probs": probs})
    save_history(history)

    # 9. Write forecast.json  (groups + bracket meta included for index.html)
    forecast = {
        "updated": ts,
        "n_sims": N_SIMS,
        "market_weight": MARKET_WEIGHT,
        "played_matches": played,
        "groups": GROUPS,
        "bracket": BRACKET_R32,
        "hosts": sorted(HOSTS),
        "host_adv": {"group": HOST_ADV_GROUP, "ko": HOST_ADV_KO},
        "poisson": {"a": round(a, 5), "b": round(b, 5)},
        "elo": {t: round(elo_blend[t], 1) for t in ALL_TEAMS},
        "elo_raw": {t: round(elo.get(t, 1500), 1) for t in ALL_TEAMS},
        "fixtures": fixtures,
        "probs": probs,
        "history": history,          # full history in one file for simplicity
    }
    with open(OUT, "w") as f:
        json.dump(forecast, f, separators=(",", ":"))
    print(f"forecast.json written ({os.path.getsize(OUT)//1024} KB)")
    print(f"Timestamp: {ts}")

    # Quick summary
    champs = sorted(probs.items(), key=lambda kv: -kv[1]["W"])
    print("\nTop 10 champion odds:")
    for t, p in champs[:10]:
        print(f"  {t:<30} {p['W']:6.1%}")

if __name__ == "__main__":
    main()
