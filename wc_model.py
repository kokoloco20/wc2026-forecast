"""
World Cup 2026 Elo-Poisson Monte Carlo Simulator
=================================================
Pipeline:
  1. Compute World-Football-style Elo ratings from all international results (1872-present)
  2. Calibrate a Poisson goal model: expected goals as a function of Elo difference
  3. Backtest on the 2018 and 2022 World Cups (log loss / Brier vs baselines)
  4. Simulate the 2026 tournament N times (real groups, real format, third-place rules)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from collections import defaultdict

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. ELO ENGINE (World Football Elo rules: K by importance, margin
#    multiplier, +100 home advantage)
# ----------------------------------------------------------------------
HOME_ADV = 100.0

def k_factor(tournament: str) -> float:
    t = tournament.lower()
    if t == "fifa world cup":
        return 60
    if any(s in t for s in ["euro", "copa américa", "copa america", "africa cup",
                            "asian cup", "gold cup", "confederations"]):
        return 50 if "qualification" not in t else 40
    if "qualification" in t:
        return 40
    if "friendly" in t:
        return 20
    return 30

def margin_mult(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    if gd == 3: return 1.75
    return 1.75 + (gd - 3) / 8.0

def compute_elo(df: pd.DataFrame):
    """Returns final ratings dict + per-match pre-game elo columns."""
    elo = defaultdict(lambda: 1500.0)
    pre_h, pre_a = [], []
    for row in df.itertuples(index=False):
        rh, ra = elo[row.home_team], elo[row.away_team]
        pre_h.append(rh); pre_a.append(ra)
        adv = 0.0 if row.neutral else HOME_ADV
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + adv) - ra) / 400.0))
        res_h = 0.5 if row.home_score == row.away_score else float(row.home_score > row.away_score)
        delta = k_factor(row.tournament) * margin_mult(row.home_score - row.away_score) * (res_h - exp_h)
        elo[row.home_team] = rh + delta
        elo[row.away_team] = ra - delta
    df = df.copy()
    df["elo_h"], df["elo_a"] = pre_h, pre_a
    return dict(elo), df

# ----------------------------------------------------------------------
# 2. POISSON GOAL MODEL
#    lambda_team = exp(a + b * (own_adjusted_elo - opp_adjusted_elo)/400)
#    Fit by Poisson MLE on competitive matches since 2010.
# ----------------------------------------------------------------------
def fit_poisson(df_elo: pd.DataFrame):
    d = df_elo[(df_elo.date >= "2010-01-01") & (~df_elo.tournament.str.contains("Friendly", case=False))]
    adv = np.where(d.neutral, 0.0, HOME_ADV)
    x_h = ((d.elo_h + adv) - d.elo_a) / 400.0          # home perspective
    x_a = -x_h                                          # away perspective
    x = np.concatenate([x_h, x_a])
    g = np.concatenate([d.home_score.values, d.away_score.values]).astype(float)

    def nll(p):
        lam = np.exp(p[0] + p[1] * x)
        return -(g * np.log(lam) - lam).sum()

    res = minimize(nll, x0=[0.2, 0.8], method="Nelder-Mead")
    a, b = res.x
    print(f"Poisson fit on {len(d):,} competitive matches since 2010:")
    print(f"  lambda = exp({a:.4f} + {b:.4f} * elo_diff/400)")
    print(f"  -> evenly matched teams: {np.exp(a):.2f} xG each | +400 Elo edge: "
          f"{np.exp(a+b):.2f} vs {np.exp(a-b):.2f}")
    return a, b

MAX_G = 10
def match_probs(elo1, elo2, a, b, adv1=0.0):
    """Return (P(team1 win), P(draw), P(team2 win), lam1, lam2) over 90 min."""
    x = ((elo1 + adv1) - elo2) / 400.0
    lam1, lam2 = np.exp(a + b * x), np.exp(a - b * x)
    p1 = poisson.pmf(np.arange(MAX_G + 1), lam1)
    p2 = poisson.pmf(np.arange(MAX_G + 1), lam2)
    grid = np.outer(p1, p2)
    grid /= grid.sum()
    pw = np.tril(grid, -1).sum()   # team1 scores more
    pd_ = np.trace(grid)
    pl = np.triu(grid, 1).sum()
    return pw, pd_, pl, lam1, lam2

# ----------------------------------------------------------------------
# 3. BACKTEST on 2018 + 2022 World Cups
# ----------------------------------------------------------------------
def backtest(df_all: pd.DataFrame, a, b):
    print("\n=== BACKTEST: model trained pre-tournament, scored on WC matches ===")
    for year, start in [(2018, "2018-06-01"), (2022, "2022-11-01")]:
        hist = df_all[df_all.date < start]
        elo, _ = compute_elo(hist)
        wc = df_all[(df_all.tournament == "FIFA World Cup") & (df_all.date >= start)
                    & (df_all.date < f"{year + 1}-01-01")]
        ll_m, ll_u, br_m, br_u, n, correct = 0, 0, 0, 0, 0, 0
        for r in wc.itertuples(index=False):
            pw, pdr, pl, *_ = match_probs(elo.get(r.home_team, 1500), elo.get(r.away_team, 1500), a, b)
            probs = np.array([pw, pdr, pl])
            out = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
            y = np.zeros(3); y[out] = 1
            ll_m += -np.log(max(probs[out], 1e-12)); ll_u += -np.log(1/3)
            br_m += ((probs - y) ** 2).sum();        br_u += ((np.ones(3)/3 - y) ** 2).sum()
            correct += int(probs.argmax() == out);   n += 1
        print(f"  WC {year} ({n} matches, 90-min result): "
              f"log loss {ll_m/n:.4f} (uniform baseline {ll_u/n:.4f}) | "
              f"Brier {br_m/n:.4f} (baseline {br_u/n:.4f}) | "
              f"top-pick accuracy {correct/n:.1%}")

# ----------------------------------------------------------------------
# 4. 2026 TOURNAMENT SIMULATION
# ----------------------------------------------------------------------
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
HOSTS = {"United States", "Mexico", "Canada"}
HOST_ADV_GROUP, HOST_ADV_KO = 100.0, 50.0   # hosts play group games at home; KO venues vary

def sim_goals(elo1, elo2, a, b, adv1=0.0, adv2=0.0, factor=1.0):
    x = ((elo1 + adv1) - (elo2 + adv2)) / 400.0
    return (rng.poisson(np.exp(a + b * x) * factor),
            rng.poisson(np.exp(a - b * x) * factor))

def host_adv(team, stage):
    if team not in HOSTS: return 0.0
    return HOST_ADV_GROUP if stage == "group" else HOST_ADV_KO

def play_ko(t1, t2, elo, a, b):
    """Knockout match: 90 min -> extra time (1/3 rate) -> penalties (50/50)."""
    a1, a2 = host_adv(t1, "ko"), host_adv(t2, "ko")
    g1, g2 = sim_goals(elo[t1], elo[t2], a, b, a1, a2)
    if g1 != g2: return t1 if g1 > g2 else t2
    e1, e2 = sim_goals(elo[t1], elo[t2], a, b, a1, a2, factor=1/3)
    if e1 != e2: return t1 if e1 > e2 else t2
    return t1 if rng.random() < 0.5 else t2

# Official FIFA 2026 bracket, in bracket order (top-left to bottom-left,
# then top-right to bottom-right). "3rd:XYZ" = third-place team that must
# come from one of those groups.
BRACKET_R32 = [
    ("1E", "3rd:ABCDF"), ("1I", "3rd:CDFGH"),   # left half
    ("2A", "2B"),        ("1F", "2C"),
    ("2K", "2L"),        ("1H", "2J"),
    ("1D", "3rd:BEFIJ"), ("1G", "3rd:AEHIJ"),
    ("1C", "2F"),        ("2E", "2I"),          # right half
    ("1A", "3rd:CEFHI"), ("1L", "3rd:EHIJK"),
    ("1J", "2H"),        ("2D", "2G"),
    ("1B", "3rd:EFGIJ"), ("1K", "3rd:DEIJL"),
]
THIRD_SLOTS = [(i, j, set(s.split(":")[1]))
               for i, pair in enumerate(BRACKET_R32)
               for j, s in enumerate(pair) if s.startswith("3rd")]

def assign_thirds(qualified):
    """Backtracking match of the 8 qualified third-place groups to the 8
    constrained bracket slots (any valid assignment, like FIFA's table)."""
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
    if not bt(0):  # shouldn't happen; FIFA's table covers all combinations
        assignment.clear()
        for (i, j, _), g in zip(slots, list(qualified)):
            assignment[(i, j)] = g
    return assignment

def simulate_once(elo, a, b):
    winners, runners, thirds = {}, {}, []
    for grp, teams in GROUPS.items():
        pts = defaultdict(int); gf = defaultdict(int); ga = defaultdict(int)
        for i in range(4):
            for j in range(i + 1, 4):
                t1, t2 = teams[i], teams[j]
                g1, g2 = sim_goals(elo[t1], elo[t2], a, b,
                                   host_adv(t1, "group"), host_adv(t2, "group"))
                gf[t1] += g1; ga[t1] += g2; gf[t2] += g2; ga[t2] += g1
                if g1 > g2:   pts[t1] += 3
                elif g2 > g1: pts[t2] += 3
                else:         pts[t1] += 1; pts[t2] += 1
        order = sorted(teams, key=lambda t: (pts[t], gf[t]-ga[t], gf[t], rng.random()), reverse=True)
        winners[grp], runners[grp] = order[0], order[1]
        thirds.append((pts[order[2]], gf[order[2]]-ga[order[2]], gf[order[2]], rng.random(),
                       grp, order[2]))
    ranked = sorted(thirds, reverse=True)[:8]
    third_team = {t[4]: t[5] for t in ranked}            # group letter -> team
    slot_map = assign_thirds(list(third_team.keys()))    # (match, side) -> group

    def resolve(i, j, code):
        if code.startswith("3rd"): return third_team[slot_map[(i, j)]]
        return winners[code[1]] if code[0] == "1" else runners[code[1]]

    r32 = [(resolve(i, 0, p[0]), resolve(i, 1, p[1])) for i, p in enumerate(BRACKET_R32)]
    rnd = [play_ko(t1, t2, elo, a, b) for t1, t2 in r32]
    results = {t: "R32" for pair in r32 for t in pair}
    for stage in ["R16", "QF", "SF", "F"]:
        for t in rnd: results[t] = stage
        rnd = [play_ko(rnd[i], rnd[i + 1], elo, a, b) for i in range(0, len(rnd), 2)]
    champ = rnd[0]
    results[champ] = "W"
    return results, champ

def run_simulation(elo_now, a, b, n_sims=20000):
    stages = ["R32", "R16", "QF", "SF", "F", "W"]
    counts = {t: defaultdict(int) for g in GROUPS.values() for t in g}
    for _ in range(n_sims):
        res, _ = simulate_once(elo_now, a, b)
        for team in counts:
            reached = res.get(team)
            if reached is None: continue
            for s in stages[:stages.index(reached) + 1]:
                counts[team][s] += 1
    rows = []
    for t, c in counts.items():
        rows.append([t] + [c[s] / n_sims for s in ["R32", "R16", "QF", "SF", "F", "W"]])
    out = pd.DataFrame(rows, columns=["Team", "KO stage", "R16", "QF", "SF", "Final", "Champion"])
    return out.sort_values("Champion", ascending=False).reset_index(drop=True)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("results.csv", parse_dates=["date"]).dropna(subset=["home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df):,} completed matches ({df.date.min().date()} -> {df.date.max().date()})")

    elo_now, df_elo = compute_elo(df)
    print("\nTop 10 current Elo:")
    for t, r in sorted(elo_now.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {t:<15} {r:7.0f}")

    a, b = fit_poisson(df_elo)
    backtest(df, a, b)

    print("\n=== SIMULATING WORLD CUP 2026 (20,000 tournaments) ===")
    table = run_simulation(elo_now, a, b, n_sims=20000)
    pd.set_option("display.float_format", lambda v: f"{v:6.1%}")
    print(table.head(20).to_string(index=False))
    table.to_csv("wc2026_forecast.csv", index=False)
    print("\nFull table saved to wc2026_forecast.csv")
