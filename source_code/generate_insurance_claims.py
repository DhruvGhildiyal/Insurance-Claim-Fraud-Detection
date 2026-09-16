"""
Synthetic Insurance Claim Fraud Detection dataset generator.

Pipeline (one function per stage, so each step is easy to explain):
    1. build_provider_pool()     -> ~200 hospitals / garages / contractors (10 colluding)
    2. generate_base_claims()    -> claimants, policies and *genuine-looking* claims
    3. inject_fraud_patterns()   -> label ~5% of claims as fraud and plant 1-4 red flags each
    4. add_noise()               -> plant single "false-flag" red flags on genuine claims
    5. assemble_dataset()        -> derive features, add missingness, run sanity checks
    6. print_summary()           -> re-detect every red flag from the FINAL table

Design notes
- All dates are handled internally as integer day numbers (days since 1970-01-01).
- History features (claimant_avg_past_claim, claimant_claim_count_6m) are computed
  strictly from claims filed BEFORE the current one, so there is no look-ahead leakage.
- policy_renewal_date is "as of the incident": the most recent renewal on or before
  the incident date (empty if the claim happened in the policy's first term).
- is_fraud is a label only. It must not be used as a model feature.
"""

import math

import numpy as np
import pandas as pd
from faker import Faker

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
N_CLAIMS = 10_000
FRAUD_RATE = 0.05
SNAPSHOT_DATE = "2026-08-31"          # data-extract date: nothing happens after this
OUTPUT_CSV = "insurance_claims.csv"
PROVIDER_CSV = "provider_master.csv"  # lookup table: code -> provider name / city

YEAR = 365
WINDOW_6M = 180

CLAIM_TYPES = ["health", "auto", "property"]
CLAIM_TYPE_PROBS = [0.40, 0.35, 0.25]

# Truncated log-normal claim severity per type (INR). `sigma` is the marginal spread;
# part of it is a claimant-level effect so repeat claimants are self-consistent.
SEVERITY = {
    "health":   dict(median=45_000,  sigma=1.00, low=5_000,  high=1_000_000),
    "auto":     dict(median=60_000,  sigma=0.80, low=15_000, high=500_000),
    "property": dict(median=130_000, sigma=0.85, low=20_000, high=1_500_000),
}
CLAIMANT_SEVERITY_SD = 0.5            # share of log-amount spread explained by the claimant

# Sum-insured / IDV tiers per claim type and their market share. The top tier equals
# the type's maximum claim, so even near-limit claims stay inside the realistic range.
COVERAGE_TIERS = {
    "health":   ([200_000, 300_000, 500_000, 750_000, 1_000_000],
                 [0.15, 0.25, 0.30, 0.18, 0.12]),
    "auto":     ([200_000, 250_000, 300_000, 400_000, 500_000],
                 [0.15, 0.20, 0.25, 0.25, 0.15]),
    "property": ([500_000, 750_000, 1_000_000, 1_250_000, 1_500_000],
                 [0.20, 0.25, 0.25, 0.18, 0.12]),
}

# claim_type: (code prefix, #providers, name suffixes, #high-risk colluding providers)
PROVIDER_POOL = {
    "health":   ("HSP", 80, ["Hospital", "Multispeciality Hospital", "Medical Centre",
                             "Nursing Home", "Healthcare"], 4),
    "auto":     ("GRG", 70, ["Motors", "Auto Works", "Car Care", "Service Centre",
                             "Automobiles"], 4),
    "property": ("CTR", 50, ["Builders", "Constructions", "Restoration Services",
                             "Civil Works", "Infra Projects"], 2),
}

# Claims per claimant: ~18% of claimants file more than once
CLAIMS_PER_CLAIMANT = ([1, 2, 3, 4, 5, 6], [0.82, 0.12, 0.04, 0.012, 0.005, 0.003])

PATTERNS = {
    1: "early claim (15-30d after policy start)",
    2: "large claim <=30d after renewal",
    3: "amount 2.5-5x claimant's past average",
    4: "round claim amount",
    5: "amount within 2-5% of coverage limit",
    6: "3+ claims in 6 months",
    7: "high-risk (colluding) provider",
    8: "long incident->filing delay (15-45d)",
    9: "address/bank details changed recently",
}
# Relative prevalence of each red flag among fraud cases (renormalised per claim
# over the patterns that are actually feasible for that claim)
FRAUD_PATTERN_WEIGHTS = {1: .11, 2: .09, 3: .15, 4: .11, 5: .08, 6: .12, 7: .14, 8: .10, 9: .10}
# How many red flags each fraud case gets: mostly 2-3, some weak single-flag cases
FRAUD_N_PATTERNS = ([1, 2, 3, 4], [0.15, 0.42, 0.30, 0.13])

# Share of genuine claims that receive exactly one false-flag red flag
NOISE_SHARE = 0.25
NOISE_PATTERN_WEIGHTS = {1: .10, 2: .08, 3: .10, 4: .12, 5: .07, 6: .06, 7: .15, 8: .18, 9: .14}

SNAP = int(np.datetime64(SNAPSHOT_DATE, "D").astype(np.int64))


# =============================================================================
# Helpers
# =============================================================================
def truncated_lognormal(rng, mu, sigma, low, high, max_iter=100):
    """Log-normal draws restricted to [low, high] by re-sampling (avoids clipping spikes)."""
    mu, sigma, low, high = np.broadcast_arrays(*(np.asarray(a, dtype=float)
                                                 for a in (mu, sigma, low, high)))
    out = rng.lognormal(mu, sigma)
    bad = (out < low) | (out > high)
    for _ in range(max_iter):
        if not bad.any():
            break
        out[bad] = rng.lognormal(mu[bad], sigma[bad])
        bad = (out < low) | (out > high)
    return np.clip(out, low, high)


def normal_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def to_datetime(day_numbers):
    return pd.to_datetime(pd.Series(day_numbers, dtype="float"), unit="D")


# =============================================================================
# Stage 1: provider pool
# =============================================================================
def build_provider_pool(fake, rng):
    """~200 providers with heavy-tailed popularity; a few mid-sized ones collude."""
    rows = []
    for ctype, (prefix, n, suffixes, n_risky) in PROVIDER_POOL.items():
        popularity = rng.lognormal(0.0, 1.0, n)   # a few big hospitals/garages dominate volume
        # Colluding providers are mid-sized outfits (not the big brand names, not tiny
        # ones), so they also serve plenty of genuine customers.
        mid_band = np.argsort(popularity)[n // 4: 3 * n // 4]
        risky = set(rng.choice(mid_band, n_risky, replace=False).tolist())
        for i in range(n):
            rows.append(dict(
                provider_code=f"{prefix}{i + 1:04d}",
                provider_name=f"{fake.last_name()} {rng.choice(suffixes)}",
                city=fake.city(),
                claim_type=ctype,
                weight=popularity[i],
                high_risk=i in risky,
            ))
    return pd.DataFrame(rows)


# =============================================================================
# Stage 2: claimants, policies and base (genuine-looking) claims
# =============================================================================
def generate_base_claims(rng, providers):
    """Return (claimants, policies, claims) tables. Every claim starts out genuine."""
    # --- 2a. Claimants and how many claims each one files -----------------------
    counts, probs = CLAIMS_PER_CLAIMANT
    guess = int(N_CLAIMS / np.dot(counts, probs) * 1.1)
    k = rng.choice(counts, size=guess, p=probs)
    cum = np.cumsum(k)
    n_claimants = int(np.searchsorted(cum, N_CLAIMS)) + 1
    k = k[:n_claimants].copy()
    k[-1] -= cum[n_claimants - 1] - N_CLAIMS          # trim so the total is exactly N_CLAIMS

    claimants = pd.DataFrame({
        "n_claims": k,
        "severity_z": rng.normal(0, 1, n_claimants),        # drives claim size AND cover bought
        "delay_scale": rng.lognormal(0, 0.3, n_claimants),  # habitually prompt vs slow reporters
        "fraud_propensity": rng.normal(0, 1, n_claimants),  # latent; only used to pick fraud rows
    })

    # --- 2b. Policies: repeat claimants may hold 1-3 policies -------------------
    n_pol = np.where(k == 1, 1, rng.choice([1, 2, 3], size=n_claimants, p=[0.6, 0.3, 0.1]))
    n_pol = np.minimum(n_pol, k)
    pol_claimant = np.repeat(np.arange(n_claimants), n_pol)
    P = len(pol_claimant)
    pol = pd.DataFrame({"claimant_idx": pol_claimant})
    pol["claim_type"] = rng.choice(CLAIM_TYPES, size=P, p=CLAIM_TYPE_PROBS)
    # Inception dates spread over the ~3 years before the snapshot
    pol["start_d"] = rng.integers(SNAP - 3 * YEAR - 60, SNAP - 30 + 1, size=P)

    # Annual renewals with ~85% retention; the first lapse ends the policy
    max_renew = np.minimum((SNAP - 1 - pol["start_d"].to_numpy()) // YEAR, 3)
    streak = np.cumprod(rng.random((P, 3)) < 0.85, axis=1).sum(axis=1)
    pol["n_renewals"] = np.minimum(streak, max_renew)
    pol["term_end_d"] = pol["start_d"] + YEAR * (pol["n_renewals"] + 1)   # exclusive

    # Coverage limit: Gaussian copula so higher-severity claimants buy higher tiers
    z_pol = claimants["severity_z"].to_numpy()[pol_claimant]
    u = normal_cdf(0.7 * z_pol + np.sqrt(1 - 0.7 ** 2) * rng.normal(0, 1, P))
    limits = np.empty(P, dtype=np.int64)
    ptype = pol["claim_type"].to_numpy()
    for ctype, (tiers, tier_probs) in COVERAGE_TIERS.items():
        m = ptype == ctype
        idx = np.searchsorted(np.cumsum(tier_probs), u[m], side="right")
        limits[m] = np.asarray(tiers)[np.minimum(idx, len(tiers) - 1)]
    pol["coverage_limit"] = limits

    # --- 2c. Claims: every policy gets >=1 claim, extras land on a random policy
    pol_offset = np.concatenate([[0], np.cumsum(n_pol)[:-1]])
    extra_claimant = np.repeat(np.arange(n_claimants), k - n_pol)
    extra_policy = pol_offset[extra_claimant] + (
        rng.random(len(extra_claimant)) * n_pol[extra_claimant]).astype(int)
    policy = np.concatenate([np.arange(P), extra_policy])
    N = len(policy)

    clm = pd.DataFrame({"policy_idx": policy})
    clm["claimant_idx"] = pol["claimant_idx"].to_numpy()[policy]
    clm["claim_type"] = ptype[policy]
    ct = clm["claim_type"].to_numpy()
    who = clm["claimant_idx"].to_numpy()
    start = pol["start_d"].to_numpy()[policy]
    term_end = pol["term_end_d"].to_numpy()[policy]

    # --- 2d. Dates: reporting delay mostly 0-10 days with a long right tail ----
    r = rng.random(N)
    short = np.clip(np.ceil(rng.gamma(2.0, 1.9 * claimants["delay_scale"].to_numpy()[who])), 1, 10)
    long_tail = np.clip(np.round(rng.lognormal(np.log(20), 0.45, N)), 11, 75)
    delay = np.where(r < 0.05, 0, np.where(r < 0.93, short, long_tail)).astype(np.int64)
    delay = np.where(start > SNAP - delay, SNAP - start, delay)   # very new policy: shorter delay
    latest_incident = np.minimum(term_end - 1, SNAP - delay)
    incident = start + (rng.random(N) * (latest_incident - start + 1)).astype(np.int64)
    clm["incident_d"] = incident
    clm["filed_d"] = incident + delay

    # --- 2e. Claim amounts: truncated log-normal per type, claimant-consistent --
    sev = pd.DataFrame(SEVERITY).T.loc[ct]
    limit = limits[policy]
    z_clm = claimants["severity_z"].to_numpy()[who]
    mu = np.log(sev["median"].to_numpy(float)) + CLAIMANT_SEVERITY_SD * z_clm
    sigma_within = np.sqrt(sev["sigma"].to_numpy(float) ** 2 - CLAIMANT_SEVERITY_SD ** 2)
    high = np.minimum(sev["high"].to_numpy(float), 0.92 * limit)   # genuine claims stay under cover
    amount = np.round(truncated_lognormal(rng, mu, sigma_within, sev["low"].to_numpy(float), high))
    # Contractor quotes / garage estimates are often rounded a little; hospital bills are itemised
    q = rng.random(N)
    amount = np.where((ct == "property") & (q < 0.35), np.round(amount, -2), amount)
    amount = np.where((ct == "auto") & (q < 0.25), np.round(amount, -1), amount)
    clm["claim_amount"] = amount.astype(np.int64)

    # --- 2f. Providers: popularity-weighted, claimants tend to reuse their usual one
    provider = np.empty(N, dtype=object)
    for ctype in CLAIM_TYPES:
        g = providers[providers["claim_type"] == ctype]
        p = g["weight"].to_numpy() / g["weight"].sum()
        codes = g["provider_code"].to_numpy()
        m = np.flatnonzero(ct == ctype)
        usual = rng.choice(codes, size=n_claimants, p=p)
        fresh = rng.choice(codes, size=len(m), p=p)
        provider[m] = np.where(rng.random(len(m)) < 0.55, usual[who[m]], fresh)
    clm["provider_code"] = provider

    # --- 2g. KYC changes at a genuine base rate (people who move update both) --
    addr = rng.random(N) < 0.025
    clm["addr_changed"] = addr
    clm["bank_changed"] = np.where(addr, rng.random(N) < 0.30, rng.random(N) < 0.012)

    clm["is_fraud"] = 0
    clm["date_locked"] = False      # set once a pattern has fixed this claim's filing date
    return claimants, pol, clm


def build_context(claimants, pol, clm, providers):
    """Lookups shared by the fraud-injection and noise stages."""
    high_risk = {t: providers.loc[(providers["claim_type"] == t) & providers["high_risk"],
                                  "provider_code"].to_numpy() for t in CLAIM_TYPES}
    return dict(
        claims_per_claimant=claimants["n_claims"].to_numpy(),
        claimant_rows=clm.groupby("claimant_idx").indices,   # claimant -> row positions
        high_risk=high_risk,
        high_risk_codes=set(np.concatenate(list(high_risk.values())).tolist()),
        type_amounts={t: clm.loc[clm["claim_type"] == t, "claim_amount"].to_numpy()
                      for t in CLAIM_TYPES},
        ring={},                                             # (claimant, type) -> colluding provider
    )


# =============================================================================
# Shared red-flag machinery (used for BOTH real fraud and false-flag noise)
# =============================================================================
def pattern_feasibility(clm, pol, ctx):
    """Bool matrix [n_claims x 9]: can pattern p physically be planted on this claim?"""
    N = len(clm)
    policy = clm["policy_idx"].to_numpy()
    start = pol["start_d"].to_numpy()[policy]
    term_end = pol["term_end_d"].to_numpy()[policy]
    filed = clm["filed_d"].to_numpy()
    locked = clm["date_locked"].to_numpy()

    order = np.lexsort((np.arange(N), filed))
    prior_claims = np.empty(N, dtype=int)
    prior_claims[order] = clm.iloc[order].groupby("claimant_idx").cumcount().to_numpy()

    feas = np.ones((N, len(PATTERNS)), dtype=bool)                     # column j == pattern j+1
    feas[:, 0] = ~locked                                               # 1: needs to move filing date
    feas[:, 1] = (~locked & (pol["n_renewals"].to_numpy()[policy] >= 1)
                  & (start + YEAR + 30 <= SNAP))                       # 2: needs a renewal
    feas[:, 2] = prior_claims >= 1                                     # 3: needs claim history
    feas[:, 5] = ctx["claims_per_claimant"][clm["claimant_idx"].to_numpy()] >= 3   # 6
    feas[:, 7] = (filed - start >= 15) | (start + 15 <= np.minimum(SNAP, term_end))  # 8
    return feas


def choose_pattern_sets(rows, feas, weights, n_patterns, rng):
    """For each row draw `n` distinct feasible patterns, weighted by prevalence."""
    ids = np.array(list(PATTERNS))
    w = np.array([weights[i] for i in ids], dtype=float)
    sets = {}
    for r, n in zip(rows, n_patterns):
        avail = feas[r].copy()
        chosen = set()
        for _ in range(n):
            p = w * avail
            if p.sum() == 0:
                break
            pick = int(rng.choice(ids, p=p / p.sum()))
            chosen.add(pick)
            avail[pick - 1] = False
            if pick in (1, 2):          # "early after purchase" and "after renewal" exclude each other
                avail[(3 - pick) - 1] = False
        sets[int(r)] = chosen
    return sets


def apply_patterns(clm, pol, ctx, sets, rng, sticky_provider):
    """Plant the requested red flags. Returns {pattern: [rows where it was applied]}.

    Order matters: date patterns first (they change claim ordering and therefore
    history), then amount patterns in chronological order (so "past average" sees
    already-modified earlier claims), then provider / KYC flags.
    """
    applied = {p: [] for p in PATTERNS}
    with_p = {p: sorted(r for r, s in sets.items() if p in s) for p in PATTERNS}

    pol_start = pol["start_d"].to_numpy()
    pol_end = pol["term_end_d"].to_numpy()
    pol_nren = pol["n_renewals"].to_numpy()
    pol_limit = pol["coverage_limit"].to_numpy()
    policy = clm["policy_idx"].to_numpy()
    claimant = clm["claimant_idx"].to_numpy()
    ctype = clm["claim_type"].to_numpy()
    members = ctx["claimant_rows"]

    filed = clm["filed_d"].to_numpy().copy()
    incident = clm["incident_d"].to_numpy().copy()
    locked = clm["date_locked"].to_numpy().copy()

    # ---- Pattern 1: claim filed 15-30 days after the policy was bought --------
    # Fraudster buys a policy for a loss that has already happened / is planned.
    for r in with_p[1]:
        if locked[r]:
            continue
        s0 = pol_start[policy[r]]
        f = s0 + rng.integers(15, 31)
        d = rng.integers(15, f - s0 + 1) if 8 in sets[r] else rng.integers(1, 6)
        filed[r], incident[r], locked[r] = f, f - d, True
        applied[1].append(r)

    # ---- Pattern 2 (timing half): incident & filing within 30 days of a renewal
    renewal_timed = set()
    for r in with_p[2]:
        if locked[r]:
            continue
        p = policy[r]
        renewals = pol_start[p] + YEAR * np.arange(1, pol_nren[p] + 1)
        renewals = renewals[renewals + 30 <= SNAP]
        if len(renewals) == 0:
            continue
        ren = rng.choice(renewals)
        d = rng.integers(15, 29) if 8 in sets[r] else rng.integers(1, 8)
        incident[r] = ren + rng.integers(0, 30 - d + 1)
        filed[r], locked[r] = incident[r] + d, True
        renewal_timed.add(r)

    # ---- Pattern 6: frequency abuse (3+ claims inside a 6-month window) --------
    # Pull the claimant's other claims into the 180 days before this one.
    for r in with_p[6]:
        t = filed[r]
        sib = members[claimant[r]]
        sib = sib[sib != r]
        in_window = (filed[sib] > t - WINDOW_6M) & (filed[sib] <= t)
        need = 2 - int(in_window.sum()) + int(rng.random() < 0.3)   # sometimes 4 claims
        moved = 0
        for s in rng.permutation(sib[~in_window]):
            if moved >= need:
                break
            if locked[s]:
                continue
            d, p = filed[s] - incident[s], policy[s]
            lo = max(t - WINDOW_6M + 5, pol_start[p] + d)
            hi = min(t - 1, pol_end[p] - 1 + d, SNAP)
            if lo > hi:
                continue
            filed[s] = rng.integers(lo, hi + 1)
            incident[s], locked[s] = filed[s] - d, True
            moved += 1
        if in_window.sum() + moved >= 2:
            locked[r] = True
            applied[6].append(r)

    # ---- Pattern 8: long incident->filing delay (15-45 days) -------------------
    # Staged / fabricated losses are reported late (time to "arrange" evidence).
    for r in with_p[8]:
        p = policy[r]
        if 15 <= filed[r] - incident[r] <= 45:          # already set by pattern 1 / 2
            applied[8].append(r)
            continue
        d = rng.integers(15, 46)
        if filed[r] - d >= pol_start[p]:                 # keep filing date, push incident back
            incident[r] = filed[r] - d
        elif not locked[r] and pol_start[p] + d <= SNAP:
            incident[r] = pol_start[p] + rng.integers(0, min(SNAP - pol_start[p] - d, 10) + 1)
            filed[r] = incident[r] + d
        elif filed[r] - pol_start[p] >= 15:
            incident[r] = pol_start[p]
        else:
            continue
        applied[8].append(r)

    clm["filed_d"], clm["incident_d"], clm["date_locked"] = filed, incident, locked

    # ---- Amount patterns (2, 3, 5, 4) in chronological order ---------------------
    amount = clm["claim_amount"].to_numpy().astype(float)
    limit = pol_limit[policy]

    def prior_mean(r):
        sib = members[claimant[r]]
        prior = sib[(filed[sib] < filed[r]) | ((filed[sib] == filed[r]) & (sib < r))]
        return amount[prior].mean() if len(prior) else np.nan

    amount_rows = sorted((r for r, s in sets.items() if s & {2, 3, 4, 5}),
                         key=lambda r: (filed[r], r))
    for r in amount_rows:
        s, lim = sets[r], limit[r]

        # Pattern 2 (amount half): the post-renewal claim is large for its type
        if 2 in s and r in renewal_timed:
            target = np.quantile(ctx["type_amounts"][ctype[r]], rng.uniform(0.80, 0.97))
            amount[r] = max(amount[r], min(target, 0.92 * lim))
            applied[2].append(r)

        # Pattern 3: inflate to 2.5-5x the claimant's own historical average
        spiked_avg = np.nan
        if 3 in s:
            avg = prior_mean(r)
            if not np.isnan(avg):
                hi_mult = min(5.0, 0.98 * lim / avg)
                if hi_mult >= 2.55:
                    amount[r] = max(avg * rng.uniform(2.55, hi_mult), SEVERITY[ctype[r]]["low"])
                    spiked_avg = avg
                    applied[3].append(r)

        # Pattern 5: park the claim just under the coverage limit (95-98% of it)
        if 5 in s:
            amount[r] = lim * rng.uniform(0.951, 0.979)
            applied[5].append(r)

        # Pattern 4: suspiciously round amount (multiple of 1,000 / 10,000 / 1,00,000).
        # Rounding must not break any other amount pattern planted on the same claim.
        if 4 in s:
            unit = int(rng.choice([1_000, 10_000, 100_000], p=[0.55, 0.35, 0.10]))
            v, best = amount[r], None
            for u in (x for x in (100_000, 10_000, 1_000) if x <= unit):
                for cand in (np.round(v / u) * u, np.floor(v / u) * u, np.ceil(v / u) * u):
                    if cand < 1_000 or cand > lim:
                        continue
                    if 5 in s and not 0.95 * lim <= cand <= 0.98 * lim:
                        continue
                    if not np.isnan(spiked_avg) and cand < 2.5 * spiked_avg:
                        continue
                    best = cand
                    break
                if best is not None:
                    break
            if best is not None:
                amount[r] = best
                applied[4].append(r)

    clm["claim_amount"] = np.round(amount).astype(np.int64)

    # ---- Pattern 7: colluding provider -----------------------------------------
    # Fraud rings keep going back to the same provider; noise rows pick one at random.
    provider = clm["provider_code"].to_numpy().copy()
    ring = ctx["ring"] if sticky_provider else {}
    for r in with_p[7]:
        pool = ctx["high_risk"][ctype[r]]
        key = (claimant[r], ctype[r])
        if key not in ring:
            ring[key] = str(rng.choice(pool))
        provider[r] = ring[key] if rng.random() < 0.75 else str(rng.choice(pool))
        applied[7].append(r)
    clm["provider_code"] = provider

    # ---- Pattern 9: address and/or bank details changed shortly before filing ---
    # Classic account-takeover / payout-diversion signal.
    addr = clm["addr_changed"].to_numpy().copy()
    bank = clm["bank_changed"].to_numpy().copy()
    for r in with_p[9]:
        u = rng.random()
        if u < 0.5:
            bank[r] = True
        elif u < 0.8:
            addr[r] = True
        else:
            bank[r] = addr[r] = True
        applied[9].append(r)
    clm["addr_changed"], clm["bank_changed"] = addr, bank
    return applied


# =============================================================================
# Stage 3: fraud injection
# =============================================================================
def inject_fraud_patterns(clm, pol, claimants, ctx, rng):
    """Pick ~5% of claims as fraud, then plant 1-4 overlapping red flags on each."""
    N = len(clm)
    n_fraud = int(round(N * FRAUD_RATE))

    # Fraud is not random: it clusters in fraud-prone claimants (so a claimant's
    # claims behave consistently), repeat claimants and health/auto lines.
    who = clm["claimant_idx"].to_numpy()
    propensity = claimants["fraud_propensity"].to_numpy()[who]
    repeat = ctx["claims_per_claimant"][who] > 1
    frequent = ctx["claims_per_claimant"][who] >= 3
    type_effect = clm["claim_type"].map({"health": 0.15, "auto": 0.10, "property": -0.25}).to_numpy()
    w = np.exp(0.9 * propensity + 0.5 * repeat + 0.6 * frequent + type_effect)
    fraud_rows = rng.choice(N, size=n_fraud, replace=False, p=w / w.sum())
    clm.loc[fraud_rows, "is_fraud"] = 1

    # Most fraud cases carry 2-3 red flags; ~15% are "weak" single-flag cases
    n_flags = rng.choice(FRAUD_N_PATTERNS[0], size=n_fraud, p=FRAUD_N_PATTERNS[1])
    feas = pattern_feasibility(clm, pol, ctx)
    sets = choose_pattern_sets(fraud_rows, feas, FRAUD_PATTERN_WEIGHTS, n_flags, rng)
    return apply_patterns(clm, pol, ctx, sets, rng, sticky_provider=True)


# =============================================================================
# Stage 4: noise (keeps the classes from being cleanly separable)
# =============================================================================
def add_noise(clm, pol, ctx, rng):
    """Give ~25% of genuine claims exactly one red flag.

    Real books are full of innocent anomalies: a chronic patient with many claims,
    a garage quote rounded to 50,000, a customer who moved house, an accident in
    week two of a new policy. Together with the natural base rates already in the
    genuine data (long delays, popular high-risk providers, repeat claimants) and
    the single-flag fraud cases, this makes the problem genuinely probabilistic.
    """
    genuine = np.flatnonzero(clm["is_fraud"].to_numpy() == 0)
    rows = rng.choice(genuine, size=int(NOISE_SHARE * len(genuine)), replace=False)
    feas = pattern_feasibility(clm, pol, ctx)
    sets = choose_pattern_sets(rows, feas, NOISE_PATTERN_WEIGHTS, np.ones(len(rows), int), rng)
    return apply_patterns(clm, pol, ctx, sets, rng, sticky_provider=False)


# =============================================================================
# Stage 5: final assembly
# =============================================================================
def assemble_dataset(clm, pol, claimants, rng):
    """Derive features, assign IDs, add realistic missingness, validate, order columns."""
    N = len(clm)
    policy = clm["policy_idx"].to_numpy()
    start = pol["start_d"].to_numpy()[policy]
    incident = clm["incident_d"].to_numpy()
    filed = clm["filed_d"].to_numpy()

    # --- Logical consistency checks ------------------------------------------
    assert (incident >= start).all(), "incident before policy start"
    assert (incident < pol["term_end_d"].to_numpy()[policy]).all(), "incident outside cover"
    assert (filed >= incident).all(), "claim filed before incident"
    assert (filed <= SNAP).all(), "claim filed after snapshot"
    assert (clm["claim_amount"].to_numpy() <= pol["coverage_limit"].to_numpy()[policy]).all()

    # --- IDs: claim_id in filing order, policy_id in inception order ----------
    order = np.lexsort((np.arange(N), filed))           # tie-break == injection's tie-break
    df = clm.iloc[order].reset_index(drop=True)
    df["claim_id"] = [f"CLM{i + 1:05d}" for i in range(N)]
    pol_rank = np.empty(len(pol), dtype=int)
    pol_rank[np.lexsort((np.arange(len(pol)), pol["start_d"].to_numpy()))] = np.arange(len(pol))
    claimant_perm = rng.permutation(len(claimants))
    df["policy_id"] = [f"POL{pol_rank[p] + 1:05d}" for p in df["policy_idx"]]
    df["claimant_id"] = [f"CLT{claimant_perm[c] + 1:05d}" for c in df["claimant_idx"]]

    # --- Dates and date-derived features -------------------------------------
    policy = df["policy_idx"].to_numpy()
    start = pol["start_d"].to_numpy()[policy]
    incident = df["incident_d"].to_numpy()
    filed = df["filed_d"].to_numpy()
    k = np.minimum((incident - start) // YEAR, pol["n_renewals"].to_numpy()[policy])
    renewal = np.where(k >= 1, start + k * YEAR, np.nan)   # most recent renewal <= incident

    df["policy_start_date"] = to_datetime(start)
    df["policy_renewal_date"] = to_datetime(renewal)
    df["incident_date"] = to_datetime(incident)
    df["claim_filed_date"] = to_datetime(filed)
    df["policy_coverage_limit"] = pol["coverage_limit"].to_numpy()[policy]
    df["is_weekend_filed"] = df["claim_filed_date"].dt.dayofweek >= 5
    df["days_policy_to_claim"] = filed - start
    df["days_incident_to_claim"] = filed - incident

    # --- Claimant history (strictly prior claims; df is already in filing order)
    g = df.groupby("claimant_idx", sort=False)
    prior_n = g.cumcount()
    prior_sum = g["claim_amount"].cumsum() - df["claim_amount"]
    df["claimant_avg_past_claim"] = (prior_sum / prior_n.replace(0, np.nan)).round(2)  # null = first claim

    count_6m = np.empty(N, dtype=int)                    # claims in trailing 180 days, incl. this one
    for rows in g.indices.values():
        f = filed[rows]
        count_6m[rows] = np.arange(len(rows)) - np.searchsorted(f, f - WINDOW_6M + 1) + 1
    df["claimant_claim_count_6m"] = count_6m

    # --- Realistic missing data (~1-2%) ---------------------------------------
    # Provider code not captured on the claim form; slightly more common on
    # fraudulent files (incomplete paperwork) - a weak, realistic signal.
    miss_provider = rng.random(N) < np.where(df["is_fraud"] == 1, 0.025, 0.013)
    df.loc[miss_provider, "provider_code"] = np.nan
    # History lookup failed for ~1% of returning claimants (legacy-system migration)
    returning = df["claimant_avg_past_claim"].notna().to_numpy()
    df.loc[returning & (rng.random(N) < 0.01), "claimant_avg_past_claim"] = np.nan

    df = df.rename(columns={
        "provider_code": "hospital_garage_code",
        "addr_changed": "claimant_address_changed_recently",
        "bank_changed": "claimant_bank_details_changed_recently",
    })
    columns = [
        "claim_id", "policy_id", "claimant_id", "policy_start_date", "policy_renewal_date",
        "incident_date", "claim_filed_date", "claim_type", "claim_amount",
        "policy_coverage_limit", "claimant_avg_past_claim", "claimant_claim_count_6m",
        "hospital_garage_code", "claimant_address_changed_recently",
        "claimant_bank_details_changed_recently", "is_weekend_filed",
        "days_policy_to_claim", "days_incident_to_claim", "is_fraud",
    ]
    df = df[columns].copy()
    for c in ("claimant_address_changed_recently", "claimant_bank_details_changed_recently"):
        df[c] = df[c].astype(bool)
    return df


# =============================================================================
# Stage 6: summary
# =============================================================================
def detect_patterns(df, high_risk_codes):
    """Re-derive every red flag from the FINAL table - what a rules engine would see."""
    amount = df["claim_amount"]
    since_renewal = (df["claim_filed_date"] - df["policy_renewal_date"]).dt.days
    p75 = df.groupby("claim_type")["claim_amount"].transform(lambda s: s.quantile(0.75))
    return pd.DataFrame({
        1: df["days_policy_to_claim"].between(15, 30),
        2: since_renewal.between(0, 30) & (amount >= p75),
        3: amount >= 2.5 * df["claimant_avg_past_claim"],
        4: amount % 1000 == 0,
        5: (amount / df["policy_coverage_limit"]).between(0.95, 0.98),
        6: df["claimant_claim_count_6m"] >= 3,
        7: df["hospital_garage_code"].isin(high_risk_codes),
        8: df["days_incident_to_claim"].between(15, 45),
        9: df["claimant_address_changed_recently"] | df["claimant_bank_details_changed_recently"],
    })


def print_summary(df, fraud_applied, noise_applied, high_risk_codes):
    fraud = df["is_fraud"] == 1
    n_fraud, n_gen = int(fraud.sum()), int((~fraud).sum())
    claims_per = df["claimant_id"].value_counts()
    line = "=" * 92

    print(line)
    print("INSURANCE CLAIMS DATASET SUMMARY")
    print(line)
    print(f"Total rows            : {len(df):,}")
    print(f"Fraud cases           : {n_fraud:,}  ({100 * n_fraud / len(df):.2f}%)")
    print(f"Unique claimants      : {len(claims_per):,}  "
          f"({100 * (claims_per > 1).mean():.1f}% filed more than one claim)")
    print(f"Unique policies       : {df['policy_id'].nunique():,}  "
          f"({100 * df['policy_renewal_date'].notna().mean():.1f}% of claims on a renewed term)")
    print(f"Unique providers used : {df['hospital_garage_code'].nunique():,}")
    print(f"Date range (filed)    : {df['claim_filed_date'].min().date()} -> "
          f"{df['claim_filed_date'].max().date()}")

    print("\nClaim amount by type (INR)")
    stats = df.groupby("claim_type")["claim_amount"].agg(
        share=lambda s: len(s) / len(df), mean="mean", median="median", min="min", max="max")
    stats["fraud_mean"] = df[fraud].groupby("claim_type")["claim_amount"].mean()
    stats["genuine_mean"] = df[~fraud].groupby("claim_type")["claim_amount"].mean()
    stats["fraud_rate"] = df.groupby("claim_type")["is_fraud"].mean()
    fmt = {"share": "{:.1%}".format, "fraud_rate": "{:.2%}".format}
    print(stats.to_string(formatters={c: fmt.get(c, "{:,.0f}".format) for c in stats.columns}))

    print("\nMissing values")
    missing = df.isna().mean()
    print(missing[missing > 0].map("{:.2%}".format).to_string())
    print("  (policy_renewal_date is empty when the claim is in the policy's first term;")
    print("   claimant_avg_past_claim is empty for first-time claimants)")

    flags = detect_patterns(df, high_risk_codes)
    print("\nRed-flag breakdown")
    print("  injected = planted by the generator | observed = true in the final CSV")
    print(f"  {'#':<2} {'pattern':<40} {'injected':>8} {'observed':>9} {'% fraud':>8} "
          f"{'noise':>6} {'% genuine':>9} {'lift':>6}")
    for p, name in PATTERNS.items():
        f_rate = flags.loc[fraud, p].mean()
        g_rate = flags.loc[~fraud, p].mean()
        lift = f_rate / g_rate if g_rate > 0 else float("inf")
        print(f"  {p:<2} {name:<40} {len(fraud_applied[p]):>8} {int(flags.loc[fraud, p].sum()):>9} "
              f"{f_rate:>8.1%} {len(noise_applied[p]):>6} {g_rate:>9.1%} {lift:>6.1f}")

    n_flags = flags.sum(axis=1).clip(upper=5)
    dist = pd.crosstab(n_flags, df["is_fraud"], normalize="columns")
    dist.index = [f"{i}{'+' if i == 5 else ''} flags" for i in dist.index]
    dist.columns = ["genuine", "fraud"]
    print("\nObserved red flags per claim (share of each class)")
    print(dist.to_string(formatters={c: "{:.1%}".format for c in dist.columns}))
    print(f"\nFraud with 2-3 flags: {dist.loc[['2 flags', '3 flags'], 'fraud'].sum():.1%} | "
          f"genuine with >=2 flags: {(n_flags[~fraud] >= 2).mean():.1%} "
          f"({int((n_flags[~fraud] >= 2).sum())} rows vs {int((n_flags[fraud] >= 2).sum())} fraud rows)")
    print(line)


# =============================================================================
# Main
# =============================================================================
def main():
    rng = np.random.default_rng(SEED)
    Faker.seed(SEED)
    fake = Faker("en_IN")

    providers = build_provider_pool(fake, rng)
    claimants, pol, clm = generate_base_claims(rng, providers)
    ctx = build_context(claimants, pol, clm, providers)
    fraud_applied = inject_fraud_patterns(clm, pol, claimants, ctx, rng)
    noise_applied = add_noise(clm, pol, ctx, rng)
    df = assemble_dataset(clm, pol, claimants, rng)

    df.to_csv(OUTPUT_CSV, index=False, date_format="%Y-%m-%d")
    providers[["provider_code", "provider_name", "city", "claim_type"]].to_csv(PROVIDER_CSV, index=False)
    print(f"Saved {OUTPUT_CSV} ({len(df):,} rows) and {PROVIDER_CSV} ({len(providers)} providers)\n")
    print_summary(df, fraud_applied, noise_applied, ctx["high_risk_codes"])


if __name__ == "__main__":
    main()
