# SPDX-License-Identifier: AGPL-3.0-or-later
"""Q-01 — Spécificité du dommage aux positions incertaines.

Protocole pré-enregistré : experiments/EXP-2026-08-21-specificite-dommage-incertaines.md

La corrélation dommage/entropie (+0.394, P5) est-elle une propriété de la LECTURE
de M, ou toute perturbation de même norme injectée aux mêmes positions la
reproduit-elle ? Variable manipulée UNIQUE : la nature du vecteur injecté à la
couche 6, à profil de norme apparié position par position — cinq niveaux :

  baseline  M=0, aucune injection — NLL_base(t), entropie H(t), ‖h_t‖ de référence
  readM     lecture réelle de M (protocole E1 : FACT_TEMPLATE force_write,
            clear_context D7, lecture active / écriture coupée), 10 secrets — ancre
  iid       ±ε gaussien frais par position (paires antithétiques, Generator torch
            dédié), renormé à ‖r_t‖ en fp32 avant cast — 10 seeds appariés aux secrets
  fixe      une direction gaussienne par run, constante, norme appariée — 10
            tirages (règle adaptative → 20 si σ inter-tirages des corrélations > 0.08)
  rbar      ±r̄ constant, norme appariée — choix Builder documenté : r̄ = moyenne
            empirique des lectures r_t enregistrées (fp32, post-cap) des 10 runs
            readM du MÊME texte, normalisée — 10 runs par bras, appariés aux profils

Mécanique : override d'INSTANCE de `FastWeightMemory.read` (précédent
`force_cycle_gate`, eval/gate_cycle.py) — AUCUNE modification du moteur engram/,
aucun flag EngramConfig (diagnostic pur).

Portes de validité : V1 (dommage moyen readM texte A = +0.1354 ± 0.005 — seule
porte dure ; échec ⇒ STOP) ; V2 (writes > 0 à chaque injection ; ‖r'_t‖ = ‖r_t‖
en fp32, tol 1e-4 ; aucun write sous hook pendant les mesures ; clear_context()
assert avant chaque mesure ; dérive ‖h'_t‖/‖h_t‖ ∈ [0.9, 1.1]).

Statistiques pré-enregistrées : par run Pearson + Spearman (co-primaire) +
winsorisation 1/99 % (sensibilité) + corr partielle (D,H|NLL_base) (diagnostic) ;
block bootstrap circulaire intra-run (L≈25, B=2000, Fisher-z) ; P2 décisionnelle :
R = corr_pair(iid)/corr(readM), corr_pair sur D_pair = [D(+ε)+D(−ε)]/2, contraste
apparié par secret/seed (Fisher-z, t apparié + Wilcoxon, signe constant ≥ 9/10) ;
P6 partage confiantes/incertaines à la médiane d'entropie ; P3/P4 (null
empirique)/P5'/P7/P-ord ; instrumentation ρ₁, corr(‖r_t‖,H_t), fraction de
saturation du cap, dérive des normes cachées.

Usage :
  python eval/perturb_position.py                              # protocole complet
  python eval/perturb_position.py --texts A --conditions readM # une condition
  (--positions > 0 ou --secrets < 10 = mode SMOKE : portes non contraignantes)

Les imports lourds (engram → transformers) sont faits dans main() pour que les
tests CPU (tests/test_perturb_position.py) importent ce module en millisecondes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------- constantes

V1_ANCHOR = 0.1354   # E3 none λ2/cap0.5, reproduit par gate_anomaly.py ET gate_cycle.py
V1_TOL = 0.005
P5_ANCHOR = 0.394    # V3 indicative (provenance inconnue — décision PI §13)
FP32_TOL = 1e-4      # tolérance d'appariement de norme, en fp32 (V2)
BOOT_L = 25          # block bootstrap circulaire : longueur de bloc
BOOT_B = 2000        # ... nombre de rééchantillons
ADAPT_SIGMA = 0.08   # règle adaptative random-fixe : 10 → 20 tirages
DRIFT_BAND = (0.9, 1.1)
MIN_VALID_POSITIONS = 300

# Seeds du Generator dédié (isolé du seed global 0 et de G) — un espace par condition
# et par texte, documenté pour reproduction à la ligne de commande près.
SEED_BASE = {"iid": {"A": 1000, "B": 2000}, "fixe": {"A": 3000, "B": 4000}}
BOOT_SEED = 777

# Texte neutre B (~450 tokens GPT-2) : même registre que NEUTRAL_TEXT
# (eval/collateral.py), vocabulaire disjoint des 10 SECRETS de eval/fact_injection.py.
TEXT_B = """
The valley town woke slowly under a pale grey sky. At the small station, the
first train of the morning arrived a few minutes behind schedule, and the
travelers on the platform hardly seemed to mind. A woman sold coffee and warm
rolls from a cart near the ticket office, and the stationmaster checked his
watch against the clock above the door. Beyond the tracks, the river moved
quietly under the old stone bridge, carrying leaves and small branches down
toward the lowlands.

By midday, the square in front of the town hall had filled with people. Farmers
had brought carts of grain and baskets of eggs, and shoppers moved from stall
to stall, comparing prices and exchanging news. Two old friends sat on a bench
near the fountain, arguing gently about the harvest and whether the cold would
come early this year. A clerk hurried past with a folder of papers under his
arm, nodding to the few faces he recognized on his way back to the office.

In the middle of the afternoon, the schoolhouse released its students, who
scattered down the narrow streets in twos and threes. Some stopped at the
bakery window to look at the cakes; others went straight to the field behind
the church to play until supper. The teacher stayed behind to correct the day's
exercises, and the scratch of her pen was the only sound in the empty
classroom. Outside, a delivery van made its slow rounds, and a carpenter
repaired the wooden shutters of the pharmacy.

As evening settled over the valley, the lamps along the main road came on one
by one. The station saw its last departure of the day, a nearly empty train
heading north, and the stationmaster locked the waiting room behind him. Smoke
rose from the chimneys, thin and straight in the still air. The river kept its
steady course under the bridge, and the town, having finished its business,
folded itself into another quiet night, certain that the morning would bring
back the same unhurried round of small events it had always known.
""".strip()


# ------------------------------------------------- primitives testables (CPU)

def make_generator(seed: int) -> torch.Generator:
    """Generator torch DÉDIÉ (CPU), isolé du RNG global (seed=0 de l'engine) et
    du Generator de G (pièges numériques Math, protocole §5.3)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def draw_eps(seed: int, n_pos: int, d: int) -> torch.Tensor:
    """[n_pos, d] gaussien iid fp32, déterministe par seed, indépendant du RNG global."""
    return torch.randn(n_pos, d, generator=make_generator(seed), dtype=torch.float32)


def scale_to_norm(v: torch.Tensor, target: float) -> torch.Tensor:
    """Renorme v à ‖·‖ = target, TOUT en fp32 (V2 : appariement avant cast).
    Assert |‖sortie‖ − target| ≤ 1e-4 — la tolérance serait intenable en fp16."""
    v = v.float()
    n = float(v.norm())
    if target == 0.0 or n == 0.0:
        return torch.zeros_like(v)
    out = v * (target / n)
    err = abs(float(out.norm()) - target)
    assert err <= FP32_TOL, f"V2 appariement fp32 violé : err={err:.3e} (target={target:.4f})"
    return out


class InjectionReader:
    """Override d'instance de FastWeightMemory.read : à l'appel t, renvoie
    directions[t] renormée à targets[t] (fp32), castée au dtype de h.
    Ne lit jamais M, n'écrit jamais (le write reste dans l'engine, hors hook)."""

    def __init__(self, directions: torch.Tensor, targets: list[float]):
        assert directions.shape[0] == len(targets)
        self.directions = directions.float()
        self.targets = targets
        self.t = 0
        self.applied_norms: list[float] = []

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        i = self.t
        self.t += 1
        r = scale_to_norm(self.directions[i].to(h.device), self.targets[i])
        self.applied_norms.append(float(r.norm()))
        return r.to(h.dtype)


class RecordingReader:
    """Wrappe la lecture RÉELLE de M : sortie bit à bit identique (contrôle §5.7),
    enregistre ‖r_t‖ post-cap (fp32), le flag de saturation du cap, et accumule
    Σ r_t (fp32) pour la direction moyenne r̄ de la condition ±r̄."""

    def __init__(self, memory):
        self.memory = memory
        self.norms: list[float] = []
        self.saturated: list[bool] = []
        self.sum_r: torch.Tensor | None = None

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        r = type(self.memory).read(self.memory, h)  # méthode de classe originale
        rf = r.float()
        n = float(rf.norm())
        cap = self.memory.cfg.max_read_norm * float(h.float().norm())
        self.norms.append(n)
        self.saturated.append(n >= cap * (1.0 - 1e-3))
        self.sum_r = rf.clone() if self.sum_r is None else self.sum_r + rf
        return r


# ------------------------------------------------------------- statistiques

def pearson(x: list[float], y: list[float]) -> float:
    a = torch.tensor(x, dtype=torch.float64)
    b = torch.tensor(y, dtype=torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = float(a.norm() * b.norm())
    return float(a @ b) / den if den > 0 else math.nan


def rankdata(xs: list[float]) -> list[float]:
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def winsorize(x: list[float], lo: float = 0.01, hi: float = 0.99) -> list[float]:
    t = torch.tensor(x, dtype=torch.float64)
    return t.clamp(torch.quantile(t, lo), torch.quantile(t, hi)).tolist()


def partial_corr(x: list[float], y: list[float], z: list[float]) -> float:
    """corr(x, y | z) par résidualisation (avec intercept)."""
    zt = torch.tensor(z, dtype=torch.float64)
    design = torch.stack([torch.ones_like(zt), zt], dim=1)

    def resid(v: list[float]) -> list[float]:
        vt = torch.tensor(v, dtype=torch.float64)
        beta = torch.linalg.lstsq(design, vt.unsqueeze(1)).solution
        return (vt - (design @ beta).squeeze(1)).tolist()

    return pearson(resid(x), resid(y))


def lag1_autocorr(x: list[float]) -> float:
    return pearson(x[:-1], x[1:])


def fisher_z(r: float) -> float:
    r = max(-0.999999, min(0.999999, r))
    return 0.5 * math.log((1 + r) / (1 - r))


def inv_fisher(z: float) -> float:
    return math.tanh(z)


def block_bootstrap_corr(x: list[float], y: list[float], L: int = BOOT_L,
                         B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    """Block bootstrap CIRCULAIRE intra-run des paires (x_t, y_t) : IC 95 % de la
    corrélation de Pearson (percentiles), SE en Fisher-z. Vectorisé."""
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    n = len(x)
    g = make_generator(seed)
    nb = math.ceil(n / L)
    starts = torch.randint(0, n, (B, nb), generator=g)
    idx = (starts.unsqueeze(-1) + torch.arange(L)).reshape(B, nb * L)[:, :n] % n
    xs = xt[idx]
    ys = yt[idx]
    xs = xs - xs.mean(dim=1, keepdim=True)
    ys = ys - ys.mean(dim=1, keepdim=True)
    r = (xs * ys).sum(dim=1) / (xs.norm(dim=1) * ys.norm(dim=1) + 1e-12)
    lo, hi = torch.quantile(r, torch.tensor([0.025, 0.975], dtype=torch.float64))
    z = torch.atanh(r.clamp(-0.999999, 0.999999))
    return {"ci_lo": float(lo), "ci_hi": float(hi), "se_z": float(z.std())}


# --- inférence classique sans scipy : incomplete beta → CDF de Student ---

def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    if math.isnan(t):
        return math.nan
    x = df / (df + t * t)
    return betainc_reg(df / 2.0, 0.5, x)


def t_quantile_975(df: int) -> float:
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def paired_t(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    if sd == 0.0:
        return math.nan, math.nan
    t = m / (sd / math.sqrt(n))
    return t, t_two_sided_p(t, n - 1)


def wilcoxon_signed(diffs: list[float]) -> tuple[float, float]:
    """Wilcoxon signé, bilatéral : exact (DP) si pas d'ex æquo et n ≤ 25,
    sinon approximation normale avec correction des ex æquo."""
    d = [x for x in diffs if x != 0.0]
    n = len(d)
    if n == 0:
        return math.nan, math.nan
    absd = [abs(x) for x in d]
    ranks = rankdata(absd)
    wplus = sum(r for r, x in zip(ranks, d) if x > 0)
    ties = len(set(absd)) != n
    if not ties and n <= 25:
        maxw = n * (n + 1) // 2
        counts = [0] * (maxw + 1)
        counts[0] = 1
        for r in range(1, n + 1):
            for w in range(maxw, r - 1, -1):
                counts[w] += counts[w - r]
        total = float(2 ** n)
        w_int = int(round(wplus))
        cdf_le = sum(counts[: w_int + 1]) / total
        cdf_ge = sum(counts[w_int:]) / total
        p = min(1.0, 2.0 * min(cdf_le, cdf_ge))
    else:
        mu = n * (n + 1) / 4.0
        from collections import Counter
        tie_term = sum(t ** 3 - t for t in Counter(absd).values())
        sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
        z = (wplus - mu - math.copysign(0.5, wplus - mu)) / math.sqrt(sigma2)
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))
    return wplus, p


def mean_sd(xs: list[float]) -> tuple[float, float]:
    return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)


# ------------------------------------------------------------ boucle de mesure

def measured_stream(engine, token_ids: list[int], *, read: bool):
    """Streame token par token, écriture coupée. Retourne (nlls, entropies, hnorms)
    alignés sur token_ids (indice 0 : nll/entropie None — pas de logits antérieurs).
    Asserts V2 : clear_context effectif, aucun write pendant la mesure."""
    engine.clear_context()
    assert engine._context_len == 0 and engine._past is None, "V2 : clear_context inopérant"
    w0 = engine.memory.write_count
    nlls, ents, hnorms = [], [], []
    for tid in token_ids:
        if engine._last_logits is None:
            ents.append(None)
        else:
            logp = torch.log_softmax(engine._last_logits.float(), dim=-1)
            ents.append(float(-(logp.exp() * logp).sum()))
        rec = engine._consume(tid, read=read, write=False, force_write=False)
        assert not rec.wrote
        nlls.append(rec.nll)
        hnorms.append(float(engine.cortex.last_h_pre.float().norm()))
    assert engine.memory.write_count == w0, "V2 : write pendant une mesure"
    return nlls, ents, hnorms


def inject_fact(engine, template: str, secret: str) -> int:
    """Protocole E1 : reset M, contexte vierge, injection du fait avec write forcé."""
    engine.reset_memory()
    engine.clear_context()
    recs = engine.stream(template.format(secret=secret), force_write=True)
    writes = sum(r.wrote for r in recs)
    return writes


def drift_stats(hnorms_run: list[float], hnorms_base: list[float]) -> dict:
    ratios = [a / b for a, b in zip(hnorms_run, hnorms_base)]
    inband = [DRIFT_BAND[0] <= r <= DRIFT_BAND[1] for r in ratios]
    return {"min": min(ratios), "max": max(ratios),
            "mean": statistics.mean(ratios),
            "frac_in_band": statistics.mean([float(b) for b in inband])}


# --------------------------------------------------------------------- main

def run_metrics(D: list[float], H: list[float], nll_base: list[float],
                h_median: float) -> dict:
    """Statistiques par run sur les positions valides (protocole §10)."""
    conf = [d for d, h in zip(D, H) if h <= h_median]
    unc = [d for d, h in zip(D, H) if h > h_median]
    return {
        "pearson": pearson(D, H),
        "spearman": spearman(D, H),
        "pearson_winsor": pearson(winsorize(D), winsorize(H)),
        "partial_DH_given_nllbase": partial_corr(D, H, nll_base),
        "boot": block_bootstrap_corr(D, H),
        "mean_damage": statistics.mean(D),
        "damage_confident": statistics.mean(conf),
        "damage_uncertain": statistics.mean(unc),
        "rho1_D": lag1_autocorr(D),
    }


def save_run_json(out_dir: Path, name: str, payload: dict) -> None:
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runs" / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--positions", type=int, default=0,
                        help="tronquer les textes à N tokens (0 = complet ; >0 = SMOKE)")
    parser.add_argument("--secrets", type=int, default=10)
    parser.add_argument("--texts", default="A,B")
    parser.add_argument("--conditions", default="readM,iid,fixe,rbar")
    parser.add_argument("--out", default=None,
                        help="dossier de sortie (défaut : experiments/results/specificite-dommage-incertaines)")
    args = parser.parse_args()

    t_start = time.time()
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out) if args.out else root / "experiments" / "results" / "specificite-dommage-incertaines"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = "python eval/perturb_position.py " + " ".join(sys.argv[1:])

    # Imports lourds ici (voir docstring).
    from collateral import NEUTRAL_TEXT  # noqa: E402
    from fact_injection import FACT_TEMPLATE, SECRETS  # noqa: E402
    from engram import EngramConfig, EngramEngine  # noqa: E402

    smoke = args.positions > 0 or args.secrets < len(SECRETS)
    secrets = SECRETS[: args.secrets]
    texts_sel = [t.strip() for t in args.texts.split(",") if t.strip()]
    conds_sel = [c.strip() for c in args.conditions.split(",") if c.strip()]

    # §7 : read_gate="none" (écart au défaut keysim, comme P5 et gate_cycle.py) ;
    # tout le reste = défauts EngramConfig (λ=2, cap=0.5, η=0.2, decay=1e-3, thr=4,
    # dg=8192/64, prune 512/0.10, seed 0).
    cfg = EngramConfig(read_gate="none")
    if args.model:
        cfg.model_name = args.model
    if args.layer is not None:
        cfg.layer_index = args.layer

    print(f"[perturb_position / Q-01] {cfg.summary()}")
    if smoke:
        print("  !! MODE SMOKE : portes V1/positions non contraignantes")
    engine = EngramEngine(cfg)
    device = str(engine.cortex.device)
    print(f"  device : {device}")
    if device == "cpu":
        print("  !! ANOMALIE : repli CPU (CUDA indisponible) — à rapporter")

    d_model = engine.cortex.d_model
    texts = {"A": NEUTRAL_TEXT, "B": TEXT_B}
    for name, text in texts.items():
        low = text.lower()
        clash = [s for s in SECRETS if s in low]
        assert not clash, f"texte {name} : vocabulaire non disjoint des secrets {clash}"

    ids = {}
    for name in texts_sel:
        tok = engine.tokenizer.encode(texts[name])
        if args.positions > 0:
            tok = tok[: args.positions]
        ids[name] = tok
        n_valid = len(tok) - 1
        print(f"  texte {name} : {len(tok)} tokens ({n_valid} positions valides)")
        if not smoke and n_valid < MIN_VALID_POSITIONS:
            print(f"  !! INVALIDE : < {MIN_VALID_POSITIONS} positions valides sur le texte {name}")
            sys.exit(2)

    stream_count = 0
    injection_count = 0
    data: dict[str, dict] = {}   # texte -> {baseline, h_median, runs: {cond: [run dicts]}}

    def base_run(text: str) -> dict:
        nonlocal stream_count
        engine.reset_memory()
        t0 = time.time()
        nlls, ents, hnorms = measured_stream(engine, ids[text], read=False)
        dur = time.time() - t0
        stream_count += 1
        valid = [i for i in range(len(nlls)) if nlls[i] is not None]
        h_valid = [ents[i] for i in valid]
        entry = {"nlls": nlls, "ents": ents, "hnorms": hnorms, "valid": valid,
                 "h_median": statistics.median(h_valid), "duration_s": dur}
        print(f"  [baseline-{text}] NLL {statistics.mean([nlls[i] for i in valid]):.4f} "
              f"| H méd {entry['h_median']:.4f} | {dur:.1f}s")
        save_run_json(out_dir, f"baseline-{text}", {
            "command": command, "condition": "baseline", "text": text, "seed": None,
            "config": cfg.summary(), "device": device, "duration_s": dur,
            "writes": 0, "metrics": {
                "nll_base": nlls, "H": ents, "hnorm": hnorms,
                "mean_nll": statistics.mean([nlls[i] for i in valid]),
                "h_median": entry["h_median"]},
            "stdout": f"baseline {text} ok"})
        return entry

    def perturbed_run(text: str, cond_name: str, seed_label, reader, *,
                      writes: int | None, extra: dict | None = None) -> dict:
        """Un stream de mesure avec un lecteur (RecordingReader ou InjectionReader)
        installé en override d'instance. Retourne le dict du run (D_t inclus)."""
        nonlocal stream_count
        base = data[text]
        engine.memory.read = reader          # override d'instance (précédent gate_cycle)
        try:
            t0 = time.time()
            nlls, ents, hnorms = measured_stream(engine, ids[text], read=True)
            dur = time.time() - t0
        finally:
            del engine.memory.read           # retire l'override : lecture normale
        stream_count += 1
        if isinstance(reader, InjectionReader):
            assert reader.t == len(ids[text]), "hook et positions désalignés"
            rnorms = reader.applied_norms
            saturated = None
        else:
            rnorms = reader.norms
            saturated = reader.saturated
        valid = base["valid"]
        D_full = [None if nlls[i] is None else nlls[i] - base["nlls"][i]
                  for i in range(len(nlls))]
        D = [D_full[i] for i in valid]
        H = [base["ents"][i] for i in valid]
        nll_b = [base["nlls"][i] for i in valid]
        met = run_metrics(D, H, nll_b, base["h_median"])
        met["corr_rnorm_H"] = pearson([rnorms[i] for i in valid], H)
        met["drift"] = drift_stats(hnorms, base["hnorms"])
        if saturated is not None:
            met["cap_saturation_frac"] = statistics.mean([float(s) for s in saturated])
        name = f"{cond_name}-{text}-{seed_label}"
        line = (f"  [{name}] D̄ {met['mean_damage']:+.4f} | r(D,H) {met['pearson']:+.4f} "
                f"| ρ {met['spearman']:+.4f} | conf {met['damage_confident']:+.4f} "
                f"| incert {met['damage_uncertain']:+.4f} | {dur:.1f}s")
        print(line)
        run = {"command": command, "condition": cond_name, "text": text,
               "seed": seed_label, "config": cfg.summary(), "device": device,
               "duration_s": dur, "writes": writes,
               "metrics": {**{k: v for k, v in met.items()},
                           "D": D_full, "H": base["ents"], "rnorm": rnorms},
               "stdout": line}
        if extra:
            run.update(extra)
        save_run_json(out_dir, name, run)
        # copie en mémoire, sans les gros tableaux dupliqués pour l'analyse
        return {"name": name, "seed": seed_label, "D": D, "H": H, "met": met,
                "rnorms": rnorms, "writes": writes, "nlls": nlls}

    # ---------------------------------------------------------------- Phase 1 : ancre A
    rho1_H = {}
    if "A" in texts_sel:
        data["A"] = {}
        data["A"].update(base_run("A"))
        data["A"]["runs"] = {}
        rho1_H["A"] = lag1_autocorr([data["A"]["ents"][i] for i in data["A"]["valid"]])

        readM_A = []
        for si, secret in enumerate(secrets):
            w = inject_fact(engine, FACT_TEMPLATE, secret)
            injection_count += 1
            print(f"  injection {secret} : {w} writes", end="")
            if w == 0:
                print("  !! 0 write : run INVALIDE")
                sys.exit(2)
            print()
            frob_before = float(engine.memory.M.norm())
            rec = RecordingReader(engine.memory)
            run = perturbed_run("A", "readM", si, rec, writes=w,
                                extra={"secret": secret})
            assert abs(float(engine.memory.M.norm()) - frob_before) < 1e-6, \
                "V2 : M modifiée pendant la mesure"
            run["secret"] = secret
            run["sum_r"] = rec.sum_r
            readM_A.append(run)
        data["A"]["runs"]["readM"] = readM_A

        # ---- Porte V1 (seule porte dure) ----
        v1 = statistics.mean(r["met"]["mean_damage"] for r in readM_A)
        v1_ok = abs(v1 - V1_ANCHOR) <= V1_TOL
        print(f"\n  PORTE V1 : dommage moyen readM texte A = {v1:+.4f} "
              f"(ancre {V1_ANCHOR:+.4f} ± {V1_TOL}) → {'OK' if v1_ok else 'ÉCHEC'}")
        if not v1_ok and not smoke:
            print("  V1 HORS BANDE : run INVALIDE — STOP (protocole §6)")
            with open(out_dir / "V1_FAILURE.json", "w", encoding="utf-8") as f:
                json.dump({"v1": v1, "anchor": V1_ANCHOR, "tol": V1_TOL,
                           "per_secret": [r["met"]["mean_damage"] for r in readM_A]}, f)
            sys.exit(2)

        # ---- Contrôle §5.7 : override off = identité bit à bit (au niveau modèle) ----
        w = inject_fact(engine, FACT_TEMPLATE, secrets[0])
        injection_count += 1
        nlls_plain, _, _ = measured_stream(engine, ids["A"], read=True)  # sans override
        stream_count += 1
        # comparaison directe aux NLL brutes du run readM-A-0 (mesuré via wrapper) :
        max_dev = max(abs(a - b) for a, b in zip(nlls_plain[1:], readM_A[0]["nlls"][1:]))
        identity_ok = max_dev == 0.0
        print(f"  contrôle override-off (secret 0, texte A) : écart NLL max = {max_dev:.3e} "
              f"→ {'identité bit à bit' if identity_ok else 'ÉCART (anomalie)'}")
        save_run_json(out_dir, "identity-A", {
            "command": command, "condition": "identity_control", "text": "A",
            "seed": 0, "config": cfg.summary(), "device": device,
            "duration_s": None, "writes": w,
            "metrics": {"max_nll_deviation": max_dev, "identity": identity_ok},
            "stdout": "contrôle 5.7"})

    # ---------------------------------------------------------------- Phase 2 : ancre B
    if "B" in texts_sel:
        data["B"] = {}
        data["B"].update(base_run("B"))
        data["B"]["runs"] = {}
        rho1_H["B"] = lag1_autocorr([data["B"]["ents"][i] for i in data["B"]["valid"]])
        readM_B = []
        for si, secret in enumerate(secrets):
            w = inject_fact(engine, FACT_TEMPLATE, secret)
            injection_count += 1
            if w == 0:
                print(f"  injection {secret} : 0 write — INVALIDE")
                sys.exit(2)
            rec = RecordingReader(engine.memory)
            run = perturbed_run("B", "readM", si, rec, writes=w, extra={"secret": secret})
            run["secret"] = secret
            run["sum_r"] = rec.sum_r
            readM_B.append(run)
        data["B"]["runs"]["readM"] = readM_B

    # ------------------------------------------------------- Phase 3 : perturbations
    engine.reset_memory()  # M=0 pour toutes les conditions sans M
    rbar_unit = {}
    for text in texts_sel:
        if "readM" not in data[text]["runs"]:
            continue
        total = None
        for r in data[text]["runs"]["readM"]:
            total = r["sum_r"].clone() if total is None else total + r["sum_r"]
        rbar_unit[text] = (total / float(total.norm())).to(engine.cortex.device)

    if len(rbar_unit) == 2:
        cos_ab = float(rbar_unit["A"].cpu() @ rbar_unit["B"].cpu())
        print(f"\n  r̄ : direction moyenne empirique des lectures readM (choix Builder) ; "
              f"cos(r̄_A, r̄_B) = {cos_ab:+.4f}")
    else:
        cos_ab = None

    for text in texts_sel:
        T = len(ids[text])
        profiles = [r["rnorms"] for r in data[text]["runs"]["readM"]]

        if "iid" in conds_sel:
            for s in range(len(secrets)):
                eps = draw_eps(SEED_BASE["iid"][text] + s, T, d_model).to(engine.cortex.device)
                for arm, sign in (("iidplus", 1.0), ("iidminus", -1.0)):
                    engine.reset_memory()
                    inj = InjectionReader(sign * eps, profiles[s])
                    run = perturbed_run(text, arm, s, inj, writes=0)
                    data[text]["runs"].setdefault(arm, []).append(run)

        if "fixe" in conds_sel:
            def fixe_draw(j: int) -> None:
                u = draw_eps(SEED_BASE["fixe"][text] + j, 1, d_model)[0]
                u = (u / float(u.norm())).to(engine.cortex.device)
                engine.reset_memory()
                inj = InjectionReader(u.unsqueeze(0).expand(T, d_model), profiles[j % len(secrets)])
                run = perturbed_run(text, "fixe", j, inj, writes=0)
                data[text]["runs"].setdefault("fixe", []).append(run)

            for j in range(len(secrets)):
                fixe_draw(j)
            corrs = [r["met"]["pearson"] for r in data[text]["runs"]["fixe"]]
            sigma = statistics.stdev(corrs) if len(corrs) > 1 else 0.0
            print(f"  [fixe-{text}] σ inter-tirages (10) = {sigma:.4f} "
                  f"(règle adaptative : > {ADAPT_SIGMA} → 20 tirages)")
            if sigma > ADAPT_SIGMA and not smoke:
                for j in range(len(secrets), 2 * len(secrets)):
                    fixe_draw(j)
                corrs = [r["met"]["pearson"] for r in data[text]["runs"]["fixe"]]
                print(f"  [fixe-{text}] étendu à 20 tirages ; σ = {statistics.stdev(corrs):.4f}")

        if "rbar" in conds_sel and text in rbar_unit:
            for s in range(len(secrets)):
                for arm, sign in (("rbarplus", 1.0), ("rbarminus", -1.0)):
                    engine.reset_memory()
                    inj = InjectionReader((sign * rbar_unit[text]).unsqueeze(0).expand(T, d_model),
                                          profiles[s])
                    run = perturbed_run(text, arm, s, inj, writes=0)
                    data[text]["runs"].setdefault(arm, []).append(run)

    # ------------------------------------------------------------------ Analyse
    print("\n================ ANALYSE (statistiques pré-enregistrées) ================")
    analysis: dict = {"command": command, "config": cfg.summary(), "device": device,
                      "smoke": smoke, "streams": stream_count,
                      "injections": injection_count, "rho1_H": rho1_H,
                      "cos_rbar_AB": cos_ab, "texts": {}}

    csv_rows = []

    def condition_summary(text: str, cond: str, runs: list[dict]) -> dict:
        out = {}
        for key in ("mean_damage", "pearson", "spearman", "pearson_winsor",
                    "partial_DH_given_nllbase", "damage_confident",
                    "damage_uncertain", "rho1_D", "corr_rnorm_H",
                    "cap_saturation_frac"):
            vals = [r["met"][key] for r in runs if not math.isnan(r["met"].get(key, math.nan))]
            if vals:
                m, sd = mean_sd(vals)
                out[key] = {"mean": m, "sd": sd, "n": len(vals)}
                csv_rows.append([cond, text, key, f"{m:.6f}", f"{sd:.6f}", len(vals)])
        # signes des splits (P6)
        out["n_conf_neg"] = sum(1 for r in runs if r["met"]["damage_confident"] < 0)
        out["n_unc_pos"] = sum(1 for r in runs if r["met"]["damage_uncertain"] > 0)
        out["n_corr_pos"] = sum(1 for r in runs if r["met"]["pearson"] > 0)
        _, out["wilcoxon_p_conf"] = wilcoxon_signed([r["met"]["damage_confident"] for r in runs])
        _, out["wilcoxon_p_unc"] = wilcoxon_signed([r["met"]["damage_uncertain"] for r in runs])
        out["boot_se_z_mean"] = statistics.mean(r["met"]["boot"]["se_z"] for r in runs)
        drifts = [r["met"]["drift"] for r in runs if "drift" in r["met"]]
        if drifts:
            out["drift_mean"] = statistics.mean(d["mean"] for d in drifts)
            out["drift_min"] = min(d["min"] for d in drifts)
            out["drift_max"] = max(d["max"] for d in drifts)
            out["drift_frac_in_band"] = statistics.mean(d["frac_in_band"] for d in drifts)
        return out

    for text in texts_sel:
        tx: dict = {"conditions": {}}
        runs_by = data[text]["runs"]

        # D_pair pour iid
        pair_runs = []
        if "iidplus" in runs_by and "iidminus" in runs_by:
            for rp, rm in zip(runs_by["iidplus"], runs_by["iidminus"]):
                Dp = [(a + b) / 2 for a, b in zip(rp["D"], rm["D"])]
                H = rp["H"]
                nll_b = [data[text]["nlls"][i] for i in data[text]["valid"]]
                met = run_metrics(Dp, H, nll_b, data[text]["h_median"])
                met["corr_rnorm_H"] = rp["met"]["corr_rnorm_H"]
                pair_runs.append({"name": f"iidpair-{text}-{rp['seed']}",
                                  "seed": rp["seed"], "D": Dp, "H": H, "met": met})
            runs_by["iidpair"] = pair_runs

        for cond, runs in runs_by.items():
            tx["conditions"][cond] = condition_summary(text, cond, runs)

        # ---- P2 : décisionnelle unique ----
        if "readM" in runs_by and "iidpair" in runs_by:
            cr = [r["met"]["pearson"] for r in runs_by["readM"]]
            cp = [r["met"]["pearson"] for r in runs_by["iidpair"]]
            ratios = [p / q for p, q in zip(cp, cr)]
            n = len(ratios)
            mr, sdr = mean_sd(ratios)
            tq = t_quantile_975(n - 1)
            ci = (mr - tq * sdr / math.sqrt(n), mr + tq * sdr / math.sqrt(n))
            zdiff = [fisher_z(p) - fisher_z(q) for p, q in zip(cp, cr)]
            t_stat, t_p = paired_t(zdiff)
            w_stat, w_p = wilcoxon_signed(zdiff)
            signs = [math.copysign(1, d) for d in zdiff]
            n_modal = max(signs.count(1.0), signs.count(-1.0))
            R_of_means = statistics.mean(cp) / statistics.mean(cr)
            # Spearman co-primaire
            crs = [r["met"]["spearman"] for r in runs_by["readM"]]
            cps = [r["met"]["spearman"] for r in runs_by["iidpair"]]
            ratios_s = [p / q for p, q in zip(cps, crs)]
            mrs, sdrs = mean_sd(ratios_s)
            ci_s = (mrs - tq * sdrs / math.sqrt(n), mrs + tq * sdrs / math.sqrt(n))
            if mr <= 0.25 and ci[1] < 0.5:
                rule = "H fausse (spécifique) : R ≤ 0.25, IC exclut 0.5"
            elif mr >= 0.5 and ci[0] > 0.25:
                rule = "H vraie (générique) : R ≥ 0.5, IC exclut 0.25"
            else:
                rule = "zone grise → P6"
            tx["P2"] = {"corr_readM": cr, "corr_iidpair": cp,
                        "R_per_pair": ratios, "R_mean": mr, "R_ci95": ci,
                        "R_of_means": R_of_means,
                        "R_spearman_mean": mrs, "R_spearman_ci95": ci_s,
                        "fisherz_paired_t": {"t": t_stat, "p": t_p},
                        "fisherz_wilcoxon": {"W": w_stat, "p": w_p},
                        "sign_consistency": f"{n_modal}/{n}", "rule": rule}
            print(f"\n  --- P2 texte {text} ---")
            print(f"  corr readM      : {statistics.mean(cr):+.4f} ± {mean_sd(cr)[1]:.4f} (N={n})")
            print(f"  corr iid_pair   : {statistics.mean(cp):+.4f} ± {mean_sd(cp)[1]:.4f}")
            print(f"  R par paire     : {mr:+.4f} ± {sdr:.4f}  IC95 [{ci[0]:+.4f}, {ci[1]:+.4f}]")
            print(f"  R (ratio moy.)  : {R_of_means:+.4f} | R Spearman {mrs:+.4f} IC95 [{ci_s[0]:+.4f}, {ci_s[1]:+.4f}]")
            print(f"  Fisher-z t app. : t={t_stat:.3f} p={t_p:.4g} | Wilcoxon W={w_stat} p={w_p:.4g}"
                  f" | signes {n_modal}/{n}")
            print(f"  règle P2        : {rule}")

        # ---- P3 ----
        if "fixe" in runs_by and "iidpair" in runs_by:
            cf = [r["met"]["pearson"] for r in runs_by["fixe"]]
            cp = [r["met"]["pearson"] for r in runs_by["iidpair"]]
            tx["P3"] = {"abs_diff_means": abs(statistics.mean(cf) - statistics.mean(cp)),
                        "sigma_fixe": mean_sd(cf)[1], "sigma_iidpair": mean_sd(cp)[1],
                        "n_fixe": len(cf)}
            print(f"  P3 |corr_fixe − corr_iid_pair| = {tx['P3']['abs_diff_means']:.4f} ; "
                  f"σ_fixe {tx['P3']['sigma_fixe']:.4f} vs σ_iid {tx['P3']['sigma_iidpair']:.4f}")

        # ---- P4 : similarité inter-secrets vs null empirique ----
        def profile_similarity(runs: list[dict]) -> tuple[float, float]:
            cs = []
            for i in range(len(runs)):
                for j in range(i + 1, len(runs)):
                    cs.append(pearson(runs[i]["D"], runs[j]["D"]))
            return mean_sd(cs)

        if "readM" in runs_by:
            m_r, s_r = profile_similarity(runs_by["readM"])
            tx["P4"] = {"readM_intersecret": {"mean": m_r, "sd": s_r}}
            if "iidpair" in runs_by:
                m_n, s_n = profile_similarity(runs_by["iidpair"])
                tx["P4"]["null_iidpair"] = {"mean": m_n, "sd": s_n}
                print(f"  P4 similarité profils : readM {m_r:.4f} ± {s_r:.4f} "
                      f"vs null iid_pair {m_n:.4f} ± {s_n:.4f}")

        # ---- P5' ----
        if "readM" in runs_by and "iidpair" in runs_by:
            dm_r = statistics.mean(r["met"]["mean_damage"] for r in runs_by["readM"])
            dm_i = statistics.mean(r["met"]["mean_damage"] for r in runs_by["iidpair"])
            tx["P5prime"] = {"damage_readM": dm_r, "damage_iidpair": dm_i,
                             "ratio": dm_r / dm_i if dm_i != 0 else math.nan}
            print(f"  P5' dommage readM {dm_r:+.4f} / iid_pair {dm_i:+.4f} "
                  f"→ ratio {tx['P5prime']['ratio']:.2f}")

        # ---- P7 ----
        if "rbarplus" in runs_by and "readM" in runs_by:
            prof_corrs = [pearson(rp["D"], rm["D"])
                          for rp, rm in zip(runs_by["rbarplus"], runs_by["readM"])]
            m_pc, s_pc = mean_sd(prof_corrs)
            tx["P7"] = {"profile_corr_rbarplus_readM": {"mean": m_pc, "sd": s_pc,
                                                        "per_secret": prof_corrs}}
            print(f"  P7 corr profils (+r̄ vs readM) : {m_pc:+.4f} ± {s_pc:.4f}")

        # ---- P-ord ----
        order = {}
        for cond in ("readM", "fixe", "iidpair", "rbarplus", "rbarminus"):
            if cond in runs_by:
                order[cond] = statistics.mean(r["met"]["mean_damage"] for r in runs_by[cond])
        tx["P_ord"] = order
        print("  P-ord dommages moyens : " +
              " | ".join(f"{c} {v:+.4f}" for c, v in order.items()))

        # rho1 H
        tx["rho1_H"] = rho1_H.get(text)
        analysis["texts"][text] = tx

    # ---- V1/V3 récapitulatif ----
    if "A" in texts_sel:
        readM_A = data["A"]["runs"]["readM"]
        v1 = statistics.mean(r["met"]["mean_damage"] for r in readM_A)
        per_secret_corr = [r["met"]["pearson"] for r in readM_A]
        mean_profile = [statistics.mean(col) for col in zip(*[r["D"] for r in readM_A])]
        v3_mean_profile = pearson(mean_profile, readM_A[0]["H"])
        analysis["V1"] = {"value": v1, "anchor": V1_ANCHOR, "tol": V1_TOL,
                          "ok": abs(v1 - V1_ANCHOR) <= V1_TOL}
        analysis["V3"] = {"per_secret_corr": per_secret_corr,
                          "mean_of_corrs": statistics.mean(per_secret_corr),
                          "corr_of_mean_profile": v3_mean_profile,
                          "anchor_indicative": P5_ANCHOR}
        print(f"\n  V3 (indicative) : corr par secret moy {statistics.mean(per_secret_corr):+.4f}, "
              f"corr du profil moyen {v3_mean_profile:+.4f} (ancre indicative +{P5_ANCHOR})")

    # ---- V2 récapitulatif (les asserts durs — norme fp32 tol 1e-4, aucun write
    # sous hook, clear_context, writes>0 à l'injection — sont passés si on arrive ici)
    all_drifts = [r["met"]["drift"] for t in data.values()
                  for rs in t["runs"].values() for r in rs if "drift" in r["met"]]
    if all_drifts:
        analysis["V2"] = {
            "asserts_passes": True,
            "drift_min": min(d["min"] for d in all_drifts),
            "drift_max": max(d["max"] for d in all_drifts),
            "drift_frac_in_band_mean": statistics.mean(d["frac_in_band"] for d in all_drifts),
            "drift_all_in_band": all(d["frac_in_band"] == 1.0 for d in all_drifts),
        }
        print(f"  V2 : asserts passés ; dérive ‖h'‖/‖h‖ globale "
              f"[{analysis['V2']['drift_min']:.4f}, {analysis['V2']['drift_max']:.4f}], "
              f"fraction en bande [0.9,1.1] = {analysis['V2']['drift_frac_in_band_mean']:.4f}")

    analysis["duration_s"] = time.time() - t_start
    analysis["total_writes_all_injections"] = sum(
        r.get("writes") or 0 for t in data.values() for rs in t["runs"].values() for r in rs)
    print(f"\n  streams : {stream_count} | injections : {injection_count} | "
          f"durée totale : {analysis['duration_s']:.0f}s")

    with open(out_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=1, default=str)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "text", "metric", "mean", "sd", "N"])
        w.writerows(csv_rows)
    print(f"  résultats bruts : {out_dir}")


if __name__ == "__main__":
    main()
