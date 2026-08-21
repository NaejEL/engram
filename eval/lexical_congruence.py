# SPDX-License-Identifier: AGPL-3.0-or-later
"""Q-01b — Le terme signé de la lecture est-il porté par un canal de congruence ?

Protocole pré-enregistré : experiments/EXP-2026-08-21-congruence-lexicale.md

LECTURE SEULE sur les bruts de Q-01
(`experiments/results/specificite-dommage-incertaines/runs/*.json`).

Convention fixée avant mesure : impair_t = [D_t(+r̄) − D_t(−r̄)]/2 ;
pair_t = [D_t(+r̄) + D_t(−r̄)]/2 ; **impair négatif = le bras +r̄ aide**.
H1 prédit une pente NÉGATIVE de impair sur la log-fréquence unigramme de la cible.

Ossature (§4 du protocole, dans l'ordre d'exécution imposé) :
  V0   σ_pos (écart-type des impair_t aux confiantes) AVANT tout calcul de
       corrélation. σ_pos > 0.17 sur au moins un texte ⇒ F reste une ESTIMATION ;
       σ_pos ≤ 0.17 sur les deux ⇒ F est promue co-décisionnelle.
  manifeste SHA-256 écrit AVANT toute statistique.
  V1   reproduction des 8 valeurs publiées depuis les séries brutes (± 0.001) et
       des identités pair ± impair (1e-9).
  V2   intégrité : len(D_full) == len(ids), alignement x_t = logfreq(ids[t]),
       config identique, writes == 0 / > 0, comptage des JSON `fixe-*`.
  V3   couverture lexicale ; V3b dé-duplication ; V3c masse au plancher add-1.
  V4   ≥ 100 confiantes/texte, sd(logfreq) > 0 par strate, ≥ 15 par bin.
  V5   placebo d'estimateur : pipeline COMPLET sur impair_iid.
  V6   porte du replay r̄ (dommage moyen readM texte A = +0.1354 ± 0.005).
  P1   DÉCISIONNELLE UNIQUE : ρ de Spearman(impair, logfreq), toutes positions
       valides, IC 95 % par block bootstrap circulaire, intersection-union A ∧ B.
  P2/P2′  estimation de F (OLS primaire, post-stratification en sensibilité),
       sous la porte P3 (|Δ_z| ≥ 0.15 SD).
  P8/P8b  portes d'interprétation (congruence exacte sur W_U).
  N-P1..N-P5, D1-D4, sensibilités, contrôles §5.

INTERDITS PORTÉS PAR CE SCRIPT (protocole §4, §5, §10) :
  * il est INTERDIT d'instancier `EngramEngine` ici — ce script est un analyseur
    de bruts ; le seul contact avec un modèle est le chargement CPU de W_U
    (unembedding) SANS forward, pour P8/P8b, sous `--with-instrument` ;
  * il est INTERDIT d'utiliser le prior du cortex comme régresseur : il est
    ENDOGÈNE (corr(W_U·r, logfreq) = +0.484 EST le canal mesuré) ;
  * il est INTERDIT de calculer le bivarié impair ~ KL ;
  * aucun flag `EngramConfig`, moteur `engram/` strictement intact.

Usage :
  python eval/lexical_congruence.py
  python eval/lexical_congruence.py --with-instrument --rbar-file <rbar_unit.json>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from perturb_position import (  # noqa: E402
    SEED_BASE, draw_eps, fisher_z, lag1_autocorr, mean_sd, pearson, spearman,
    partial_corr, t_quantile_975, t_two_sided_p,
)

# ------------------------------------------------------------------ constantes
# Toutes fixées par le protocole pré-enregistré (§4, §7). Aucune n'est ajustable
# après mesure.
V0_SIGMA_THRESHOLD = 0.17     # V0 : bascule estimation / co-décisionnelle
V1_TOL = 0.001                # V1 : bande sur les 8 valeurs publiées
V1_IDENTITY_TOL = 1e-9        # V1 : identités pair ± impair
V3_COVERAGE_MIN = 0.95        # V3 : couverture lexicale exigée
V3_COVERAGE_STOP = 0.80       # V3 : sous ce seuil après repli ⇒ non concluant
V3C_FLOOR = 2                 # V3c/D4 : « cible à compte ≤ 2 »
V3C_MAX_FRAC = 0.05           # V3c : au-delà, sensibilité obligatoire
V4_MIN_CONF = 100             # V4 : positions confiantes par texte
V4_MIN_BIN = 15               # V4 : positions par bin après fusion
V5_RHO_ABS_MAX = 0.10         # V5 : |ρ_null| toléré
V6_ANCHOR = 0.1354            # V6 : ancre V1 de Q-01
V6_TOL = 0.005
P1_RHO_THRESHOLD = -0.20      # P1 : seuil du verdict « canal établi »
P1_NULL_BAND = 0.10           # P1 : bande du verdict « canal absent »
P3_MIN_DELTA_Z = 0.15         # P3 : porte de l'estimation, en SD
BOOT_B = 2000                 # A2 : nombre de réplicats
BOOT_SEED = 777               # A2 : graine
DELTA_RAW_FLOOR = 0.03        # A2 : plancher de |Δ_raw| dans un réplicat
DELTA_RAW_INVALID_FRAC = 0.05  # A2 : au-delà, IC de F invalidée
NGRAM_N = 10                  # V3b : longueur des n-grammes de dé-duplication
N_BINS = 5                    # quintiles
D2_BLOCKS = 4                 # D2 : blocs intra-texte
CHUNK_CHARS = 200_000         # découpage du corpus (aux sauts de ligne) pour la BPE

CORPUS_FULL = ["rfc9293.txt", "pnp_narrative.txt", "pg1342.txt"]
CORPUS_NARRATIVE = ["pnp_narrative.txt", "pg1342.txt"]


# --------------------------------------------------------- primitives testables

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def block_length(rho1: float, n_full: int) -> int:
    """Règle corrigée (Math Q4) : L = max(25, ⌈3/(−ln ρ̂₁)⌉ arrondi à la dizaine
    supérieure), plafonné à N_full/10. ρ̂₁ ≤ 0 ⇒ pas de dépendance mesurable ⇒ 25."""
    if not (0.0 < rho1 < 1.0) or math.isnan(rho1):
        base = 25
    else:
        raw = math.ceil(3.0 / (-math.log(rho1)))
        base = max(25, int(math.ceil(raw / 10.0) * 10))
    return max(1, min(base, int(n_full // 10)))


def circular_block_indices(n: int, L: int, B: int, seed: int) -> np.ndarray:
    """[B, n] indices de block bootstrap CIRCULAIRE (mécanique A2 : tirage sur la
    série PLEINE). Déterministe pour (n, L, B, seed)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    nb = math.ceil(n / L)
    starts = torch.randint(0, n, (B, nb), generator=g)
    idx = (starts.unsqueeze(-1) + torch.arange(L)).reshape(B, nb * L)[:, :n] % n
    return idx.numpy()


def check_alignment(targets: list[int], ids: list[int]) -> bool:
    """V2 : la cible de la position t EST ids[t] (jamais ids[t−1]).
    Un décalage d'un cran fait échouer cette clause."""
    if len(targets) != len(ids):
        return False
    return all(int(a) == int(b) for a, b in zip(targets, ids))


def ols(y: np.ndarray, cols: list[np.ndarray]) -> dict:
    """OLS avec intercept. Retourne {'beta': [b0, b1, ...], 'r2', 'resid'}."""
    y = np.asarray(y, dtype=np.float64)
    X = np.column_stack([np.ones_like(y)] + [np.asarray(c, dtype=np.float64) for c in cols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    resid = y - fit
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else math.nan
    return {"beta": beta.tolist(), "r2": r2, "resid": resid}


def ols_full(y: np.ndarray, cols: list[np.ndarray], names: list[str]) -> dict:
    """OLS avec intercept + inférence classique (SE, t, p) et R².
    Utilisé UNIQUEMENT par le bloc descriptif post-hoc `posthoc_absorption`."""
    y = np.asarray(y, dtype=np.float64)
    X = np.column_stack([np.ones_like(y)] + [np.asarray(c, dtype=np.float64) for c in cols])
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = float((resid ** 2).sum()) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), 0.0))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else math.nan
    out = {"n": n, "k": k, "dof": dof, "r2": r2, "terms": {}}
    for i, nm in enumerate(["intercept"] + names):
        t_i = beta[i] / se[i] if se[i] > 0 else math.nan
        out["terms"][nm] = {"beta": float(beta[i]), "se": float(se[i]),
                            "t": float(t_i), "p": t_two_sided_p(t_i, dof)}
    return out


def unique_contribution(y: np.ndarray, cols: dict[str, np.ndarray]) -> dict:
    """ΔR² unique et F partiel de chaque terme (R²_complet − R²_sans_le_terme)."""
    names = list(cols)
    full = ols(y, [cols[n] for n in names])
    r2f, n, k = full["r2"], len(y), len(names) + 1
    out = {"r2_full": r2f, "per_term": {}}
    for nm in names:
        rest = [cols[o] for o in names if o != nm]
        r2r = ols(y, rest)["r2"] if rest else 0.0
        d = r2f - r2r
        dof = n - k
        out["per_term"][nm] = {"delta_r2": d, "r2_reduced": r2r,
                               "F_partial": (d / (1.0 - r2f)) * dof if r2f < 1 else math.nan,
                               "dof_resid": dof}
    return out


def repetition_indicator(targets: list[int], valid: list[int]) -> np.ndarray:
    """`rep_t` : le token cible est-il DÉJÀ apparu plus tôt dans le même stream
    (fenêtre = tout le préfixe) ? Modérateur de repli (§4) et régresseur du bloc
    descriptif post-hoc."""
    seen: set[int] = set()
    rep = []
    for pos in valid:
        rep.append(1.0 if targets[pos] in seen else 0.0)
        seen.update(targets[:pos + 1])
    return np.asarray(rep)


def ols_slope(y: np.ndarray, x: np.ndarray) -> float:
    return ols(y, [x])["beta"][1]


def merge_bins(edges: list[float], xa: np.ndarray, xb: np.ndarray,
               min_count: int = V4_MIN_BIN) -> list[float]:
    """Fusionne les bins adjacents tant qu'un bin compte < min_count positions
    dans l'UN des deux textes (V4)."""
    edges = list(edges)
    while len(edges) > 2:
        ca = np.histogram(xa, bins=edges)[0]
        cb = np.histogram(xb, bins=edges)[0]
        bad = [k for k in range(len(ca)) if ca[k] < min_count or cb[k] < min_count]
        if not bad:
            break
        k = bad[0]
        # supprime la borne intérieure qui fusionne k avec son voisin
        drop = k + 1 if k + 1 < len(edges) - 1 else k
        if drop <= 0 or drop >= len(edges) - 1:
            drop = len(edges) - 2
        edges.pop(drop)
    return edges


ALIGN_MARGIN = 0.10   # V2 : marge exigée entre le décalage 0 et ses voisins ±1


def alignment_evidence(nll_valid: list[float], lf_shift: dict[int, list[float]],
                       margin: float = ALIGN_MARGIN) -> dict:
    """V2 — clause d'alignement SUBSTANTIELLE (et non tautologique).

    `check_alignment` ne compare que deux listes construites par le même chemin :
    sur les vraies données elle ne peut pas échouer. La clause qui MORD est
    empirique : le coût du cortex à la position t est le coût de prédire ids[t],
    donc corr(NLL_base_t, logfreq(ids[t])) doit être NÉGATIVE et STRICTEMENT
    meilleure (plus basse) que pour ids[t−1] et ids[t+1], avec une marge.
    Un décalage réel d'un cran fait donc échouer la porte sur les VRAIES données.

    NLL_base n'est utilisé ici QUE comme instrument d'alignement d'une porte de
    validité. Il n'entre dans aucune décisionnelle ni dans aucune estimation :
    l'interdit d'endogénéité (§4, §5) porte sur les régresseurs de `impair`,
    et le §5 classe déjà « toute statistique conditionnée sur NLL_base » comme
    explicitement non décisionnelle. Aucune de ces valeurs n'est un résultat."""
    r = {s: pearson(nll_valid, lf_shift[s]) for s in sorted(lf_shift)}
    r0 = r[0]
    neigh = [r[s] for s in r if s != 0]
    ok = r0 < 0.0 and all(r0 <= x - margin for x in neigh)
    return {"corr_nll_logfreq_by_shift": r, "margin_required": margin,
            "margin_observed": {s: r[s] - r0 for s in r if s != 0}, "ok": bool(ok)}


def quantile_edges(values, n_bins: int, xa=None, xb=None,
                   min_count: int = V4_MIN_BIN) -> list[float]:
    """Bornes de quintiles (mécanique A2 : RECALCULÉES dans chaque réplicat).
    Fusionne ensuite les bins < min_count si xa/xb sont fournis (V4)."""
    e = list(np.quantile(np.asarray(values, dtype=np.float64),
                         np.linspace(0.0, 1.0, n_bins + 1)))
    e[0] -= 1e-9
    e[-1] += 1e-9
    if xa is not None and xb is not None:
        e = merge_bins(e, xa, xb, min_count)
    return [float(x) for x in e]


def bootstrap_F(yA, xA, cA, yB, xB, cB, idxA, idxB, n_bins: int) -> dict:
    """Mécanique bootstrap A2 intégrale : blocs circulaires tirés sur la série
    PLEINE (idxA/idxB), puis DANS CHAQUE RÉPLICAT recalcul du sous-ensemble
    confiant, des bornes de quintiles, de Δ_raw, Δ_adj, E et F.
    h_median et le nombre de bins restent FIXÉS hors réplicat.
    IC invalidée si > 5 % des réplicats ont |Δ_raw| < 0.03."""
    B = idxA.shape[0]
    Fo, Fp, drs = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        ia, ib = idxA[b], idxB[b]
        ya, xa, ca = yA[ia], xA[ia], cA[ia]
        yb, xb, cb = yB[ib], xB[ib], cB[ib]
        if ca.sum() < V4_MIN_BIN or cb.sum() < V4_MIN_BIN:
            Fo[b] = Fp[b] = drs[b] = math.nan
            continue
        dr = float(ya[ca].mean() - yb[cb].mean())
        drs[b] = dr
        eb = quantile_edges(np.concatenate([xa[ca], xb[cb]]), n_bins, xa[ca], xb[cb])
        Fo[b] = ols_F(np.concatenate([ya, yb]), np.concatenate([xa, xb]),
                      xa[ca], xb[cb], dr)["F"] if dr != 0 else math.nan
        Fp[b] = poststrat_F(ya[ca], xa[ca], yb[cb], xb[cb], eb)["F"]
    ok_dr = drs[~np.isnan(drs)]
    frac_small = float(np.mean(np.abs(ok_dr) < DELTA_RAW_FLOOR)) if len(ok_dr) else math.nan
    Fo_c = np.clip(Fo[~np.isnan(Fo)], -1.0, 2.0)      # troncature [−1, 2] (A1)
    Fp_c = np.clip(Fp[~np.isnan(Fp)], -1.0, 2.0)
    def pc(a):
        return {"ci_lo": float(np.percentile(a, 2.5)), "ci_hi": float(np.percentile(a, 97.5)),
                "median": float(np.median(a))}
    return {"F_ols_ci95": pc(Fo_c), "F_poststrat_ci95": pc(Fp_c),
            "frac_replicates_delta_raw_below_0.03": frac_small,
            "ci_valid": bool(frac_small <= DELTA_RAW_INVALID_FRAC),
            "delta_raw_boot": {"mean": float(np.nanmean(drs)),
                               "sd": float(np.nanstd(drs, ddof=1))},
            "truncated_to": [-1.0, 2.0], "B": B}


def check_run_lengths(runs: dict, ids_by_text: dict[str, list[int]]) -> bool:
    """V2 : `len(D_full) == len(ids)` dans TOUS les runs qui portent une série D.
    Un JSON tronqué fait échouer cette clause."""
    for r in runs.values():
        D = r.get("metrics", {}).get("D")
        if D is None:
            continue
        if len(D) != len(ids_by_text.get(r["text"], [])):
            return False
    return True


def poststrat_F(ya: np.ndarray, xa: np.ndarray, yb: np.ndarray, xb: np.ndarray,
                edges: list[float]) -> dict:
    """Post-stratification par bins de x (poids = proportions poolées A ∪ B).
    Δ_raw = ȳ_A − ȳ_B ; Δ_adj = Σ w_k (ȳ_A,k − ȳ_B,k) ; E = Δ_raw − Δ_adj ;
    F = E / Δ_raw. ESS de Kish sur les poids effectifs."""
    ea = np.asarray(edges, dtype=np.float64)
    ka = np.clip(np.searchsorted(ea[1:-1], xa, side="right"), 0, len(ea) - 2)
    kb = np.clip(np.searchsorted(ea[1:-1], xb, side="right"), 0, len(ea) - 2)
    nk = len(ea) - 1
    delta_raw = float(ya.mean() - yb.mean())
    w, dif = [], []
    for k in range(nk):
        ma, mb = ka == k, kb == k
        if ma.sum() == 0 or mb.sum() == 0:
            continue
        w.append(float(ma.sum() + mb.sum()))
        dif.append(float(ya[ma].mean() - yb[mb].mean()))
    if not w:
        return {"delta_raw": delta_raw, "delta_adj": math.nan, "E": math.nan,
                "F": math.nan, "kish_ess": math.nan, "n_cells": 0}
    w = np.asarray(w) / float(np.sum(w))
    delta_adj = float(np.dot(w, np.asarray(dif)))
    E = delta_raw - delta_adj
    F = E / delta_raw if delta_raw != 0.0 else math.nan
    ess = float((w.sum() ** 2) / (w ** 2).sum())
    return {"delta_raw": delta_raw, "delta_adj": delta_adj, "E": E, "F": F,
            "kish_ess": ess, "n_cells": len(w)}


def ols_F(impair_pooled: np.ndarray, reg_pooled: np.ndarray,
          reg_conf_a: np.ndarray, reg_conf_b: np.ndarray,
          delta_raw: float) -> dict:
    """Estimateur PRIMAIRE de F (Math Q2, conflit tranché) : β̂ OLS sur TOUTES les
    positions valides (A ∪ B), multiplié par l'écart du régresseur aux confiantes.
    F = β̂ · Δ(régresseur, confiantes, A−B) / Δ_raw."""
    beta = ols_slope(impair_pooled, reg_pooled)
    d_reg = float(reg_conf_a.mean() - reg_conf_b.mean())
    F = beta * d_reg / delta_raw if delta_raw != 0.0 else math.nan
    return {"beta": beta, "delta_reg": d_reg, "delta_raw": delta_raw, "F": F}


def bpe_counts(texts: list[str], encode) -> Counter:
    """Comptage unigramme au niveau token BPE, déterministe. Le découpage se fait
    aux sauts de ligne (BPE local) pour éviter les séquences géantes."""
    c: Counter = Counter()
    for text in texts:
        buf = []
        size = 0
        for line in text.splitlines(keepends=True):
            buf.append(line)
            size += len(line)
            if size >= CHUNK_CHARS:
                c.update(encode("".join(buf)))
                buf, size = [], 0
        if buf:
            c.update(encode("".join(buf)))
    return c


def ngram_overlap(a: list[int], b: list[int], n: int = NGRAM_N) -> float:
    """Fraction des n-grammes de `a` présents dans `b` (V3b)."""
    if len(a) < n or len(b) < n:
        return 0.0
    sb = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    hits = sum(1 for i in range(len(a) - n + 1) if tuple(a[i:i + n]) in sb)
    return hits / float(len(a) - n + 1)


def load_rbar_unit(path: Path) -> dict[str, torch.Tensor]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {t: torch.tensor(v, dtype=torch.float32)
            for t, v in d["rbar_unit"].items()}


def save_rbar_unit(path: Path, rbar: dict[str, torch.Tensor], d_model: int,
                   cos_ab=None) -> None:
    Path(path).write_text(json.dumps(
        {"cos_rbar_AB": cos_ab, "d_model": d_model,
         "rbar_unit": {t: v.detach().cpu().float().tolist() for t, v in rbar.items()}},
        ensure_ascii=False), encoding="utf-8")


def series_mean(series: list[list]) -> list[float]:
    """Moyenne de série d'abord (clause statistique non négociable (ii))."""
    return [statistics.mean(col) for col in zip(*series)]


def spearman_ci(x: list[float], y: list[float], L: int, B: int = BOOT_B,
                seed: int = BOOT_SEED) -> dict:
    """IC 95 % percentile de ρ de Spearman par block bootstrap circulaire."""
    idx = circular_block_indices(len(x), L, B, seed)
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    rs = np.empty(B)
    for b in range(B):
        i = idx[b]
        rs[b] = spearman(xa[i].tolist(), ya[i].tolist())
    return {"rho": spearman(list(xa), list(ya)),
            "ci_lo": float(np.percentile(rs, 2.5)),
            "ci_hi": float(np.percentile(rs, 97.5)),
            "boot_mean": float(np.mean(rs)), "boot_sd": float(np.std(rs, ddof=1)),
            "L": L, "B": B}


def block_permutation_p(x: list[float], y: list[float], L: int, B: int = BOOT_B,
                        seed: int = BOOT_SEED + 1) -> dict:
    """Contrôle §5.7 : permutation en BLOCS des étiquettes y (ici logfreq)."""
    n = len(x)
    obs = spearman(x, y)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    nb = math.ceil(n / L)
    ya = np.asarray(y, dtype=np.float64)
    blocks = [ya[i * L:(i + 1) * L] for i in range(nb)]
    cnt = 0
    null = np.empty(B)
    for b in range(B):
        perm = torch.randperm(nb, generator=g).tolist()
        yp = np.concatenate([blocks[p] for p in perm])[:n]
        null[b] = spearman(x, yp.tolist())
        if abs(null[b]) >= abs(obs):
            cnt += 1
    return {"observed": obs, "p_two_sided": (cnt + 1) / (B + 1),
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_q025": float(np.percentile(null, 2.5)),
            "null_q975": float(np.percentile(null, 97.5))}


# ----------------------------------------------------------------- chargement

def load_runs(runs_dir: Path) -> dict:
    runs: dict[str, dict] = {}
    for p in sorted(runs_dir.glob("*.json")):
        runs[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return runs


def arm_series(runs: dict, cond: str, text: str, n_seeds: int = 10) -> list[list]:
    out = []
    for s in range(n_seeds):
        key = f"{cond}-{text}-{s}"
        if key in runs:
            out.append(runs[key]["metrics"]["D"])
    return out


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--corpus", default="complet", choices=["complet", "narratif"])
    ap.add_argument("--texts", default="A,B")
    ap.add_argument("--bins", type=int, default=N_BINS)
    ap.add_argument("--boot", type=int, default=BOOT_B)
    ap.add_argument("--with-instrument", action="store_true",
                    help="charge W_U en CPU SANS forward (P8/P8b)")
    ap.add_argument("--rbar-file", default=None, help="rbar_unit.json du replay (P8b)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t_start = time.time()
    root = Path(__file__).resolve().parents[1]
    runs_dir = Path(args.runs_dir) if args.runs_dir else \
        root / "experiments" / "results" / "specificite-dommage-incertaines" / "runs"
    out_dir = Path(args.out) if args.out else \
        root / "experiments" / "results" / "congruence-lexicale"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    command = "python eval/lexical_congruence.py " + " ".join(sys.argv[1:])
    texts_sel = [t.strip() for t in args.texts.split(",") if t.strip()]
    B = args.boot

    print(f"[lexical_congruence / Q-01b] {command}")
    print(f"  bruts : {runs_dir}")
    runs = load_runs(runs_dir)
    print(f"  {len(runs)} JSON chargés | corpus : {args.corpus} | bins {args.bins} | B {B}")

    res: dict = {"command": command, "runs_dir": str(runs_dir), "corpus": args.corpus,
                 "bins": args.bins, "boot": B, "texts": {}}

    # ---- séries de base ---------------------------------------------------
    base = {t: runs[f"baseline-{t}"] for t in texts_sel}
    valid = {t: [i for i, v in enumerate(base[t]["metrics"]["nll_base"]) if v is not None]
             for t in texts_sel}
    Hfull = {t: base[t]["metrics"]["H"] for t in texts_sel}
    hmed = {t: base[t]["metrics"]["h_median"] for t in texts_sel}
    nllb = {t: base[t]["metrics"]["nll_base"] for t in texts_sel}

    impair, pair, readM, dplus, dminus, rnorm_avg = {}, {}, {}, {}, {}, {}
    for t in texts_sel:
        sp = arm_series(runs, "rbarplus", t)
        sm = arm_series(runs, "rbarminus", t)
        rm = arm_series(runs, "readM", t)
        assert len(sp) == len(sm) == len(rm) == 10, f"texte {t} : bras incomplets"
        dplus[t] =[statistics.mean(r[i] for r in sp) for i in valid[t]]
        dminus[t] = [statistics.mean(r[i] for r in sm) for i in valid[t]]
        readM[t] = [statistics.mean(r[i] for r in rm) for i in valid[t]]
        impair[t] = [(a - b) / 2.0 for a, b in zip(dplus[t], dminus[t])]
        pair[t] = [(a + b) / 2.0 for a, b in zip(dplus[t], dminus[t])]
        rnorm_avg[t] = [statistics.mean(runs[f"rbarplus-{t}-{s}"]["metrics"]["rnorm"][i]
                                        for s in range(10)) for i in valid[t]]

    Hv = {t: [Hfull[t][i] for i in valid[t]] for t in texts_sel}
    nllv = {t: [nllb[t][i] for i in valid[t]] for t in texts_sel}
    conf = {t: np.asarray([h <= hmed[t] for h in Hv[t]]) for t in texts_sel}

    # =================================================================== V0
    print("\n================ V0 (AVANT tout calcul de corrélation) ================")
    v0 = {}
    for t in texts_sel:
        vals = [x for x, c in zip(impair[t], conf[t]) if c]
        v0[t] = {"sigma_pos": statistics.stdev(vals), "n_conf": len(vals),
                 "mean_impair_conf": statistics.mean(vals)}
        print(f"  σ_pos (texte {t}, {len(vals)} confiantes) = {v0[t]['sigma_pos']:.4f}")
    promoted = all(v0[t]["sigma_pos"] <= V0_SIGMA_THRESHOLD for t in texts_sel)
    v0_rule = ("F PROMUE co-décisionnelle (σ_pos ≤ 0.17 sur les deux textes)"
               if promoted else
               "F reste une ESTIMATION (σ_pos > 0.17 sur au moins un texte)")
    print(f"  seuil {V0_SIGMA_THRESHOLD} → {v0_rule}")
    res["V0"] = {"per_text": v0, "threshold": V0_SIGMA_THRESHOLD,
                 "F_promoted": promoted, "rule_applied": v0_rule}

    # ======================================== manifeste SHA-256 (avant stats)
    man_paths = sorted(runs_dir.glob("*.json")) + \
        [data_dir / n for n in CORPUS_FULL if (data_dir / n).exists()] + \
        [root / "eval" / "lexical_congruence.py", root / "eval" / "perturb_position.py"]
    if args.rbar_file and Path(args.rbar_file).exists():
        man_paths.append(Path(args.rbar_file).resolve())

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(root)).replace("\\", "/")
        except ValueError:
            return str(p.resolve()).replace("\\", "/")

    manifest = {"generated_before_analysis": True, "command": command,
                "files": {rel(p): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                          for p in man_paths}}
    with open(out_dir / "manifest_sha256.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n  manifeste SHA-256 : {len(manifest['files'])} fichiers → "
          f"{out_dir / 'manifest_sha256.json'}")
    res["manifest_files"] = len(manifest["files"])

    # =================================================================== V1
    print("\n================ V1 (reproduction depuis les séries brutes) ================")
    published = {
        "A": {"impair_conf": -0.229, "pair_conf": 0.137,
              "readM_conf": -0.0646, "dplus_conf": -0.0918},
        "B": {"impair_conf": -0.134, "pair_conf": 0.157,
              "readM_conf": 0.0370, "dplus_conf": 0.0230},
    }
    v1 = {"values": {}, "identities": {}, "ok": True}
    for t in texts_sel:
        c = conf[t]
        got = {
            "impair_conf": float(np.asarray(impair[t])[c].mean()),
            "pair_conf": float(np.asarray(pair[t])[c].mean()),
            "readM_conf": float(np.asarray(readM[t])[c].mean()),
            "dplus_conf": float(np.asarray(dplus[t])[c].mean()),
        }
        dmin_conf = float(np.asarray(dminus[t])[c].mean())
        for k, v in got.items():
            ref = published[t][k]
            ok = abs(v - ref) <= V1_TOL
            v1["values"][f"{t}.{k}"] = {"measured": v, "published": ref,
                                        "abs_dev": abs(v - ref), "ok": ok}
            v1["ok"] &= ok
            print(f"  {t} {k:<12} mesuré {v:+.6f} | publié {ref:+.4f} | "
                  f"écart {abs(v - ref):.6f} → {'OK' if ok else 'ÉCHEC'}")
        id1 = abs((got["pair_conf"] + got["impair_conf"]) - got["dplus_conf"])
        id2 = abs((got["pair_conf"] - got["impair_conf"]) - dmin_conf)
        v1["identities"][t] = {"pair_plus_impair_minus_Dplus": id1,
                               "pair_minus_impair_minus_Dminus": id2,
                               "ok": id1 <= V1_IDENTITY_TOL and id2 <= V1_IDENTITY_TOL}
        v1["ok"] &= v1["identities"][t]["ok"]
        print(f"  {t} identités : |pair+impair−D(+r̄)| = {id1:.3e} ; "
              f"|pair−impair−D(−r̄)| = {id2:.3e} → "
              f"{'OK' if v1['identities'][t]['ok'] else 'ÉCHEC'}")
    res["V1"] = v1
    if not v1["ok"]:
        print("\n  V1 ÉCHEC → ANALYSE INVALIDE (protocole §6). STOP.")
        with open(out_dir / "analysis.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=str)
        sys.exit(2)

    # =================================================================== V2
    print("\n================ V2 (intégrité, alignement, D7) ================")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer  # noqa: E402
    from collateral import NEUTRAL_TEXT  # noqa: E402
    from perturb_position import TEXT_B  # noqa: E402

    tok = AutoTokenizer.from_pretrained("gpt2")
    raw_texts = {"A": NEUTRAL_TEXT, "B": TEXT_B}
    ids = {t: tok.encode(raw_texts[t]) for t in texts_sel}
    targets = {t: list(ids[t]) for t in texts_sel}   # cible de la position t = ids[t]

    v2: dict = {"clauses": {}, "ok": True}

    def clause(name: str, ok: bool, detail=None) -> None:
        v2["clauses"][name] = {"ok": bool(ok), "detail": detail}
        v2["ok"] &= bool(ok)
        print(f"  {name:<34} {'OK' if ok else 'ÉCHEC'}  {detail if detail is not None else ''}")

    clause("len(D_full)==len(ids) (tous runs)", check_run_lengths(runs, ids),
           {t: len(ids[t]) for t in texts_sel})
    for t in texts_sel:
        clause(f"alignement structurel x_t=ids[t] [{t}]",
               check_alignment(targets[t], ids[t]), f"{len(targets[t])} cibles")
        clause(f"re-tokenisation réversible [{t}]",
               tok.decode(ids[t]) == raw_texts[t], "decode(ids) == texte source")
        clause(f"D_full[0] is None [{t}]", runs[f"rbarplus-{t}-0"]["metrics"]["D"][0] is None)

    cfgs = {r["config"] for r in runs.values()}
    clause("config identique partout", len(cfgs) == 1, next(iter(cfgs)))

    zero_w = [k for k, r in runs.items()
              if r["condition"] in ("iidplus", "iidminus", "fixe", "rbarplus", "rbarminus")]
    clause("writes==0 sur iid/fixe/±r̄",
           all(runs[k]["writes"] == 0 for k in zero_w), f"{len(zero_w)} runs")
    readm_keys = [k for k, r in runs.items() if r["condition"] == "readM"]
    clause("writes>0 sur les 20 readM (D7)",
           len(readm_keys) == 20 and all(runs[k]["writes"] > 0 for k in readm_keys),
           f"{len(readm_keys)} runs, writes {sorted({runs[k]['writes'] for k in readm_keys})}")
    n_fixe = {t: len([k for k in runs if k.startswith(f"fixe-{t}-")]) for t in texts_sel}
    clause("comptage JSON fixe-* (10 ou 20)",
           all(v in (10, 20) for v in n_fixe.values()), str(n_fixe))
    v2["n_fixe"] = n_fixe
    print("  (clause d'alignement SUBSTANTIELLE : évaluée après construction du "
          "corpus, avant toute statistique décisionnelle — voir « V2.align »)")
    res["V2"] = v2
    if not v2["ok"]:
        print("\n  V2 ÉCHEC → ANALYSE INVALIDE (protocole §6). STOP.")
        with open(out_dir / "analysis.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=str)
        sys.exit(2)

    # ============================================================ V3 / V3b / V3c
    print("\n================ V3 / V3b / V3c (corpus) ================")
    corpus_txt = {n: (data_dir / n).read_text(encoding="utf-8", errors="replace")
                  for n in CORPUS_FULL if (data_dir / n).exists()}
    missing = [n for n in CORPUS_FULL if n not in corpus_txt]
    if missing:
        print(f"  !! ANOMALIE : corpus absents {missing}")

    # --- V3b : dé-duplication ---
    corp_ids = {n: bpe_counts([txt], tok.encode) for n, txt in corpus_txt.items()}
    corp_tok = {n: [] for n in corpus_txt}
    for n, txt in corpus_txt.items():
        acc, buf, size = [], [], 0
        for line in txt.splitlines(keepends=True):
            buf.append(line)
            size += len(line)
            if size >= CHUNK_CHARS:
                acc.extend(tok.encode("".join(buf)))
                buf, size = [], 0
        if buf:
            acc.extend(tok.encode("".join(buf)))
        corp_tok[n] = acc
    v3b = {}
    if "pnp_narrative.txt" in corp_tok and "pg1342.txt" in corp_tok:
        ov = ngram_overlap(corp_tok["pnp_narrative.txt"], corp_tok["pg1342.txt"], NGRAM_N)
        v3b["pnp_in_pg1342_10gram_frac"] = ov
        print(f"  V3b pnp_narrative ⊂ pg1342 ({NGRAM_N}-grammes) : {ov:.4f}")
    for n in corp_tok:
        for t in texts_sel:
            v3b[f"{n}_vs_text{t}_10gram_frac"] = ngram_overlap(ids[t], corp_tok[n], NGRAM_N)
    dedup_drop = []
    if v3b.get("pnp_in_pg1342_10gram_frac", 0.0) > 0.5:
        dedup_drop = ["pnp_narrative.txt"]
        print(f"  V3b → dé-duplication : {dedup_drop} retiré (Austen compté deux fois)")
    else:
        print("  V3b → aucun chevauchement majeur : corpus complet conservé")
    v3b["dropped"] = dedup_drop
    for t in texts_sel:
        for n in corp_tok:
            k = f"{n}_vs_text{t}_10gram_frac"
            if v3b[k] > 0.0:
                print(f"  !! V3b chevauchement texte {t} / {n} : {v3b[k]:.4f}")
    res["V3b"] = v3b

    all_targets = [x for t in texts_sel for x in [targets[t][i] for i in valid[t]]]

    def coverage_of(files: list[str]) -> tuple[Counter, float, float]:
        c = bpe_counts([corpus_txt[n] for n in files], tok.encode)
        cov_pos = statistics.mean([1.0 if c.get(x, 0) >= 1 else 0.0 for x in all_targets])
        cov_typ = statistics.mean([1.0 if c.get(x, 0) >= 1 else 0.0 for x in set(all_targets)])
        return c, cov_pos, cov_typ

    # Ordre de repli PRÉ-DÉCLARÉ (§7, PI §13.2) : complet → narratif → STOP.
    # Jamais post hoc : la chaîne est parcourue dans cet ordre, on s'arrête au
    # premier corpus qui franchit 0.95 ; sinon on STOPPE la chaîne et le corpus
    # PRIMAIRE pré-désigné (complet dé-dupliqué) est conservé, V3 non franchie.
    chain = [("complet", [n for n in CORPUS_FULL if n in corpus_txt and n not in dedup_drop]),
             ("narratif", [n for n in CORPUS_NARRATIVE if n in corpus_txt and n not in dedup_drop])]
    if args.corpus == "narratif":
        chain = chain[1:]
    fallback_log, chosen = [], None
    for label, files in chain:
        c, cp, ct = coverage_of(files)
        fallback_log.append({"corpus": label, "files": files,
                             "coverage_positions": cp, "coverage_types": ct,
                             "passes_0.95": cp >= V3_COVERAGE_MIN})
        print(f"  repli [{label}] {files} → couverture positions {cp:.4f} / types {ct:.4f}"
              f" {'✓ ≥ 0.95' if cp >= V3_COVERAGE_MIN else '✗ < 0.95'}")
        if chosen is None and cp >= V3_COVERAGE_MIN:
            chosen = (label, files, c, cp, ct)
    if chosen is None:
        label, files = chain[0]
        c, cp, ct = coverage_of(files)
        chosen = (label, files, c, cp, ct)
        print("  → STOP de la chaîne de repli : aucun corpus ne franchit 0.95 ; "
              "le corpus PRIMAIRE pré-désigné (complet dé-dupliqué) est conservé, "
              "V3 NON FRANCHIE (rapportée telle quelle)")
    corpus_label, sel_files, counts, cov_tokens, cov_types = chosen
    # Circularité lexicale (§5.8) : A et B sont EXCLUS du comptage — écart assumé
    # et documenté vis-à-vis de eval/marginal_pull.py::unigram_logfreq (qui inclut A).
    total_tokens = sum(counts.values())
    print(f"  corpus retenu : [{corpus_label}] {sel_files} | {total_tokens} tokens BPE | "
          f"{len(counts)} types")
    print(f"  V3 couverture (positions) {cov_tokens:.4f} | (types) {cov_types:.4f} "
          f"| exigée ≥ {V3_COVERAGE_MIN} → {'OK' if cov_tokens >= V3_COVERAGE_MIN else 'NON FRANCHIE'}")
    v3 = {"corpus_files": sel_files, "corpus_label": corpus_label,
          "corpus_tokens": total_tokens, "corpus_types": len(counts),
          "coverage_positions": cov_tokens, "coverage_types": cov_types,
          "fallback_chain": fallback_log, "ok": cov_tokens >= V3_COVERAGE_MIN}
    res["V3"] = v3
    if cov_tokens < V3_COVERAGE_STOP:
        print("  V3 < 0.80 après repli ⇒ NON CONCLUANT. STOP.")
        with open(out_dir / "analysis.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=str)
        sys.exit(2)

    logfreq = {t: [math.log(counts.get(x, 0) + 1.0) for x in
                   [targets[t][i] for i in valid[t]]] for t in texts_sel}
    rawcount = {t: [counts.get(x, 0) for x in [targets[t][i] for i in valid[t]]]
                for t in texts_sel}
    floor_frac = statistics.mean([1.0 if counts.get(x, 0) <= V3C_FLOOR else 0.0
                                  for x in all_targets])
    print(f"  V3c masse au plancher add-1 (compte ≤ {V3C_FLOOR}) : {floor_frac:.4f} "
          f"{'→ sensibilité obligatoire' if floor_frac > V3C_MAX_FRAC else ''}")
    res["V3c"] = {"frac_count_le_2": floor_frac,
                  "sensitivity_required": floor_frac > V3C_MAX_FRAC}

    # --- arbitrage du PI consigné : V3 non franchie n'invalide pas l'analyse ---
    if not v3["ok"]:
        res["V3"]["PI_arbitration"] = (
            "V3 non franchie (couverture %.4f < %.2f) mais l'analyse CONTINUE : "
            "le §6 ne nomme comme invalidant que « V3 < 0.80 après repli ». "
            "Limite consignée : proxy AFFAIBLI (couverture %.4f + %.4f de cibles "
            "au plancher add-1) — à lire sous la clause d'asymétrie déjà "
            "pré-enregistrée (un ρ̂ bas ne falsifie pas H1, il falsifie « le canal "
            "est lexical au sens de la fréquence marginale d'un corpus local »)."
            % (cov_tokens, V3_COVERAGE_MIN, cov_tokens, floor_frac))
        print(f"  ARBITRAGE PI : {res['V3']['PI_arbitration']}")

    # ================================================== V2.align (substantielle)
    print("\n================ V2.align (clause d'alignement substantielle) ================")
    valign = {"ok": True, "per_text": {}}
    for t in texts_sel:
        n = len(ids[t])
        shifts = {s: [math.log(counts.get(ids[t][min(max(i + s, 0), n - 1)], 0) + 1.0)
                      for i in valid[t]] for s in (-1, 0, 1)}
        ev = alignment_evidence(nllv[t], shifts)
        valign["per_text"][t] = ev
        valign["ok"] &= ev["ok"]
        rr = ev["corr_nll_logfreq_by_shift"]
        print(f"  {t} : corr(NLL_base, logfreq) décalage −1 {rr[-1]:+.4f} | "
              f"0 {rr[0]:+.4f} | +1 {rr[+1]:+.4f} | marges "
              f"{ev['margin_observed'][-1]:+.4f} / {ev['margin_observed'][+1]:+.4f} "
              f"(exigée ≥ {ALIGN_MARGIN}) → {'OK' if ev['ok'] else 'ÉCHEC'}")
    res["V2"]["align_substantive"] = valign
    res["V2"]["ok"] = bool(res["V2"]["ok"] and valign["ok"])
    if not valign["ok"]:
        print("\n  V2.align ÉCHEC → ANALYSE INVALIDE (protocole §6). STOP.")
        with open(out_dir / "analysis.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=str)
        sys.exit(2)

    # =================================================================== V4
    print("\n================ V4 ================")
    v4 = {"ok": True, "per_text": {}}
    conf_lf = {t: np.asarray(logfreq[t])[conf[t]] for t in texts_sel}
    pooled_conf_lf = np.concatenate([conf_lf[t] for t in texts_sel])
    edges = quantile_edges(pooled_conf_lf, args.bins,
                           conf_lf[texts_sel[0]], conf_lf[texts_sel[-1]])
    for t in texts_sel:
        n_conf = int(conf[t].sum())
        unc_lf = np.asarray(logfreq[t])[~conf[t]]
        e = {"n_conf": n_conf, "n_unc": int((~conf[t]).sum()),
             "sd_logfreq_conf": float(conf_lf[t].std(ddof=1)),
             "sd_logfreq_unc": float(unc_lf.std(ddof=1))}
        e["ok"] = n_conf >= V4_MIN_CONF and e["sd_logfreq_conf"] > 0 and e["sd_logfreq_unc"] > 0
        v4["per_text"][t] = e
        v4["ok"] &= e["ok"]
        print(f"  {t} : {n_conf} confiantes, sd(logfreq) conf {e['sd_logfreq_conf']:.4f} "
              f"/ incert {e['sd_logfreq_unc']:.4f} → {'OK' if e['ok'] else 'ÉCHEC'}")
    bincounts = {t: np.histogram(conf_lf[t], bins=edges)[0].tolist() for t in texts_sel}
    v4["edges"] = [float(x) for x in edges]
    v4["bin_counts"] = bincounts
    bins_ok = all(min(c) >= V4_MIN_BIN for c in bincounts.values())
    v4["ok"] &= bins_ok
    print(f"  bins après fusion ({len(edges) - 1}) : {bincounts} → "
          f"{'OK' if bins_ok else 'ÉCHEC'}")
    res["V4"] = v4

    # ------------------------------------------------ règle de bloc (A2 / Q4)
    L = {}
    for t in texts_sel:
        r1 = max(lag1_autocorr(impair[t]), lag1_autocorr(logfreq[t]))
        L[t] = block_length(r1, len(impair[t]))
        print(f"  bloc L[{t}] = {L[t]} (ρ̂₁ = {r1:.4f}, N = {len(impair[t])})")
    res["block_length"] = {t: {"L": L[t],
                               "rho1_impair": lag1_autocorr(impair[t]),
                               "rho1_logfreq": lag1_autocorr(logfreq[t])}
                           for t in texts_sel}

    # =================================================================== V5
    print("\n================ V5 (placebo d'estimateur) ================")
    impair_iid_per_seed = {}
    for t in texts_sel:
        ip = arm_series(runs, "iidplus", t)
        im = arm_series(runs, "iidminus", t)
        impair_iid_per_seed[t] = [[(a[i] - b[i]) / 2.0 for i in valid[t]]
                                  for a, b in zip(ip, im)]
    v5 = {"per_seed_rho": {}, "pipeline": {}}
    for t in texts_sel:
        rr = [spearman(s, logfreq[t]) for s in impair_iid_per_seed[t]]
        v5["per_seed_rho"][t] = {"values": rr, "mean": statistics.mean(rr),
                                 "sd": statistics.stdev(rr)}
        avg = series_mean(impair_iid_per_seed[t])
        ci = spearman_ci(avg, logfreq[t], L[t], B)
        v5["pipeline"][t] = ci
        print(f"  {t} : ρ_null (série moyennée) {ci['rho']:+.4f} "
              f"IC95 [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}] | "
              f"par seed {statistics.mean(rr):+.4f} ± {statistics.stdev(rr):.4f}")
    # F_null : pipeline complet d'estimation sur impair_iid
    iid_avg = {t: series_mean(impair_iid_per_seed[t]) for t in texts_sel}
    tA, tB = texts_sel[0], texts_sel[-1]
    dr_null = float(np.asarray(iid_avg[tA])[conf[tA]].mean()
                    - np.asarray(iid_avg[tB])[conf[tB]].mean())
    f_null_ols = ols_F(np.concatenate([np.asarray(iid_avg[t]) for t in texts_sel]),
                       np.concatenate([np.asarray(logfreq[t]) for t in texts_sel]),
                       conf_lf[tA], conf_lf[tB], dr_null)
    f_null_ps = poststrat_F(np.asarray(iid_avg[tA])[conf[tA]], conf_lf[tA],
                            np.asarray(iid_avg[tB])[conf[tB]], conf_lf[tB], edges)
    v5["F_null_ols"] = f_null_ols
    v5["F_null_poststrat"] = f_null_ps
    print(f"  F_null OLS {f_null_ols['F']:+.4f} (β {f_null_ols['beta']:+.5f}, "
          f"Δ_raw_null {dr_null:+.5f}) | F_null post-strat {f_null_ps['F']:+.4f}")
    idxA0 = circular_block_indices(len(impair[tA]), L[tA], B, BOOT_SEED)
    idxB0 = circular_block_indices(len(impair[tB]), L[tB], B, BOOT_SEED + 1)
    f_null_boot = bootstrap_F(np.asarray(iid_avg[tA]), np.asarray(logfreq[tA]), conf[tA],
                              np.asarray(iid_avg[tB]), np.asarray(logfreq[tB]), conf[tB],
                              idxA0, idxB0, args.bins)
    v5["F_null_bootstrap"] = f_null_boot
    fn = f_null_boot["F_ols_ci95"]
    f_null_ci_ok = fn["ci_lo"] <= 0.0 <= fn["ci_hi"]
    print(f"  F_null IC95 (OLS) [{fn['ci_lo']:+.4f}, {fn['ci_hi']:+.4f}] → "
          f"{'contient 0 (OK)' if f_null_ci_ok else 'NE CONTIENT PAS 0 (ÉCHEC)'}"
          f" | réplicats |Δ_raw_null| < 0.03 : "
          f"{f_null_boot['frac_replicates_delta_raw_below_0.03']:.4f}")

    rho_ok = all(abs(v5["pipeline"][t]["rho"]) <= V5_RHO_ABS_MAX
                 and v5["pipeline"][t]["ci_lo"] <= 0 <= v5["pipeline"][t]["ci_hi"]
                 for t in texts_sel) and f_null_ci_ok
    # contrôle positif : pente lexicale du bras fixe
    fixe_slopes = {}
    for t in texts_sel:
        sl = []
        for j in range(n_fixe[t]):
            k = f"fixe-{t}-{j}"
            if k not in runs:
                continue
            Dj = np.asarray([runs[k]["metrics"]["D"][i] for i in valid[t]], dtype=np.float64)
            sl.append(ols_slope(Dj, np.asarray(logfreq[t])))
        m, sd = mean_sd(sl)
        tstat = m / (sd / math.sqrt(len(sl))) if sd > 0 else math.nan
        fixe_slopes[t] = {"slopes": sl, "mean": m, "sd": sd, "n": len(sl),
                          "t": tstat, "p": t_two_sided_p(tstat, len(sl) - 1),
                          "n_nonzero_sign_pos": sum(1 for x in sl if x > 0)}
        print(f"  contrôle positif fixe-{t} : pente lexicale moyenne {m:+.5f} ± {sd:.5f} "
              f"(N={len(sl)}, t={tstat:.2f}, p={fixe_slopes[t]['p']:.3g})")
    v5["fixe_lexical_slope"] = fixe_slopes
    # Le contrôle positif exige une pente lexicale du bras *fixe* CENTRÉE SUR 0
    # entre tirages tout en étant non nulle par tirage. Les deux composantes de
    # V5 sont rapportées SÉPARÉMENT : la porte n'est redéfinie dans aucun sens.
    pos_ctrl_ok = all(fixe_slopes[t]["p"] > 0.05 for t in texts_sel)
    pvals = "NON — p = " + ", ".join(f"{fixe_slopes[t]['p']:.3g}" for t in texts_sel)
    per_draw_nonzero = all(fixe_slopes[t]["sd"] > 0 for t in texts_sel)
    v5["placebo_antithetique_ok"] = bool(rho_ok)
    v5["controle_positif_fixe_ok"] = bool(pos_ctrl_ok)
    v5["controle_positif_fixe_per_draw_nonzero"] = bool(per_draw_nonzero)
    v5["status"] = ("FRANCHIE" if rho_ok and pos_ctrl_ok
                    else "PARTIELLEMENT ÉCHOUÉE" if rho_ok
                    else "ÉCHOUÉE")
    v5["ok"] = bool(rho_ok)          # composante invalidante (impair_iid) seule
    v5["PI_arbitration"] = (
        "Arbitrage du PI (résolution d'ambiguïté du pré-enregistrement, PAS une "
        "modification de prédiction) : le placebo antithétique passe ⇒ l'analyse "
        "reste VALIDE (P1 et P8b sont antithétiques par construction, non touchés "
        "par la composante fautive) ; le contrôle positif du bras *fixe* ÉCHOUE "
        "⇒ P8 est REQUALIFIÉ en DESCRIPTIF NON FIABLE et ne peut plus servir de "
        "porte d'interprétation. P8b devient la seule porte d'interprétation.")
    print(f"  V5 composante 1 — placebo antithétique (impair_iid) : "
          f"{'OK' if rho_ok else 'ÉCHEC'} "
          f"(ρ_null en bande, IC de ρ_null et de F_null contenant 0)")
    print(f"  V5 composante 2 — contrôle positif (pente lexicale du bras fixe) : "
          f"{'OK' if pos_ctrl_ok else 'ÉCHEC'} "
          f"(non nulle par tirage : {'oui' if per_draw_nonzero else 'non'} ; "
          f"centrée sur 0 entre tirages : {'oui' if pos_ctrl_ok else pvals})")
    print(f"  V5 STATUT GLOBAL : {v5['status']}")
    print(f"  ARBITRAGE PI : {v5['PI_arbitration']}")
    res["V5"] = v5
    if not rho_ok:
        print("\n  V5 VIOLÉ → ANALYSE INVALIDE (protocole §6). STOP.")
        with open(out_dir / "analysis.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=str)
        sys.exit(2)

    # =================================================================== P1
    print("\n================ P1 (DÉCISIONNELLE UNIQUE) ================")
    p1 = {}
    for t in texts_sel:
        ci = spearman_ci(impair[t], logfreq[t], L[t], B)
        ci["n"] = len(impair[t])
        ci["pearson"] = pearson(impair[t], logfreq[t])
        p1[t] = ci
        print(f"  {t} : ρ_S {ci['rho']:+.4f} IC95 [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}] "
              f"(N={ci['n']}, L={L[t]}, r_P {ci['pearson']:+.4f})")
    est = all(p1[t]["rho"] <= P1_RHO_THRESHOLD and p1[t]["ci_hi"] < 0 for t in texts_sel)
    absent = (all(p1[t]["ci_lo"] <= 0 <= p1[t]["ci_hi"] and abs(p1[t]["rho"]) < P1_NULL_BAND
                  for t in texts_sel)
              or any(p1[t]["rho"] >= P1_NULL_BAND for t in texts_sel))
    verdict = ("canal établi (H1a soutenue)" if est else
               "canal absent au sens du proxy" if absent else
               "inconclusif — cause : puissance (MDE ≈ −0.19)")
    print(f"  VERDICT P1 (intersection-union A ∧ B) : {verdict}")
    res["P1"] = {"per_text": p1, "verdict": verdict,
                 "threshold": P1_RHO_THRESHOLD, "null_band": P1_NULL_BAND}

    # contrôles §5.6 (dérive ordinale) et §5.7 (permutation en blocs)
    print("\n  --- contrôles §5.6 / §5.7 ---")
    ctrl = {}
    for t in texts_sel:
        tt = np.arange(len(impair[t]), dtype=np.float64)
        c_it = pearson(impair[t], tt.tolist())
        c_lt = pearson(logfreq[t], tt.tolist())
        ri = ols(np.asarray(impair[t]), [tt])["resid"]
        rl = ols(np.asarray(logfreq[t]), [tt])["resid"]
        rho_det = spearman(ri.tolist(), rl.tolist())
        perm = block_permutation_p(impair[t], logfreq[t], L[t], B)
        ctrl[t] = {"corr_impair_t": c_it, "corr_logfreq_t": c_lt,
                   "rho_detrended": rho_det, "delta_rho": abs(rho_det - p1[t]["rho"]),
                   "block_permutation": perm}
        print(f"  {t} : corr(impair,t) {c_it:+.4f} | corr(logfreq,t) {c_lt:+.4f} | "
              f"ρ détrendé {rho_det:+.4f} (Δ {ctrl[t]['delta_rho']:.4f}) | "
              f"permutation p={perm['p_two_sided']:.4g}")
    res["controls"] = ctrl

    # ================================================================== P3/P2/P2'
    print("\n================ P3 / P2 / P2′ (estimation) ================")
    delta_raw = float(np.asarray(impair[tA])[conf[tA]].mean()
                      - np.asarray(impair[tB])[conf[tB]].mean())
    print(f"  Δ_raw (impair confiantes, {tA}−{tB}) = {delta_raw:+.5f}")

    regs = {"logfreq": {t: np.asarray(logfreq[t]) for t in texts_sel},
            "logfreq_plus_H": {t: np.asarray(logfreq[t]) + np.asarray(Hv[t])
                               for t in texts_sel}}
    est_out = {}
    for rname, rv in regs.items():
        rc = {t: rv[t][conf[t]] for t in texts_sel}
        pooled_conf = np.concatenate([rc[t] for t in texts_sel])
        sd_pooled = float(pooled_conf.std(ddof=1))
        d_reg = float(rc[tA].mean() - rc[tB].mean())
        dz = d_reg / sd_pooled if sd_pooled > 0 else math.nan
        p3_ok = abs(dz) >= P3_MIN_DELTA_Z
        pooled_imp = np.concatenate([np.asarray(impair[t]) for t in texts_sel])
        pooled_reg = np.concatenate([rv[t] for t in texts_sel])
        fo = ols_F(pooled_imp, pooled_reg, rc[tA], rc[tB], delta_raw)
        e2 = quantile_edges(pooled_conf, args.bins, rc[tA], rc[tB])
        fp = poststrat_F(np.asarray(impair[tA])[conf[tA]], rc[tA],
                         np.asarray(impair[tB])[conf[tB]], rc[tB], e2)
        beta_by_text = {t: ols_slope(np.asarray(impair[t]), rv[t]) for t in texts_sel}
        # variante avec indicatrice de texte (sensibilité de spécification)
        dummy = np.concatenate([np.full(len(impair[t]), 1.0 if t == tA else 0.0)
                                for t in texts_sel])
        beta_dummy = ols(pooled_imp, [pooled_reg, dummy])["beta"][1]
        est_out[rname] = {
            "P3": {"delta_reg": d_reg, "sd_pooled": sd_pooled, "delta_z": dz,
                   "ok": p3_ok, "threshold": P3_MIN_DELTA_Z},
            "OLS": fo, "poststrat": fp, "edges": [float(x) for x in e2],
            "beta_per_text": beta_by_text, "beta_with_text_dummy": beta_dummy,
        }
        print(f"  [{rname}] Δ_z = {dz:+.4f} SD → porte P3 "
              f"{'FRANCHIE' if p3_ok else 'ÉCHOUÉE (F non rapportée)'}")
        if p3_ok:
            print(f"    F(OLS) = {fo['F']:+.4f} (β {fo['beta']:+.5f}, Δ_reg {d_reg:+.5f})")
            print(f"    F(post-strat, sensibilité) = {fp['F']:+.4f} "
                  f"(Δ_adj {fp['delta_adj']:+.5f}, E {fp['E']:+.5f}, "
                  f"ESS Kish {fp['kish_ess']:.2f}, cellules {fp['n_cells']})")
        else:
            # Protocole P3 : sous la porte, F est un 0/0 — elle n'est PAS rapportée.
            # Les composantes brutes sont mises de côté, jamais publiées comme F.
            print("    F NON RAPPORTÉE (0/0 — protocole P3)")
            # Les valeurs numériques de F sont DÉTRUITES, pas archivées : garder
            # un `withheld_...` chiffré laisserait un chemin de citation
            # accidentelle d'une quantité pré-enregistrée comme non rapportable.
            est_out[rname].pop("OLS")
            est_out[rname].pop("poststrat")
            reason = (f"NON RAPPORTÉE — porte P3 échouée : |Δ_z| = "
                      f"{abs(dz):.5f} < {P3_MIN_DELTA_Z} SD (F est un 0/0). "
                      "Valeurs numériques non conservées (protocole §4, P3).")
            est_out[rname]["OLS"] = {"F": None, "reason": reason}
            est_out[rname]["poststrat"] = {"F": None, "reason": reason}
        print(f"    β par texte : " +
              " | ".join(f"{t} {beta_by_text[t]:+.5f}" for t in texts_sel) +
              f" | β avec indicatrice texte {beta_dummy:+.5f}")

    # --- IC bootstrap de F (mécanique A2 intégrale) ---
    print("\n  --- IC bootstrap de F (A2 : blocs sur la série pleine, "
          "quintiles recalculés par réplicat) ---")
    idxA = circular_block_indices(len(impair[tA]), L[tA], B, BOOT_SEED)
    idxB = circular_block_indices(len(impair[tB]), L[tB], B, BOOT_SEED + 1)
    for rname, rv in regs.items():
        if not est_out[rname]["P3"]["ok"]:
            est_out[rname]["bootstrap"] = {"skipped": "porte P3 échouée"}
            continue
        bt = bootstrap_F(np.asarray(impair[tA]), rv[tA], conf[tA],
                         np.asarray(impair[tB]), rv[tB], conf[tB],
                         idxA, idxB, args.bins)
        est_out[rname]["bootstrap"] = bt
        bo, bp = bt["F_ols_ci95"], bt["F_poststrat_ci95"]
        frac_small, valid_ci = bt["frac_replicates_delta_raw_below_0.03"], bt["ci_valid"]
        print(f"  [{rname}] F(OLS) IC95 tronqué [−1,2] "
              f"[{bo['ci_lo']:+.4f}, {bo['ci_hi']:+.4f}] (méd {bo['median']:+.4f}) | "
              f"F(post-strat) [{bp['ci_lo']:+.4f}, {bp['ci_hi']:+.4f}]")
        print(f"           réplicats |Δ_raw| < 0.03 : {frac_small:.4f} → IC "
              f"{'VALIDE' if valid_ci else 'INVALIDÉE (A2)'}")
    res["estimation"] = est_out
    res["delta_raw"] = delta_raw
    F_point = est_out["logfreq"]["OLS"]["F"] if est_out["logfreq"]["P3"]["ok"] else None
    Fp_point = est_out["logfreq_plus_H"]["OLS"]["F"] \
        if est_out["logfreq_plus_H"]["P3"]["ok"] else None

    # ============================================== descriptives N-P1..N-P5, D1-D4
    print("\n================ Descriptives obligatoires ================")
    desc: dict = {}
    npd = {}
    for t in texts_sel:
        c = conf[t]
        ic = np.asarray(impair[t])[c]
        iu = np.asarray(impair[t])[~c]
        npd[t] = {
            "N_P1_rho_conf": spearman(ic.tolist(), conf_lf[t].tolist()),
            "N_P1_rho_unc": spearman(iu.tolist(),
                                     np.asarray(logfreq[t])[~c].tolist()),
            "N_P1b_impair_conf": float(ic.mean()),
            "N_P1b_impair_unc": float(iu.mean()),
            "N_P1b_diff_unc_minus_conf": float(iu.mean() - ic.mean()),
            "N_P4_corr_pair_logfreq": pearson(pair[t], logfreq[t]),
            "N_P4_corr_pair_H": pearson(pair[t], Hv[t]),
        }
        print(f"  N-P1 {t} : ρ_S(impair, logfreq) confiantes {npd[t]['N_P1_rho_conf']:+.4f} "
              f"| incertaines {npd[t]['N_P1_rho_unc']:+.4f}")
        print(f"  N-P1b {t} : impair confiantes {npd[t]['N_P1b_impair_conf']:+.4f} vs "
              f"incertaines {npd[t]['N_P1b_impair_unc']:+.4f} "
              f"(diff {npd[t]['N_P1b_diff_unc_minus_conf']:+.4f})")
        print(f"  N-P4 {t} : corr(pair, logfreq) {npd[t]['N_P4_corr_pair_logfreq']:+.4f} "
              f"| corr(pair, H) {npd[t]['N_P4_corr_pair_H']:+.4f}")
    desc["N_P1_P1b_P4"] = npd
    d_lf = float(conf_lf[tA].mean() - conf_lf[tB].mean())
    desc["N_P3_delta_logfreq_conf_AmB"] = d_lf
    print(f"  N-P3 : Δ logfreq(A−B) aux confiantes = {d_lf:+.5f}")
    desc["N_P2_F"] = F_point
    desc["N_P2prime_Fprime"] = Fp_point
    if F_point is not None and Fp_point is not None:
        print(f"  N-P2 F = {F_point:+.4f} | N-P2′ F′ = {Fp_point:+.4f} "
              f"| F′ − F = {Fp_point - F_point:+.4f}")

    # N-P5 : impair ~ logfreq + H + logfreq×H (UNE seule interaction)
    np5 = {}
    for t in texts_sel + ["pooled"]:
        if t == "pooled":
            y = np.concatenate([np.asarray(impair[x]) for x in texts_sel])
            lf = np.concatenate([np.asarray(logfreq[x]) for x in texts_sel])
            hh = np.concatenate([np.asarray(Hv[x]) for x in texts_sel])
        else:
            y, lf, hh = np.asarray(impair[t]), np.asarray(logfreq[t]), np.asarray(Hv[t])
        m = ols(y, [lf, hh, lf * hh])
        b = m["beta"]
        ratio = b[1] / b[2] if b[2] != 0 else math.nan
        np5[t] = {"beta_intercept": b[0], "beta_logfreq": b[1], "beta_H": b[2],
                  "beta_interaction": b[3], "ratio_logfreq_over_H": ratio,
                  "r2": m["r2"],
                  "additive": ols(y, [lf, hh])["beta"],
                  "r2_additive": ols(y, [lf, hh])["r2"]}
        print(f"  N-P5 {t} : β_logfreq {b[1]:+.5f} | β_H {b[2]:+.5f} | "
              f"β_inter {b[3]:+.5f} | rapport {ratio:+.3f} | R² {m['r2']:.4f}")
    desc["N_P5"] = np5

    # D1 : décomposition du contraste conf/incertaines
    d1 = {}
    for t in texts_sel:
        c = conf[t]
        badd = np5[t]["additive"]
        dlf = float(np.asarray(logfreq[t])[~c].mean() - np.asarray(logfreq[t])[c].mean())
        dh = float(np.asarray(Hv[t])[~c].mean() - np.asarray(Hv[t])[c].mean())
        d1[t] = {"beta_logfreq": badd[1], "beta_H": badd[2],
                 "delta_logfreq_unc_minus_conf": dlf, "delta_H_unc_minus_conf": dh,
                 "contrib_logfreq": badd[1] * dlf, "contrib_H": badd[2] * dh,
                 "observed_diff": npd[t]["N_P1b_diff_unc_minus_conf"]}
        d1[t]["ratio_contrib"] = (d1[t]["contrib_logfreq"] / d1[t]["contrib_H"]
                                  if d1[t]["contrib_H"] != 0 else math.nan)
        print(f"  D1 {t} : β_lf·Δlf {d1[t]['contrib_logfreq']:+.5f} vs "
              f"β_H·ΔH {d1[t]['contrib_H']:+.5f} "
              f"(somme {d1[t]['contrib_logfreq'] + d1[t]['contrib_H']:+.5f} vs "
              f"observé {d1[t]['observed_diff']:+.5f})")
    desc["D1"] = d1

    # D2 : lecture H2 gratuite — 4 blocs intra-texte
    d2 = {}
    for t in texts_sel:
        n = len(impair[t])
        cut = [round(k * n / D2_BLOCKS) for k in range(D2_BLOCKS + 1)]
        bl = []
        for k in range(D2_BLOCKS):
            sl = slice(cut[k], cut[k + 1])
            m = np.asarray(impair[t])[sl][conf[t][sl]]
            bl.append(float(m.mean()) if len(m) else math.nan)
        d2[t] = {"block_means_impair_conf": bl, "sd_between_blocks": statistics.stdev(bl),
                 "n_blocks": D2_BLOCKS, "block_sizes": [cut[k + 1] - cut[k]
                                                        for k in range(D2_BLOCKS)]}
        print(f"  D2 {t} : blocs {[f'{x:+.4f}' for x in bl]} | "
              f"sd intra {d2[t]['sd_between_blocks']:.4f}")
    d2["delta_raw_inter_texte"] = delta_raw
    d2["se_intra_pooled"] = math.sqrt(sum(d2[t]["sd_between_blocks"] ** 2 / D2_BLOCKS
                                          for t in texts_sel))
    print(f"  D2 : |Δ_raw inter-textes| {abs(delta_raw):.4f} vs SE intra poolée "
          f"{d2['se_intra_pooled']:.4f} → rapport "
          f"{abs(delta_raw) / d2['se_intra_pooled']:.2f}")
    desc["D2"] = d2

    # D3 : Fisher-z poolé (descriptif)
    zs = [fisher_z(p1[t]["rho"]) for t in texts_sel]
    ns = [p1[t]["n"] for t in texts_sel]
    zbar = sum(z * n for z, n in zip(zs, ns)) / sum(ns)
    desc["D3_fisherz_pooled"] = {"z": zbar, "rho": math.tanh(zbar),
                                 "per_text_z": dict(zip(texts_sel, zs))}
    print(f"  D3 : Fisher-z poolé z̄ {zbar:+.4f} → ρ {math.tanh(zbar):+.4f}")

    # D4 : cibles à compte ≤ 2 + sensibilité
    d4 = {"frac_count_le_2": floor_frac, "per_text": {}}
    for t in texts_sel:
        keep = np.asarray([c > V3C_FLOOR for c in rawcount[t]])
        d4["per_text"][t] = {
            "frac": float(1.0 - keep.mean()),
            "rho_excluding_floor": spearman(np.asarray(impair[t])[keep].tolist(),
                                            np.asarray(logfreq[t])[keep].tolist()),
            "n_kept": int(keep.sum())}
        print(f"  D4 {t} : {d4['per_text'][t]['frac']:.4f} au plancher | "
              f"ρ hors plancher {d4['per_text'][t]['rho_excluding_floor']:+.4f}")
    desc["D4"] = d4
    res["descriptives"] = desc

    # ------------------------------------------------------------ sensibilités
    print("\n================ Sensibilités (jamais décisionnelles) ================")
    sens: dict = {"poststrat": {r: est_out[r]["poststrat"] for r in est_out}}
    # variante impair / (NLL_base + 0.5) — SIGNE seulement
    sens["normalized_by_nllbase"] = {}
    for t in texts_sel:
        yn = [a / (b + 0.5) for a, b in zip(impair[t], nllv[t])]
        r = spearman(yn, logfreq[t])
        sens["normalized_by_nllbase"][t] = {"rho": r, "sign": int(math.copysign(1, r))}
        print(f"  impair/(NLL_base+0.5) {t} : ρ {r:+.4f} (signe seulement)")
    # repondération jointe 2D (H, logfreq) — SIGNE seulement
    sens["joint_2d_reweight"] = {}
    hq = {t: np.quantile(np.asarray(Hv[t])[conf[t]], np.linspace(0, 1, 4)) for t in texts_sel}
    for lab in ["2d"]:
        cells = {}
        for t in texts_sel:
            hb = np.clip(np.searchsorted(hq[t][1:-1], np.asarray(Hv[t])[conf[t]],
                                         side="right"), 0, 2)
            lb = np.clip(np.searchsorted(np.asarray(edges[1:-1]), conf_lf[t],
                                         side="right"), 0, len(edges) - 2)
            cells[t] = (hb, lb, np.asarray(impair[t])[conf[t]])
        num, den = 0.0, 0.0
        for i in range(3):
            for j in range(len(edges) - 1):
                ma = (cells[tA][0] == i) & (cells[tA][1] == j)
                mb = (cells[tB][0] == i) & (cells[tB][1] == j)
                if ma.sum() < 3 or mb.sum() < 3:
                    continue
                w = float(ma.sum() + mb.sum())
                num += w * float(cells[tA][2][ma].mean() - cells[tB][2][mb].mean())
                den += w
        d_adj2 = num / den if den > 0 else math.nan
        sens["joint_2d_reweight"][lab] = {
            "delta_adj_2d": d_adj2, "delta_raw": delta_raw,
            "F_2d": (delta_raw - d_adj2) / delta_raw if delta_raw else math.nan,
            "sign": int(math.copysign(1, (delta_raw - d_adj2) / delta_raw))
            if delta_raw and not math.isnan(d_adj2) else None}
        print(f"  repondération jointe (H, logfreq) : Δ_adj {d_adj2:+.5f} → "
              f"F_2d {sens['joint_2d_reweight'][lab]['F_2d']:+.4f} (signe seulement)")
    # corpus narratif seul
    counts_narr = bpe_counts([corpus_txt[n] for n in CORPUS_NARRATIVE
                              if n in corpus_txt and n not in dedup_drop], tok.encode)
    sens["narrative_only"] = {}
    for t in texts_sel:
        lfn = [math.log(counts_narr.get(x, 0) + 1.0)
               for x in [targets[t][i] for i in valid[t]]]
        sens["narrative_only"][t] = {"rho": spearman(impair[t], lfn)}
        print(f"  corpus narratif seul {t} : ρ {sens['narrative_only'][t]['rho']:+.4f}")
    sens["exclude_floor_counts"] = d4["per_text"]
    res["sensitivities"] = sens

    # `rep_t` calculé une seule fois : sert au modérateur de repli ET au bloc
    # descriptif POST-HOC ci-dessous.
    rep_ind = {t: repetition_indicator(targets[t], valid[t]) for t in texts_sel}

    # -------------------------------- modérateur de repli (si P1 échoue seulement)
    if not est:
        print("\n================ Modérateur de repli (P1 non établi) ================")
        mod = {}
        for t in texts_sel:
            rep = rep_ind[t]
            c = conf[t]
            mod[t] = {"repetition_rate_conf": float(rep[c].mean()),
                      "repetition_rate_unc": float(rep[~c].mean()),
                      "mean_impair_repeated_conf":
                          float(np.asarray(impair[t])[c & (rep == 1)].mean()),
                      "mean_impair_new_conf":
                          float(np.asarray(impair[t])[c & (rep == 0)].mean()),
                      "corr_impair_repetition": pearson(impair[t], rep.tolist())}
            print(f"  {t} : taux de répétition confiantes {mod[t]['repetition_rate_conf']:.4f} "
                  f"| impair répétées {mod[t]['mean_impair_repeated_conf']:+.4f} vs "
                  f"nouvelles {mod[t]['mean_impair_new_conf']:+.4f}")
        res["fallback_moderator"] = mod
    else:
        res["fallback_moderator"] = "non activé (P1 établi)"

    # ============================================ DESCRIPTIF POST-HOC (Math)
    # NON PRÉ-ENREGISTRÉ. Demandé en phase d'interprétation pour distinguer un
    # canal « lexical » d'un canal de NOUVEAUTÉ. Ne modifie AUCUN verdict,
    # AUCUNE porte, AUCUNE valeur existante.
    print("\n================ DESCRIPTIF POST-HOC (non pré-enregistré) ================")
    print("  impair ~ logfreq + H + rep — demandé par Math en interprétation")
    ph: dict = {"STATUT": "POSTHOC_NON_PREENREGISTRE — descriptif demandé en phase "
                          "d'interprétation (Math). Ne modifie aucun verdict, aucune "
                          "porte, aucune valeur pré-enregistrée.",
                "per_unit": {}}
    units = texts_sel + ["pooled"]
    for t in units:
        if t == "pooled":
            y = np.concatenate([np.asarray(impair[x]) for x in texts_sel])
            lf = np.concatenate([np.asarray(logfreq[x]) for x in texts_sel])
            hh = np.concatenate([np.asarray(Hv[x]) for x in texts_sel])
            rp = np.concatenate([rep_ind[x] for x in texts_sel])
        else:
            y, lf, hh, rp = (np.asarray(impair[t]), np.asarray(logfreq[t]),
                             np.asarray(Hv[t]), rep_ind[t])
        m_lf = ols_full(y, [lf], ["logfreq"])
        m_lfh = ols_full(y, [lf, hh], ["logfreq", "H"])
        m_full = ols_full(y, [lf, hh, rp], ["logfreq", "H", "rep"])
        uc_lfh = unique_contribution(y, {"logfreq": lf, "H": hh})
        uc_full = unique_contribution(y, {"logfreq": lf, "H": hh, "rep": rp})
        b_before = m_lfh["terms"]["logfreq"]["beta"]
        b_after = m_full["terms"]["logfreq"]["beta"]
        e = {
            "n": int(len(y)),
            "model_logfreq_H": {k: m_lfh["terms"][k] for k in ("logfreq", "H")},
            "model_logfreq_H_rep": {k: m_full["terms"][k] for k in ("logfreq", "H", "rep")},
            "r2_logfreq_only": m_lf["r2"], "r2_logfreq_H": m_lfh["r2"],
            "r2_logfreq_H_rep": m_full["r2"],
            "beta_logfreq_before_rep": b_before, "beta_logfreq_after_rep": b_after,
            "absorbed_fraction_of_beta_logfreq":
                (b_before - b_after) / b_before if b_before != 0 else math.nan,
            "unique_contrib_logfreq_H": uc_lfh,
            "unique_contrib_logfreq_H_rep": uc_full,
            "corr_rep_logfreq": pearson(rp.tolist(), lf.tolist()),
            "corr_logfreq_H": pearson(lf.tolist(), hh.tolist()),
            "corr_rep_H": pearson(rp.tolist(), hh.tolist()),
            "r_pointbiserial_rep_impair": pearson(rp.tolist(), y.tolist()),
            "rep_rate": float(rp.mean()),
        }
        ph["per_unit"][t] = e
        mf, ml = m_full["terms"], m_lfh["terms"]
        print(f"  [{t}] N={e['n']} | rep_rate {e['rep_rate']:.4f}")
        print(f"    β_logfreq {mf['logfreq']['beta']:+.5f} (SE {mf['logfreq']['se']:.5f}, "
              f"p {mf['logfreq']['p']:.3g}) | β_H {mf['H']['beta']:+.5f} "
              f"(SE {mf['H']['se']:.5f}, p {mf['H']['p']:.3g}) | β_rep "
              f"{mf['rep']['beta']:+.5f} (SE {mf['rep']['se']:.5f}, p {mf['rep']['p']:.3g})")
        print(f"    β_logfreq AVANT rep {b_before:+.5f} (SE {ml['logfreq']['se']:.5f}, "
              f"p {ml['logfreq']['p']:.3g}) → APRÈS {b_after:+.5f} | fraction absorbée "
              f"{e['absorbed_fraction_of_beta_logfreq']:+.4f}")
        u2, u3 = uc_lfh["per_term"], uc_full["per_term"]
        print(f"    ΔR² unique [logfreq+H] : logfreq {u2['logfreq']['delta_r2']:.5f} "
              f"(F partiel {u2['logfreq']['F_partial']:.2f}, dof {u2['logfreq']['dof_resid']}) | "
              f"H {u2['H']['delta_r2']:.5f} (F {u2['H']['F_partial']:.2f}) | "
              f"R² {uc_lfh['r2_full']:.5f}")
        print(f"    ΔR² unique [logfreq+H+rep] : logfreq {u3['logfreq']['delta_r2']:.5f} | "
              f"H {u3['H']['delta_r2']:.5f} | rep {u3['rep']['delta_r2']:.5f} "
              f"(F {u3['rep']['F_partial']:.2f}) | R² {uc_full['r2_full']:.5f}")
        print(f"    corr(rep, logfreq) {e['corr_rep_logfreq']:+.4f} | "
              f"corr(logfreq, H) {e['corr_logfreq_H']:+.4f} | "
              f"corr(rep, H) {e['corr_rep_H']:+.4f} | "
              f"r_pb(rep, impair) {e['r_pointbiserial_rep_impair']:+.4f}")
    res["posthoc_absorption"] = ph

    # ================================================================= P8 / P8b
    res["P8"] = None
    res["P8b"] = None
    s_star: dict[str, np.ndarray] = {}
    if args.with_instrument:
        print("\n================ P8 / P8b (instrument W_U, CPU, sans forward) ================")
        t_wu = time.time()
        from transformers import AutoModelForCausalLM  # noqa: E402
        mdl = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
        W_U = mdl.lm_head.weight.detach().float()
        d_model = W_U.shape[1]
        del mdl
        print(f"  W_U {tuple(W_U.shape)} chargée en {time.time() - t_wu:.1f}s "
              f"(aucun forward, aucun EngramEngine)")

        p8 = {}
        for t in texts_sel:
            tgt = torch.tensor([targets[t][i] for i in valid[t]])
            Wt = W_U[tgt]                       # [n_pos, d_model]
            betas_s, betas_H, r2s = [], [], []
            for j in range(n_fixe[t]):
                k = f"fixe-{t}-{j}"
                if k not in runs:
                    continue
                u = draw_eps(SEED_BASE["fixe"][t] + j, 1, d_model)[0]
                u = u / u.norm()
                proj = (Wt @ u).numpy().astype(np.float64)
                rn = np.asarray([runs[k]["metrics"]["rnorm"][i] for i in valid[t]])
                s = rn * proj
                Dj = np.asarray([runs[k]["metrics"]["D"][i] for i in valid[t]])
                m = ols(Dj, [s, np.asarray(Hv[t])])
                betas_s.append(m["beta"][1])
                betas_H.append(m["beta"][2])
                r2s.append(m["r2"])
            ms, sds = mean_sd(betas_s)
            tst = ms / (sds / math.sqrt(len(betas_s))) if sds > 0 else math.nan
            p8[t] = {"n_runs": len(betas_s), "beta_s_mean": ms, "beta_s_sd": sds,
                     "beta_s_t": tst, "beta_s_p": t_two_sided_p(tst, len(betas_s) - 1),
                     "beta_s_values": betas_s, "n_neg": sum(1 for x in betas_s if x < 0),
                     "beta_H_mean": mean_sd(betas_H)[0], "r2_mean": statistics.mean(r2s)}
            print(f"  P8 {t} : β_s {ms:+.6g} ± {sds:.3g} (N={len(betas_s)}, "
                  f"t={tst:.2f}, p={p8[t]['beta_s_p']:.3g}, "
                  f"{p8[t]['n_neg']}/{len(betas_s)} négatifs) | β_H {p8[t]['beta_H_mean']:+.5f} "
                  f"| R² moyen {p8[t]['r2_mean']:.4f}")
        p8["STATUT"] = ("DESCRIPTIF NON FIABLE — contrôle positif de V5 échoué "
                        "(pente lexicale du bras fixe non centrée sur 0 entre "
                        "tirages). Requalifié par arbitrage du PI : P8 n'est PLUS "
                        "une porte d'interprétation ; P8b est la seule porte.")
        print(f"  P8 STATUT : {p8['STATUT']}")
        res["P8"] = p8

        # ---- porte V6 : le replay doit reproduire l'ancre V1 de Q-01 ----
        v6 = None
        if args.rbar_file:
            rep_an = Path(args.rbar_file).parent / "analysis.json"
            if rep_an.exists():
                a = json.loads(rep_an.read_text(encoding="utf-8"))
                val = a.get("V1", {}).get("value")
                v6 = {"replay_readM_damage_A": val, "anchor": V6_ANCHOR, "tol": V6_TOL,
                      "ok": val is not None and abs(val - V6_ANCHOR) <= V6_TOL}
                print(f"  PORTE V6 : dommage moyen readM texte A (replay) = {val:+.5f} "
                      f"(ancre {V6_ANCHOR:+.4f} ± {V6_TOL}) → "
                      f"{'OK' if v6['ok'] else 'ÉCHEC — P8b tombe, P8 reste'}")
        res["V6"] = v6
        if args.rbar_file and Path(args.rbar_file).exists() and (v6 is None or v6["ok"]):
            rbar = load_rbar_unit(Path(args.rbar_file))
            norms = {t: float(v.norm()) for t, v in rbar.items()}
            print(f"  r̄ rechargé : normes {norms}")
            p8b = {"rbar_norms": norms, "rbar_file": str(args.rbar_file), "per_text": {}}
            for t in texts_sel:
                if t not in rbar:
                    continue
                tgt = torch.tensor([targets[t][i] for i in valid[t]])
                proj = (W_U[tgt] @ rbar[t]).numpy().astype(np.float64)
                s = np.asarray(rnorm_avg[t]) * proj
                y = np.asarray(impair[t])
                m = ols(y, [s, np.asarray(Hv[t])])
                r2_s = ols(y, [s])["r2"]
                r2_lf = ols(y, [np.asarray(logfreq[t])])["r2"]
                ci_s = spearman_ci(y.tolist(), s.tolist(), L[t], B)
                s_star[t] = s
                p8b["per_text"][t] = {
                    "beta_s_star": m["beta"][1], "beta_H": m["beta"][2], "r2_joint": m["r2"],
                    "r2_s_star_alone": r2_s, "r2_logfreq_alone": r2_lf,
                    "F_ceiling_r2_ratio": r2_lf / r2_s if r2_s > 0 else math.nan,
                    "rho_S_impair_s_star": ci_s["rho"],
                    "rho_ci95": [ci_s["ci_lo"], ci_s["ci_hi"]],
                    "corr_s_star_logfreq": pearson(s.tolist(), logfreq[t]),
                    "mean_proj_WU_rbar": float(np.mean(proj)),
                }
                e = p8b["per_text"][t]
                print(f"  P8b {t} : β_s* {e['beta_s_star']:+.6g} | β_H {e['beta_H']:+.5f} "
                      f"| R²(s*) {r2_s:.4f} vs R²(logfreq) {r2_lf:.4f} "
                      f"→ plafond de F {e['F_ceiling_r2_ratio']:.4f}")
                print(f"        ρ_S(impair, s*) {e['rho_S_impair_s_star']:+.4f} "
                      f"IC95 [{ci_s['ci_lo']:+.4f}, {ci_s['ci_hi']:+.4f}] | "
                      f"corr(s*, logfreq) {e['corr_s_star_logfreq']:+.4f}")
            res["P8b"] = p8b
        else:
            print("  P8b : pas de --rbar-file exploitable → non exécutée")

    # ================== DESCRIPTIF POST-HOC 2 — borne asymétrique de ΔNLL ==========
    # NON PRÉ-ENREGISTRÉ. Demandé par Neuro en phase d'interprétation : D_t ≥ −NLL_base_t,
    # donc le gain est borné par la surprise de base et le dommage ne l'est pas.
    # Ne modifie AUCUN verdict, AUCUNE porte, AUCUNE valeur pré-enregistrée.
    print("\n============ DESCRIPTIF POST-HOC 2 — borne asymétrique (non pré-enregistré) ============")
    pb: dict = {"STATUT": "POSTHOC_NON_PREENREGISTRE — descriptif demandé en phase "
                          "d'interprétation (Neuro). Ne modifie aucun verdict, aucune "
                          "porte, aucune valeur pré-enregistrée.",
                "rival": "D_t = NLL_perturbé − NLL_base ≥ −NLL_base_t",
                "per_text": {}}
    for t in texts_sel:
        y = np.asarray(impair[t])
        lf = np.asarray(logfreq[t])
        hh = np.asarray(Hv[t])
        nb = np.asarray(nllv[t])
        rp = rep_ind[t]
        e: dict = {}

        # (6) force directe du rival
        e["corr_logfreq_nllbase"] = pearson(lf.tolist(), nb.tolist())
        e["corr_rep_nllbase"] = pearson(rp.tolist(), nb.tolist())
        e["mean_nllbase_rep1"] = float(nb[rp == 1].mean())
        e["mean_nllbase_rep0"] = float(nb[rp == 0].mean())
        e["frac_positions_bound_active"] = float(np.mean(y <= -nb / 2.0))

        # (1) appariement par décile de NLL_base
        dq = np.quantile(nb, np.linspace(0.0, 1.0, 11))
        dq[0] -= 1e-9
        dq[-1] += 1e-9
        kb = np.clip(np.searchsorted(dq[1:-1], nb, side="right"), 0, 9)
        cells, wsum, num = [], 0.0, 0.0
        for k in range(10):
            m1 = (kb == k) & (rp == 1)
            m0 = (kb == k) & (rp == 0)
            c = {"decile": k + 1, "n_rep1": int(m1.sum()), "n_rep0": int(m0.sum()),
                 "nllbase_range": [float(dq[k]), float(dq[k + 1])],
                 "impair_rep1": float(y[m1].mean()) if m1.sum() else None,
                 "impair_rep0": float(y[m0].mean()) if m0.sum() else None}
            if m1.sum() and m0.sum():
                c["diff"] = c["impair_rep1"] - c["impair_rep0"]
                w = float(m1.sum() + m0.sum())
                num += w * c["diff"]
                wsum += w
            else:
                c["diff"] = None
            cells.append(c)
        e["decile_matching"] = {
            "cells": cells,
            "unmatched_diff": float(y[rp == 1].mean() - y[rp == 0].mean()),
            "matched_diff": num / wsum if wsum else math.nan,
            "n_cells_both": sum(1 for c in cells if c["diff"] is not None)}

        # (2) restriction NLL_base ∈ [1, 3] nats
        keep = (nb >= 1.0) & (nb <= 3.0)
        e["restriction_nllbase_1_3"] = {
            "n_kept": int(keep.sum()), "n_total": int(len(nb)),
            "rho_S_impair_logfreq": spearman(y[keep].tolist(), lf[keep].tolist()),
            "rho_S_full_series": p1[t]["rho"]}
        if t in s_star:
            ss = s_star[t]
            m_r = ols_full(y[keep], [ss[keep], hh[keep]], ["s_star", "H"])
            e["restriction_nllbase_1_3"]["beta_s_star"] = m_r["terms"]["s_star"]["beta"]
            e["restriction_nllbase_1_3"]["beta_s_star_se"] = m_r["terms"]["s_star"]["se"]
            e["restriction_nllbase_1_3"]["beta_s_star_p"] = m_r["terms"]["s_star"]["p"]
            e["restriction_nllbase_1_3"]["beta_s_star_full"] = \
                res["P8b"]["per_text"][t]["beta_s_star"]

        # (3) IC et permutation en blocs sur β_s* (machinerie A2)
        if t in s_star:
            ss = s_star[t]
            idx = circular_block_indices(len(y), L[t], B, BOOT_SEED)
            bs = np.empty(B)
            for b in range(B):
                i = idx[b]
                bs[b] = ols(y[i], [ss[i], hh[i]])["beta"][1]
            obs = ols(y, [ss, hh])["beta"][1]
            # permutation en blocs des étiquettes s* (H suit s*, comme en P1)
            g = torch.Generator(device="cpu")
            g.manual_seed(BOOT_SEED + 1)
            nbk = math.ceil(len(y) / L[t])
            blocks = [np.arange(k * L[t], min((k + 1) * L[t], len(y))) for k in range(nbk)]
            null = np.empty(B)
            for b in range(B):
                perm = torch.randperm(nbk, generator=g).tolist()
                order = np.concatenate([blocks[p] for p in perm])[:len(y)]
                null[b] = ols(y, [ss[order], hh[order]])["beta"][1]
            e["beta_s_star_inference"] = {
                "observed": obs, "L": L[t], "B": B, "seed": BOOT_SEED,
                "ci_lo": float(np.percentile(bs, 2.5)),
                "ci_hi": float(np.percentile(bs, 97.5)),
                "boot_sd": float(np.std(bs, ddof=1)),
                "perm_p_two_sided": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (B + 1)),
                "perm_null_mean": float(null.mean()), "perm_null_sd": float(null.std(ddof=1))}

        # (4) vérification Simpson : intra vs inter-strate du modérateur binaire
        r_marg = pearson(y.tolist(), lf.tolist())
        r_within_partial = partial_corr(y.tolist(), lf.tolist(), rp.tolist())
        rho_s1 = spearman(y[rp == 1].tolist(), lf[rp == 1].tolist())
        rho_s0 = spearman(y[rp == 0].tolist(), lf[rp == 0].tolist())
        n1, n0 = int((rp == 1).sum()), int((rp == 0).sum())
        z_pooled = (fisher_z(rho_s1) * (n1 - 3) + fisher_z(rho_s0) * (n0 - 3)) / (n1 + n0 - 6)
        e["simpson"] = {
            "corr_rep_logfreq": pearson(rp.tolist(), lf.tolist()),
            "rho_S_marginal": p1[t]["rho"],
            "rho_S_within_rep1": rho_s1, "n_rep1": n1,
            "rho_S_within_rep0": rho_s0, "n_rep0": n0,
            "rho_S_pooled_within_fisherz": math.tanh(z_pooled),
            "pearson_marginal": r_marg,
            "pearson_within_partial_given_rep": r_within_partial,
            "composition_share_pearson": (r_marg - r_within_partial) / r_marg
            if r_marg != 0 else math.nan}

        # (5) IC bootstrap du rapport β_logfreq / β_H (identité pré-enregistrée : 1)
        idx5 = circular_block_indices(len(y), L[t], B, BOOT_SEED)
        for lab, cols in (("additif", lambda i: [lf[i], hh[i]]),
                          ("N-P5_avec_interaction",
                           lambda i: [lf[i], hh[i], lf[i] * hh[i]])):
            rat = np.empty(B)
            for b in range(B):
                i = idx5[b]
                bb = ols(y[i], cols(i))["beta"]
                rat[b] = bb[1] / bb[2] if bb[2] != 0 else math.nan
            rr = rat[~np.isnan(rat)]
            full = np.arange(len(y))
            bb0 = ols(y, cols(full))["beta"]
            e[f"ratio_beta_lf_over_H[{lab}]"] = {
                "observed": bb0[1] / bb0[2], "ci_lo": float(np.percentile(rr, 2.5)),
                "ci_hi": float(np.percentile(rr, 97.5)),
                "frac_replicates_below_1": float(np.mean(rr < 1.0)), "B": int(len(rr))}
        e["ratio_beta_lf_over_H"] = e["ratio_beta_lf_over_H[additif]"]

        pb["per_text"][t] = e
        dm = e["decile_matching"]
        print(f"  [{t}] corr(logfreq, NLL_base) {e['corr_logfreq_nllbase']:+.4f} | "
              f"corr(rep, NLL_base) {e['corr_rep_nllbase']:+.4f} | "
              f"NLL_base moyen rep1 {e['mean_nllbase_rep1']:.4f} vs rep0 "
              f"{e['mean_nllbase_rep0']:.4f}")
        print(f"    (1) écart impair rep1−rep0 : NON apparié {dm['unmatched_diff']:+.4f} → "
              f"APPARIÉ par décile de NLL_base {dm['matched_diff']:+.4f} "
              f"({dm['n_cells_both']}/10 déciles exploitables)")
        for c in dm["cells"]:
            print(f"        décile {c['decile']:>2} NLL_base [{c['nllbase_range'][0]:.3f}, "
                  f"{c['nllbase_range'][1]:.3f}] n={c['n_rep1']:>3}/{c['n_rep0']:>3} "
                  f"impair {('%+.4f' % c['impair_rep1']) if c['impair_rep1'] is not None else '  n/a  '}"
                  f" / {('%+.4f' % c['impair_rep0']) if c['impair_rep0'] is not None else '  n/a  '}"
                  f" → {('%+.4f' % c['diff']) if c['diff'] is not None else 'n/a'}")
        rst = e["restriction_nllbase_1_3"]
        print(f"    (2) NLL_base ∈ [1,3] : N {rst['n_kept']}/{rst['n_total']} | "
              f"ρ_S {rst['rho_S_impair_logfreq']:+.4f} (série pleine "
              f"{rst['rho_S_full_series']:+.4f})"
              + (f" | β_s* {rst['beta_s_star']:+.6g} (SE {rst['beta_s_star_se']:.6g}, "
                 f"p {rst['beta_s_star_p']:.3g} ; série pleine "
                 f"{rst['beta_s_star_full']:+.6g})" if "beta_s_star" in rst else ""))
        if "beta_s_star_inference" in e:
            bi = e["beta_s_star_inference"]
            print(f"    (3) β_s* {bi['observed']:+.6g} IC95 [{bi['ci_lo']:+.6g}, "
                  f"{bi['ci_hi']:+.6g}] (L={bi['L']}, B={bi['B']}) | permutation en blocs "
                  f"p={bi['perm_p_two_sided']:.4g} (null {bi['perm_null_mean']:+.6g} ± "
                  f"{bi['perm_null_sd']:.6g})")
        sp_ = e["simpson"]
        print(f"    (4) corr(rep, logfreq) {sp_['corr_rep_logfreq']:+.4f} | ρ_S marginal "
              f"{sp_['rho_S_marginal']:+.4f} | intra rep=1 {sp_['rho_S_within_rep1']:+.4f} "
              f"(n={sp_['n_rep1']}) | intra rep=0 {sp_['rho_S_within_rep0']:+.4f} "
              f"(n={sp_['n_rep0']}) | intra poolé (Fisher-z) "
              f"{sp_['rho_S_pooled_within_fisherz']:+.4f}")
        print(f"        Pearson marginal {sp_['pearson_marginal']:+.4f} → partiel | rep "
              f"{sp_['pearson_within_partial_given_rep']:+.4f} → part de composition "
              f"{sp_['composition_share_pearson']:+.4f}")
        for lab in ("additif", "N-P5_avec_interaction"):
            rt = e[f"ratio_beta_lf_over_H[{lab}]"]
            print(f"    (5) rapport β_lf/β_H [{lab}] {rt['observed']:+.4f} IC95 "
                  f"[{rt['ci_lo']:+.4f}, {rt['ci_hi']:+.4f}] | fraction de réplicats "
                  f"< 1 : {rt['frac_replicates_below_1']:.4f}")
    res["posthoc_bound_test"] = pb

    # ------------------------------------------------------------------ sorties
    res["duration_s"] = time.time() - t_start
    with open(out_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)

    rows = [["bloc", "texte", "metrique", "valeur", "ic_lo", "ic_hi"]]
    for t in texts_sel:
        rows.append(["V0", t, "sigma_pos", f"{v0[t]['sigma_pos']:.6f}", "", ""])
        rows.append(["P1", t, "rho_spearman", f"{p1[t]['rho']:.6f}",
                     f"{p1[t]['ci_lo']:.6f}", f"{p1[t]['ci_hi']:.6f}"])
        rows.append(["N-P1", t, "rho_conf", f"{npd[t]['N_P1_rho_conf']:.6f}", "", ""])
        rows.append(["N-P1b", t, "impair_conf", f"{npd[t]['N_P1b_impair_conf']:.6f}", "", ""])
        rows.append(["N-P1b", t, "impair_unc", f"{npd[t]['N_P1b_impair_unc']:.6f}", "", ""])
        rows.append(["N-P4", t, "corr_pair_logfreq",
                     f"{npd[t]['N_P4_corr_pair_logfreq']:.6f}", "", ""])
        rows.append(["N-P4", t, "corr_pair_H", f"{npd[t]['N_P4_corr_pair_H']:.6f}", "", ""])
        rows.append(["N-P5", t, "beta_logfreq", f"{np5[t]['beta_logfreq']:.6f}", "", ""])
        rows.append(["N-P5", t, "beta_H", f"{np5[t]['beta_H']:.6f}", "", ""])
        rows.append(["N-P5", t, "beta_interaction", f"{np5[t]['beta_interaction']:.6f}", "", ""])
        rows.append(["V5", t, "rho_null", f"{v5['pipeline'][t]['rho']:.6f}",
                     f"{v5['pipeline'][t]['ci_lo']:.6f}", f"{v5['pipeline'][t]['ci_hi']:.6f}"])
    for rname in est_out:
        b = est_out[rname].get("bootstrap", {})
        ci = b.get("F_ols_ci95", {}) if isinstance(b, dict) else {}
        ok3 = est_out[rname]["P3"]["ok"]
        fo = est_out[rname]["OLS"]["F"]
        fp = est_out[rname]["poststrat"]["F"]
        rows.append(["P2", "A-B", f"F_ols[{rname}]",
                     f"{fo:.6f}" if ok3 else "NON_RAPPORTEE_P3",
                     f"{ci.get('ci_lo', '')}", f"{ci.get('ci_hi', '')}"])
        rows.append(["P2", "A-B", f"F_poststrat[{rname}]",
                     f"{fp:.6f}" if ok3 else "NON_RAPPORTEE_P3", "", ""])
        rows.append(["P3", "A-B", f"delta_z[{rname}]",
                     f"{est_out[rname]['P3']['delta_z']:.6f}",
                     "", "porte_franchie" if ok3 else "porte_echouee"])
    rows.append(["P2", "A-B", "delta_raw", f"{delta_raw:.6f}", "", ""])
    rows.append(["V5", "-", "statut_global", v5["status"], "", ""])
    rows.append(["V5", "-", "placebo_antithetique", "OK" if v5["placebo_antithetique_ok"]
                 else "ECHEC", "", ""])
    rows.append(["V5", "-", "controle_positif_fixe", "OK" if v5["controle_positif_fixe_ok"]
                 else "ECHEC", "", ""])
    rows.append(["V3", "-", "couverture_positions", f"{cov_tokens:.6f}", "",
                 "OK" if v3["ok"] else "NON_FRANCHIE_analyse_continue_PI"])
    for t in texts_sel:
        ev = valign["per_text"][t]["corr_nll_logfreq_by_shift"]
        rows.append(["V2.align", t, "corr_nll_logfreq_shift0", f"{ev[0]:.6f}",
                     f"{ev[-1]:.6f}", f"{ev[+1]:.6f}"])
    if res["P8"]:
        for t in [x for x in res["P8"] if x != "STATUT"]:
            rows.append(["P8", t, "beta_s_mean", f"{res['P8'][t]['beta_s_mean']:.8f}", "",
                         "DESCRIPTIF_NON_FIABLE_V5"])
    if res["P8b"]:
        for t in res["P8b"]["per_text"]:
            e = res["P8b"]["per_text"][t]
            rows.append(["P8b", t, "beta_s_star", f"{e['beta_s_star']:.8f}", "", ""])
            rows.append(["P8b", t, "F_ceiling_r2_ratio",
                         f"{e['F_ceiling_r2_ratio']:.6f}", "", ""])
    # --- descriptifs POST-HOC (non pré-enregistrés) ---
    for t in units:
        e = ph["per_unit"][t]
        m = e["model_logfreq_H_rep"]
        for nm in ("logfreq", "H", "rep"):
            rows.append(["POSTHOC_NON_PREENREGISTRE", t, f"beta_{nm}[lf+H+rep]",
                         f"{m[nm]['beta']:.6f}", f"se={m[nm]['se']:.6f}",
                         f"p={m[nm]['p']:.4g}"])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "beta_logfreq_avant_rep",
                     f"{e['beta_logfreq_before_rep']:.6f}", "", ""])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "beta_logfreq_apres_rep",
                     f"{e['beta_logfreq_after_rep']:.6f}", "", ""])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "fraction_beta_lf_absorbee",
                     f"{e['absorbed_fraction_of_beta_logfreq']:.6f}", "", ""])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "corr_rep_logfreq",
                     f"{e['corr_rep_logfreq']:.6f}", "", ""])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "corr_logfreq_H",
                     f"{e['corr_logfreq_H']:.6f}", "", ""])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "r_pb_rep_impair",
                     f"{e['r_pointbiserial_rep_impair']:.6f}", "", ""])
        u2 = e["unique_contrib_logfreq_H"]["per_term"]
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "delta_r2_logfreq[lf+H]",
                     f"{u2['logfreq']['delta_r2']:.6f}",
                     f"F={u2['logfreq']['F_partial']:.4f}", ""])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "delta_r2_H[lf+H]",
                     f"{u2['H']['delta_r2']:.6f}", f"F={u2['H']['F_partial']:.4f}", ""])
        u3 = e["unique_contrib_logfreq_H_rep"]["per_term"]
        for nm in ("logfreq", "H", "rep"):
            rows.append(["POSTHOC_NON_PREENREGISTRE", t, f"delta_r2_{nm}[lf+H+rep]",
                         f"{u3[nm]['delta_r2']:.6f}", f"F={u3[nm]['F_partial']:.4f}", ""])
    for t in texts_sel:
        e = pb["per_text"][t]
        dm = e["decile_matching"]
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "impair_rep1_moins_rep0_non_apparie",
                     f"{dm['unmatched_diff']:.6f}", "", ""])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "impair_rep1_moins_rep0_APPARIE_decile",
                     f"{dm['matched_diff']:.6f}", f"cellules={dm['n_cells_both']}/10", ""])
        rst = e["restriction_nllbase_1_3"]
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "rho_S_restr_nllbase_1_3",
                     f"{rst['rho_S_impair_logfreq']:.6f}", f"N={rst['n_kept']}",
                     f"serie_pleine={rst['rho_S_full_series']:.6f}"])
        if "beta_s_star" in rst:
            rows.append(["POSTHOC_NON_PREENREGISTRE", t, "beta_s_star_restr_nllbase_1_3",
                         f"{rst['beta_s_star']:.8f}", f"se={rst['beta_s_star_se']:.8f}",
                         f"serie_pleine={rst['beta_s_star_full']:.8f}"])
        if "beta_s_star_inference" in e:
            bi = e["beta_s_star_inference"]
            rows.append(["POSTHOC_NON_PREENREGISTRE", t, "beta_s_star_IC95",
                         f"{bi['observed']:.8f}", f"{bi['ci_lo']:.8f}", f"{bi['ci_hi']:.8f}"])
            rows.append(["POSTHOC_NON_PREENREGISTRE", t, "beta_s_star_perm_p",
                         f"{bi['perm_p_two_sided']:.6f}", "", ""])
        sp_ = e["simpson"]
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "rho_S_intra_rep1",
                     f"{sp_['rho_S_within_rep1']:.6f}", f"n={sp_['n_rep1']}", ""])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "rho_S_intra_rep0",
                     f"{sp_['rho_S_within_rep0']:.6f}", f"n={sp_['n_rep0']}", ""])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "rho_S_intra_poole_fisherz",
                     f"{sp_['rho_S_pooled_within_fisherz']:.6f}", "", ""])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "part_composition_pearson",
                     f"{sp_['composition_share_pearson']:.6f}", "", ""])
        for lab in ("additif", "N-P5_avec_interaction"):
            rt = e[f"ratio_beta_lf_over_H[{lab}]"]
            rows.append(["POSTHOC_NON_PREENREGISTRE", t, f"ratio_beta_lf_sur_H[{lab}]",
                         f"{rt['observed']:.6f}", f"{rt['ci_lo']:.6f}", f"{rt['ci_hi']:.6f}"])
            rows.append(["POSTHOC_NON_PREENREGISTRE", t,
                         f"ratio_frac_replicats_inf_1[{lab}]",
                         f"{rt['frac_replicates_below_1']:.6f}", "", ""])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "corr_logfreq_nllbase",
                     f"{e['corr_logfreq_nllbase']:.6f}", "", ""])
        rows.append(["POSTHOC_NON_PREENREGISTRE", t, "corr_rep_nllbase",
                     f"{e['corr_rep_nllbase']:.6f}", "", ""])

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"\n  durée totale : {res['duration_s']:.0f}s")
    print(f"  sorties : {out_dir}")


if __name__ == "__main__":
    main()
