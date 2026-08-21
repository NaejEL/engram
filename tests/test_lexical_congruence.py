# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests CPU des primitives de eval/lexical_congruence.py (protocole Q-01b,
EXP-2026-08-21-congruence-lexicale) — aucun réseau, aucun modèle HF, W_U factice.

Les dix tests exigés par le §10 « Livrables » du protocole :
  (1) identité impair + pair = D(+r̄) ;
  (2) post-stratification exacte : F = 1.00 ± 1e-6 quand A et B ne diffèrent QUE
      par la distribution du régresseur, F = 0 quand elles sont identiques ;
  (3) déterminisme du bootstrap (graine 777) ET recalcul effectif des bornes de
      quintiles dans le réplicat ;
  (4) règle de bloc : ρ₁ = 0.06 → L = 25 ; ρ₁ = 0.9 → L croît ; plafond N/10 ;
  (5) comptage unigramme reproductible ET exclusion effective de A/B ;
  (6) détection de chevauchement V3b ;
  (7) assert d'alignement : un décalage volontaire d'un cran fait échouer V2 ;
  (8) reproduction bit-exacte des directions du bras *fixe* par `draw_eps` ;
  (9) échec attendu de V2 sur un JSON tronqué ;
 (10) `rbar_unit` sérialisé et rechargeable, norme 1 à 1e-6.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from lexical_congruence import (  # noqa: E402
    alignment_evidence, block_length, bpe_counts, check_alignment, check_run_lengths,
    circular_block_indices, load_rbar_unit, ngram_overlap, ols_F, poststrat_F,
    quantile_edges, save_rbar_unit,
)
from perturb_position import SEED_BASE, draw_eps  # noqa: E402


# ------------------------------------------------------------------------ (1)
def test_identity_impair_plus_pair_equals_dplus():
    rng = np.random.default_rng(0)
    dp = rng.normal(size=200)
    dm = rng.normal(size=200)
    impair = (dp - dm) / 2.0
    pair = (dp + dm) / 2.0
    assert np.max(np.abs((pair + impair) - dp)) <= 1e-12
    assert np.max(np.abs((pair - impair) - dm)) <= 1e-12


# ------------------------------------------------------------------------ (2)
def test_poststrat_exact_F_one_and_zero():
    # Cinq valeurs discrètes de régresseur ; y = f(x) DÉTERMINISTE et identique
    # dans les deux textes ⇒ les moyennes intra-cellule coïncident ⇒ Δ_adj = 0.
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = np.array([0.5, 0.2, -0.1, -0.4, -0.7])
    na = [40, 30, 20, 10, 10]
    nb = [10, 10, 20, 30, 40]
    xa = np.concatenate([np.full(n, xs[k]) for k, n in enumerate(na)])
    ya = np.concatenate([np.full(n, f[k]) for k, n in enumerate(na)])
    xb = np.concatenate([np.full(n, xs[k]) for k, n in enumerate(nb)])
    yb = np.concatenate([np.full(n, f[k]) for k, n in enumerate(nb)])
    edges = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    out = poststrat_F(ya, xa, yb, xb, edges)
    assert abs(out["delta_adj"]) <= 1e-12
    assert abs(out["F"] - 1.0) <= 1e-6

    # Distributions du régresseur IDENTIQUES, y décalé d'une constante ⇒ F = 0.
    yb2 = np.concatenate([np.full(n, f[k] - 0.3) for k, n in enumerate(na)])
    out0 = poststrat_F(ya, xa, yb2, xa.copy(), edges)
    assert abs(out0["delta_raw"] - 0.3) <= 1e-12
    assert abs(out0["F"]) <= 1e-6

    # L'estimateur OLS de F est cohérent sur le même cas trivial (pente exacte).
    fo = ols_F(np.concatenate([ya, yb]), np.concatenate([xa, xb]), xa, xb,
               float(ya.mean() - yb.mean()))
    assert fo["F"] > 0.9


# ------------------------------------------------------------------------ (3)
def test_bootstrap_determinism_and_replicate_quantiles():
    a = circular_block_indices(343, 25, 50, 777)
    b = circular_block_indices(343, 25, 50, 777)
    assert np.array_equal(a, b)                      # déterminisme graine 777
    assert not np.array_equal(a, circular_block_indices(343, 25, 50, 778))
    assert a.shape == (50, 343)
    assert a.min() >= 0 and a.max() < 343

    # Les bornes de quintiles DOIVENT être recalculées dans chaque réplicat :
    # deux rééchantillonnages différents donnent des bornes différentes.
    rng = np.random.default_rng(1)
    x = rng.normal(size=343)
    e0 = quantile_edges(x, 5)
    e1 = quantile_edges(x[a[0]], 5)
    e2 = quantile_edges(x[a[1]], 5)
    assert e1 != e2
    assert e1 != e0
    assert len(e0) == 6 and e0 == sorted(e0)


# ------------------------------------------------------------------------ (4)
def test_block_length_rule():
    # ρ₁ ≈ 0.06 (valeur mesurée en Q-01) → L = 25
    assert block_length(0.06, 343) == 25
    assert block_length(0.06, 440) == 25
    # dépendance forte → L croît (la forme du brouillon faisait l'inverse)
    assert block_length(0.9, 10_000) > 25
    assert block_length(0.95, 10_000) > block_length(0.9, 10_000)
    # plafond N_full/10 respecté
    assert block_length(0.95, 200) <= 20
    assert block_length(0.9, 343) <= 34
    # ρ₁ ≤ 0 : pas de dépendance mesurable → plancher
    assert block_length(-0.1, 343) == 25
    assert block_length(0.0, 343) == 25


# ------------------------------------------------------------------------ (5)
def test_unigram_counts_reproducible_and_excludes_texts():
    enc = lambda s: [abs(hash(w)) % 1000 for w in s.split()]  # noqa: E731
    corpus = ["alpha beta gamma\n" * 50, "delta beta\n" * 20]
    text_a = "zulu yankee"
    c1 = bpe_counts(corpus, enc)
    c2 = bpe_counts(corpus, enc)
    assert c1 == c2                                   # reproductible
    ids_a = set(enc(text_a))
    assert not (ids_a & set(c1)), "A doit être EXCLU du comptage (§5.8)"
    # l'inclure changerait le comptage : le contrôle de circularité est effectif
    c3 = bpe_counts(corpus + [text_a], enc)
    assert c3 != c1 and (ids_a & set(c3))


# ------------------------------------------------------------------------ (6)
def test_v3b_overlap_detection():
    rng = np.random.default_rng(2)
    big = rng.integers(0, 500, size=5000).tolist()
    extract = big[1000:3000]                          # extrait littéral
    assert ngram_overlap(extract, big, 10) > 0.99
    unrelated = rng.integers(0, 500, size=2000).tolist()
    assert ngram_overlap(unrelated, big, 10) < 0.01


# ------------------------------------------------------------------------ (7)
def test_alignment_assert_catches_one_step_shift():
    ids = [10, 11, 12, 13, 14]
    assert check_alignment(list(ids), ids)            # x_t = logfreq(ids[t])
    shifted = [ids[0]] + ids[:-1]                     # cible = ids[t−1] : INTERDIT
    assert not check_alignment(shifted, ids)
    assert not check_alignment(ids[:-1], ids)

    # La clause structurelle ci-dessus est tautologique sur les vraies données
    # (targets est construit depuis ids). La clause qui MORD est empirique :
    # NLL_base_t doit suivre la log-fréquence de ids[t], pas celle des voisins.
    rng = np.random.default_rng(7)
    n = 400
    lf = rng.normal(size=n + 2)
    nll = -0.6 * lf[1:n + 1] + 0.4 * rng.normal(size=n)   # aligné sur le décalage 0
    aligned = {-1: lf[:n].tolist(), 0: lf[1:n + 1].tolist(), 1: lf[2:].tolist()}
    ev = alignment_evidence(nll.tolist(), aligned)
    assert ev["ok"] and ev["corr_nll_logfreq_by_shift"][0] < -0.5
    # décalage réel d'un cran : la porte doit ÉCHOUER
    off = {-1: aligned[0], 0: aligned[-1], 1: aligned[0]}
    assert not alignment_evidence(nll.tolist(), off)["ok"]
    # bruit pur : la porte doit ÉCHOUER aussi
    noise = {s: rng.normal(size=n).tolist() for s in (-1, 0, 1)}
    assert not alignment_evidence(nll.tolist(), noise)["ok"]


# ------------------------------------------------------------------------ (8)
def test_fixe_directions_bit_exact_from_draw_eps():
    d = 768
    for text in ("A", "B"):
        for j in (0, 7, 19):
            u1 = draw_eps(SEED_BASE["fixe"][text] + j, 1, d)[0]
            u2 = draw_eps(SEED_BASE["fixe"][text] + j, 1, d)[0]
            assert torch.equal(u1, u2)                # bit à bit
            assert u1.dtype == torch.float32 and u1.shape == (d,)
            n = (u1 / u1.norm()).norm()
            assert abs(float(n) - 1.0) <= 1e-6
    # espaces de seeds disjoints entre textes et entre tirages
    assert not torch.equal(draw_eps(SEED_BASE["fixe"]["A"], 1, d),
                           draw_eps(SEED_BASE["fixe"]["B"], 1, d))
    assert not torch.equal(draw_eps(SEED_BASE["fixe"]["A"], 1, d),
                           draw_eps(SEED_BASE["fixe"]["A"] + 1, 1, d))


# ------------------------------------------------------------------------ (9)
def test_v2_fails_on_truncated_json():
    ids = {"A": list(range(344))}
    runs = {"rbarplus-A-0": {"text": "A", "metrics": {"D": [None] + [0.0] * 343}}}
    assert check_run_lengths(runs, ids)
    runs["rbarplus-A-0"]["metrics"]["D"] = [None] + [0.0] * 300   # JSON tronqué
    assert not check_run_lengths(runs, ids)


# ----------------------------------------------------------------------- (10)
def test_rbar_unit_roundtrip(tmp_path):
    g = torch.Generator(device="cpu")
    g.manual_seed(3)
    rb = {}
    for t in ("A", "B"):
        v = torch.randn(768, generator=g, dtype=torch.float32)
        rb[t] = v / v.norm()
    p = tmp_path / "rbar_unit.json"
    save_rbar_unit(p, rb, 768, cos_ab=float(rb["A"] @ rb["B"]))
    back = load_rbar_unit(p)
    for t in ("A", "B"):
        assert abs(float(back[t].norm()) - 1.0) <= 1e-6
        assert float((back[t] - rb[t]).abs().max()) <= 1e-6
