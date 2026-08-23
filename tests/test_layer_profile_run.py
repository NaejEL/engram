# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests CPU de l'ORCHESTRATION d'I2 (`eval/layer_profile.py`, section §9).

Le protocole `experiments/EXP-2026-08-22-layer-profile.md` fixe `B = 10 000`,
trois modèles et quatre variantes : l'exécution demande des formes vectorisées
des mêmes statistiques. **Chaque forme vectorisée est ici confrontée à son
implémentation de référence** — celle que le banc et les tests §10 vérifient
déjà. Un écart est un défaut de l'orchestration, jamais une nouvelle convention.

Aucun GPU, aucun modèle : les états sont synthétiques.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import layer_profile as lp  # noqa: E402


@pytest.fixture(scope="module")
def petit():
    """Un corpus (a) réduit à N = 20 unités et des états synthétiques.

    N = 20 garde des composantes d'owner et d'entité non triviales tout en
    rendant les tests instantanés ; les fonctions testées ne dépendent pas de N.
    """
    n = 20
    c = lp.corpus_a(n)
    g = np.random.default_rng(7)
    base = g.normal(size=(n, 16))
    etats = {e: (np.repeat(base, 3, axis=0)
                 + (0.6 + 0.3 * e) * g.normal(size=(3 * n, 16))
                 ).astype(np.float32) for e in range(4)}
    return c, etats, lp.cos_par_couche(etats)


def test_les_index_vectorises_reproduisent_paires_intra_inter(petit):
    c, _, _ = petit
    n = len(c["slots"])
    intra, inter = lp.paires_intra_inter(n)
    u, ra, rb, _ = lp.index_intra(n)
    _, _, _, _, ea, eb = lp.index_inter(n)
    assert list(zip(ra.tolist(), rb.tolist())) == intra
    assert list(zip(ea.tolist(), eb.tolist())) == inter
    assert (u == np.repeat(np.arange(n), 3)).all()


def test_les_classes_vectorisees_reproduisent_classe_identite(petit):
    c, _, _ = petit
    ui, uj, _, _, _, _ = lp.index_inter(len(c["slots"]))
    cls = lp.classes_des_paires_inter(c["slots"], ui, uj)
    for k in range(0, ui.size, 37):
        assert cls[k] == lp.classe_identite(c["slots"][ui[k]], c["slots"][uj[k]])


def test_r1_vectorise_egale_r1_36_de_reference(petit):
    c, _, cos3 = petit
    req, cand = lp.matrice_candidats_R1(c["slots"], lp.P_OWN)
    assert cand.shape[1] == 37
    S = lp.r1_vectorise(cos3, req, cand)
    for e in range(cos3.shape[0]):
        ref = lp.r1_36(cos3[e].astype(np.float64), c["slots"], lp.P_OWN)
        assert S[e] == pytest.approx(ref["succes"])
        assert float(S[e].mean()) == pytest.approx(ref["R1"])


def test_auc_du_profil_vectorise_egale_auc_par_couche(petit):
    c, etats, cos3 = petit
    prof = lp.profil_vectorise(cos3, c["slots"], etats, avec_H=False)
    intra, inter = lp.paires_intra_inter(len(c["slots"]))
    for e in range(cos3.shape[0]):
        S = cos3[e]
        ref = lp.auc_par_couche([S[a, b] for a, b in intra],
                                [S[a, b] for a, b in inter])
        assert prof["auc"][e] == pytest.approx(ref)
    assert prof["n_intra"] == len(intra) and prof["n_inter"] == len(inter)


def test_auc_ponderee_egale_la_statistique_de_reference_sur_un_reechantillon(petit):
    """`auc_ponderee` doit rendre EXACTEMENT ce que rend `auc_stat_fn` (la
    statistique du bootstrap par composante de slot déjà en place)."""
    c, _, cos3 = petit
    n = len(c["slots"])
    prep = lp.prep_bootstrap_auc(cos3, c["slots"])
    cl = lp.clusters_de_strate(c["slots"], lp.P_OWN)
    membres = [np.array(cl["membres"][k]) for k in sorted(cl["membres"])]
    u, ra, rb, _ = lp.index_intra(n)
    ui, uj, _, _, ea, eb = lp.index_inter(n)
    rng = np.random.default_rng(3)
    for _ in range(4):
        tir = rng.integers(0, len(membres), size=len(membres))
        m = np.zeros(n)
        for k in tir:
            m[list(membres[int(k)])] += 1.0
        a_vec = lp.auc_ponderee(prep, m)
        for e in range(cos3.shape[0]):
            A = cos3[e][ra, rb].reshape(n, 3)
            B = np.zeros((n, n, 9))
            pu, pv = ui[::9], uj[::9]
            B[pu, pv] = cos3[e][ea, eb].reshape(-1, 9)
            B[pv, pu] = B[pu, pv]
            ref = lp.auc_stat_fn(A, B, {k: tuple(v) for k, v
                                        in enumerate(membres)})(tir)
            assert a_vec[e] == pytest.approx(ref, abs=1e-12)


def test_auc_permutee_egale_le_calcul_direct(petit):
    """La permutation des étiquettes d'unité à couche fixée : la forme par rangs
    doit rendre l'AUC exacte, égalités à ½ crédit comprises."""
    c, _, cos3 = petit
    n = len(c["slots"])
    prep = lp.prep_permutation_auc(cos3)
    rng = np.random.default_rng(11)
    for _ in range(3):
        p = rng.permutation(3 * n)
        a_vec = lp.auc_permutee(prep, p)
        g = p.reshape(n, 3)
        intra = [(g[i, s], g[i, t]) for i in range(n)
                 for s in range(3) for t in range(s + 1, 3)]
        ens = set(map(frozenset, intra))
        inter = [(i, j) for i in range(3 * n) for j in range(i + 1, 3 * n)
                 if frozenset((i, j)) not in ens]
        assert len(intra) == 3 * n and len(inter) == 3 * n * (3 * n - 1) // 2 - 3 * n
        for e in range(cos3.shape[0]):
            S = cos3[e]
            ref = lp.auc_par_couche([S[a, b] for a, b in intra],
                                    [S[a, b] for a, b in inter])
            assert a_vec[e] == pytest.approx(ref, abs=1e-12)


def test_r1_full_l2_egale_la_reference(petit):
    _, etats, cos3 = petit
    n = cos3.shape[1] // 3
    lab = np.repeat(np.arange(n), 3)
    for e in etats:
        assert lp.recall_at_1_l2(etats[e], lab) == pytest.approx(
            lp.recall_at_1(etats[e], lab, "l2"))
        assert lp.recall_at_1_cos(cos3[e], lab) == pytest.approx(
            lp.recall_at_1(etats[e], lab, "cos"))


def test_le_plancher_retenu_est_le_plus_haut_a_chaque_couche():
    """§5 : le **plancher le plus haut** entre dans `ΔR1`. La nulle mélangée
    déclarée permissive en est exclue."""
    L, nq = 3, 12
    prof_a = {"succes_R1_P-own": np.full((L + 1, nq), 0.9)}
    prof_a["succes_R1_P-own"][0] = 0.1
    cadre = {"succes_R1_P-own": np.full((L + 1, nq), 0.2)}
    cadre["succes_R1_P-own"][2] = 0.8
    mel = {"succes_R1_P-own": np.full((L + 1, nq), 0.95)}
    lex = {"succes": [0.3] * nq}
    succ, noms, moy = lp.succes_planchers_R1(prof_a, cadre, mel, lex, L, False)
    assert noms[2] == "4_ordre_nulle_melangee"
    assert succ[2] == pytest.approx(0.95)
    succ2, noms2, _ = lp.succes_planchers_R1(prof_a, cadre, mel, lex, L, True)
    assert noms2[2] == "2_capture_nulle_de_cadre" and succ2[2] == pytest.approx(0.8)
    assert noms2[1] == "1_materiel_lexical"
    assert moy["5_statistique"][0] == pytest.approx(lp.HASARD_R1)


def test_le_couloir_bootstrap_reselectionne_l_argmax():
    """M-15 : l'argmax de `ΔR1` est RE-SÉLECTIONNÉ dans chaque rééchantillon ;
    l'estimateur débiaisé diffère du max brut."""
    g = np.random.default_rng(4)
    L, nq = 5, 60
    reel = (g.random((L + 1, nq)) < 0.3).astype(float)
    plancher = np.full((L + 1, nq), 1.0 / 37)
    blocs = [np.arange(k * 10, (k + 1) * 10) for k in range(6)]
    d, r, a = lp.bootstrap_couloir(reel, plancher, blocs, b=200, seed=0)
    assert d.shape == (200,) and set(a.tolist()) <= set(range(1, L + 1))
    assert len(set(a.tolist())) > 1                     # argmax re-sélectionné
    jd, jr = lp.jackknife_couloir(reel, plancher, blocs)
    assert jd.shape == (6,) and jr.shape == (6,)
    obs, ell, r1 = lp._stat_couloir(reel, plancher, np.arange(nq))
    ic = lp.ic_du_max(d, float(obs), jd)
    assert ic["theta_debiaise"] != pytest.approx(float(obs))
    assert ic["ic_bas"] <= ic["ic_haut"]
    assert 1 <= ell <= L and 0.0 <= r1 <= 1.0


def test_permutation_du_couloir_conserve_l_appariement():
    reel = np.tile(np.arange(12) / 12.0, (4, 1))
    plancher = np.zeros((4, 12))
    ech = lp.permutation_couloir(reel, plancher, b=50, seed=0)
    assert ech.shape == (50,)
    assert np.allclose(ech, reel[1:].mean(axis=1).max())


def test_les_clusters_de_requetes_couvrent_toutes_les_requetes(petit):
    c, _, _ = petit
    blocs, info = lp.clusters_des_requetes(c["slots"], lp.P_OWN)
    tout = np.sort(np.concatenate(blocs))
    assert (tout == np.arange(3 * len(c["slots"]))).all()
    assert len(blocs) == info["K_composantes_totales"]


def test_courbes_H_sont_appariees_entre_couches(petit):
    _, etats, _ = petit
    H, l1 = lp.courbes_H(etats, 3, n_a=20, b=5, seed=0)
    assert H.shape == (5, 4) and l1.shape == (5, 4)
    assert np.isfinite(H).all() and ((0 < l1) & (l1 <= 1)).all()
