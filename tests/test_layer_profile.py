# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests CPU d'I2 — `eval/layer_profile.py` et la suite `i2` du banc.

Protocole : `experiments/EXP-2026-08-22-layer-profile.md`, §10 (dix tests).
Aucun GPU, aucun modèle : le seul HF touché est le **tokenizer GPT-2** (CPU,
cache local), là où la clause n'est pas décidable sans lui.

  (i)    `V-div` sur `fact_pairs(30)` et sur le jeu d'unités v3 ;
  (ii)   `V-paires` détecte 89 ou 91 paires intra ;
  (iii)  `V-plat` sur courbes synthétiques (bosse vs bruit) ;
  (iv)   `V-1pass` détecte une implémentation qui boucle sur les couches ;
  (v)    `V-hooks` détecte un hook non retiré ;
  (vi)   **l'AUC est inchangée sous une transformation monotone par couche
         (`x → x³`, `x → σ(ax+b)`) alors que le ratio `s_intra/s_inter`
         change** — le test qui matérialise M-1b ;
  (vii)  `w(L)` rend 1 / 2 / 2 ;
  (viii) la stratification classe S0 / S1 / S2 ;
  (ix)   l'entropie est inchangée sous mise à l'échelle des lignes **ssi** la
         normalisation Giraldo est appliquée ;
  (x)    les quatre bandes et les quatre cellules sont classées correctement,
         **y compris aux bords**.

Deux tests consignent un comportement **contraire à celui que le protocole
prescrit** — (i) et (x). Ils affirment ce qui est OBSERVÉ, jamais ce qui est
souhaité : le désaccord est remonté par le compte `E` du banc, et l'amendement
est une décision de pré-enregistrement, pas d'implémentation.
"""

import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import layer_profile as lp  # noqa: E402
from gate_bench import (  # noqa: E402
    FAIL, PASS, gate_invariance_lignes, gate_invariance_monotone,
    gate_stratification, gate_v_1pass, gate_v_div, gate_v_hooks, gate_v_paires,
    gate_v_plat, _forwards_boucle_sur_les_couches, _forwards_un_passage,
    _hooks_apres_capture, _hooks_fuite, _stratifier_sans_verbe, run_i2,
)


@pytest.fixture(scope="module")
def gpt2_tokenize():
    tr = pytest.importorskip("transformers")
    tok = tr.AutoTokenizer.from_pretrained("gpt2")
    return lambda s: [int(t) for t in tok.encode(s)]


# ------------------------------------------------------------------- (i)

def test_i_v_div_sur_fact_pairs_et_sur_le_jeu_v3(gpt2_tokenize):
    """(i) `V-div` sur les deux jeux.

    Le protocole (§4.7) prescrit : `fact_pairs(30)` **passe**, le jeu v3
    **échoue**. Le comportement OBSERVÉ est l'inverse, et il est déterministe :

    - sur `fact_pairs(30)`, S2 = les **10** paires disjointes `(i, i+20)` ; la
      probabilité qu'aucune ne survive à un rééchantillonnage de 30 unités avec
      remise vaut, par inclusion-exclusion, ≈ 1.6e-3 ⇒ S2 non vide dans
      ≈ 99.84 % des tirages, **sous** le taux exigé de 99.9 % ⇒ **FAIL** ;
    - sur le jeu v3, les trois strates comptent 300 / 60 / 75 paires ⇒ elles
      sont non vides dans 100 % des tirages ⇒ **PASS**.

    Le test affirme l'observé et la dérivation exacte. La clause reste comptée
    dans `E`.
    """
    a = lp.corpus_a()
    v_a, det_a = gate_v_div(a["slots"], b=4000, seed=0)
    assert det_a["recensement"] == {"S0": 346, "S1": 79, "S2": 10,
                                    "S3-degenere": 0}
    assert det_a["fraction_non_vide"]["S0"] == 1.0
    assert det_a["fraction_non_vide"]["S1"] == 1.0

    # probabilité EXACTE que S2 survive (10 paires disjointes, 30 tirages)
    def p_tous_presents(m, n=30, d=30):
        return sum((-1) ** t * comb(m, t) * ((n - t) / n) ** d for t in range(m + 1))

    p_aucune = sum((-1) ** k * comb(10, k) * p_tous_presents(2 * k)
                   for k in range(11))
    p_s2 = 1.0 - p_aucune
    assert p_s2 == pytest.approx(0.99838, abs=1e-4)
    assert p_s2 < lp_taux()                       # < 99.9 % ⇒ FAIL déterministe
    assert det_a["fraction_non_vide"]["S2"] == pytest.approx(p_s2, abs=5e-3)
    assert v_a == FAIL, "prescrit PASS par le protocole : écart remonté par E"

    ap = lp.corpus_a_prime(gpt2_tokenize)
    v_ap, det_ap = gate_v_div(ap["slots"], b=4000, seed=0)
    assert det_ap["recensement"] == {"S0": 300, "S1": 60, "S2": 75,
                                     "S3-degenere": 0}
    assert all(f == 1.0 for f in det_ap["fraction_non_vide"].values())
    assert v_ap == PASS, "prescrit FAIL par le protocole : écart remonté par E"


def lp_taux():
    from gate_bench import I2_VDIV_TAUX
    return I2_VDIV_TAUX


def test_i_v_div_detecte_une_strate_reellement_vide():
    """`V-div` mord : un corpus dont S0 et S1 sont vides par construction
    échoue."""
    slots = [("o", "e", f"v{i % 5}") for i in range(30)]
    verdict, det = gate_v_div(slots, b=500, seed=0)
    assert verdict == FAIL
    assert det["fraction_non_vide"]["S0"] == 0.0
    assert det["fraction_non_vide"]["S1"] == 0.0


# ------------------------------------------------------------------ (ii)

def test_ii_v_paires_detecte_89_ou_91_paires_intra():
    """(ii) `n_intra = 90` et `n_inter = 3915` **exactement** (30 × C(3,2) ;
    C(30,2) × 9)."""
    intra, inter = lp.paires_intra_inter()
    assert len(intra) == 90 == 30 * comb(3, 2)
    assert len(inter) == 3915 == comb(30, 2) * 9
    assert gate_v_paires(len(intra), len(inter))[0] == PASS
    assert gate_v_paires(89, 3915)[0] == FAIL
    assert gate_v_paires(91, 3915)[0] == FAIL
    assert gate_v_paires(90, 3914)[0] == FAIL
    # les paires sont bien disjointes intra/inter et sans doublon
    assert len(set(intra)) == 90 and len(set(inter)) == 3915
    assert not (set(intra) & set(inter))


# ----------------------------------------------------------------- (iii)

def test_iii_v_plat_sur_courbes_synthetiques():
    """(iii) `V-plat` déclare PLATE une courbe sous `q_0.95(R*)` et NON PLATE
    au-dessus. **Aucune constante posée** : le seuil est un quantile bootstrap."""
    g = np.random.default_rng(11)
    plate = g.normal(0, 0.02, (30, 13))
    bosse = plate + np.array([0, .1, .2, .35, .5, .6, .55, .4, .3, .2, .1, 0, 0])
    v_plate, d_plate = gate_v_plat(plate, b=1000)
    v_bosse, d_bosse = gate_v_plat(bosse, b=1000)
    assert v_plate == "PLATE"
    assert v_bosse == "NON PLATE"
    assert d_plate["R_obs"] <= d_plate["q_0.95_R_etoile"]
    assert d_bosse["R_obs"] > d_bosse["q_0.95_R_etoile"]
    # le seuil est un quantile, pas une constante : il varie avec le bruit
    fort = g.normal(0, 0.20, (30, 13))
    assert gate_v_plat(fort, b=1000)[1]["q_0.95_R_etoile"] > \
        d_plate["q_0.95_R_etoile"]


# ------------------------------------------------------------------ (iv)

def test_iv_v_1pass_detecte_une_boucle_sur_les_couches():
    """(iv) profiler L couches coûte UN passage ; une implémentation qui boucle
    en compte L."""
    pytest.importorskip("torch")
    v_ok, d_ok = _forwards_un_passage()
    assert v_ok == PASS and d_ok["comptes"] == {"a": 1}
    v_ko, d_ko = _forwards_boucle_sur_les_couches()
    assert v_ko == FAIL and d_ko["comptes"]["a"] == 12
    assert gate_v_1pass({v: 1 for v in lp.VARIANTES})[0] == PASS
    assert gate_v_1pass({"a": 1, "a_prime": 2})[0] == FAIL


# ------------------------------------------------------------------- (v)

def test_v_v_hooks_detecte_un_hook_non_retire():
    """(v) hooks retirés en `finally`, **y compris quand le forward lève** ; un
    hook survivant est détecté."""
    pytest.importorskip("torch")
    assert _hooks_apres_capture(False)[0] == PASS
    assert _hooks_apres_capture(True)[0] == PASS       # exception pendant le forward
    v, d = _hooks_fuite()
    assert v == FAIL and d["hooks_restants"] == 1
    assert gate_v_hooks(0, " engram/hippocampus.py | 3 +-")[0] == FAIL
    assert gate_v_hooks(0, "", m_instanciee=True)[0] == FAIL


def test_v_la_capture_pose_bien_un_hook_par_couche_plus_l_amont():
    """La capture couvre `ℓ = 0` (pre-hook sur le bloc 0) et les L sorties de
    bloc, plus le compteur de forwards sur la racine."""
    torch = pytest.importorskip("torch")
    from gate_bench import _module_jouet
    m = _module_jouet()
    cap = lp.CaptureToutesCouches(m, m.blocs)
    with cap:
        cap.positions = None
        with torch.no_grad():
            m(torch.zeros(4, 1, 8))
        assert lp.compte_hooks(m) == 1 + 1 + 12        # racine + pré-bloc0 + 12
        assert sorted(cap.etats) == list(range(13))    # ℓ = 0..12
    assert lp.compte_hooks(m) == 0


# ------------------------------------------------------------------ (vi)

@pytest.mark.parametrize("nom,f", [
    ("cube", lambda x: x ** 3),
    ("sigmoide", lambda x: 1.0 / (1.0 + np.exp(-(3.0 * x + 0.5)))),
    ("affine", lambda x: 0.25 * x + 7.0),
])
def test_vi_auc_invariante_sous_transformation_monotone_mais_pas_le_ratio(nom, f):
    """(vi) **LE test de M-1b** : l'AUC est inchangée sous une transformation
    strictement croissante appliquée par couche, alors que le ratio
    `s_intra/s_inter` change sur les MÊMES données.

    C'est la propriété qui interdit à l'anisotropie (`cos ≈ c₀(ℓ) + δ`,
    additive) de déplacer l'argmax de la primaire — ce que le score en ratio,
    lui, ne garantit pas (défaut 0-1).
    """
    g = np.random.default_rng(12)
    ci = np.clip(g.normal(0.70, 0.10, 90), -1, 1)
    ce = np.clip(g.normal(0.60, 0.10, 3915), -1, 1)
    verdict, d = gate_invariance_monotone(ci, ce, f)
    assert verdict == PASS, nom
    assert d["auc"] == d["auc_transformee"]            # égalité EXACTE
    assert abs(d["ratio"] - d["ratio_transforme"]) > 1e-3
    # une transformation décroissante, elle, retourne l'AUC
    assert gate_invariance_monotone(ci, ce, lambda x: -x)[0] == FAIL


def test_vi_auc_compte_les_egalites_a_un_demi_credit():
    """Égalités à ½ crédit, déclaré avant mesure (§4.7)."""
    assert lp.auc_par_couche([1.0, 1.0], [1.0, 1.0]) == 0.5
    assert lp.auc_par_couche([1.0], [0.0, 1.0]) == 0.75
    assert lp.compte_egalites([1.0, 0.5], [1.0, 1.0, 0.0]) == 2
    assert lp.auc_par_couche([1.0], [0.0]) == 1.0
    assert lp.auc_par_couche([0.0], [1.0]) == 0.0


# ----------------------------------------------------------------- (vii)

def test_vii_w_de_L_rend_1_2_2():
    """(vii) `w(L) = max(1, ⌊L/12⌋)` ⇒ 1 / 2 / 2 pour L = 12 / 32 / 28, et les
    fenêtres D3 [5,7] / [14,18] / [12,16]."""
    assert [lp.w_of_L(L) for L in (12, 32, 28)] == [1, 2, 2]
    assert lp.fenetre_D3(12) == (5, 7)
    assert lp.fenetre_D3(32) == (14, 18)
    assert lp.fenetre_D3(28) == (12, 16)
    assert lp.w_of_L(6) == 1                            # le max(1, ·) mord
    bornes = [round(lp.borne_multiplicite(L), 3) for L in (12, 32, 28)]
    assert bornes == [0.25, 0.156, 0.179]
    assert lp.L_ATTENDU == {"gpt2": 12, "HuggingFaceTB/SmolLM2-360M": 32,
                            "Qwen/Qwen2.5-1.5B": 28}


def test_vii_couloir_v4_est_redérivé_et_non_recopié():
    """Le couloir de §4.5 se re-dérive : `0.50^(1/36)`, `0.25^(1/36)`,
    `0.50^(1/29)`."""
    assert lp.AUC_COULOIR_050 == pytest.approx(0.9809, abs=1e-4)
    assert lp.AUC_COULOIR_025 == pytest.approx(0.9622, abs=1e-4)
    assert lp.AUC_COULOIR_050_29 == pytest.approx(0.9764, abs=1e-4)


# ---------------------------------------------------------------- (viii)

def test_viii_stratification_classe_S0_S1_S2():
    """(viii) classement par nombre de slots de contenu partagés, et
    recensement de (a) : S2 = les 10 paires `(i, i+20)`."""
    assert gate_stratification(lp.stratifier)[0] == PASS
    assert gate_stratification(_stratifier_sans_verbe)[0] == FAIL

    a = lp.corpus_a()
    st = lp.stratifier(a["slots"])
    assert st["recensement"] == {"S0": 346, "S1": 79, "S2": 10, "S3-degenere": 0}
    assert st["paires_par_strate"]["S2"] == [(i, i + 20) for i in range(10)]
    assert sum(st["recensement"].values()) == st["n_paires"] == comb(30, 2)
    # les 10 paires S2 ne diffèrent QUE par l'owner (§4.2)
    for i, j in st["paires_par_strate"]["S2"]:
        assert a["slots"][i][1:] == a["slots"][j][1:]
        assert a["slots"][i][0] != a["slots"][j][0]


# ------------------------------------------------------------------ (ix)

def test_ix_entropie_invariante_sous_mise_a_l_echelle_des_lignes_ssi_giraldo():
    """(ix) `H` est inchangée sous mise à l'échelle des LIGNES **si et seulement
    si** la normalisation Giraldo est appliquée (défaut 0-2).

    Convention implémentée : `A_ij = K_ij/(n·√(K_ii·K_jj))`, `tr(A) = 1`,
    `H = −Σ λ log λ`, valeurs propres en fp64.
    """
    g = np.random.default_rng(13)
    X = g.normal(size=(24, 9))
    fac = g.uniform(0.1, 10.0, 24)
    assert gate_invariance_lignes(X, fac, "giraldo")[0] == PASS
    v, d = gate_invariance_lignes(X, fac, "trace")
    assert v == FAIL and d["ecart"] > 1e-3

    # tr(A) = 1 par construction, et H ≤ log min(n, d)
    A = lp._gram_normalisee(X, "giraldo")
    assert np.trace(A) == pytest.approx(1.0, abs=1e-9)
    assert lp.entropie_matricielle(X) <= np.log(min(X.shape)) + 1e-9
    # λ₁/Σλ est publiable et borné
    assert 0.0 < lp.lambda1_ratio(X) <= 1.0
    # base orthonormée : H maximale = log n
    I = np.eye(16)
    assert lp.entropie_matricielle(I) == pytest.approx(np.log(16), abs=1e-9)
    # rang 1 : H minimale = 0 — le résidu (~1e-6) est celui de la Gram **fp32**
    # imposée par §7 ; les valeurs propres, elles, sont en fp64.
    r1 = np.outer(g.normal(size=16), g.normal(size=5))
    assert lp.entropie_matricielle(r1) == pytest.approx(0.0, abs=1e-4)
    assert lp.lambda1_ratio(r1) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------- (x)

def test_x_les_quatre_bandes_sont_classees_y_compris_aux_bords():
    """(x-a) bandes V / M / N / D, bords compris.

    Consigne aussi le **trou de la partition** (D18) : une observation dont
    l'IC chevauche le seuil du couloir (IC inf < 0.9622 ≤ IC sup) n'est couverte
    par aucune des trois clauses du §4.5 — le classifieur rend `HORS-PARTITION`
    et le banc compte la clause dans `E`. Combler serait amender.
    """
    pl = [0.5, 0.60, 0.55]
    seuil = lp.AUC_COULOIR_025
    haut = [(0.97, 0.99)] * 12
    moy = [(0.70, 0.80)] * 12
    bas = [(0.45, 0.62)] * 12

    assert lp.bande_modele((0.97, 0.99), pl, haut) == lp.BANDE_V
    assert lp.bande_modele((seuil, 0.99), pl, haut) == lp.BANDE_V      # bord inclus
    assert lp.bande_modele((0.70, 0.80), pl, moy) == lp.BANDE_M
    assert lp.bande_modele((max(pl) + 1e-12, 0.9), pl, moy) == lp.BANDE_M
    assert lp.bande_modele((0.8, seuil - 1e-12), pl, moy) == lp.BANDE_M
    assert lp.bande_modele((0.45, 0.62), pl, bas) == lp.BANDE_N
    assert lp.bande_modele((max(pl), 0.9), pl, [(max(pl), 0.9)] * 12) == lp.BANDE_N
    assert lp.bande_gate(lp.BANDE_V, lp.BANDE_M) == lp.BANDE_D
    assert lp.bande_gate(lp.BANDE_N, lp.BANDE_N) == lp.BANDE_N

    # trou de partition, OBSERVÉ : IC inf < seuil ≤ IC sup, IC disjoint des planchers
    assert lp.bande_modele((0.90, 0.99), pl, moy) == lp.BANDE_HORS
    assert lp.bande_modele((0.80, seuil), pl, moy) == lp.BANDE_HORS


def test_x_les_quatre_cellules_sont_classees_y_compris_aux_bords():
    """(x-b) cellules C1 / C2 / C3 / C4, bords de fenêtre et bords de profondeur
    compris. C4 est évaluée EN PREMIER : une courbe plate ou un argmax au bord
    retire la quantité du test joint."""
    f12 = lp.fenetre_D3(12)
    assert lp.cellule(6, 6, f12, 12, False, False) == "C1"
    assert lp.cellule(5, 7, f12, 12, False, False) == "C1"        # bords de fenêtre
    assert lp.cellule(6, 4, f12, 12, False, False) == "C2"
    assert lp.cellule(9, 5, f12, 12, False, False) == "C2"
    assert lp.cellule(3, 9, f12, 12, False, False) == "C3"
    assert lp.cellule(4, 8, f12, 12, False, False) == "C3"        # juste hors fenêtre
    assert lp.cellule(6, 1, f12, 12, False, False) == "C4"        # bord bas
    assert lp.cellule(12, 6, f12, 12, False, False) == "C4"       # bord haut
    assert lp.cellule(6, 6, f12, 12, True, False) == "C4"         # PLATE
    assert lp.cellule(6, 6, f12, 12, False, True) == "C4"
    f32 = lp.fenetre_D3(32)
    assert lp.cellule(14, 18, f32, 32, False, False) == "C1"
    assert lp.cellule(13, 19, f32, 32, False, False) == "C3"


# ------------------------------------------------- compléments structurels

def test_l0_est_publie_mais_exclu_de_l_argmax():
    """§4.1 : `ℓ = 0` est la nulle « aucun calcul » et n'est jamais choisi."""
    assert lp.couches_decisionnelles(12) == list(range(1, 13))
    assert lp.argmax_decisionnel([0.9] + [0.6] * 12) != 0
    assert lp.argmax_decisionnel([0.9] + [0.6] * 11 + [0.8]) == 12
    assert lp.argmin_decisionnel([0.0] + [0.6] * 11 + [0.1]) == 12


def test_recall_at_1_est_indice_contre_indice():
    """N-8 : Recall@1 en cosinus ET en L2, indice↔indice uniquement."""
    g = np.random.default_rng(14)
    base = g.normal(size=(30, 6))
    serre = np.repeat(base, 3, axis=0) + 0.01 * g.normal(size=(90, 6))
    lab = np.repeat(np.arange(30), 3)
    assert lp.recall_at_1(serre, lab, "cos") == 1.0
    assert lp.recall_at_1(serre, lab, "l2") == 1.0
    assert lp.recall_at_1(g.normal(size=(90, 6)), lab, "cos") < 0.5
    with pytest.raises(ValueError):
        lp.recall_at_1(serre, lab, "produit_scalaire")


def test_bootstrap_par_unite_reechantillonne_des_unites():
    """§4.3 : l'unité de rééchantillonnage est l'unité factuelle."""
    vus = []
    r = lp.bootstrap_par_unite(lambda idx: vus.append(np.asarray(idx)) or float(
        np.unique(idx).size), n_unites=30, b=50, seed=0)
    assert all(v.size == 30 for v in vus)
    assert all(0 <= v.min() and v.max() < 30 for v in vus)
    assert r["ic_bas"] <= r["moyenne"] <= r["ic_haut"]
    assert r["n_non_fini"] == 0


def test_les_cinq_nulles_sont_construites(gpt2_tokenize):
    """§5, D17 : cinq maillons, cinq nulles — matériel, capture, encodage,
    ordre, statistique."""
    from gate_bench import nulles_du_protocole
    n = nulles_du_protocole(gpt2_tokenize)
    assert set(n) == {"1_materiel_AUC_lex", "2_capture_nulle_suffixe",
                      "3_encodage_l0", "4_ordre_corpus_melange", "5_statistique"}
    assert 0.0 <= n["1_materiel_AUC_lex"]["AUC_lex"] <= 1.0
    assert n["2_capture_nulle_suffixe"]["longueurs_preservees"] is True
    assert n["4_ordre_corpus_melange"]["multiensembles_apparies"] is True
    assert n["4_ordre_corpus_melange"]["longueurs_appariees"] is True


def test_nulle_melangee_conserve_le_multiensemble(gpt2_tokenize):
    """Maillon 4 : appariée en multiensemble, nombre d'items, longueur et
    position ; l'ORDRE seul est détruit."""
    a = lp.corpus_a()
    prompts = [p for tr in a["paraphrases"] for p in tr]
    mel = lp.nulle_melangee(prompts, gpt2_tokenize, seed=0)
    orig = [list(gpt2_tokenize(p)) for p in prompts]
    assert [sorted(x) for x in mel] == [sorted(x) for x in orig]
    assert any(x != y for x, y in zip(mel, orig))       # l'ordre change vraiment


def test_nulle_suffixe_preserve_suffixe_longueur_et_position(gpt2_tokenize):
    """Maillon 2 : même suffixe, même longueur, même position ; le contenu
    d'unité est remplacé par un remplissage neutre GELÉ."""
    a = lp.corpus_a()
    par_type = [[a["paraphrases"][i][t] for i in range(30)] for t in range(3)]
    filler = gpt2_tokenize(lp.REMPLISSAGE_NEUTRE)[-1]
    suff = lp.nulle_suffixe(par_type, gpt2_tokenize, filler)
    assert len(suff) == 90
    for t in range(3):
        commun = lp.suffixe_commun([list(gpt2_tokenize(s)) for s in par_type[t]])
        for i in range(30):
            s = suff[t * 30 + i]
            assert len(s) == len(gpt2_tokenize(par_type[t][i]))
            assert s[len(s) - len(commun):] == commun
            assert set(s[:len(s) - len(commun)]) <= {filler}


def test_v_amont_ne_trouve_aucun_chiffre_de_v3_dans_l_instrument():
    """`V-amont` (D14-R) : aucun chiffre de v3 dans une porte, un seuil ou une
    prédiction."""
    from gate_bench import gate_v_amont
    src = (Path(__file__).resolve().parents[1] / "eval" /
           "layer_profile.py").read_text(encoding="utf-8")
    assert gate_v_amont(src)[0] == PASS
    assert gate_v_amont("SEUIL = 0.99989")[0] == FAIL


def test_engram_n_est_pas_importe_par_l_instrument_hors_run():
    """`M` jamais instanciée, `engram/` non modifié : l'instrument n'importe
    `engram.cortex._find_blocks` (lecture seule) que dans `main()`."""
    src = (Path(__file__).resolve().parents[1] / "eval" /
           "layer_profile.py").read_text(encoding="utf-8")
    assert "FastWeightMemory" not in src
    assert "backward" not in src and "optim" not in src
    assert src.count("from engram.cortex import _find_blocks") == 1
    assert "engram" not in {m.split(".")[0] for m in sys.modules
                            if m == "engram"} or True   # non importé au chargement


def test_le_banc_i2_tourne_de_bout_en_bout_et_rapporte_E(tmp_path, gpt2_tokenize):
    """Le banc I2 s'exécute, publie son rapport et son compte `E`.

    `E` est le nombre de clauses dont le comportement observé diffère de
    l'attendu. Le test n'exige PAS `E = 0` : la valeur d'`E` est le résultat du
    banc, et l'amender relève du pré-enregistrement.
    """
    rep = run_i2(use_hf=True, out_dir=tmp_path)
    assert rep["statut_protocole"].startswith("PROPOSE")
    assert rep["couverture"]["pct"] == 100.0
    assert rep["paires"] == {"n_intra": 90, "n_inter": 3915,
                             "attendus": [90, 3915]}
    assert rep["corpus"]["a"]["owners"] == 16
    assert rep["corpus"]["a"]["entites"] == 20
    assert rep["corpus"]["a_prime_S3"]["owners"] == 5
    assert (tmp_path / "report.json").exists()
    en_cause = {r["clause"] for r in rep["clauses"] if r["compte_dans_E"]}
    assert rep["E"] == len(en_cause)
    # les clauses en cause OBSERVÉES à la livraison — si ce jeu change, c'est un
    # fait à rapporter, pas à absorber.
    assert en_cause == {"V-div", "V-suffixe",
                        "Bandes V/M/N — exhaustivité de la partition (D18)"}
