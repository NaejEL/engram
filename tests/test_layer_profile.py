# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests CPU d'I2 — `eval/layer_profile.py` et la suite `i2` du banc.

Protocole : `experiments/EXP-2026-08-22-layer-profile.md`, §10 (quatorze tests),
version **CONSOLIDÉE** (N = 80, partition par slot d'identité, couloir sur
`R1_36`, bandes N/M/I/V).

Aucun GPU, aucun modèle : le seul HF touché est le **tokenizer GPT-2** (CPU,
cache local), là où la clause n'est pas décidable sans lui.

  (i)    **`V-diversité` échoue sur le jeu v3 et passe sur `fact_pairs(80)`** ;
  (ii)   `V-puissance` rend K = 14 (N=30 FAIL), 16 (N=80 PASS), **FAIL sous
         décision AUC à tout N ≤ 80** ;
  (iii)  recensement **411/14/10/0** et **2880/160/120/0**, `P-both = ∅` prouvé
         par `lcm(16,20) = 80` ;
  (iv)   `V-paires` détecte 239 ou 241 ;
  (v)    **`V-suffixe` rend 1.0000/0.18987/1.0000 à N = 80 et
         1.0000/0.17241/1.0000 à N = 30** ;
  (vi)   `V-plat` ;
  (vii)  `V-1pass` ;
  (viii) `V-hooks` ;
  (ix)   **l'AUC est inchangée sous transformation monotone par couche alors que
         le ratio change** ;
  (x)    `w(L)` = 1 / 2 / 2 ;
  (xi)   l'entropie est inchangée sous mise à l'échelle des lignes **ssi**
         normalisation Giraldo ;
  (xii)  le jeu `R1_36` a **exactement 37 éléments**, est **déterministe** et
         **indépendant des similarités mesurées** ;
  (xiii) **l'estimateur débiaisé diffère du max brut à effet nul**, « max des
         IC » rejeté ;
  (xiv)  `V-bandes` classe quatre bandes, trois bords, et rend **I** quand un
         modèle est I et l'autre V.

Les tests affirment ce qui est OBSERVÉ, jamais ce qui est souhaité : un désaccord
est remonté par le compte `E` du banc, et l'amendement est une décision de
pré-enregistrement, pas d'implémentation.
"""

import sys
from math import comb, lcm
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import layer_profile as lp  # noqa: E402
from gate_bench import (  # noqa: E402
    FAIL, PASS, gate_invariance_lignes, gate_invariance_monotone, gate_jeu_r1,
    gate_nulle_cadre, gate_recensement_avec_classifieur,
    gate_recensement_identite, gate_v_1pass, gate_v_bandes, gate_v_diversite,
    gate_v_hooks, gate_v_paires, gate_v_plat, gate_v_puissance, gate_v_source,
    gate_v_suffixe, make_offsets, _cas_estimateur_debiaise,
    _cas_max_des_ic_rejete, _forwards_boucle_sur_les_couches,
    _forwards_un_passage, _hooks_apres_capture, _hooks_fuite,
    _stratifier_avec_verbe, run_i2,
)

REC_80 = {"P-0": 2880, "P-own": 160, "P-ent": 120, "P-both": 0}
REC_30 = {"P-0": 411, "P-own": 14, "P-ent": 10, "P-both": 0}


@pytest.fixture(scope="module")
def gpt2_tokenize():
    tr = pytest.importorskip("transformers")
    tok = tr.AutoTokenizer.from_pretrained("gpt2")
    return lambda s: [int(t) for t in tok.encode(s)]


@pytest.fixture(scope="module")
def corpus80():
    return lp.corpus_a()


# ------------------------------------------------------------------- (i)

def test_i_v_diversite_passe_sur_fact_pairs_80_et_echoue_sur_le_jeu_v3(
        gpt2_tokenize, corpus80):
    """(i) **Direction INVERSÉE du défaut 0-7.**

    `V-diversité` est combinatoire : PASS ssi `(#owners, #entités, #verbes) =
    (min(N,16), min(N,20), min(N,5))`. `fact_pairs(80)` rend (16, 20, 5) et
    PASSE ; le jeu v3 rend (5, 6, ·) et ÉCHOUE. L'ancienne `V-div` mesurait un
    cardinal de strate et récompensait la dégénérescence : elle faisait
    exactement l'inverse.
    """
    v, det = gate_v_diversite(corpus80["slots"])
    assert det["observe"] == [16, 20, 5] == det["attendu"]
    assert v == PASS

    b_v3 = lp.corpus_b_v3(gpt2_tokenize)
    v3, det3 = gate_v_diversite(b_v3["slots"])
    assert det3["observe"][:2] == [5, 6]
    assert v3 == FAIL

    # zéro bootstrap, zéro GPU : la porte est une égalité de recensement
    assert lp.diversite_attendue(80) == (16, 20, 5)
    assert lp.diversite_attendue(10) == (10, 10, 5)
    # un corpus dégénéré échoue
    assert gate_v_diversite([("o", "e", "v")] * 80)[0] == FAIL


# ------------------------------------------------------------------ (ii)

def test_ii_v_puissance_14_fail_16_pass_et_fail_sous_AUC(corpus80):
    """(ii) `K_S ≥ (1.96·σ₀/|θ−T|)²` : σ₀ = 0.5, |θ−T| = 0.25 ⇒ **K ≥ 16**.

    `P-own` : K = 14 à N = 30 (**FAIL**), K = 16 à N = 80 (**PASS à
    l'égalité — marge nulle**). Sous décision AUC, K requis = 429 ⇒ **FAIL à
    tout N ≤ 80**, et la porte **nomme** l'indécidabilité.
    """
    assert lp.k_requis() == 16
    assert lp.k_requis(0.5, 0.25) == 16

    k80 = lp.clusters_de_strate(corpus80["slots"], lp.P_OWN)["K"]
    k30 = lp.clusters_de_strate(lp.corpus_a(30)["slots"], lp.P_OWN)["K"]
    assert (k80, k30) == (16, 14)
    assert lp.clusters_de_strate(corpus80["slots"], lp.P_ENT)["K"] == 20
    assert lp.clusters_de_strate(lp.corpus_a(30)["slots"], lp.P_ENT)["K"] == 10

    v80, d80 = gate_v_puissance(k80)
    assert v80 == PASS and d80["K"] == d80["K_requis"] == 16   # égalité EXACTE
    v30, d30 = gate_v_puissance(k30)
    assert v30 == FAIL and "puissance du DESIGN" in d30["cause_du_FAIL"]

    # décision AUC : FAIL à tout N ≤ 80, indécidabilité NOMMÉE
    for k in (14, 16, 20):
        v, d = gate_v_puissance(k, statistique="AUC",
                                k_requis=lp.K_REQUIS_AUC_PROTOCOLE)
        assert v == FAIL
        assert d["indecidable_a_tout_N"] is True
        assert "aucune répétition ne lève cette porte" in \
            d["nom_de_l_indecidabilite"]
    assert lp.K_REQUIS_AUC_PROTOCOLE == 429


# ----------------------------------------------------------------- (iii)

def test_iii_recensement_identite_et_P_both_vide(corpus80):
    """(iii) Recensement par slot d'IDENTITÉ : **2880/160/120/0** à N = 80,
    **411/14/10/0** à N = 30 ; `P-both = ∅` **prouvé par `lcm(16,20) = 80`**.

    Le VERBE n'est pas un slot : deux unités n'en différant que par le verbe
    désignent le même fait (N-9, défaut 0-12).
    """
    p80 = lp.partition_identite(corpus80["slots"])
    p30 = lp.partition_identite(lp.corpus_a(30)["slots"])
    assert p80["recensement"] == REC_80
    assert p30["recensement"] == REC_30
    assert sum(p80["recensement"].values()) == comb(80, 2) == 3160
    assert sum(p30["recensement"].values()) == comb(30, 2) == 435

    # P-both : arithmétiquement impossible
    pb = lp.p_both_impossible()
    assert pb["lcm"] == lcm(16, 20) == 80
    assert pb["possible"] is False
    assert "l'arithmétique du pool le lui a IMPOSÉ" in pb["clause"]
    assert p80["recensement"]["P-both"] == p30["recensement"]["P-both"] == 0

    assert gate_recensement_identite(corpus80["slots"], REC_80)[0] == PASS
    assert gate_recensement_identite(lp.corpus_a(30)["slots"], REC_30)[0] == PASS
    # un classifieur qui compte le verbe comme slot d'identité échoue
    assert gate_recensement_avec_classifieur(
        _stratifier_avec_verbe, corpus80["slots"], REC_80)[0] == FAIL

    # les paires P-own ne diffèrent QUE par l'entité (le verbe est covariable)
    for i, j in p80["paires_par_classe"]["P-own"]:
        assert corpus80["slots"][i][0] == corpus80["slots"][j][0]
        assert corpus80["slots"][i][1] != corpus80["slots"][j][1]
    assert lp.ESPACE_IDENTITE == 16 * 20 == 320


# ------------------------------------------------------------------ (iv)

def test_iv_v_paires_detecte_239_ou_241():
    """(iv) `n_intra = 240` et `n_inter = 28 440` **exactement**."""
    intra, inter = lp.paires_intra_inter()
    assert len(intra) == 240 == 80 * comb(3, 2)
    assert len(inter) == 28_440 == comb(80, 2) * 9
    rec = lp.partition_identite(lp.corpus_a()["slots"])["recensement"]
    assert gate_v_paires(len(intra), len(inter), rec)[0] == PASS
    assert gate_v_paires(239, 28_440, rec)[0] == FAIL
    assert gate_v_paires(241, 28_440, rec)[0] == FAIL
    assert gate_v_paires(240, 28_439, rec)[0] == FAIL
    assert gate_v_paires(240, 28_440, dict(rec, **{"P-both": 1}))[0] == FAIL
    assert len(set(intra)) == 240 and len(set(inter)) == 28_440
    assert not (set(intra) & set(inter))


# ------------------------------------------------------------------- (v)

def test_v_v_suffixe_rend_les_constantes_derivees(gpt2_tokenize, corpus80):
    """(v) `V-suffixe` **re-dérivée** (0-8) : para1 = **1.0000**,
    para3 = **1.0000**, para2 = `#{paires : 5|d}/C(N,2)` = **0.18987** à N = 80
    (**0.17241** à N = 30).
    """
    d80 = lp.partage_suffixe_derive(80)
    assert d80["para1"] == d80["para3"] == 1.0
    assert d80["n_paires_5_divise_d"] == 600 and d80["C_n_2"] == 3160
    assert d80["para2"] == pytest.approx(0.18987, abs=1e-5)
    d30 = lp.partage_suffixe_derive(30)
    assert (d30["n_paires_5_divise_d"], d30["C_n_2"]) == (75, 435)
    assert d30["para2"] == pytest.approx(0.17241, abs=1e-5)

    mes = lambda c, n: {t: lp.partage_dernier_token(                    # noqa: E731
        [c["paraphrases"][i][k] for i in range(n)], gpt2_tokenize)
        for k, t in enumerate(lp.POOL_PARAPHRASE_TYPES)}
    obs80, obs30 = mes(corpus80, 80), mes(lp.corpus_a(30), 30)
    assert obs80["para1"] == 1.0 and obs80["para3"] == 1.0
    assert obs80["para2"] == pytest.approx(0.18987, abs=1e-5)
    assert obs30["para2"] == pytest.approx(0.17241, abs=1e-5)
    assert gate_v_suffixe(obs80, 80)[0] == PASS
    assert gate_v_suffixe(obs30, 30)[0] == PASS
    # la porte mord : l'ancienne attente périmée (para1 ≈ 0) échoue
    assert gate_v_suffixe({"para1": 0.0, "para2": 0.0, "para3": 1.0}, 80)[0] == FAIL
    assert gate_v_suffixe({"para1": 1.0, "para2": 1.0, "para3": 1.0}, 80)[0] == FAIL


def test_v_la_nulle_du_maillon_2_conserve_le_verbe(gpt2_tokenize, corpus80):
    """La nulle du maillon 2 est re-spécifiée : **verbe conservé**, slots de
    contenu remplacés, appariée en longueur et position. La clause « même
    suffixe » est abandonnée (elle était **vacuée** pour para2 : suffixe commun
    vide)."""
    off = make_offsets(True)
    filler = gpt2_tokenize(lp.REMPLISSAGE_NEUTRE)[-1]
    v, det = gate_nulle_cadre(corpus80["slots"], gpt2_tokenize, filler, off)
    assert v == PASS
    assert det["longueurs_appariees"] is True
    assert det["verbe"]["conserve"] is True
    assert det["n_sequences"] == 240
    assert det["tokens_remplaces_min_max"][0] >= 1
    # une nulle qui efface AUSSI le verbe est détectée
    assert gate_nulle_cadre(corpus80["slots"], gpt2_tokenize, filler, off,
                            effacer_le_verbe=True)[0] == FAIL
    # le suffixe commun de para2 est VIDE : la contrainte y était vacuée
    par_type2 = [corpus80["paraphrases"][i][1] for i in range(80)]
    assert lp.suffixe_commun([list(gpt2_tokenize(s)) for s in par_type2]) == []


# ------------------------------------------------------------------ (vi)

def test_vi_v_plat_sur_courbes_synthetiques():
    """(vi) `V-plat` : PLATE ssi `R_obs ≤ q_0.95(R*)`. **Aucune constante
    posée** — le seuil est un quantile bootstrap."""
    g = np.random.default_rng(11)
    plate = g.normal(0, 0.02, (30, 13))
    bosse = plate + np.array([0, .1, .2, .35, .5, .6, .55, .4, .3, .2, .1, 0, 0])
    v_plate, d_plate = gate_v_plat(plate, b=1000)
    v_bosse, d_bosse = gate_v_plat(bosse, b=1000)
    assert v_plate == "PLATE" and v_bosse == "NON PLATE"
    assert d_plate["R_obs"] <= d_plate["q_0.95_R_etoile"]
    assert d_bosse["R_obs"] > d_bosse["q_0.95_R_etoile"]
    fort = g.normal(0, 0.20, (30, 13))
    assert gate_v_plat(fort, b=1000)[1]["q_0.95_R_etoile"] > \
        d_plate["q_0.95_R_etoile"]


# ----------------------------------------------------------------- (vii)

def test_vii_v_1pass_detecte_une_boucle_sur_les_couches():
    """(vii) profiler L couches coûte UN passage."""
    pytest.importorskip("torch")
    v_ok, d_ok = _forwards_un_passage()
    assert v_ok == PASS and d_ok["comptes"] == {"a": 1}
    v_ko, d_ko = _forwards_boucle_sur_les_couches()
    assert v_ko == FAIL and d_ko["comptes"]["a"] == 12
    assert gate_v_1pass({v: 1 for v in lp.VARIANTES})[0] == PASS
    assert gate_v_1pass({"a": 1, "b_v3": 2})[0] == FAIL


# ---------------------------------------------------------------- (viii)

def test_viii_v_hooks_detecte_un_hook_non_retire():
    """(viii) hooks retirés en `finally`, **y compris quand le forward lève**."""
    pytest.importorskip("torch")
    assert _hooks_apres_capture(False)[0] == PASS
    assert _hooks_apres_capture(True)[0] == PASS
    v, d = _hooks_fuite()
    assert v == FAIL and d["hooks_restants"] == 1
    assert gate_v_hooks(0, " engram/hippocampus.py | 3 +-")[0] == FAIL
    assert gate_v_hooks(0, "", m_instanciee=True)[0] == FAIL


def test_viii_la_capture_pose_un_hook_par_couche_plus_l_amont():
    torch = pytest.importorskip("torch")
    from gate_bench import _module_jouet
    m = _module_jouet()
    cap = lp.CaptureToutesCouches(m, m.blocs)
    with cap:
        cap.positions = None
        with torch.no_grad():
            m(torch.zeros(4, 1, 8))
        assert lp.compte_hooks(m) == 1 + 1 + 12
        assert sorted(cap.etats) == list(range(13))
    assert lp.compte_hooks(m) == 0


# ------------------------------------------------------------------ (ix)

@pytest.mark.parametrize("nom,f", [
    ("cube", lambda x: x ** 3),
    ("sigmoide", lambda x: 1.0 / (1.0 + np.exp(-(3.0 * x + 0.5)))),
    ("affine", lambda x: 0.25 * x + 7.0),
])
def test_ix_auc_invariante_sous_transformation_monotone_mais_pas_le_ratio(nom, f):
    """(ix) **LE test du défaut 0-1** : l'AUC est inchangée sous une
    transformation strictement croissante appliquée par couche, alors que le
    ratio `s_intra/s_inter` change sur les MÊMES données."""
    g = np.random.default_rng(12)
    ci = np.clip(g.normal(0.70, 0.10, 240), -1, 1)
    ce = np.clip(g.normal(0.60, 0.10, 5000), -1, 1)
    verdict, d = gate_invariance_monotone(ci, ce, f)
    assert verdict == PASS, nom
    assert d["auc"] == d["auc_transformee"]            # égalité EXACTE
    assert abs(d["ratio"] - d["ratio_transforme"]) > 1e-3
    assert gate_invariance_monotone(ci, ce, lambda x: -x)[0] == FAIL


def test_ix_auc_compte_les_egalites_a_un_demi_credit():
    assert lp.auc_par_couche([1.0, 1.0], [1.0, 1.0]) == 0.5
    assert lp.auc_par_couche([1.0], [0.0, 1.0]) == 0.75
    assert lp.compte_egalites([1.0, 0.5], [1.0, 1.0, 0.0]) == 2


# ------------------------------------------------------------------- (x)

def test_x_w_de_L_rend_1_2_2():
    """(x) `w(L) = max(1, ⌊L/12⌋)` ⇒ 1 / 2 / 2 et fenêtres [5,7]/[14,18]/[12,16]."""
    assert [lp.w_of_L(L) for L in (12, 32, 28)] == [1, 2, 2]
    assert lp.fenetre_D3(12) == (5, 7)
    assert lp.fenetre_D3(32) == (14, 18)
    assert lp.fenetre_D3(28) == (12, 16)
    assert [round(lp.borne_multiplicite(L), 3) for L in (12, 32, 28)] == \
        [0.25, 0.156, 0.179]
    assert lp.L_ATTENDU == {"gpt2": 12, "HuggingFaceTB/SmolLM2-360M": 32,
                            "Qwen/Qwen2.5-1.5B": 28}


def test_x_les_reperes_en_AUC_ne_sont_pas_decisionnels():
    """§4.5, note historique : `A ≥ 0.9622` / `0.9809` sont des **repères de
    publication**, jamais un critère. La décision porte sur `R1_36` et `T = 0.25`.
    """
    assert lp.AUC_REPERE_025 == pytest.approx(0.9622, abs=1e-4)
    assert lp.AUC_REPERE_050 == pytest.approx(0.9809, abs=1e-4)
    assert lp.T_COULOIR == 0.25 and lp.T_COULOIR_PLUS == 0.50
    assert lp.TAILLE_JEU_R1 == 37
    assert lp.HASARD_R1 == pytest.approx(0.02703, abs=1e-5)
    src = (Path(__file__).resolve().parents[1] / "eval"
           / "layer_profile.py").read_text(encoding="utf-8")
    # aucun repère en AUC n'entre dans `bande_modele`
    debut = src.index("def bande_modele(")
    fin = src.index("def delta_r1_negatif(")
    assert "AUC_REPERE" not in src[debut:fin]


# ------------------------------------------------------------------ (xi)

def test_xi_entropie_invariante_sous_mise_a_l_echelle_des_lignes_ssi_giraldo():
    """(xi) `H` est inchangée sous mise à l'échelle des LIGNES **ssi** la
    normalisation Giraldo est appliquée (défaut 0-2)."""
    g = np.random.default_rng(13)
    X = g.normal(size=(24, 9))
    fac = g.uniform(0.1, 10.0, 24)
    assert gate_invariance_lignes(X, fac, "giraldo")[0] == PASS
    v, d = gate_invariance_lignes(X, fac, "trace")
    assert v == FAIL and d["ecart"] > 1e-3

    A = lp._gram_normalisee(X, "giraldo")
    assert np.trace(A) == pytest.approx(1.0, abs=1e-9)
    assert lp.entropie_matricielle(X) <= np.log(min(X.shape)) + 1e-9
    assert 0.0 < lp.lambda1_ratio(X) <= 1.0
    I = np.eye(16)
    assert lp.entropie_matricielle(I) == pytest.approx(np.log(16), abs=1e-9)
    r1 = np.outer(g.normal(size=16), g.normal(size=5))
    assert lp.entropie_matricielle(r1) == pytest.approx(0.0, abs=1e-4)
    assert lp.lambda1_ratio(r1) == pytest.approx(1.0, abs=1e-6)


def test_xi_v_source_exige_le_PDF_de_la_source_primaire():
    """`V-source` (0-10) : Giraldo et al. 2014 pour la normalisation des lignes ;
    Skean Eq. 1 **ne la porte pas** ; lire du HTML est un motif d'arrêt."""
    assert gate_v_source()[0] == PASS
    assert all(c["support"] == "PDF" for c in lp.CITATIONS)
    assert any("Giraldo" in c["source_primaire"] for c in lp.CITATIONS)
    assert gate_v_source([{"equation": "e", "source_primaire": "Giraldo 2014",
                           "support": "HTML"}])[0] == FAIL
    assert gate_v_source([{"equation": "e", "source_primaire": "",
                           "support": "PDF"}])[0] == FAIL


# ----------------------------------------------------------------- (xii)

def test_xii_le_jeu_R1_36_a_37_elements_deterministe_et_independant(corpus80):
    """(xii) 1 cible + 36 concurrents = **37 éléments exactement**, jeu
    **déterministe** et **indépendant des similarités mesurées** — permuter les
    valeurs de similarité NE CHANGE PAS le jeu."""
    slots = corpus80["slots"]
    j = lp.jeu_candidats_R1(0, 0, slots)
    assert j["taille"] == 37
    assert len(set([j["cible"]] + list(j["concurrents"]))) == 37
    assert j["cible"] == 1                      # règle cyclique para1 → para2
    assert lp.jeu_candidats_R1(0, 2, slots)["cible"] == 0     # para3 → para1

    # remplissage LE PLUS DUR D'ABORD : 12 owner, 9 entité, 15 P-0
    comp = lp.composantes(slots)
    o, e = slots[0][0], slots[0][1]
    bloc_own = {u * 3 + k for u in comp["owner"][o] if u != 0 for k in range(3)}
    bloc_ent = {u * 3 + k for u in comp["entite"][e] if u != 0 for k in range(3)}
    conc = list(j["concurrents"])
    assert set(conc[:12]) == bloc_own and len(bloc_own) == 12
    assert set(conc[12:21]) == bloc_ent and len(bloc_ent) == 9
    assert len(conc[21:]) == 15
    assert not (set(conc[21:]) & (bloc_own | bloc_ent))

    v, det = gate_jeu_r1(slots)
    assert v == PASS
    assert det["tailles_observees"] == [37]
    assert det["deterministe"] is True
    assert det["independant_des_similarites"] is True
    # ... et la STATISTIQUE, elle, dépend bien des similarités
    assert det["la_statistique_depend_bien_de_S"] is True
    assert det["hasard"] == pytest.approx(1 / 37)


def test_xii_r1_36_est_1_sur_une_geometrie_parfaite_et_apparie_delta(corpus80):
    """`R1_36` = 1.0 quand les trois indices d'une unité coïncident ; `ΔR1` est
    appariée par requête."""
    slots = corpus80["slots"]
    g = np.random.default_rng(2)
    base = g.normal(size=(80, 12))
    X = np.repeat(base, 3, axis=0) + 0.001 * g.normal(size=(240, 12))
    r = lp.r1_36(lp.cosinus_matrice(X).astype(np.float64), slots)
    assert r["R1"] == 1.0 and r["n_requetes"] == 240
    hasard = lp.r1_36(lp.cosinus_matrice(g.normal(size=(240, 12))
                                         ).astype(np.float64), slots)
    assert hasard["R1"] < 0.25
    d = lp.delta_r1(r["succes"], hasard["succes"])
    assert d.shape == (240,) and d.mean() > 0
    with pytest.raises(ValueError):
        lp.delta_r1(r["succes"], hasard["succes"][:10])
    # cluster de rééchantillonnage = composante d'owner, effectifs préservés
    cl = lp.clusters_de_strate(slots, lp.P_OWN)
    assert cl["K"] == 16 and cl["cle"] == "owner"
    assert {len(v) for v in cl["membres"].values()} == {5}


# ---------------------------------------------------------------- (xiii)

def test_xiii_estimateur_debiaise_et_rejet_du_max_des_ic():
    """(xiii) **À effet nul**, `θ̂ = 2·max_obs − mean_b(θ*_b)` diffère du max
    brut (biais de sélection du max, M-15) ; « max des IC par couche » est
    **rejeté** mécaniquement."""
    v, d = _cas_estimateur_debiaise()
    assert v == PASS
    assert d["theta_debiaise"] != d["max_brut"]
    assert d["ecart_debiaise_vs_max"] > 1e-9
    assert d["argmax_re_selectionne"] is True
    # le biais est positif : le max bootstrap moyen dépasse le max observé ou
    # s'en approche ; l'estimateur débiaisé corrige DANS UNE DIRECTION
    assert lp.estimateur_debiaise(1.0, [0.9, 0.95, 1.0]) == pytest.approx(1.05)

    verdict, det = _cas_max_des_ic_rejete()
    assert verdict == "REJETÉ"
    assert "motif d'invalidation" in det["message"]
    with pytest.raises(ValueError):
        lp.ic_max_des_ic_par_couche([(0.1, 0.9)] * 12)


def test_xiii_bca_est_requis_au_dela_de_095():
    """M-16 : le percentile sous-couvre près de la borne 1 ⇒ **BCa** dès qu'une
    borne dépasse 0.95."""
    g = np.random.default_rng(5)
    ech = np.clip(g.beta(20, 1.0, 400), 0, 1)
    jack = np.clip(g.beta(20, 1.0, 16), 0, 1)
    haut = lp.ic_du_max(ech, float(ech.max()), jack)
    assert haut["methode"] == "BCa"
    bas = lp.ic_du_max(g.normal(0.4, 0.05, 400), 0.55, g.normal(0.4, 0.05, 16))
    assert bas["methode"] == "percentile"
    assert lp.SEUIL_BCA == 0.95


def test_xiii_permutation_a_couche_fixee_recalcule_le_max():
    """Bande N par permutation des étiquettes d'unité **à couche fixée avec
    recalcul de `max_ℓ`** (FWER exact)."""
    g = np.random.default_rng(6)
    nul = g.normal(0, 1, (16, 12))

    def stat(ell, rng):
        return float(nul[rng.permutation(16), ell].mean())

    r = lp.permutation_max_couches(stat, range(12), b=200, seed=6)
    assert r["B"] == 200 and np.isfinite(r["q_0.95"])
    assert max(float(nul[:, e].mean()) for e in range(12)) <= r["q_0.95"]


# ----------------------------------------------------------------- (xiv)

def test_xiv_les_quatre_bandes_trois_bords_et_la_precedence_de_D():
    """(xiv) Partition **N / M / I / V** exhaustive et mutuellement exclusive,
    **bords inclus** ; overlay `D` : **un modèle I et l'autre V ⇒ I**."""
    T = lp.T_COULOIR
    # une bande par classe
    assert lp.bande_modele((-0.10, 0.10), (0.05, 0.60)) == lp.BANDE_N
    assert lp.bande_modele((0.05, 0.40), (0.05, 0.20)) == lp.BANDE_M
    assert lp.bande_modele((0.05, 0.40), (0.05, 0.60)) == lp.BANDE_I
    assert lp.bande_modele((0.05, 0.40), (0.55, 0.90)) == lp.BANDE_V

    # les TROIS bords gravés
    assert lp.bande_modele((0.05, 0.40), (0.05, T)) == lp.BANDE_I     # IC_sup = T
    assert lp.bande_modele((0.05, 0.40), (T, 0.60)) == lp.BANDE_V     # IC_inf = T
    assert lp.bande_modele((0.0, 0.30), (0.55, 0.90)) == lp.BANDE_N   # ΔR1 touche 0

    # précédence de l'overlay D
    assert lp.bande_gate(lp.BANDE_I, lp.BANDE_V) == lp.BANDE_I
    assert lp.bande_gate(lp.BANDE_V, lp.BANDE_I) == lp.BANDE_I
    assert lp.bande_gate(lp.BANDE_I, lp.BANDE_N) == lp.BANDE_I
    assert lp.bande_gate(lp.BANDE_V, lp.BANDE_M) == lp.BANDE_D
    assert lp.bande_gate(lp.BANDE_N, lp.BANDE_N) == lp.BANDE_N

    # mention V⁺ et sous-étiquette descriptive
    assert lp.mention_v_plus((0.55, 0.90)) is True
    assert lp.mention_v_plus((0.30, 0.90)) is False
    assert lp.delta_r1_negatif((-0.20, -0.05)) is True

    # exhaustivité : la porte balaie et ne trouve aucun trou
    v, det = gate_v_bandes()
    assert v == PASS
    assert det["hors_partition"] == []
    assert set(det["bandes_rencontrees"]) <= set(lp.BANDES)
    assert det["bords_conformes"] and det["precedence_conforme"]


def test_xiv_les_quatre_cellules_sont_classees_y_compris_aux_bords():
    """Cellules C1 / C2 / C3 / C4 (§4.8). C4 est évaluée EN PREMIER."""
    f12 = lp.fenetre_D3(12)
    assert lp.cellule(6, 6, f12, 12, False, False) == "C1"
    assert lp.cellule(5, 7, f12, 12, False, False) == "C1"
    assert lp.cellule(6, 4, f12, 12, False, False) == "C2"
    assert lp.cellule(3, 9, f12, 12, False, False) == "C3"
    assert lp.cellule(6, 1, f12, 12, False, False) == "C4"
    assert lp.cellule(12, 6, f12, 12, False, False) == "C4"
    assert lp.cellule(6, 6, f12, 12, True, False) == "C4"
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


def test_les_cinq_nulles_sont_construites(gpt2_tokenize):
    """§5, D17 : cinq maillons, cinq nulles — matériel, capture, encodage,
    ordre, statistique."""
    from gate_bench import nulles_du_protocole
    n = nulles_du_protocole(gpt2_tokenize, make_offsets(True))
    assert set(n) == {"1_materiel_AUC_lex", "2_capture_nulle_de_cadre",
                      "3_encodage_l0", "4_ordre_corpus_melange", "5_statistique"}
    assert 0.0 <= n["1_materiel_AUC_lex"]["AUC_lex"] <= 1.0
    assert 0.0 <= n["1_materiel_AUC_lex"]["R1_lex"] <= 1.0
    assert n["2_capture_nulle_de_cadre"]["longueurs_appariees"] is True
    assert n["2_capture_nulle_de_cadre"]["verbe_conserve"] is True
    assert n["4_ordre_corpus_melange"]["multiensembles_apparies"] is True
    assert n["4_ordre_corpus_melange"]["longueurs_appariees"] is True
    assert n["5_statistique"]["plancher_R1_36"] == pytest.approx(1 / 37)


def test_nulle_melangee_conserve_le_multiensemble(gpt2_tokenize, corpus80):
    """Maillon 4 : appariée en multiensemble, longueur et position ; l'ORDRE
    seul est détruit. La clause « permissive » est pré-enregistrée."""
    prompts = [p for tr in corpus80["paraphrases"] for p in tr]
    mel = lp.nulle_melangee(prompts, gpt2_tokenize, seed=0)
    orig = [list(gpt2_tokenize(p)) for p in prompts]
    assert [sorted(x) for x in mel] == [sorted(x) for x in orig]
    assert any(x != y for x, y in zip(mel, orig))
    assert lp.nulle_melangee_permissive(0.45)["permissive"] is True
    assert lp.nulle_melangee_permissive(0.49)["permissive"] is False


def test_v_amont_ne_trouve_aucun_chiffre_de_v3_dans_l_instrument():
    """`V-amont` (D14-R) : aucun chiffre de v3 dans une porte, un seuil ou une
    prédiction. `B-v3` reste descriptif."""
    from gate_bench import gate_v_amont
    src = (Path(__file__).resolve().parents[1] / "eval" /
           "layer_profile.py").read_text(encoding="utf-8")
    assert gate_v_amont(src)[0] == PASS
    assert gate_v_amont("SEUIL = 0.99989")[0] == FAIL
    assert "B-v3" in src and "jamais décisionnel" in src


def test_engram_n_est_pas_importe_par_l_instrument_hors_run():
    """`M` jamais instanciée, `engram/` non modifié."""
    src = (Path(__file__).resolve().parents[1] / "eval" /
           "layer_profile.py").read_text(encoding="utf-8")
    assert "FastWeightMemory" not in src
    assert "backward" not in src and "optim" not in src
    assert src.count("from engram.cortex import _find_blocks") == 1


def test_le_banc_i2_tourne_de_bout_en_bout_et_rapporte_E(tmp_path, gpt2_tokenize):
    """Le banc I2 s'exécute, publie son rapport et son compte `E`.

    Le test n'exige PAS `E = 0` : la valeur d'`E` est le RÉSULTAT du banc, et
    l'amender relève du pré-enregistrement. Il consigne le jeu de clauses en
    cause OBSERVÉ — s'il change, c'est un fait à rapporter, pas à absorber.
    """
    rep = run_i2(use_hf=True, out_dir=tmp_path)
    assert rep["statut_protocole"].startswith("PROPOSE")
    assert rep["couverture"]["pct"] == 100.0
    assert rep["paires"] == {"n_intra": 240, "n_inter": 28_440,
                             "attendus": [240, 28_440]}
    assert rep["corpus"]["a"]["diversite_(owners,entites,verbes)"] == [16, 20, 5]
    assert rep["corpus"]["a"]["recensement_identite"] == REC_80
    assert rep["corpus"]["a_N30"]["recensement_identite"] == REC_30
    assert rep["corpus"]["B-v3"]["diversite_(owners,entites,verbes)"][:2] == [5, 6]
    assert rep["partition_identite"]["clusters"]["P-own"]["K_N80"] == 16
    assert rep["partition_identite"]["clusters"]["P-own"]["K_N30"] == 14
    assert rep["puissance"]["K_requis_R1_36"] == 16
    assert (tmp_path / "report.json").exists()
    en_cause = {r["clause"] for r in rep["clauses"] if r["compte_dans_E"]}
    assert rep["E"] == len(en_cause)
    assert en_cause == set()
