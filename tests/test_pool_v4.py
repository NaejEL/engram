# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests CPU du matériau v4 et de son noyau décisionnel.

Protocole : `experiments/EXP-2026-08-23-v4-materiel.md`.

Aucun modèle, aucun GPU, **aucun tokenizer HF** : les portes qui dépendent du
BPE sont exercées avec un encodeur FEINT (un mot = un token, sensible à la
casse), ce qui suffit pour tester la STRUCTURE. La qualification BPE réelle est
exercée par `eval/pool_v4.py` et par la suite v4 du banc.

Couverture demandée par la méthode : **flag off ⇒ sorties identiques à l'ancien
code ; flag on ⇒ propriété attendue.**
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import materiel_v4 as m4  # noqa: E402
import pool_v4 as p4  # noqa: E402
from engram.config import EngramConfig  # noqa: E402


# ---------------------------------------------------------------- encodeur feint

def _encs_feints():
    """Un mot = un token, sensible à la casse. Ids déterministes et stables."""
    vocab = {}

    def enc(s):
        out = []
        for w in s.split():
            if w not in vocab:
                vocab[w] = 1000 + 7919 * len(vocab) % 40000
            out.append(vocab[w])
        return out

    return {m: (enc, 50000) for m in p4.MODELES_TOK}


@pytest.fixture(scope="module")
def mat():
    return p4.construire(EngramConfig(dataset="pool_v4"), _encs_feints())


# =========================================================================
#  Flag de configuration — défaut = comportement courant, bit à bit
# =========================================================================

def test_dataset_defaut_est_fact_pairs():
    assert EngramConfig().dataset == "fact_pairs"


def test_summary_inchange_au_defaut():
    """Flag OFF ⇒ la ligne de résumé est celle d'avant v4-matériel."""
    s = EngramConfig().summary()
    assert "dataset=" not in s
    assert s.startswith("model=gpt2 layer=6 lam=2.0")


def test_summary_expose_le_flag_quand_il_est_arme():
    assert "dataset=pool_v4" in EngramConfig(dataset="pool_v4").summary()


def test_aucun_defaut_existant_modifie():
    c = EngramConfig()
    assert (c.lam, c.max_read_norm, c.eta, c.decay, c.dg_dim, c.dg_topk,
            c.read_gate, c.surprise_threshold, c.seed) == (
        2.0, 0.5, 0.2, 1e-3, 8192, 64, "keysim", 4.0, 0)


# =========================================================================
#  Structure du matériau
# =========================================================================

def test_cardinaux_de_structure(mat):
    assert len(mat["tiges_pontees"]) == p4.N_TIGES_PONTEES
    assert len(mat["tiges_simples"]) == p4.N_TIGES_SIMPLES
    assert len(mat["unites"]) == p4.N_ENTITES == 72
    assert len(mat["unites_decisionnelles"]) == p4.N_UNITES_DEC == 60
    assert len(mat["unites_reserve"]) == p4.N_UNITES_RESERVE == 12
    assert len(mat["cellules"]) == 6
    assert len(mat["paires_cadre"]) == p4.N_PAIRES_CADRE == 40


def test_chaque_tige_pontee_porte_deux_domaines(mat):
    for t in mat["tiges_pontees"]:
        doms = {f["domaine"] for f in mat["familles"] if f["tige"] == t}
        assert len(doms) == 2


def test_cinq_familles_par_domaine(mat):
    for d in p4.DOMAINES:
        n = sum(1 for f in mat["familles"] if f["domaine"] == d and f["pontee"])
        assert n == p4.N_FAMILLES_PAR_DOMAINE


def test_suffixes_globalement_uniques(mat):
    s = [u["suffixe"] for u in mat["unites"]]
    assert len(set(s)) == len(s)


def test_strates_S3_60_et_S2_90(mat):
    c = p4.compte_strates(mat["unites_decisionnelles"])
    assert c["S3"] == 60 and c["S2"] == 90
    assert sum(c.values()) == 60 * 59 // 2


def test_m1_vaut_5_pour_toutes_les_unites_decisionnelles(mat):
    u = mat["unites"]
    for i, x in enumerate(u):
        if x["pontee"]:
            assert p4.pool_p2(i, u, 0)["m1"] == p4.M1_ATTENDU


def test_composition_gravee_du_pool_P2(mat):
    u = mat["unites"]
    for i, x in enumerate(u):
        if not x["pontee"]:
            continue
        p = p4.pool_p2(i, u, 0)
        assert (len(p["tige_partages"]), len(p["meme_domaine"]),
                len(p["autre_domaine"])) == (5, 12, 19)
        assert len(p["pool"]) == p4.TAILLE_POOL


def test_C5_eligibles_au_moins_60(mat):
    u = mat["unites"]
    assert min(len(p4.eligibles_c5(i, u)) for i in range(len(u))) >= 60


def test_coloration_des_tiges_est_propre(mat):
    couleur = {f["tige"]: f["groupe_de_bandes"] for f in mat["familles"]
               if f["pontee"]}
    for d in p4.DOMAINES:
        cs = [couleur[f["tige"]] for f in mat["familles"]
              if f["pontee"] and f["domaine"] == d]
        assert len(set(cs)) == len(cs) == p4.N_FAMILLES_PAR_DOMAINE


def test_les_deux_familles_dune_tige_partagent_leur_groupe_de_bandes(mat):
    for t in mat["tiges_pontees"]:
        g = {f["groupe_de_bandes"] for f in mat["familles"] if f["tige"] == t}
        assert len(g) == 1


def test_cascade_suit_lordre_grave(mat):
    assert mat["cascade_executee"] == list(p4.CASCADE_GRAVEE[:7])
    assert p4.v_ordre(mat)[0] == "PASS"


def test_cardinal_D24b_a_t_moins_1_vaut_14(mat):
    encs = _encs_feints()
    e = encs["gpt2"][0]
    for cle in mat["cellules"]:
        assert len({p4.tokens_tronques(u, cle, mat, e, -1)
                    for u in mat["unites"]}) == p4.CARD_T1 == 14
        assert len({p4.tokens_tronques(u, cle, mat, e)
                    for u in mat["unites"]}) == 72


def test_ensemble_restreint_dune_tige_a_6_sequences_a_1_token_decart(mat):
    encs = _encs_feints()
    e = encs["gpt2"][0]
    for t in mat["tiges_pontees"]:
        membres = [u for u in mat["unites"] if u["tige"] == t]
        seqs = [p4.tokens_tronques(membres[i], "T1-A", mat, e)
                for i in range(len(membres))]
        assert len(set(seqs)) == 6
        for a in range(6):
            for b in range(a + 1, 6):
                diff = sum(1 for x, y in zip(seqs[a], seqs[b]) if x != y)
                assert diff == 1        # drapeau « contraste minimal : 1 token »


def test_un_seul_type_capitalise(mat):
    caps = [c for c, v in mat["cellules"].items() if v["capitalise"]]
    assert len(caps) == 2                # T3-A et T3-B, soit UN type sur trois
    assert {c.split("-")[0] for c in caps} == {"T3"}


def test_nulle_de_cadre_sans_token_partage(mat):
    mots = [w for p in mat["paires_cadre"] for w in p]
    assert len(set(mots)) == len(mots)
    entites = ({u["tige"].lower() for u in mat["unites"]}
               | {u["suffixe"] for u in mat["unites"]})
    assert not (set(w.lower() for w in mots) & entites)


def test_fact_pairs_echoue_au_contre_exemple():
    fp = p4.soumettre_fact_pairs(_encs_feints())
    assert "FAIL" in (fp["C1"], fp["C2"], fp["S-1"])


def test_pool_py_nest_pas_touche():
    """`eval/pool.py` est GELÉ : `pool_v4` ne fait que le lire."""
    src = (ROOT / "eval" / "pool_v4.py").read_text(encoding="utf-8")
    assert "from pool import fact_pairs" in src
    assert "pool.OWNERS =" not in src and "pool.SECRETS_80 =" not in src


# =========================================================================
#  Noyau décisionnel — partitions, ordres, exclusivité
# =========================================================================

@pytest.mark.parametrize("ic,attendu", [
    ((0.20, 0.60), "N-b"),
    ((-0.05, 0.05), "N-a"),
    ((-0.50, 0.60), "N-ind"),
    ((-0.40, -0.20), "INVALIDE-INSTRUMENT"),
    ((0.01, 0.09), "N-b"),           # 0-68 : N-b ET N-a sans ordre gravé
    ((-0.14, -0.01), "N-a"),         # 0-86 : jamais INVALIDE-INSTRUMENT
])
def test_calibrateur_ordre_grave(ic, attendu):
    assert m4.classe_calibrateur(ic) == attendu


@pytest.mark.parametrize("ic,somme_m,k_sup,attendu", [
    ((0.30, 0.60), 200, 10, "C+"),
    ((-0.60, -0.30), 200, 10, "C−"),
    ((-0.20, 0.20), 200, 10, "C-0"),
    ((0.30, 0.60), 40, 10, "C-ind"),      # famine GLOBALE
    ((0.30, 0.60), 90, 5, "C-ind"),       # famine PARTIELLE concentrée (0-74)
    ((0.02, 0.50), 200, 10, "C-ind"),     # C-ind de RÉSOLUTION (0-89)
    ((0.02, 0.18), 200, 10, "C-0"),       # 0-83 : jamais C+
    ((-0.18, -0.02), 200, 10, "C-0"),     # 0-83 : jamais C−
])
def test_bandes_primaire_ordre_grave(ic, somme_m, k_sup, attendu):
    assert m4.bande_primaire(ic, 0.215, somme_m, k_sup) == attendu


@pytest.mark.parametrize("ic,attendu", [
    ((0.15, 0.40), "+"),
    ((-0.40, -0.15), "−"),
    ((-0.08, 0.08), "0-résolu"),
    ((-0.05, 0.35), "ind"),
    ((0.01, 0.05), "0-résolu"),      # 0-88 : jamais `+` sous ±ε_M
    ((0.20, 0.60), "+"),             # 0-78 : effet FORT, jamais `ind`
])
def test_etats_de_maillon(ic, attendu):
    assert m4.etat_maillon(ic, 0.10) == attendu


def test_espace_ORD_resolu_compte_27_cellules():
    c = m4.espace_ord_resolu()
    assert c == {"ORD-1": 1, "ORD-4": 2, "ORD-2": 6, "ORD-3": 9, "ORD-0": 9,
                 "total": 27}


def test_partition_ORD_est_exhaustive_et_exclusive():
    etats = ("+", "−", "0-résolu", "ind")
    vues = set()
    for m1 in etats:
        for m2 in etats:
            for m3 in etats:
                c = m4.classe_ord(m1, m2, m3)
                assert c in m4.VERDICT_ORD          # exhaustive
                vues.add(c)
    assert vues == set(m4.VERDICT_ORD)              # aucune classe inatteignable


def test_aucun_verdict_ORD_jumeau():
    textes = [v[0] for v in m4.VERDICT_ORD.values()]
    assert len(set(textes)) == len(textes)


def test_routage_ind_domine():
    assert m4.classe_ord("ind", "+", "0-résolu") == "ORD-ind"
    assert m4.classe_ord("+", "+", "ind") == "ORD-ind"


def test_M2_zero_resolu_va_en_ORD_0_pas_ORD_3():
    assert m4.classe_ord("+", "0-résolu", "+") == "ORD-0"
    assert m4.classe_ord("+", "−", "+") == "ORD-3"
    assert m4.VERDICT_ORD["ORD-0"][0] != m4.VERDICT_ORD["ORD-3"][0]


def test_G_bis_licenciee_par_la_seule_cellule_N_b_x_C_ind():
    assert m4.phrase_g_bis_licenciee("N-b", "C-ind")
    for c in ("N-a", "N-ind", "INVALIDE-INSTRUMENT"):
        assert not m4.phrase_g_bis_licenciee(c, "C-ind")


def test_six_causes_de_C_ind_aucune_muette():
    causes = {m4.cause_c_ind(c, 40, 10)["cause"]
              for c in ("N-b", "N-a", "N-ind", "INVALIDE-INSTRUMENT")}
    causes.add(m4.cause_c_ind("N-b", 90, 5)["cause"])       # famine partielle
    causes.add(m4.cause_c_ind("N-b", 200, 10)["cause"])     # résolution
    assert len(causes) == 6
    for c in causes:
        assert c and isinstance(c, str)


# =========================================================================
#  Enveloppes nulles, barrière, portes de mesure
# =========================================================================

def test_epsilon_sous_le_plafond_derive():
    r = m4.eps_nulle_composition([3] * 20 + [5] * 20 + [7] * 20,
                                 [f"T{i % 10}" for i in range(60)],
                                 b=2000, seed=0)
    assert 0 < r["epsilon"] <= m4.EPS_MAX
    assert r["sous_plafond"]


def test_epsilon_est_reproductible_a_seed_egal():
    a = m4.eps_nulle_composition([4] * 60, ["T0"] * 60, b=500, seed=0)
    b = m4.eps_nulle_composition([4] * 60, ["T0"] * 60, b=500, seed=0)
    assert a["epsilon"] == b["epsilon"]


def test_D_est_une_difference_observe_attendu():
    r = m4.d_observe([2, 0, 3], [10, 0, 12])
    assert r["n_eff"] == 2                       # E3 : m = 0 exclu, PRÉ-DÉCLARÉ
    assert r["D"] == pytest.approx(2 - 10 * 5 / 36 + 3 - 12 * 5 / 36)


def test_barriere_refuse_publier_avant_sceller():
    b = m4.Barriere()
    with pytest.raises(RuntimeError):
        b.publier(0.2)


def test_barriere_ordre_conforme():
    b = m4.Barriere()
    b.sceller({"D": 1.0})
    b.publier(0.215)
    b.desceller()
    assert b.ordre() == ["sceller", "publier", "desceller"]


def test_v_compo_refuse_un_epsilon_au_dessus_du_plafond():
    assert m4.v_compo(["sceller", "publier", "desceller"], 200, True, 0.75)[0] \
        == "FAIL"
    assert m4.v_compo(["sceller", "publier", "desceller"], 200, True, 0.215)[0] \
        == "PASS"


def test_v_leak_mord_sur_B0_prime():
    assert m4.v_leak((-0.2, -0.05), (-0.03, 0.03))[0] == "PASS"
    assert m4.v_leak((0.02, 0.15), (-0.03, 0.03))[0] == "FAIL"
    assert m4.v_leak((-0.2, -0.05), (0.04, 0.12))[0] == "FAIL"


def test_v_dtype_unite_est_la_tige():
    assert m4.v_dtype({"Iron": 0.001}, {"Iron": 0.3}, 0.004)[0] == "PASS"
    v, d = m4.v_dtype({"Iron": 0.001, "North": 0.001},
                      {"Iron": 0.3, "North": 0.6}, 0.004)
    assert v == "INCONCLUSIF-précision" and d["n_tiges_touchees"] == 2


def test_v_surprise_retire_M1_si_le_signe_nest_que_dans_la_bande_haute():
    assert m4.v_surprise([1, 1, 1])[0] == "PASS"
    v, d = m4.v_surprise([0, 0, 1])
    assert v == "FAIL" and d["M1_retiree"]


def test_v_subst_interdit_la_substitution_de_famille_pontee():
    assert m4.v_subst([{"niveau": "unite"}], 10)[0] == "PASS"
    assert m4.v_subst([{"niveau": "unite"}], 9)[0] == "PASS"
    assert m4.v_subst([{"niveau": "unite"}], 8)[0] == "FAIL"
    assert m4.v_subst([{"niveau": "famille", "pontee": True}], 10)[0] == "FAIL"


def test_v_joint_interdit_le_produit_de_p_valeurs():
    assert m4.v_joint_produit_de_p([0.3, 0.3, 0.3])[0] == "FAIL"


def test_v_joint_un_seul_reechantillonnage_de_tiges():
    series = {j: [float(i) for i in range(10)] for j in ("a", "b", "c")}
    v, d = m4.v_joint(series, b=200, seed=0)
    assert v == "PASS" and len(d["IC_simultanes"]) == 3


def test_portes_de_schema_refusent_un_champ_decisionnel():
    sch = m4.schema_de_sortie("N-b", {d: "36/37" for d in p4.DOMAINES})
    assert m4.v_calib(sch)[0] == "PASS"
    assert m4.v_calib(dict(sch, verdict_hypothese="retenu"))[0] == "FAIL"
    assert m4.v_bindur({"bin_dur_statut": "DESCRIPTIF"})[0] == "PASS"
    assert m4.v_bindur({"bin_dur_statut": "DESCRIPTIF",
                        "bin_dur_verdict": "x"})[0] == "FAIL"


def test_v_plafond_refuse_un_plancher_poole():
    sch = m4.schema_de_sortie("N-a", {d: "36/37" for d in p4.DOMAINES})
    assert m4.v_plafond(sch)[0] == "PASS"
    assert m4.v_plafond(dict(sch, plancher_par_domaine="poolé"))[0] == "FAIL"


def test_v_perimetre_exige_les_trois_elements():
    b = m4.bloc_perimetre()
    assert m4.v_perimetre(b)[0] == "PASS"
    for k in b:
        assert m4.v_perimetre({x: y for x, y in b.items() if x != k})[0] == "FAIL"


def test_v_t1_interdit_toute_statistique_intra_tige():
    assert m4.v_t1([{"nom": "inter", "intra_tige": False}])[0] == "PASS"
    assert m4.v_t1([{"nom": "intra", "intra_tige": True}])[0] == "FAIL"


def test_phrases_gravees_ne_declenchent_pas_le_vocabulaire_interdit():
    """Elles NOMMENT la limite ; elles ne l'emploient pas comme évidence."""
    for ph in m4._PHRASES_GRAVEES:
        assert m4._contient_terme_interdit(ph) == []
    assert m4._contient_terme_interdit("une séparation de patterns")


def test_permutation_intra_tige_C_6_3_vaut_20():
    assert m4.permutations_intra_tige() == 20


def test_probabilites_sous_la_nulle_reproduisent_le_protocole():
    p = m4.probas_sous_nulle(50_000, 0)
    assert 0.90 <= p["bandes_primaire"]["C-0"] <= 0.95      # 92.5-93.2 % audité
    assert p["calibrateur"].get("INVALIDE-INSTRUMENT", 0.0) < 1e-3   # α ≈ 2e-4
    assert 0.75 <= p["classes_ORD"]["ORD-0"] <= 0.85        # 80.9 % audité


# =========================================================================
#  Correctifs de l'itération 2 (issues 1 à 6 du lab-verifier).
#  D29 : un correctif est une écriture — il est testé comme du code neuf.
# =========================================================================

import numpy as np  # noqa: E402


# ---- issue 1 : `δ̂` doit être MESURÉ, jamais une constante en dur -----------

def test_v_dtype_refuse_un_delta_non_mesure():
    tete = {f"T{i}": 0.5 for i in range(10)}
    coupe = {f"T{i}": 0.4 for i in range(10)}
    v, d = m4.v_dtype(tete, coupe, 2 ** -8, delta_mesure=False)
    assert v == "FAIL"
    assert "MESUR" in d["raison"]


def test_v_dtype_accepte_un_delta_mesure_et_publie_la_mesure():
    tete = {f"T{i}": 0.5 for i in range(10)}
    coupe = {f"T{i}": 0.4 for i in range(10)}
    v, d = m4.v_dtype(tete, coupe, 0.0033, delta_mesure=True,
                      detail_mesure={"delta_chapeau_mesure": 0.0033})
    assert v == "PASS" and d["delta_mesure"] is True
    assert d["mesure_de_delta"]["delta_chapeau_mesure"] == 0.0033


def test_v_dtype_unite_tige_inchangee_sous_delta_mesure():
    v, d = m4.v_dtype({"Iron": 0.001, "North": 0.001},
                      {"Iron": 0.3, "North": 0.6}, 0.004, delta_mesure=True)
    assert v == "INCONCLUSIF-précision" and d["n_tiges_touchees"] == 2


# ---- issue 3 : `V-pool` et `V-t1` lisent l'exécution, n'affirment rien ------

def test_v_pool_detecte_un_rang_calcule_dans_un_bootstrap():
    m4.reinitialiser_instrumentation()
    with m4._SousBootstrap():
        m4._r1_mid_rank([0.1, 0.9, 0.2], 1)
    v, d = m4.v_pool("h", "h", instr=m4.instrumentation())
    m4.reinitialiser_instrumentation()
    assert v == "FAIL" and d["rangs_recalcules_dans_un_bootstrap"] is True
    assert d["source"] == "instrumentation d'exécution"


def test_v_pool_passe_quand_les_rangs_sont_hors_bootstrap():
    m4.reinitialiser_instrumentation()
    m4._r1_mid_rank([0.1, 0.9, 0.2], 1)
    v, d = m4.v_pool("h", "h", instr=m4.instrumentation())
    m4.reinitialiser_instrumentation()
    assert v == "PASS" and d["rangs_calcules"] == 1
    assert d["rangs_calcules_dans_un_bootstrap"] == 0


def test_bootstrap_tiges_est_marque_comme_reechantillonnage():
    m4.reinitialiser_instrumentation()
    m4.bootstrap_tiges([1.0, 2.0, 3.0, 4.0], b=5, seed=0)
    assert m4.instrumentation()["profondeur_bootstrap"] == 0   # bien refermé
    m4.reinitialiser_instrumentation()


def test_registre_t1_est_evalue_sur_les_paires_comparees():
    m4.reinitialiser_instrumentation()
    q = m4.noter_quantite_t1("inter", [(0, 1), (0, 2)],
                             {0: "Iron", 1: "Silver", 2: "North"})
    assert q["intra_tige"] is False and q["n_paires_comparees"] == 2
    assert m4.v_t1(instr=m4.instrumentation())[0] == "PASS"
    m4.noter_quantite_t1("intra", [(0, 1)], {0: "Iron", 1: "Iron"})
    v, d = m4.v_t1(instr=m4.instrumentation())
    assert v == "FAIL" and len(d["fautes"]) == 1
    m4.reinitialiser_instrumentation()


def test_un_ensemble_peut_contenir_une_tige_sans_la_comparer():
    """La granularité est la PAIRE comparée, pas l'ensemble d'indices."""
    m4.reinitialiser_instrumentation()
    q = m4.noter_quantite_t1("inter", [(0, 2)], {0: "Iron", 1: "Iron", 2: "North"})
    assert q["intra_tige"] is False
    m4.reinitialiser_instrumentation()


# ---- issue 4 : le défaut de `b_perm` est la valeur GRAVÉE ------------------

def test_defaut_b_perm_est_la_valeur_gravee():
    import inspect
    assert inspect.signature(m4.run_mesure).parameters["b_perm"].default == m4.B_MC
    assert m4.B_MC == 10_000


# ---- issue 5 : l'enveloppe de `B0′` vient de sa PROPRE nulle ---------------

def test_enveloppe_dune_somme_nest_pas_la_moyenne_des_enveloppes():
    g = np.random.default_rng(21)
    a, b = g.normal(0, 1, 20_000), g.normal(0, 1, 20_000)
    e1 = m4.eps_m_permutation(a)["epsilon_M"]
    e2 = m4.eps_m_permutation(b)["epsilon_M"]
    ep = m4.eps_m_permutation(a + b)["epsilon_M"]
    assert ep > 1.2 * (e1 + e2) / 2          # ≈ √2 sous indépendance


# ---- issue 2 : bandes de NLL de `V-surprise` -------------------------------

def test_bandes_nll_trois_bandes_non_vides(mat):
    g = np.random.default_rng(3)
    n = len(mat["unites"]) + len(mat["paires_cadre"]) + len(mat["pseudo_mots"])
    nll = {c: g.uniform(8.0, 15.0, n) for c in mat["cellules"]}
    bn = m4.bandes_nll(mat, nll)
    assert len(bn["n_paires_par_bande"]) == 3
    assert all(x > 0 for x in bn["n_paires_par_bande"])
    assert sum(bn["n_paires_par_bande"]) == 60 * 59 // 2
    assert len(bn["coupures_tertiles"]) == 2


def test_bandes_nll_degenerees_sont_detectables(mat):
    n = len(mat["unites"]) + len(mat["paires_cadre"]) + len(mat["pseudo_mots"])
    nll = {c: np.full(n, 7.0) for c in mat["cellules"]}
    bn = m4.bandes_nll(mat, nll)
    assert sum(1 for x in bn["n_paires_par_bande"] if x > 0) == 1


def test_unite_de_bande_est_la_paire(mat):
    """La NLL de paire est la MOYENNE des deux membres (§4.7)."""
    n = len(mat["unites"]) + len(mat["paires_cadre"]) + len(mat["pseudo_mots"])
    nll = {c: np.arange(n, dtype=float) for c in mat["cellules"]}
    bn = m4.bandes_nll(mat, nll)
    assert bn["nll_par_paire"][0] == (bn["nll_par_unite"][0]
                                      + bn["nll_par_unite"][1]) / 2


# ---- issue 6 : descriptifs §4.8, `G` gelée, D8/D9 intactes ----------------

def test_A3_utilise_les_hyperparametres_de_EngramConfig():
    src = (ROOT / "eval" / "materiel_v4.py").read_text(encoding="utf-8")
    i = src.index("def descriptif_A3(")
    corps = src[i:src.index("def descriptif_A4_t1(")]
    assert "cfg.dg_dim" in corps and "cfg.dg_topk" in corps and "cfg.seed" in corps
    assert "8192" not in corps and "backward" not in corps


def test_A4_est_inter_tige_seulement():
    src = (ROOT / "eval" / "materiel_v4.py").read_text(encoding="utf-8")
    i = src.index("def descriptif_A4_t1(")
    corps = src[i:src.index("def descriptif_P1_moins_P2(")]
    assert 'dec[a]["tige"] != dec[b]["tige"]' in corps
    assert "noter_quantite_t1" in corps


def test_P1_moins_P2_lecture_asymetrique():
    src = (ROOT / "eval" / "materiel_v4.py").read_text(encoding="utf-8")
    i = src.index("def descriptif_P1_moins_P2(")
    corps = src[i:i + 4000]
    assert "NON INFORMATIVE" in corps and "strictement négative" in corps


def test_aucun_backprop_dans_les_correctifs():
    src = (ROOT / "eval" / "materiel_v4.py").read_text(encoding="utf-8")
    for interdit in ("loss.backward", ".backward()", "torch.optim", "requires_grad_(True)"):
        assert interdit not in src
