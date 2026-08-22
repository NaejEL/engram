# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests CPU du banc de satisfiabilité (D14-S) — V2-D(a) v3.

Protocole : `experiments/EXP-2026-08-22-knn-borne-logits-v3.md`, §10.
Aucun modèle. Le seul HF touché est le **tokenizer GPT-2** (CPU, en cache), et
seulement là où la porte n'est pas décidable sans lui (V-tok, V-para réelles).

Les quatorze tests du §10 au-delà des douze de v2 :

  (xiii)  V-indep détecte des clés dupliquées sur 30 ;
  (xiv)   V-para détecte un indice sous-chaîne **et une fuite croisée**, paire nommée ;
  (xv)    V-bord détecte `−log(1−λ)` vs `−log1p(−λ)` ;
  (xvi)   V1b-1 passe à `r = 1e-30` ;
  (xvii)  V1b-2 range `r = 1e-30` dans le complément ;
  (xviii) G rend NON ÉVALUABLE si `P1(λ*) = 0` **et** si aucun τ n'est admissible ;
  (xix)   `τ_promu` est le plus grand τ du **PRÉFIXE CONNEXE** — grille non connexe ;
  (xx)    table `k(n)` exacte contre référence, n = 5..30, **16 / 23 / 30 obligatoires** ;
  (xxi)   (c) bascule sur `m` quand les `pct` saturent ;
  (xxii)  repli Clopper-Pearson quand h = 0 partout ;
  (xxiii) `borne_marge = 0` ne produit pas de NaN ;
  (xxiv)  **V-tok échoue sur le jeu réel AVANT correction (`lighthouse`) et passe après** ;
  (xxv)   **`P5f-borne` passe sur une bascule portée par la valeur du voisin 5** ;
  (xxvi)  **P3 refuse un B pair**.
"""

import math
import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from gate_bench import (
    ARME, BUG, FAIL, K_BOUNDARIES, K_TABLE_REFERENCE, NON_ARME, NON_EVALUABLE,
    PASS, P10_FEASIBLE, REFUS, _deltas_from_r, _p5f_neighbor5_case, _p5f_tie_case,
    borne_marge, bord, bord_naif, gate_v1b1, gate_v1b2, gate_v_bord, gate_v_indep,
    gate_v_para, gate_v_tok, k_of_n, k_tail_ratio, margins_over_vk,
    margins_over_vnn, multikey_clause_c, tau_promu, ulp_gap, verdict_dp6,
    verdict_g, verdict_multikey, verdict_p1, verdict_p3, verdict_p5f_borne,
)
from knn_ceiling import LAMBDA_STAR
from pool import (
    ENTITIES, OWNERS, OWNER_OBJ, POOL_PARAPHRASES, POOL_UNIT_SECRETS,
    SECRETS_80, VERBS, unit_table,
)


@pytest.fixture(scope="module")
def gpt2_tokenize():
    """Tokenizer GPT-2 seul (CPU, cache local). Aucun modèle n'est chargé."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    return lambda s: tok.encode(s)


def _toy_tokenize(text):
    """Repli mot-à-mot déterministe pour les cas synthétiques."""
    return [abs(hash(w)) % 100003
            for w in text.replace(".", " .").replace("?", " ?").split()]


# ------------------------------------------------------------------ (xiii)

def test_xiii_v_indep_detects_duplicated_keys_over_30():
    """C4 du run 2 : les 30 unités partagent la même clé. V-indep (a) doit le
    voir ; et sd = 0 sur d²_min ou p₁₀ doit tomber sur (b)/(c)."""
    g = np.random.default_rng(0)
    qk = g.standard_normal((30, 4, 32)).astype(np.float32)
    d2m = g.uniform(0.1, 5.0, 30)
    p10 = g.uniform(1e-3, 5e-2, 30)
    v, det = gate_v_indep(qk, d2m, p10)
    assert v == PASS and det["n_distinctes"] == 120

    v, det = gate_v_indep(np.repeat(qk[0:1], 30, axis=0), d2m, p10)
    assert v == FAIL and det["clause_echouee"] == "(a)"
    assert det["n_distinctes"] == 4                  # 4 clés pour 120 lignes

    # une SEULE paire dupliquée suffit
    dup = qk.copy()
    dup[7, 2] = dup[0, 0]
    assert gate_v_indep(dup, d2m, p10)[0] == FAIL

    assert gate_v_indep(qk, np.full(30, 2.0), p10)[1]["clause_echouee"] == "(b)"
    assert gate_v_indep(qk, d2m, np.full(30, 0.01))[1]["clause_echouee"] == "(c)"
    # (d) est DESCRIPTIF : rapporté, jamais bloquant
    assert -1.0 <= gate_v_indep(qk, d2m, p10)[1]["d_cos_max_inter_unites"] <= 1.0


# ------------------------------------------------------------------- (xiv)

def _syn_units(cross_leak=False, substring=False):
    out = []
    fam = [("alpha", "bravo"), ("charlie", "delta"), ("echo", "foxtrot")]
    for i in range(6):
        w1, w2 = fam[i % 3]
        tag = f"u{i}"
        fact = f"{tag} {w1} {w2} zzz{i} qqq{i} {{secret}}."
        exact = f"{tag} {w1} {w2} zzz{i} qqq{i}"
        paras = [f"{tag} {w1} zzz{i}", f"{tag} {w2} qqq{i}", f"{tag} zzz{i}"]
        if substring:
            paras[0] = exact
        if cross_leak:
            j = (i + 1) % 6
            paras[0] = f"u{j} {fam[j % 3][0]} {fam[j % 3][1]} zzz{j} qqq{j}"
        out.append({"i": i, "fact_template": fact,
                    "fact_no_secret": fact.replace(" {secret}", ""),
                    "exact": exact, "paraphrases": paras})
    return out


def test_xiv_v_para_detects_substring_and_cross_leak_with_named_pair():
    """(a) un indice sous-chaîne de son fait ⇒ FAIL ; (c) une paraphrase de
    l'unité *i* recouvrant le fait de *j ≠ i* plus que le sien ⇒ FAIL croisé,
    **paire (i, j) nommée**."""
    v, det = gate_v_para(_syn_units(), _toy_tokenize)
    assert v == PASS, det
    assert len(det["matrice_30x4"]) == 6 and len(det["matrice_30x4"][0]) == 4

    v, det = gate_v_para(_syn_units(substring=True), _toy_tokenize)
    assert v == FAIL
    assert det["a_sous_chaine"] and det["a_sous_chaine"][0]["para"] == 1

    v, det = gate_v_para(_syn_units(cross_leak=True), _toy_tokenize)
    assert v == FAIL
    assert det["c_fuite_croisee_n"] > 0
    paires = {tuple(x["paire"]) for x in det["c_fuite_croisee"]}
    assert (0, 1) in paires                          # la paire est NOMMÉE
    for x in det["c_fuite_croisee"]:
        assert x["J_vers_j"] > x["J_vers_i"]


def test_xiv_v_para_on_the_real_frozen_dataset(gpt2_tokenize):
    """La porte appliquée aux **données gelées du §7**. Le résultat est un
    CONSTAT du banc, pas une correction : V-para (c) telle qu'écrite n'est pas
    satisfaite par `POOL_PARAPHRASES`."""
    v, det = gate_v_para(unit_table(30), gpt2_tokenize)
    assert det["a_sous_chaine"] == []                # (a) tenue
    assert det["b_jaccard"] == []                    # (b) tenue
    assert v == FAIL and det["c_fuite_croisee_n"] > 0
    assert det["c_par_type_de_paraphrase"]["para2"] == 0   # para2 seule est nette


# -------------------------------------------------------------------- (xv)

def test_xv_v_bord_detects_log_versus_log1p():
    """`bord := fl(−log1p(−λ))`. La forme proscrite `−log(1−λ)` doit être
    détectée bit-à-bit, avec l'écart en ULP exhibé."""
    v, gaps = gate_v_bord([0.02, LAMBDA_STAR, 0.05, 0.10, 0.25], False)
    assert v == PASS
    assert all(g["ulp"] == 0 for g in gaps.values())

    v, gaps = gate_v_bord([0.02, LAMBDA_STAR, 0.05, 0.10, 0.25], True)
    assert v == FAIL
    assert gaps[repr(0.02)]["ulp"] == 5              # écart exhibé, en ULP
    assert gaps[repr(0.05)]["ulp"] == 6
    assert gaps[repr(0.10)]["ulp"] == 2

    # CONSTAT du banc : au SEUL λ*, les deux expressions sont bit-identiques —
    # la porte y est vacuée par satisfaction. Verrouillé ici pour qu'un
    # changement silencieux de λ* ou de libm le fasse échouer.
    assert bord(LAMBDA_STAR) == bord_naif(LAMBDA_STAR)
    assert ulp_gap(bord(LAMBDA_STAR), bord_naif(LAMBDA_STAR)) == 0
    assert gate_v_bord([LAMBDA_STAR], True)[0] == PASS


# ---------------------------------------------------------- (xvi) / (xvii)

def test_xvi_v1b1_passes_at_r_1e_minus_30():
    """LA cellule qui a rendu la porte du run 2 insatisfiable en fp64 : le
    décrément `δ̂ = log1p(λr/(1−λ))` passe sous l'ULP du bord, donc
    `ΔNLL == bord` **exactement** — et `≤ bord + 4·ULP` doit PASSER."""
    d, dh, b = _deltas_from_r([1e-30] * 8, LAMBDA_STAR)
    assert all(x == b for x in d)                    # égalité EXACTE, pas ~
    assert all(0.0 < x < math.ulp(b) for x in dh)    # δ̂ > 0 mais sous l'ULP
    v, det = gate_v1b1(d, LAMBDA_STAR)
    assert v == PASS and det["n_violations"] == 0
    # ... y compris en tout régime mélangé
    assert gate_v1b1(_deltas_from_r([1e-30, 1e-16, 1e-8, 1e-3], LAMBDA_STAR)[0],
                     LAMBDA_STAR)[0] == PASS
    # le contre-exemple échouant existe bien
    bad = d.copy()
    bad[0] = b + 1e-12
    assert gate_v1b1(bad, LAMBDA_STAR)[0] == FAIL


def test_xvii_v1b2_files_r_1e_minus_30_in_the_complement():
    """Volet strict RESTREINT à `{δ̂ ≥ 8·ULP}` : à r = 1e-30 le sous-ensemble
    strict est VIDE, tout tombe dans le complément — **compte non nul attendu,
    ce n'est pas un échec**."""
    d, dh, b = _deltas_from_r([1e-30] * 16, LAMBDA_STAR)
    v, det = gate_v1b2(d, dh, LAMBDA_STAR)
    assert v == "COMPLEMENT (compte=16)"
    assert det["n_strict"] == 0 and det["n_complement"] == 16
    assert det["n_violations_strict"] == 0 and det["n_violations_complement"] == 0

    d3, dh3, _ = _deltas_from_r([1e-3] * 16, LAMBDA_STAR)
    assert all(x >= 8 * math.ulp(b) for x in dh3)
    assert gate_v1b2(d3, dh3, LAMBDA_STAR)[0] == PASS
    assert all(x < b for x in d3)                    # strictement sous le bord
    forced = d3.copy()
    forced[0] = b
    assert gate_v1b2(forced, dh3, LAMBDA_STAR)[0] == FAIL


# ----------------------------------------------------------------- (xviii)

def test_xviii_g_is_non_evaluable_without_p1_and_without_admissible_tau():
    e3 = [0.01, 0.02, 0.03, 0.035, 0.04, 0.042, 0.045, 0.06, 0.07, 0.08]
    p1 = [1, 2, 6, 6, 6, 6, 6, 6, 6, 6]
    # P1(λ*) = 0 ET P1(τ_promu) = 0
    assert verdict_g(e3, [0] * 10, 18, 12, 0)[0] == NON_EVALUABLE
    # aucun τ admissible : le 1ᵉʳ décile viole déjà
    assert verdict_g([0.2] + e3[1:], p1, 18, 12, 3)[0] == NON_EVALUABLE
    assert tau_promu([0.2] + e3[1:]) is None
    # non-vacuité : #{α=0} = 0 ⇒ τ n'est même pas évalué
    assert verdict_g(e3, p1, 30, 0, 3)[0] == "TAU NON ÉVALUÉ"
    assert verdict_g(e3, p1, 0, 30, 3)[0] == "TAU NON ÉVALUÉ"


# ------------------------------------------------------------------- (xix)

def test_xix_tau_promu_is_the_connected_admissible_prefix():
    """E-D7. `E3(τ)` n'est PAS monotone (le soulagement T2 donne `δ(p) < 0`) :
    l'ensemble admissible peut être **non connexe**. `τ_promu` est le plus grand
    τ du PRÉFIXE CONNEXE — jamais un τ situé après une violation."""
    grille_nc = [0.01, 0.02, 0.03, 0.20, 0.04, 0.041, 0.042, 0.043, 0.044, 0.30]
    #             1     2     3    VIOL   5      6      7      8      9    VIOL
    assert tau_promu(grille_nc) == 2                 # 0-based ⇒ 3ᵉ décile
    p1 = [1, 2, 6, 6, 6, 6, 6, 6, 6, 6]
    v, det = verdict_g(grille_nc, p1, 18, 12, 3)
    assert det["tau_promu_index"] == 3               # le 3ᵉ
    assert det["tau_promu_index"] != 9               # JAMAIS le 9ᵉ
    assert v == "PROMOTION ARMÉE (tau=idx3)"
    # la règle « le plus grand τ admissible » (fausse) rendrait le 9ᵉ
    naif = max(i for i, e in enumerate(grille_nc) if e < 0.05)
    assert naif == 8 and naif != tau_promu(grille_nc)
    # sous monotonie, la règle coïncide avec le sup
    monotone = [0.01, 0.02, 0.03, 0.04, 0.045, 0.06, 0.07, 0.08, 0.09, 0.10]
    assert tau_promu(monotone) == max(i for i, e in enumerate(monotone) if e < 0.05)


# -------------------------------------------------------------------- (xx)

def test_xx_k_table_is_exact_in_integers_against_the_reference():
    """`k(n)` recalculée en **entiers Python purs** (`math.comb`, `10·Σ ≤ 2ⁿ`),
    comparée à la table de référence du §3 pour n = 5..30. Les **trois
    frontières n = 16, 23, 30 sont obligatoires** : elles basculent à la 3ᵉ
    décimale et une implémentation en flottant ou avec `<` au lieu de `≤` les
    casserait."""
    for n, k in K_TABLE_REFERENCE.items():
        assert k_of_n(n) == k, (n, k_of_n(n), k)
    assert k_of_n(5) == 5

    # propriété DÉFINISSANTE re-vérifiée indépendamment de la table (n = 5..30)
    for n in range(5, 31):
        k = k_of_n(n)
        assert 10 * sum(comb(n, j) for j in range(k, n + 1)) <= 2 ** n
        assert 10 * sum(comb(n, j) for j in range(k - 1, n + 1)) > 2 ** n

    # les trois frontières, à la 3ᵉ décimale
    assert k_of_n(16) == 12 and k_of_n(23) == 16 and k_of_n(30) == 20
    for n, expected in K_BOUNDARIES.items():
        assert k_tail_ratio(n, k_of_n(n) - 1) == pytest.approx(expected, abs=5e-6)
    assert sum(comb(30, j) for j in range(19, 31)) == 107636402
    assert sum(comb(30, j) for j in range(20, 31)) == 53009102

    # aucun flottant dans la règle : k_of_n ne manipule que des entiers
    assert isinstance(k_of_n(30), int)
    # `<` au lieu de `≤` casserait n = 30 (0.049369 ≤ 0.10 est une égalité large
    # sur l'ENTIER 10·Σ ≤ 2ⁿ pour d'autres n) — on verrouille la table entière
    assert [k_of_n(n) for n in range(6, 31)] == \
        [K_TABLE_REFERENCE[n] for n in range(6, 31)]

    # et la règle branchée sur ΔP6
    assert verdict_dp6(10, 9)[0] == ARME and k_of_n(10) == 8
    assert verdict_dp6(10, 6)[0] == NON_ARME
    assert verdict_dp6(4, 4)[0] == NON_EVALUABLE     # n_disc < 5
    assert verdict_dp6(5, 5)[0] == ARME              # n_disc = 5 ⇒ unanimité
    assert verdict_dp6(5, 4)[0] == NON_ARME


# ------------------------------------------------------------------- (xxi)

def test_xxi_clause_c_falls_back_on_the_continuous_margin_when_pct_saturate():
    """Saturation : les deux `pct` à 0 (les médianes dépassent le max des 30 000
    clés) ⇒ (c) se décide sur la **marge continue `m`** — et surtout **PAS** sur
    un NON ÉVALUABLE."""
    ok, voie = multikey_clause_c([0.0] * 30, [0.0] * 30, [-0.2] * 30, [-0.1] * 30)
    assert ok is True and voie == "m"
    ko, voie = multikey_clause_c([0.0] * 30, [0.0] * 30, [-0.1] * 30, [-0.2] * 30)
    assert ko is False and voie == "m"
    # non saturé ⇒ la décision reste en pct-espace
    assert multikey_clause_c([1e-4] * 30, [5e-3] * 30, [0.0] * 30, [0.0] * 30) \
        == (True, "pct")
    # `m` négative partout est sans conséquence : comparaison d'ORDRE
    assert multikey_clause_c([0.0] * 30, [0.0] * 30,
                             [-5.0] * 30, [-4.9] * 30)[0] is True

    a = [True] * 20 + [False] * 10
    b = [True] * 19 + [False] * 11
    v, det = verdict_multikey(a, b, [0.0] * 30, [0.0] * 30, [-0.2] * 30, [-0.1] * 30)
    assert v == ARME and det["voie_de_decision_c"] == "m"
    assert v != NON_EVALUABLE
    # seule une divergence globale/stratifiée rend NON ÉVALUABLE
    assert verdict_multikey(a, b, [1e-4] * 30, [5e-3] * 30, [-0.2] * 30,
                            [-0.1] * 30,
                            strat_qualitatif_identique=False)[0] == NON_EVALUABLE


# ------------------------------------------------------------------ (xxii)

def clopper_pearson_upper(k, n, alpha=0.05):
    """Borne binomiale exacte unilatérale (repli quand h = 0 partout) : plus
    grand p tel que P(X ≤ k | Bin(n, p)) ≥ alpha. À k = 0 : `1 − alpha^(1/n)`."""
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if tail > alpha else (lo, mid)
    return (lo + hi) / 2


def test_xxii_clopper_pearson_fallback_when_h_is_zero_everywhere():
    """h = 0 partout ⇒ le bootstrap est dégénéré (toutes les répliques valent 0).
    Le repli pré-déclaré est une borne exacte sur les **90 couples groupés par
    unité**. h n'est plus décisionnelle : le repli ne décide rien."""
    h = [0.0] * 30
    draws = np.median(np.zeros((1000, 30)), axis=1)
    assert draws.min() == draws.max() == 0.0          # bootstrap dégénéré
    up = clopper_pearson_upper(0, 90)
    assert 0.0 < up < 0.05
    assert up == pytest.approx(1.0 - 0.05 ** (1 / 90), rel=1e-9)
    # la borne au niveau UNITÉ est plus large que la borne au niveau couple
    assert clopper_pearson_upper(0, 30) > up
    assert max(h) == 0.0


# ----------------------------------------------------------------- (xxiii)

def test_xxiii_borne_marge_zero_produces_no_nan():
    """`borne_marge = 0` ⇒ la clause exige **0 bascule** : satisfiable, et sans
    NaN (l'inégalité est absolue, pas un rapport)."""
    rows = np.tile(np.concatenate([[0.9], np.full(11, 0.1 / 11)]), (4, 1))
    vk = [[1, 2, 3]] * 4
    m = margins_over_vk(rows, vk)
    b = borne_marge(m)
    assert b == 0.0 and not math.isnan(b)
    assert verdict_p5f_borne(0.0, 0.0) == PASS       # 0 ≤ 0
    assert verdict_p5f_borne(1e-9, 0.0) == BUG       # la moindre bascule = bug
    assert not math.isnan(verdict_p5f_borne(0.0, 0.0) == PASS)


# ------------------------------------------------------------------ (xxiv)

def test_xxiv_v_tok_fails_before_the_correction_and_passes_after(gpt2_tokenize):
    """Le motif : `SECRETS_80[5] == ENTITIES[16] == "lighthouse"`. AVANT
    correction, V-tok doit ÉCHOUER en **nommant l'unité 5** ; APRÈS substitution
    par `SECRETS_80[30] = "walrus"` (le premier j ≥ 30 qui passe V-tok), elle
    doit PASSER. `SECRETS_80` n'est pas modifié (il gèle X9)."""
    assert SECRETS_80[5] == ENTITIES[16] == "lighthouse"
    assert SECRETS_80[30] == "walrus"

    avant = list(SECRETS_80[:30])
    v, det = gate_v_tok(avant, gpt2_tokenize)
    assert v == FAIL
    assert det["unites_fautives"] == [5]              # l'unité 5 est NOMMÉE
    assert det["b_collisions_pool"][5]["secret"] == "lighthouse"
    assert "lighthouse" in det["b_collisions_pool"][5]["mots_du_pool"]
    assert det["c_secret_dans_pool"][5] == "lighthouse"
    assert det["a_doublons"] == {}                    # (a) tenue même avant

    apres = list(POOL_UNIT_SECRETS)
    assert apres[5] == "walrus" and apres[:5] == avant[:5] and apres[6:] == avant[6:]
    v, det = gate_v_tok(apres, gpt2_tokenize)
    assert v == PASS and det["unites_fautives"] == []

    # la règle de substitution est DÉTERMINISTE : le premier j ≥ 30 qui passe
    for j in range(30, 35):
        cand = list(avant)
        cand[5] = SECRETS_80[j]
        if gate_v_tok(cand, gpt2_tokenize)[0] == PASS:
            assert SECRETS_80[j] == "walrus"
            break
    else:
        pytest.fail("aucun j ≥ 30 ne passe V-tok")

    # (a) : deux secrets à 1ᵉʳ token identique
    dup = list(apres)
    dup[9] = "walruses"
    v, det = gate_v_tok(dup, gpt2_tokenize)
    assert v == FAIL and 9 in det["a_doublons"]

    # la suspicion `catapult` / `cathedral` est INFIRMÉE (§C.2, vérifiée ici)
    assert gpt2_tokenize(" catapult")[0] != gpt2_tokenize(" cathedral")[0]
    assert gpt2_tokenize(" catapult")[0] != gpt2_tokenize(" cat")[0]


# ------------------------------------------------------------------- (xxv)

def test_xxv_p5f_borne_passes_on_a_flip_carried_by_the_fifth_neighbor():
    """E-D6. La bascule d'argmax est portée par la valeur du **voisin n° 5**,
    pas du plus proche. La borne corrigée (`min` sur `V_k`) compte la position ;
    la version écrite sur `v_nn` seul sous-compte et **déclarerait « bug » une
    implémentation correcte**."""
    c = _p5f_neighbor5_case()
    assert c["bascule_effective"] is True
    assert c["gagnant"] == 7 and c["rang_du_porteur"] == 5
    assert c["marge_Vk_pos0"] <= P10_FEASIBLE          # à risque, borne corrigée
    assert c["marge_vnn_pos0"] > P10_FEASIBLE          # invisible à la borne v_nn
    assert c["borne_Vk"] == 0.25 and c["borne_vnn"] == 0.0
    assert verdict_p5f_borne(c["taux_observe"], c["borne_Vk"]) == PASS
    assert verdict_p5f_borne(c["taux_observe"], c["borne_vnn"]) == BUG

    # la borne sur V_k domine toujours celle sur v_nn (min sur un sur-ensemble)
    rows = np.random.default_rng(3).dirichlet(np.ones(12), size=40)
    vk = [[1, 2, 3, 4, 5, 6, 7, 8]] * 40
    vnn = [1] * 40
    assert borne_marge(margins_over_vk(rows, vk)) >= \
        borne_marge(margins_over_vnn(rows, vnn))

    # ex-æquo à marge EXACTEMENT 0.0512711 : l'inégalité est LARGE
    t = _p5f_tie_case()
    assert t["egalite_bit"] is True
    assert t["borne_large"] == 1.0 and t["borne_stricte"] == 0.0
    assert verdict_p5f_borne(1.0, t["borne_large"]) == PASS
    assert verdict_p5f_borne(1.0, t["borne_stricte"]) == BUG
    assert P10_FEASIBLE == math.exp(0.05) - 1.0


# ------------------------------------------------------------------ (xxvi)

def test_xxvi_p3_refuses_an_even_b():
    """À B **pair**, la médiane est la moyenne des 10ᵉ/11ᵉ statistiques d'ordre,
    donc **demi-entière** : le banc REFUSE. B = 1 est refusé aussi (variance
    nulle par construction). B = 21 passe."""
    v, det = verdict_p3([1] * 20, b=20)
    assert v.startswith(REFUS) and "pair" in v
    assert verdict_p3([1] * 10 + [2] * 10, b=20)[0].startswith(REFUS)
    # la médiane demi-entière que le refus évite
    import statistics
    assert statistics.median([1] * 10 + [2] * 10) == 1.5

    v, _ = verdict_p3([1], b=1)
    assert v.startswith(REFUS) and "variance nulle" in v

    v, det = verdict_p3([0, 1, 1, 2, 0, 1, 3, 1, 0, 2, 1, 1, 0, 1, 2, 1, 0, 1,
                         1, 2, 1], b=21)
    assert v == PASS and det["mediane"] == 1 and det["B"] == 21
    assert float(det["mediane"]).is_integer()        # médiane ENTIÈRE à B impair
    assert verdict_p3([14] * 21)[0] == "INCONCLUSIF"
    # B déclaré ≠ nombre de permutations : refus aussi
    assert verdict_p3([1] * 5, b=21)[0].startswith(REFUS)


# ------------------------------------- gardes des données additives de pool.py

def test_pool_additions_are_strictly_additive_and_frozen():
    """`OWNERS`, `ENTITIES`, `VERBS`, `SECRETS_80`, `fact_pairs` intouchés ; les
    ajouts (OWNER_OBJ, POOL_PARAPHRASES, table d'unités) sont dérivés de règles."""
    assert len(SECRETS_80) == 80 and len(set(SECRETS_80)) == 80
    assert SECRETS_80[5] == "lighthouse" and SECRETS_80[30] == "walrus"
    assert len(OWNERS) == 16 and len(ENTITIES) == 20 and len(VERBS) == 5

    assert len(OWNER_OBJ) == 16 and set(OWNER_OBJ) == set(OWNERS)
    for k, v in OWNER_OBJ.items():
        assert v == v.strip() and "'s" not in v
    assert OWNER_OBJ["The captain's"] == "the captain"
    assert OWNER_OBJ["Her"] == "her" and OWNER_OBJ["His"] == "him"
    assert OWNER_OBJ["Our"] == "us" and OWNER_OBJ["Their"] == "them"
    assert OWNER_OBJ["My uncle's"] == "my uncle"

    assert len(POOL_PARAPHRASES) == 30
    U = unit_table(30)
    for u in U:
        i = u["i"]
        # para1 = rotation +1 mod 5 du verbe, owner et entité INCHANGÉS
        assert u["paraphrases"][0] == \
            f"{u['owner']} {u['entity']} {VERBS[(i + 1) % 5]}"
        assert u["verb"] == VERBS[i % 5]
        assert u["paraphrases"][0] != u["exact"]
        # para2 = préfixe gelé + owner minusculisé + entité + verbe D'ORIGINE
        assert u["paraphrases"][1] == (
            "Years later, everyone still remembered that "
            + u["owner"][:1].lower() + u["owner"][1:] + " " + u["entity"]
            + " " + u["verb"])
        # para3 = cadre + registre, seul intrant OWNER_OBJ
        assert u["paraphrases"][2] == (
            "So what's the name of the " + u["entity"] + " that belongs to "
            + OWNER_OBJ[u["owner"]] + "? It's ")
    # les 30 unités partagent le même token final de para3 (prior local constant)
    assert len({u["paraphrases"][2][-6:] for u in U}) == 1

    # la substitution du secret 5 est la SEULE
    assert [u["secret"] for u in U] == list(POOL_UNIT_SECRETS)
    for i, u in enumerate(U):
        assert u["secret"] == (SECRETS_80[30] if i == 5 else SECRETS_80[i])
    assert len(set(POOL_UNIT_SECRETS)) == 30

    # aliasing verbe/entité (note de design §2) : verbe = fonction de l'entité
    assert all(i % 5 == (i % 20) % 5 for i in range(30))


def test_verdict_p1_thresholds_and_grey_zone():
    assert verdict_p1(12) == PASS and verdict_p1(30) == PASS
    assert verdict_p1(5) == FAIL and verdict_p1(0) == FAIL
    for n in range(6, 12):
        assert verdict_p1(n) == "INCONCLUSIF — zone grise"
    # a = 12 est FORCÉ : puissance et risque I re-dérivés en entiers exacts
    puissance = 1 - sum(comb(30, j) for j in range(0, 12)) / 2 ** 30
    assert puissance == pytest.approx(0.8998, abs=1e-4)
    assert puissance < 0.90                          # « ≥ 0.90 » est PROSCRIT
    risque = sum(comb(30, k) * 9 ** (30 - k) for k in range(12, 31)) / 10 ** 30
    assert risque == pytest.approx(1.528e-5, rel=1e-3)
    assert 2 * risque <= 3.06e-5                     # FWER Bonferroni, 2 bras


def test_bench_runs_end_to_end_and_reports_E(tmp_path):
    """Le banc s'exécute en entier, couvre 100 % des clauses et publie `E`."""
    from gate_bench import run
    rep = run(use_hf=False, out_dir=tmp_path)
    assert rep["couverture"]["pct"] == 100.0
    assert isinstance(rep["E"], int) and rep["E"] >= 0
    assert (tmp_path / "report.json").exists()
    for r in rep["clauses"]:
        for f in ("clause", "pass_case", "fail_case", "expected", "observed", "ok"):
            assert f in r
        assert r["expected"]["pass_case"] and r["expected"]["fail_case"]
    # l'usage méta est BORNÉ par construction : aucune statistique décisionnelle
    assert rep["meta_replay"]["statistiques_decisionnelles"].startswith("INTERDITES")
    assert set(rep["meta_replay"]["portes_autorisees"]) == {
        "V-cap", "V-bord", "V1a", "V1b-1", "V1b-2", "V1c", "V-var", "V-drift"}
