# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests CPU du kNN-LM nu (V2-D(a) v2) — aucun modèle HF, quelques millisecondes.

Les DOUZE portes listées au §10 du protocole pré-enregistré
`experiments/EXP-2026-08-21-knn-borne-logits-v2.md` :

  (i)    `knn_lambda = 0` ⇒ logits bit-exacts ;
  (ii)   identité du mélange en log-espace et bord dur −log(1−λ) ;
  (iii)  **un-hot = uniforme sur l'argmin-set** (cas d'ex-æquo construit) ;
  (iv)   fp32 (régression fp16 à ‖h‖² ~ 10⁴) ;
  (v)    soustraction de d²_min ;
  (vi)   clés jamais passées par G/DG ;
  (vii)  datastore jamais rempli pendant `logprob_continuation` ;
  (viii) permutation = mêmes clés ;
  (ix)   k > taille du store ;
  (x)    cible = **premier token BPE** de `" <secret>"` ;
  (xi)   **R1v ≠ R1** sur un store où la cible apparaît deux fois ;
  (xii)  **`sup_c` = max sur la grille déclarée, c loggé**.

Plus les gardes de config (`knn_lambda` = 0.0, `knn_temp_c` = 0.0 ≡ un-hot,
`capture_final_state` inerte) et la garde de `T_q` **par requête**.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from engram.config import EngramConfig
from engram.hippocampus import FastWeightMemory
from knn_ceiling import (
    C_GRID, K_NEIGHBORS, LAMBDA_STAR, UNHOT, Datastore, count_argmin_ties,
    first_bpe_target, knn_distribution, knn_distribution_c, knn_weights,
    knn_weights_batch, mix_delta_nll, mix_logprob, mix_rank, per_query_temperature,
    permute_values, rank_of_index, rank_of_value, squared_distances, sup_over_c,
    topk_neighbors,
)

D = 64
VOCAB = 32


def rand_keys(n, d=D, scale=1.0, seed=0):
    g = np.random.default_rng(seed)
    return (g.standard_normal((n, d)) * scale).astype(np.float32)


def _logp(seed=1, scale=3.0, vocab=VOCAB):
    rng = np.random.default_rng(seed)
    return torch.log_softmax(torch.tensor(rng.standard_normal(vocab) * scale),
                             dim=-1).numpy().astype(np.float64)


# --------------------------------------------------------------------- (i)

def test_i_lambda_zero_is_bit_exact():
    """λ=0 ⇒ le mélange est l'IDENTITÉ, au bit près, sur la log-proba ET sur le rang.
    C'est le contrôle §5.4 : sans lui le kNN pollue le E1 de référence."""
    logp = _logp()
    pk = {3: 0.7, 7: 0.3}
    for target in range(VOCAB):
        assert mix_logprob(logp[target], pk.get(target, 0.0), 0.0) == logp[target]
        assert mix_delta_nll(logp[target], pk.get(target, 0.0), 0.0) == 0.0
        base_rank = int((logp > logp[target]).sum()) + 1
        assert mix_rank(logp, pk, 0.0, target) == base_rank


def test_i_config_knn_defaults_are_inert():
    cfg = EngramConfig()
    assert cfg.knn_lambda == 0.0
    assert cfg.knn_k == 8
    assert cfg.knn_temp_c == 0.0          # §10 v2 : 0.0 ≡ un-hot
    assert cfg.knn_key_layer == "final"
    assert cfg.knn_gate_tau == 0.0
    assert cfg.capture_final_state is False
    # la ligne de résumé ne mentionne pas knn tant qu'il n'est pas armé
    assert "knn" not in cfg.summary()
    assert "knn=" in EngramConfig(knn_lambda=0.25).summary()


# -------------------------------------------------------------------- (ii)

def test_ii_log_space_mixture_identity_and_hard_edge():
    """(a) le mélange en log-espace reproduit (1−λ)p_LM + λ p_kNN à 1e-9 ;
    (b) ΔNLL ≤ −log(1−λ) à TOUTE position, y compris quand p_LM ≈ e^−15 (le
    régime où la forme en probas brutes casse) ;
    (c) le bord est ATTEINT exactement ssi p_kNN(cible) = 0 (porte V1a) et
    STRICTEMENT sous le bord dès que la masse est > 0 (porte V1b)."""
    for lam in (0.02, LAMBDA_STAR, 0.10, 0.25, 0.5):
        edge = -math.log1p(-lam)
        for logp_lm in (-0.1, -3.0, -15.0, -30.0):
            for pk in (0.0, 1e-9, 0.5, 1.0):
                got = mix_logprob(logp_lm, pk, lam)
                lin = (1.0 - lam) * math.exp(logp_lm) + lam * pk
                if lin > 0 and math.log(lin) > -700:
                    assert abs(got - math.log(lin)) < 1e-9, (lam, logp_lm, pk)
                d = mix_delta_nll(logp_lm, pk, lam)
                assert d <= edge + 1e-12, (lam, logp_lm, pk, d, edge)
                if pk > 0.0:
                    assert d < edge                      # V1b : strict
    for lam in (0.02, 0.10, 0.25):
        assert abs(mix_delta_nll(-15.0, 0.0, lam) + math.log1p(-lam)) < 1e-12


def test_ii_naive_probability_mixture_would_break_but_log_space_does_not():
    """Régression : à p_LM = e^−800 la forme en probas brutes rend 0 (underflow)
    et log(0) = −inf ; la forme log-espace reste finie."""
    logp_lm = -800.0
    assert (1.0 - 0.1) * math.exp(logp_lm) == 0.0        # underflow effectif
    assert math.isfinite(mix_logprob(logp_lm, 0.0, 0.0))
    assert mix_logprob(logp_lm, 0.0, 0.1) == pytest.approx(logp_lm + math.log(0.9))


# ------------------------------------------------------------------- (iii)

def test_iii_unhot_is_uniform_over_the_argmin_set():
    """§10 : `c = 0` ≡ un-hot ≡ limite c→0⁺ ≡ **uniforme sur l'argmin-set**.
    Cas d'ex-æquo CONSTRUIT : trois voisins exactement à d²_min."""
    d2 = np.array([5.0, 5.0, 5.0, 9.0, 40.0])
    w = knn_weights(d2, 0.0)
    assert w[:3] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert w[3] == 0.0 and w[4] == 0.0
    assert w.sum() == pytest.approx(1.0)
    assert count_argmin_ties(d2) == 3
    # agrégation par token : deux des trois ex-æquo portent le même token
    pk = knn_distribution_c(d2, np.array([7, 7, 9, 1, 2]), UNHOT)
    assert pk == pytest.approx({7: 2 / 3, 9: 1 / 3})
    # limite continue : c → 0⁺ tend vers l'un-hot
    for c in (1e-3, 1e-4, 1e-5):
        wc = knn_weights(d2, per_query_temperature(d2, c))
        assert wc == pytest.approx(w, abs=1e-6)
    # sans ex-æquo, l'un-hot est un vrai one-hot
    w1 = knn_weights(np.array([1.0, 2.0, 3.0]), 0.0)
    assert w1 == pytest.approx([1.0, 0.0, 0.0])
    # version en lot : même résultat
    wb = knn_weights_batch(np.stack([d2, d2]), UNHOT)
    assert wb[0] == pytest.approx(w)


def test_iii_per_query_temperature_uses_median_over_j_ge_2():
    """`T_q = c · med_{j≥2}(d²_j − d²_min)` — PAR REQUÊTE (v2 §4.1). La médiane
    exclut le plus proche voisin (son écart vaut 0 par construction)."""
    d2 = np.array([10.0, 11.0, 12.0, 14.0, 20.0])       # écarts j≥2 : 1, 2, 4, 10
    assert per_query_temperature(d2, 1.0) == pytest.approx(3.0)   # med(1,2,4,10) = 3
    assert per_query_temperature(d2, 0.3) == pytest.approx(0.9)
    assert per_query_temperature(d2, 0.0) == 0.0
    # deux requêtes d'échelles différentes ⇒ deux T_q différentes (le point de
    # l'amendement : la médiane-de-médianes PAR BRAS du run 1 est abandonnée)
    assert per_query_temperature(d2 * 100.0, 1.0) == pytest.approx(300.0)
    # T_q dégénérée (tous les voisins à égalité) ⇒ retombe sur l'un-hot
    flat = np.array([7.0, 7.0, 7.0])
    assert per_query_temperature(flat, 1.0) == 0.0
    assert knn_weights(flat, per_query_temperature(flat, 1.0)) == \
        pytest.approx([1 / 3, 1 / 3, 1 / 3])


# -------------------------------------------------------------------- (iv)

def test_iv_distances_fp16_is_refused():
    q = rand_keys(1, scale=4.0, seed=2)[0]
    keys = rand_keys(5, scale=4.0, seed=3)
    with pytest.raises(TypeError):
        squared_distances(q.astype(np.float16), keys)
    with pytest.raises(TypeError):
        squared_distances(q, keys.astype(np.float16))


def test_iv_quasi_match_survives_in_fp32_and_dies_in_fp16():
    """‖h‖² ~ 10⁴ : la résolution du fp16 y est ~8. Un quasi-match à d² ≈ 1 doit
    rester distinguable de zéro en fp32 ; le même calcul en fp16 l'écrase."""
    rng = np.random.default_rng(4)
    v = rng.standard_normal(D).astype(np.float32)
    v *= np.float32(100.0 / np.linalg.norm(v))          # ‖v‖ = 100 ⇒ ‖v‖² = 1e4
    eps = rng.standard_normal(D).astype(np.float32)
    eps *= np.float32(1.0 / np.linalg.norm(eps))        # ‖eps‖ = 1 ⇒ d² = 1
    keys = np.stack([v + eps, v + 30.0 * eps]).astype(np.float32)
    d2 = squared_distances(v, keys)
    assert d2[0] == pytest.approx(1.0, rel=1e-3)
    assert d2[0] < d2[1]
    # Forme proscrite ‖q‖² + ‖k‖² − 2qᵀk évaluée en fp16 : chacun des trois termes
    # vaut ~10⁴, où la résolution du fp16 est de 8. Le résultat est donc un multiple
    # de 8 : la vraie valeur d² = 1 est INDISCERNABLE de zéro.
    nq = np.float16(float((v * v).sum()))
    nk = np.float16(float((keys[0] * keys[0]).sum()))
    qk = np.float16(float((v * keys[0]).sum()))
    naive16 = float(nq) + float(nk) - 2.0 * float(qk)
    assert float(np.spacing(nq)) >= 8.0                 # résolution ~8 à ‖h‖² ~ 10⁴
    assert abs(naive16 - 1.0) >= 1.0                    # le quasi-match est perdu
    assert abs(float(d2[0]) - 1.0) < 1e-3               # la forme (q−k) le préserve


# --------------------------------------------------------------------- (v)

def test_v_softmax_subtracts_d2min_no_underflow():
    """Sans soustraction de d²_min, exp(−d²/T) avec d² ~ 10⁴ underflow à 0 partout
    (0/0). Avec, la distribution est propre et somme à 1."""
    d2 = np.array([10000.0, 10003.0, 10010.0, 12000.0])
    naive = np.exp(-d2 / 1.0)
    assert naive.sum() == 0.0                       # underflow silencieux
    w = knn_weights(d2, 1.0)
    assert np.isfinite(w).all()
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2] > w[3]
    assert w[3] < 1e-100
    # même garde sur le chemin `c` (T_q par requête) et sur le chemin en lot
    wc = knn_distribution_c(d2, np.array([1, 2, 3, 4]), 1.0)
    assert sum(wc.values()) == pytest.approx(1.0)
    assert np.isfinite(knn_weights_batch(np.stack([d2, d2]), 1.0)).all()


def test_v_knn_distribution_aggregates_duplicate_values():
    d2 = np.array([100.0, 100.0, 400.0])
    pk = knn_distribution(d2, np.array([5, 5, 9]), 100.0)
    assert set(pk) == {5, 9}
    assert sum(pk.values()) == pytest.approx(1.0)
    assert pk[5] > pk[9]


# -------------------------------------------------------------------- (vi)

def test_vi_knn_keys_never_go_through_dg():
    """Les clés stockées sont bit-à-bit les états fournis, quelle que soit la
    config gyrus denté. Le datastore n'a ni G ni φ : c'est structurel (D9)."""
    keys = rand_keys(6, scale=5.0, seed=5)
    st = Datastore("t")
    st.add(keys, np.arange(6))
    st.freeze()
    assert np.array_equal(st.keys, keys)
    assert not hasattr(st, "G")
    assert not hasattr(st, "phi")
    # contraste : la projection DG, elle, transforme et normalise
    cfg = EngramConfig(dg_dim=256, dg_topk=16, read_gate="none")
    mem = FastWeightMemory(D, cfg)
    projected = mem.phi(torch.tensor(keys[0]))
    assert projected.shape[0] == 256
    assert float(projected.norm()) == pytest.approx(1.0, rel=1e-5)
    # ... et détruirait l'échelle des distances que P1 mesure
    d_raw = squared_distances(keys[0], keys)
    proj = np.stack([mem.phi(torch.tensor(k)).numpy() for k in keys])
    d_dg = squared_distances(proj[0], proj)
    assert d_dg.max() <= 4.0        # clés unitaires après DG : d² ∈ [0, 4]
    assert d_raw.max() > 100.0      # échelle réelle des états : sans commune mesure


# ------------------------------------------------------------------- (vii)

def test_vii_datastore_never_filled_during_a_measurement():
    """D7/D8 : le datastore est gelé avant la première requête ; toute écriture
    ensuite lève. C'est la garantie MÉCANIQUE qu'il ne se remplit jamais pendant
    une mesure de logprob (boucle de requêtes simulée ci-dessous)."""
    st = Datastore("t")
    st.add(rand_keys(4, seed=6), np.arange(4))
    with pytest.raises(AssertionError):
        st.query(rand_keys(1, seed=7)[0])           # requête avant freeze : refusée
    st.freeze()
    n0 = len(st)
    for i in range(5):                              # « logprob_continuation »
        st.query(rand_keys(1, seed=8 + i)[0], k=2)
        with pytest.raises(RuntimeError):
            st.add(rand_keys(1, seed=99), np.array([0]))
    assert len(st) == n0                            # aucune requête n'ajoute rien


# ------------------------------------------------------------------ (viii)

def test_viii_permutation_keeps_keys_and_distances():
    keys = rand_keys(12, scale=3.0, seed=10)
    vals = np.arange(100, 112)
    st = Datastore("t")
    st.add(keys, vals)
    st.freeze()
    q = rand_keys(1, scale=3.0, seed=11)[0]
    idx_a, d2_a, val_a, _ = st.query(q, k=5)
    perm = permute_values(vals, seed=42)
    idx_b, d2_b, val_b, _ = st.query(q, k=5, values=perm)
    assert np.array_equal(idx_a, idx_b)
    assert np.array_equal(d2_a, d2_b)               # distances RIGOUREUSEMENT égales
    assert sorted(perm.tolist()) == sorted(vals.tolist())   # multiensemble conservé
    assert np.array_equal(val_b, perm[idx_a])
    assert np.array_equal(st.keys, keys)


def test_viii_permutation_makes_target_drop_by_log1m_lambda():
    """P3 : si la cible sort du support kNN après permutation, logp(cible) baisse
    EXACTEMENT de log(1−λ) — la condition exacte étant `valeur(argmin) ≠ cible`."""
    keys = rand_keys(8, scale=3.0, seed=12)
    vals = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    st = Datastore("t")
    st.add(keys, vals)
    st.freeze()
    _, d2k, valk, _ = st.query(keys[0].copy(), k=4)
    pk = knn_distribution(d2k, valk, 10.0)
    target = 99                                      # absent du store
    for lam in (0.02, 0.10, 0.25):
        assert mix_delta_nll(-8.0, pk.get(target, 0.0), lam) == \
            pytest.approx(-math.log1p(-lam), abs=1e-12)


# -------------------------------------------------------------------- (ix)

def test_ix_k_greater_than_store_size():
    keys = rand_keys(3, seed=13)
    st = Datastore("t")
    st.add(keys, np.array([4, 5, 6]))
    st.freeze()
    idx, d2k, valk, d2_all = st.query(rand_keys(1, seed=14)[0], k=8)
    assert len(idx) == 3 and len(d2k) == 3 and len(valk) == 3
    assert sum(knn_distribution_c(d2k, valk, 1.0).values()) == pytest.approx(1.0)
    assert sum(knn_distribution_c(d2k, valk, UNHOT).values()) == pytest.approx(1.0)
    assert len(topk_neighbors(d2_all, 8)) == 3


def test_ix_single_entry_store():
    st = Datastore("t")
    st.add(rand_keys(1, seed=15), np.array([7]))
    st.freeze()
    idx, d2k, valk, _ = st.query(rand_keys(1, seed=16)[0], k=8)
    assert len(idx) == 1
    assert knn_distribution_c(d2k, valk, UNHOT) == {7: pytest.approx(1.0)}
    assert per_query_temperature(d2k, 1.0) == 0.0     # k=1 ⇒ pas de j≥2


# --------------------------------------------------------------------- (x)

class _StubTokenizer:
    """Tokeniseur BPE minimal à espace de tête, sans aucun téléchargement HF :
    reproduit la propriété qui compte — `" swordfish"` se scinde en plusieurs
    tokens et son PREMIER token diffère de celui de `"swordfish"`."""

    _VOCAB = {"The": 1, " password": 2, " is": 3, " sword": 4, "fish": 5,
              "sword": 6, " obs": 7, "idian": 8, ".": 9}

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            for piece, tid in sorted(self._VOCAB.items(), key=lambda kv: -len(kv[0])):
                if text.startswith(piece, i):
                    ids.append(tid)
                    i += len(piece)
                    break
            else:
                raise AssertionError(f"morceau non tokenisable à {i} : {text[i:]!r}")
        return ids


def test_x_target_is_first_bpe_token_of_leading_space_secret():
    """§4.1 : la cible est le **premier token BPE** de `" <secret>"`, dérivée PAR
    DIFFÉRENCE prompt vs prompt+continuation. Encoder la continuation seule
    donnerait un AUTRE token (pas d'espace de tête) — c'est le piège."""
    tok = _StubTokenizer()
    tgt, ids = first_bpe_target(tok, "The password is", " swordfish")
    assert ids == [4, 5]                     # " sword" + "fish"
    assert tgt == 4                          # le PREMIER, avec l'espace de tête
    assert tok.encode("swordfish")[0] == 6   # sans espace : un token DIFFÉRENT
    assert tgt != tok.encode("swordfish")[0]
    tgt2, ids2 = first_bpe_target(tok, "The password is", " obsidian")
    assert ids2 == [7, 8] and tgt2 == 7      # secret MULTI-TOKENS (T-multitok)
    with pytest.raises(AssertionError):
        first_bpe_target(tok, "The password is", "")


# -------------------------------------------------------------------- (xi)

def test_xi_r1v_differs_from_r1_when_target_appears_twice():
    """**R1v** = rang du premier voisin dont la VALEUR est la cible ; **R1** = rang
    d'une entrée DÉSIGNÉE. Store construit où la cible apparaît DEUX fois : une
    entrée lointaine (celle désignée) et une entrée proche."""
    q = np.zeros(D, dtype=np.float32)
    keys = np.zeros((4, D), dtype=np.float32)
    keys[0, 0] = 30.0          # loin      — valeur = cible (entrée « correcte »)
    keys[1, 0] = 1.0           # très près — valeur ≠ cible
    keys[2, 0] = 2.0           # près      — valeur = cible (le second porteur)
    keys[3, 0] = 50.0          # très loin — valeur ≠ cible
    vals = np.array([42, 7, 42, 8])
    d2 = squared_distances(q, keys)
    assert list(np.argsort(d2)) == [1, 2, 0, 3]
    assert rank_of_index(d2, 0) == 3                 # R1 = 3 (entrée désignée)
    assert rank_of_value(d2, vals, 42) == 2          # R1v = 2 (autre porteur)
    assert rank_of_value(d2, vals, 42) != rank_of_index(d2, 0)
    # cible absente du store ⇒ R1v = −1 (convention loggée)
    assert rank_of_value(d2, vals, 99) == -1
    # quand la cible n'apparaît qu'une fois, R1v == R1
    vals_once = np.array([42, 7, 3, 8])
    assert rank_of_value(d2, vals_once, 42) == rank_of_index(d2, 0) == 3


# ------------------------------------------------------------------- (xii)

def test_xii_sup_over_c_is_the_max_over_the_declared_grid_and_logs_c():
    """§4.1 / B-1 : la cellule décisionnelle est le SUPREMUM sur la grille
    DÉCLARÉE `c ∈ {un-hot, 0.03, 0.1, 0.3, 1, 3}`. Le rang retourné est le
    MINIMUM sur la grille, et le c qui l'atteint est LOGGÉ."""
    logp = _logp(seed=5, scale=2.0)
    # la cible (token 11) n'est PAS la valeur du plus proche voisin : l'un-hot lui
    # donne 0 masse — c'est l'erreur E-D1 du Directeur, que `sup_c` corrige.
    d2k = np.array([1.0, 2.0, 3.0, 4.0])
    valsk = np.array([2, 11, 11, 5])
    target = 11
    s = sup_over_c(logp, d2k, valsk, target, 0.25)
    assert set(s["per_c"].keys()) == set(C_GRID)
    ranks = [s["per_c"][c]["rank"] for c in C_GRID]
    assert s["rank"] == min(ranks)                       # sup = rang minimal
    assert s["c_sup"] in C_GRID                          # le c est loggé
    assert s["per_c"][s["c_sup"]]["rank"] == s["rank"]
    assert s["per_c"][UNHOT]["p_knn_target"] == 0.0      # l'un-hot est le MINIMUM
    assert s["p_knn_target"] > 0.0                       # le sup, lui, récupère
    assert s["c_sup"] != UNHOT
    # chaque cellule de la grille logge sa T_q PAR REQUÊTE
    for c in C_GRID:
        assert s["per_c"][c]["T_q"] == pytest.approx(per_query_temperature(d2k, c))
    # cas où l'un-hot EST le sup : la cible est la valeur du plus proche voisin
    s2 = sup_over_c(logp, d2k, np.array([11, 2, 5, 7]), target, 0.25)
    assert s2["per_c"][UNHOT]["p_knn_target"] == pytest.approx(1.0)
    assert s2["rank"] == min(s2["per_c"][c]["rank"] for c in C_GRID)


def test_xii_sup_c_grid_is_the_preregistered_one():
    assert C_GRID == [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]
    assert UNHOT == 0.0
    assert K_NEIGHBORS == 8
    assert LAMBDA_STAR == pytest.approx(0.0487706, abs=1e-7)


# ------------------------------------------- rang exact sous mélange (garde)

def test_mix_rank_matches_bruteforce():
    logp = _logp(seed=17, scale=2.0)
    pk = {2: 0.6, 11: 0.4}
    for lam in (0.0, LAMBDA_STAR, 0.25, 0.9):
        pm = (1.0 - lam) * np.exp(logp)
        for t, v in pk.items():
            pm[t] += lam * v
        for target in (0, 2, 11, 31):
            assert mix_rank(logp, pk, lam, target) == int((pm > pm[target]).sum()) + 1
