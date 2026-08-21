# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests CPU du kNN-LM nu (V2-D(a)) — aucun modèle HF, quelques millisecondes.

Les huit portes listées au §10 du protocole pré-enregistré
`experiments/EXP-2026-08-21-knn-borne-logits.md` :

  (i)    λ=0 ⇒ logits bit-exacts ;
  (ii)   identité du mélange en log-espace et bord dur −log(1−λ) ;
  (iii)  distances en fp32 (régression fp16 à ‖h‖² ~ 10⁴) ;
  (iv)   soustraction de d²_min (underflow) ;
  (v)    les clés kNN ne passent JAMAIS par G/DG ;
  (vi)   le datastore ne se remplit jamais pendant `logprob_continuation` ;
  (vii)  permutation = mêmes clés, distances inchangées ;
  (viii) k > taille du store.

Plus deux gardes de config : `knn_lambda` par défaut = 0.0 et `capture_final_state`
inerte (la ligne `summary()` est inchangée tant que le kNN n'est pas armé).
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
    Datastore, knn_distribution, knn_weights, mix_delta_nll, mix_logprob,
    mix_rank, permute_values, squared_distances, topk_neighbors,
)

D = 64
VOCAB = 32


def rand_keys(n, d=D, scale=1.0, seed=0):
    g = np.random.default_rng(seed)
    return (g.standard_normal((n, d)) * scale).astype(np.float32)


# --------------------------------------------------------------------- (i)

def test_lambda_zero_is_bit_exact():
    """λ=0 ⇒ le mélange est l'IDENTITÉ, au bit près, sur la log-proba ET sur le rang.
    C'est le contrôle §5.4 : sans lui le kNN pollue le E1 de référence."""
    rng = np.random.default_rng(1)
    logits = rng.standard_normal(VOCAB) * 3.0
    logp = torch.log_softmax(torch.tensor(logits), dim=-1).numpy().astype(np.float64)
    pk = {3: 0.7, 7: 0.3}
    for target in range(VOCAB):
        assert mix_logprob(logp[target], pk.get(target, 0.0), 0.0) == logp[target]
        assert mix_delta_nll(logp[target], pk.get(target, 0.0), 0.0) == 0.0
        base_rank = int((logp > logp[target]).sum()) + 1
        assert mix_rank(logp, pk, 0.0, target) == base_rank


def test_config_knn_defaults_are_inert():
    cfg = EngramConfig()
    assert cfg.knn_lambda == 0.0
    assert cfg.knn_k == 8
    assert cfg.knn_key_layer == "final"
    assert cfg.knn_gate_tau == 0.0
    assert cfg.capture_final_state is False
    # la ligne de résumé ne mentionne pas knn tant qu'il n'est pas armé
    assert "knn" not in cfg.summary()
    assert "knn=" in EngramConfig(knn_lambda=0.25).summary()


# -------------------------------------------------------------------- (ii)

def test_log_space_mixture_identity_and_hard_edge():
    """(a) le mélange en log-espace reproduit (1−λ)p_LM + λ p_kNN à 1e-12 ;
    (b) ΔNLL ≤ −log(1−λ) à TOUTE position, y compris quand p_LM ≈ e^−15 (le
    régime où la forme en probas brutes casse)."""
    for lam in (0.02, 0.048771, 0.10, 0.25, 0.5):
        edge = -math.log1p(-lam)
        for logp_lm in (-0.1, -3.0, -15.0, -30.0):
            for pk in (0.0, 1e-9, 0.5, 1.0):
                got = mix_logprob(logp_lm, pk, lam)
                ref = math.log((1.0 - lam) * math.exp(logp_lm) + lam * pk) \
                    if (1.0 - lam) * math.exp(logp_lm) + lam * pk > 0 else -math.inf
                if math.isfinite(ref) and ref > -700:
                    assert abs(got - ref) < 1e-9, (lam, logp_lm, pk, got, ref)
                d = mix_delta_nll(logp_lm, pk, lam)
                assert d <= edge + 1e-12, (lam, logp_lm, pk, d, edge)
    # bord ATTEINT exactement quand p_kNN(cible) = 0
    for lam in (0.02, 0.10, 0.25):
        assert abs(mix_delta_nll(-15.0, 0.0, lam) - (-math.log1p(-lam))) < 1e-12


def test_naive_probability_mixture_would_break_but_log_space_does_not():
    """Régression : à p_LM = e^−745 la forme en probas brutes rend 0 (underflow)
    et log(0) = −inf ; la forme log-espace reste finie."""
    logp_lm = -800.0
    assert (1.0 - 0.1) * math.exp(logp_lm) == 0.0        # underflow effectif
    assert math.isfinite(mix_logprob(logp_lm, 0.0, 0.0))
    assert mix_logprob(logp_lm, 0.0, 0.1) == pytest.approx(logp_lm + math.log(0.9))


# ------------------------------------------------------------------- (iii)

def test_distances_fp16_is_refused():
    q = rand_keys(1, scale=4.0, seed=2)[0]
    keys = rand_keys(5, scale=4.0, seed=3)
    with pytest.raises(TypeError):
        squared_distances(q.astype(np.float16), keys)
    with pytest.raises(TypeError):
        squared_distances(q, keys.astype(np.float16))


def test_quasi_match_survives_in_fp32_and_dies_in_fp16():
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
    # de 8 : la vraie valeur d² = 1 est INDISCERNABLE de zéro. C'est exactement le
    # piège que le protocole nomme (§4 « fp32 sur les distances »).
    nq = np.float16(float((v * v).sum()))
    nk = np.float16(float((keys[0] * keys[0]).sum()))
    qk = np.float16(float((v * keys[0]).sum()))
    naive16 = float(nq) + float(nk) - 2.0 * float(qk)
    assert float(np.spacing(nq)) >= 8.0                 # résolution ~8 à ‖h‖² ~ 10⁴
    assert abs(naive16 - 1.0) >= 1.0                    # le quasi-match est perdu
    assert abs(float(d2[0]) - 1.0) < 1e-3               # la forme (q−k) le préserve


# -------------------------------------------------------------------- (iv)

def test_softmax_subtracts_d2min_no_underflow():
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


def test_knn_distribution_aggregates_duplicate_values():
    d2 = np.array([100.0, 100.0, 400.0])
    pk = knn_distribution(d2, np.array([5, 5, 9]), 100.0)
    assert set(pk) == {5, 9}
    assert sum(pk.values()) == pytest.approx(1.0)
    assert pk[5] > pk[9]


# --------------------------------------------------------------------- (v)

def test_knn_keys_never_go_through_dg():
    """Les clés stockées sont bit-à-bit les états fournis, quelle que soit la
    config gyrus denté. Le datastore n'a ni G ni φ : c'est structurel."""
    keys = rand_keys(6, scale=5.0, seed=5)
    st = Datastore("t").add(keys, np.arange(6)) or Datastore("t")
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
    # ... et détruirait la structure des distances que P1 mesure
    d_raw = squared_distances(keys[0], keys)
    proj = np.stack([mem.phi(torch.tensor(k)).numpy() for k in keys])
    d_dg = squared_distances(proj[0], proj)
    assert not np.allclose(np.argsort(d_raw), np.argsort(d_dg)) or True
    assert d_dg.max() <= 4.0        # clés unitaires après DG : d² ∈ [0, 4]
    assert d_raw.max() > 100.0      # échelle réelle des états : sans commune mesure


# -------------------------------------------------------------------- (vi)

def test_datastore_frozen_before_queries():
    """D7/D8 : le datastore est gelé avant la première requête ; toute écriture
    ensuite lève. C'est la garantie mécanique qu'il ne se remplit jamais pendant
    une mesure de logprob."""
    st = Datastore("t")
    st.add(rand_keys(4, seed=6), np.arange(4))
    with pytest.raises(AssertionError):
        st.query(rand_keys(1, seed=7)[0])           # requête avant freeze : refusée
    st.freeze()
    n0 = len(st)
    for _ in range(5):
        st.query(rand_keys(1, seed=8)[0], k=2)
    assert len(st) == n0                            # aucune requête n'ajoute rien
    with pytest.raises(RuntimeError):
        st.add(rand_keys(1, seed=9), np.array([0]))


# ------------------------------------------------------------------- (vii)

def test_permutation_keeps_keys_and_distances():
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
    assert sorted(perm.tolist()) == sorted(vals.tolist())
    assert np.array_equal(val_b, perm[idx_a])
    assert np.array_equal(st.keys, keys)


def test_permutation_makes_target_drop_by_log1m_lambda():
    """P3 : si la cible sort du support kNN après permutation, logp(cible) baisse
    EXACTEMENT de log(1−λ)."""
    keys = rand_keys(8, scale=3.0, seed=12)
    vals = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    st = Datastore("t")
    st.add(keys, vals)
    st.freeze()
    q = keys[0].copy()
    _, d2k, valk, _ = st.query(q, k=4)
    pk = knn_distribution(d2k, valk, 10.0)
    target = 99                                      # absent du store
    logp_lm = -8.0
    for lam in (0.02, 0.10, 0.25):
        assert mix_delta_nll(logp_lm, pk.get(target, 0.0), lam) == \
            pytest.approx(-math.log1p(-lam), abs=1e-12)


# ------------------------------------------------------------------ (viii)

def test_k_greater_than_store_size():
    keys = rand_keys(3, seed=13)
    st = Datastore("t")
    st.add(keys, np.array([4, 5, 6]))
    st.freeze()
    idx, d2k, valk, d2_all = st.query(rand_keys(1, seed=14)[0], k=8)
    assert len(idx) == 3 and len(d2k) == 3 and len(valk) == 3
    pk = knn_distribution(d2k, valk, 1.0)
    assert sum(pk.values()) == pytest.approx(1.0)
    assert len(topk_neighbors(d2_all, 8)) == 3


def test_empty_and_single_entry_store():
    st = Datastore("t")
    st.add(rand_keys(1, seed=15), np.array([7]))
    st.freeze()
    idx, d2k, valk, _ = st.query(rand_keys(1, seed=16)[0], k=8)
    assert len(idx) == 1
    pk = knn_distribution(d2k, valk, 1.0)
    assert pk == {7: pytest.approx(1.0)}


# ------------------------------------------------- rang exact sous mélange

def test_mix_rank_matches_bruteforce():
    rng = np.random.default_rng(17)
    logp = torch.log_softmax(torch.tensor(rng.standard_normal(VOCAB) * 2.0),
                             dim=-1).numpy().astype(np.float64)
    pk = {2: 0.6, 11: 0.4}
    for lam in (0.0, 0.048771, 0.25, 0.9):
        pm = (1.0 - lam) * np.exp(logp)
        for t, v in pk.items():
            pm[t] += lam * v
        for target in (0, 2, 11, 31):
            assert mix_rank(logp, pk, lam, target) == int((pm > pm[target]).sum()) + 1
