# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests unitaires de l'hippocampe — CPU pur, aucun modèle HF, quelques millisecondes.

Ils vérifient les propriétés qui rendent le PoC interprétable : rappel associatif,
réécriture (le terme correctif de la delta rule), bornes de stabilité, oubli, reset.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engram.config import EngramConfig
from engram.hippocampus import FastWeightMemory

D = 64


def make_memory(**overrides) -> FastWeightMemory:
    # dg_dim=0 et read_gate="none" explicites : la base de test est le mécanisme nu
    # (mode X0, sans gate) ; les tests X1/X8 activent leurs mécanismes eux-mêmes.
    # Les défauts de EngramConfig sont dg actif + gate keysim.
    cfg = EngramConfig(**{"eta": 0.2, "decay": 0.0, "prune_every": 0, "dg_dim": 0,
                          "read_gate": "none", **overrides})
    return FastWeightMemory(D, cfg)


def unit(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(D, generator=g)
    return v / v.norm()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def test_write_then_read_recalls_association():
    mem = make_memory()
    k, v = unit(1), unit(2)
    for _ in range(30):
        mem.write(k, v)
    assert cosine(mem.read(k), v) > 0.99


def test_two_orthogonalish_pairs_coexist():
    mem = make_memory()
    k1, v1, k2, v2 = unit(1), unit(2), unit(3), unit(4)
    for _ in range(30):
        mem.write(k1, v1)
        mem.write(k2, v2)
    assert cosine(mem.read(k1), v1) > 0.8
    assert cosine(mem.read(k2), v2) > 0.8


def test_delta_rule_overwrites_old_value():
    """Une clé connue réassociée à une nouvelle valeur doit basculer vers celle-ci —
    c'est le terme correctif −M·k ; en Hebb pur les deux valeurs s'additionneraient."""
    mem = make_memory()
    k, v_old, v_new = unit(1), unit(2), unit(3)
    for _ in range(30):
        mem.write(k, v_old)
    for _ in range(30):
        mem.write(k, v_new)
    r = mem.read(k)
    assert cosine(r, v_new) > cosine(r, v_old)
    assert cosine(r, v_new) > 0.9


def test_delta_rule_bounded_hebbian_grows():
    """Sur une association répétée, la delta rule converge (l'erreur → 0) ; le Hebb
    pur accumule sans fin. C'est la justification quantitative de la décision D5."""
    delta_mem = make_memory()
    hebb_mem = make_memory(hebbian_only=True)
    k, v = unit(1), unit(2)
    for _ in range(200):
        delta_mem.write(k, v)
        hebb_mem.write(k, v)
    assert delta_mem.M.norm().item() < 1.5  # ‖v‖=1 : M converge vers ~v·kᵀ
    assert hebb_mem.M.norm().item() > 5.0   # ~η·200 = 40 en théorie


def test_read_norm_is_capped():
    mem = make_memory(lam=100.0, max_read_norm=0.5)
    k, v = unit(1), unit(2)
    for _ in range(50):
        mem.write(k, v)
    h = unit(1) * 3.0  # requête de norme 3
    assert mem.read(h).norm().item() <= 0.5 * 3.0 + 1e-4


def test_decay_erodes_memory():
    mem = make_memory(decay=0.05)
    k, v = unit(1), unit(2)
    mem.write(k, v)
    norm_after_one = mem.M.norm().item()
    for _ in range(100):
        mem.write(unit(10), unit(11))  # writes sans rapport : k s'érode par decay
    # La composante de la première association a décru.
    assert cosine(mem.read(k), v) < 1.0
    assert mem.M.norm().item() < norm_after_one * 100  # pas d'explosion


def test_prune_sparsifies_to_keep_fraction():
    mem = make_memory(prune_every=0, prune_keep=0.05)
    for s in range(20):
        mem.write(unit(2 * s), unit(2 * s + 1))
    mem.prune()
    density = (mem.M != 0).float().mean().item()
    assert density <= 0.05 + 1e-6


def test_reset_zeroes_everything():
    mem = make_memory()
    mem.write(unit(1), unit(2))
    assert mem.M.norm().item() > 0
    mem.reset()
    assert mem.M.norm().item() == 0.0
    assert mem.write_count == 0


def test_dg_phi_is_sparse_and_unit():
    mem = make_memory(dg_dim=1024, dg_topk=32)
    k = mem.phi(unit(1))
    assert k.shape == (1024,)
    assert (k != 0).sum().item() <= 32
    assert abs(k.norm().item() - 1.0) < 1e-5


def test_dg_recall_works():
    mem = make_memory(dg_dim=1024, dg_topk=32)
    k, v = unit(1), unit(2)
    for _ in range(30):
        mem.write(k, v)
    assert cosine(mem.read(k), v) > 0.95


def test_dg_projection_is_deterministic_given_seed():
    """G est gelée et seedée : deux mémoires de même config produisent les mêmes clés
    (comparabilité inter-runs)."""
    m1 = make_memory(dg_dim=1024, dg_topk=32, seed=7)
    m2 = make_memory(dg_dim=1024, dg_topk=32, seed=7)
    h = unit(3)
    assert torch.equal(m1.phi(h), m2.phi(h))


def test_dg_reduces_interference_between_similar_keys():
    """Le cœur de X1 : deux clés très corrélées (cos ≈ 0.95) associées à des valeurs
    différentes s'écrasent mutuellement en mode dense ; la séparation de patterns du
    gyrus denté doit préserver un meilleur rappel des deux."""
    k1 = unit(1)
    noise = unit(2)
    k2 = k1 + 0.33 * noise
    k2 = k2 / k2.norm()  # cos(k1, k2) ≈ 0.95
    v1, v2 = unit(3), unit(4)

    def recall_quality(mem) -> float:
        for _ in range(20):
            mem.write(k1, v1)
            mem.write(k2, v2)
        return min(cosine(mem.read(k1), v1), cosine(mem.read(k2), v2))

    dense = recall_quality(make_memory())
    dg = recall_quality(make_memory(dg_dim=1024, dg_topk=32))
    assert dg > dense


def test_track_keys_off_by_default():
    mem = make_memory()
    mem.write(unit(1), unit(2))
    mem.write(unit(3), unit(4))
    assert mem.write_similarities == []


def test_track_keys_records_max_cosine():
    mem = make_memory(track_keys=True)
    k = unit(1)
    mem.write(k, unit(2))            # première clé : pas de référence, rien enregistré
    assert mem.write_similarities == []
    mem.write(k, unit(3))            # même clé : cos ≈ 1
    mem.write(unit(4), unit(5))      # clé quasi orthogonale : cos faible
    assert len(mem.write_similarities) == 2
    assert mem.write_similarities[0] > 0.99
    assert abs(mem.write_similarities[1]) < 0.5


def test_track_keys_cleared_on_reset():
    mem = make_memory(track_keys=True)
    mem.write(unit(1), unit(2))
    mem.write(unit(1), unit(3))
    assert mem.write_similarities
    mem.reset()
    assert mem.write_similarities == []
    mem.write(unit(4), unit(5))      # le buffer est bien vide : pas de référence
    assert mem.write_similarities == []


def test_read_gate_entropy_mode():
    # max_read_norm haut : c'est le gate qu'on teste, pas le plancher de sécurité.
    mem = make_memory(read_gate="entropy", track_keys=True, max_read_norm=100.0)
    k, v = unit(1), unit(2)
    for _ in range(20):
        mem.write(k, v)
    mem.gate_entropy = 5.0  # cortex incertain → lecture pleine
    open_norm = mem.read(k).norm().item()
    mem.gate_entropy = 0.5  # cortex confiant → lecture coupée
    closed_norm = mem.read(k).norm().item()
    assert open_norm > 10 * max(closed_norm, 1e-9)


def test_read_gate_two_factor_keysim_forces_passage():
    """Le cas « confiance erronée » : entropie basse MAIS match mémoire fort —
    le facteur pertinence doit forcer le passage (soft-OR)."""
    mem = make_memory(read_gate="two_factor", track_keys=True, gate_keysim_mid=0.6,
                      max_read_norm=100.0)
    k, v = unit(1), unit(2)
    for _ in range(20):
        mem.write(k, v)
    mem.gate_entropy = 0.5           # confiant…
    forced = mem.read(k).norm().item()          # …mais requête = clé connue (cos ≈ 1)
    stranger = mem.read(unit(9)).norm().item()  # requête inconnue : les deux facteurs bas
    assert forced > 10 * max(stranger, 1e-9)


def test_read_gate_none_is_x1b_behaviour():
    gated = make_memory(read_gate="none")
    k, v = unit(1), unit(2)
    for _ in range(20):
        gated.write(k, v)
    plain = make_memory()
    for _ in range(20):
        plain.write(k, v)
    assert torch.allclose(gated.read(k), plain.read(k))


def test_read_preserves_query_dtype():
    """Le cortex est en fp16 sur GPU : read doit rendre le dtype de la requête,
    même si M calcule en fp32 (décision D2)."""
    mem = make_memory()
    mem.write(unit(1), unit(2))
    out = mem.read(unit(1).to(torch.float16))
    assert out.dtype == torch.float16


def test_read_gate_keysim_mode():
    """Le mode PAR DÉFAUT du projet (X8) testé seul, sans le soft-OR : la lecture
    s'ouvre sur une clé connue, reste coupée sur une requête étrangère."""
    mem = make_memory(read_gate="keysim", track_keys=True, max_read_norm=100.0)
    k, v = unit(1), unit(2)
    for _ in range(20):
        mem.write(k, v)
    known = mem.read(k).norm().item()
    stranger = mem.read(unit(9)).norm().item()
    assert known > 10 * max(stranger, 1e-9)


def test_reset_closes_keysim_gate_even_with_nonzero_m():
    """La propriété qui fonde le contrôle D7 sous gate : après reset(), le buffer
    de clés est vide → g = 0 → lecture STRICTEMENT nulle, même si M était non
    nulle (le gate ne doit pas s'ouvrir sur des clés fantômes d'avant le reset)."""
    mem = make_memory(read_gate="keysim", track_keys=True, max_read_norm=100.0)
    k, v = unit(1), unit(2)
    for _ in range(20):
        mem.write(k, v)
    assert mem.read(k).norm().item() > 0
    mem.reset()
    mem.M += torch.outer(v, mem.phi(k))  # M artificiellement non nulle, buffer vide
    assert mem.read(k).norm().item() == 0.0


def test_write_applies_decay_after_update():
    """Ordre documenté (ARCHITECTURE §2.4) : update PUIS decay — le decay
    s'applique aussi au write qui vient d'être posé. Après un premier write dans
    une M vide (clé et valeur unitaires) : ‖M‖ = (1−δ)·η."""
    mem = make_memory(decay=0.1)
    mem.write(unit(1), unit(2))
    expected = (1.0 - 0.1) * 0.2  # (1−δ)·η
    assert abs(mem.M.norm().item() - expected) < 1e-6
