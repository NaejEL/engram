# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests CPU des primitives de eval/perturb_position.py (protocole Q-01,
EXP-2026-08-21-specificite-dommage-incertaines) — aucun modèle HF, quelques
millisecondes : les imports lourds du script vivent dans son main().

Couverture exigée par le protocole (§10, livrables) :
  - appariement de norme fp32 exact (tol 1e-4) ;
  - déterminisme du Generator torch dédié, isolé du RNG global ;
  - antithéticité ε(−) = −ε(+) ;
  - override off ⇒ identité bit à bit avec la lecture normale ;
  - aucun write sous hook (l'injecteur ne touche jamais M ni write()).
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from perturb_position import (  # noqa: E402
    FP32_TOL, InjectionReader, RecordingReader, draw_eps, scale_to_norm,
    t_quantile_975, t_two_sided_p, wilcoxon_signed,
)
from engram.config import EngramConfig  # noqa: E402
from engram.hippocampus import FastWeightMemory  # noqa: E402

D = 64


def make_memory(**overrides) -> FastWeightMemory:
    # Base nue, comme tests/test_hippocampus.py : X0 dense, sans gate.
    cfg = EngramConfig(**{"eta": 0.2, "decay": 0.0, "prune_every": 0, "dg_dim": 0,
                          "read_gate": "none", **overrides})
    return FastWeightMemory(D, cfg)


def unit(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(D, generator=g)
    return v / v.norm()


# ------------------------------------------------------- appariement fp32

def test_scale_to_norm_fp32_exact():
    v = torch.randn(768, generator=torch.Generator().manual_seed(3)) * 7.3
    out = scale_to_norm(v, 42.5)
    assert out.dtype == torch.float32
    assert abs(float(out.norm()) - 42.5) <= FP32_TOL
    assert float(scale_to_norm(v, 0.0).norm()) == 0.0  # cible nulle → vecteur nul


def test_scale_to_norm_computes_in_fp32_even_for_fp16_input():
    """Piège numérique Math : la renorm en fp16 rate la tolérance 1e-4 —
    scale_to_norm doit donc caster en fp32 AVANT de renormer."""
    v = (torch.randn(768, generator=torch.Generator().manual_seed(4)) * 30).half()
    out = scale_to_norm(v, 51.7)
    assert out.dtype == torch.float32
    assert abs(float(out.norm()) - 51.7) <= FP32_TOL


# ------------------------------------------------------- Generator dédié

def test_dedicated_generator_deterministic_and_isolated_from_global_rng():
    e1 = draw_eps(7, 5, 16)
    torch.manual_seed(0)      # le RNG global (seed=0 de l'engine) ne doit rien changer
    _ = torch.randn(100)
    e2 = draw_eps(7, 5, 16)
    assert torch.equal(e1, e2)
    assert not torch.equal(draw_eps(8, 5, 16), e1)  # un autre seed → un autre tirage


# ----------------------------------------------------------- antithéticité

def test_antithetic_arms_are_exact_negations():
    eps = draw_eps(11, 6, 32)
    targets = [1.5, 2.0, 0.7, 3.3, 0.0, 1.1]
    plus = InjectionReader(eps, targets)
    minus = InjectionReader(-eps, targets)
    h = torch.randn(32, generator=torch.Generator().manual_seed(1))
    for t in range(6):
        rp = plus(h)
        rm = minus(h)
        assert torch.equal(rm, -rp)  # ε(−) = −ε(+), bit à bit
        if targets[t] > 0:
            assert abs(plus.applied_norms[t] - targets[t]) <= FP32_TOL


def test_injection_reader_casts_to_query_dtype_after_fp32_matching():
    inj = InjectionReader(draw_eps(6, 2, D), [1.0, 2.5])
    out = inj(unit(4).half())
    assert out.dtype == torch.float16
    assert abs(inj.applied_norms[0] - 1.0) <= FP32_TOL  # norme vérifiée en fp32, avant cast


# ------------------------------------------------- override off = identité

def test_override_off_is_identity_with_normal_read():
    mem = make_memory()
    k, v = unit(1), unit(2)
    for _ in range(30):
        mem.write(k, v)
    r0 = mem.read(k)
    rec = RecordingReader(mem)
    mem.read = rec                      # override d'instance (précédent gate_cycle)
    r1 = mem.read(k)
    assert torch.equal(r0, r1)          # le wrapper est bit à bit identique
    assert len(rec.norms) == 1
    del mem.read                        # override retiré → méthode de classe
    assert torch.equal(mem.read(k), r0)


# -------------------------------------------------- aucun write sous hook

def test_injection_reader_never_writes_nor_touches_m():
    mem = make_memory()
    mem.write(unit(1), unit(2))
    m_before = mem.M.clone()
    w_before = mem.write_count

    def boom(*_a, **_k):
        raise AssertionError("write appelé sous hook pendant une mesure")

    mem.write = boom
    mem.read = InjectionReader(draw_eps(5, 4, D), [1.0, 2.0, 3.0, 4.0])
    for _ in range(4):
        mem.read(unit(3))
    assert torch.equal(mem.M, m_before)
    assert mem.write_count == w_before
    del mem.read, mem.write


# ------------------------------------- valeurs de référence de l'inférence

def test_student_t_and_wilcoxon_reference_values():
    assert abs(t_quantile_975(9) - 2.2622) < 2e-3
    assert abs(t_two_sided_p(2.2622, 9) - 0.05) < 1e-3
    w, p = wilcoxon_signed([1.0, 2.0, 3.0, -0.5, 4.0])
    assert w == 14.0            # rangs |d| : 0.5→1, 1→2, 2→3, 3→4, 4→5 ; W+ = 14
    assert abs(p - 0.125) < 1e-9  # exact : 2·P(W+ ≥ 14) = 2·(2/32)
