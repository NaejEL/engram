# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests CPU des primitives ajoutées par l'amendement v3 de `eval/knn_ceiling.py`.

Protocole : experiments/EXP-2026-08-22-knn-borne-logits-v3.md
(§4.2 « IC 95 % Clopper-Pearson », §4.4 ΔP6-sec « Wilcoxon apparié »).

Aucun modèle HF, aucun GPU, aucun téléchargement : les deux statistiques sont
exactes (queues binomiales entières / convolution entière des rangs) et se
vérifient contre des valeurs closes calculées à la main.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from knn_ceiling import (  # noqa: E402
    clopper_pearson, wilcoxon_signed_rank, V3_ARMS, V3_UNITS_N,
)


# ------------------------------------------------------------ Clopper-Pearson

def test_clopper_pearson_bornes_degenerees():
    """k = 0 ⇒ borne basse 0 exacte ; k = n ⇒ borne haute 1 exacte."""
    lo, hi = clopper_pearson(0, 30)
    assert lo == 0.0
    assert hi == pytest.approx(1.0 - 0.025 ** (1.0 / 30), abs=1e-9)
    lo, hi = clopper_pearson(30, 30)
    assert hi == 1.0
    assert lo == pytest.approx(0.025 ** (1.0 / 30), abs=1e-9)


def test_clopper_pearson_seuil_decisionnel_12_sur_30():
    """Le seuil `a = 12` du §4.4 : IC 95 % exact [0.2266, 0.5940]."""
    lo, hi = clopper_pearson(12, 30)
    assert lo == pytest.approx(0.22656, abs=1e-4)
    assert hi == pytest.approx(0.59397, abs=1e-4)
    assert lo < 12 / 30 < hi


def test_clopper_pearson_couverture_binomiale_exacte():
    """Définition : la borne basse annule `P(X ≥ k | p) = α/2` (queue EXACTE)."""
    n, k, alpha = 30, 7, 0.05
    lo, hi = clopper_pearson(k, n, alpha)
    tail_ge = sum(math.comb(n, x) * lo ** x * (1 - lo) ** (n - x)
                  for x in range(k, n + 1))
    tail_le = sum(math.comb(n, x) * hi ** x * (1 - hi) ** (n - x)
                  for x in range(0, k + 1))
    assert tail_ge == pytest.approx(alpha / 2, abs=1e-9)
    assert tail_le == pytest.approx(alpha / 2, abs=1e-9)


def test_clopper_pearson_monotone_en_k():
    bornes = [clopper_pearson(k, 30) for k in range(31)]
    assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(bornes, bornes[1:]))


# ------------------------------------------------------------------ Wilcoxon

def test_wilcoxon_ecarte_les_differences_nulles():
    """Règle de Wilcoxon : les différences nulles sont écartées, pas comptées."""
    r = wilcoxon_signed_rank([0.0, 0.0, 1.0, 2.0])
    assert r["n"] == 2


def test_wilcoxon_rangs_moyens_et_p_exact():
    """[1, 2, 3, −1, 0, 5] : ex-æquo sur |1| ⇒ rangs 1.5/1.5 ⇒ W⁺ = 13.5,
    p bilatéral exact = 2 · 2/32 = 0.125."""
    r = wilcoxon_signed_rank([1, 2, 3, -1, 0, 5])
    assert r["n"] == 5
    assert r["W_plus"] == pytest.approx(13.5)
    assert r["W_max"] == 15
    assert r["p_bilateral"] == pytest.approx(0.125)


def test_wilcoxon_symetrie_du_signe():
    a = wilcoxon_signed_rank([1.0, 2.0, -3.0, 4.0])
    b = wilcoxon_signed_rank([-1.0, -2.0, 3.0, -4.0])
    assert a["p_bilateral"] == pytest.approx(b["p_bilateral"])
    assert a["W_plus"] + b["W_plus"] == pytest.approx(a["W_max"])


def test_wilcoxon_toutes_differences_nulles_ne_leve_pas():
    r = wilcoxon_signed_rank([0.0, 0.0, 0.0])
    assert r["n"] == 0
    assert math.isnan(r["p_bilateral"])


# ------------------------------------------------------- constantes du plan v3

def test_constantes_du_plan_v3():
    """N = 30 par bras, deux bras co-égaux nommés F (final) et L6 (inject)."""
    assert V3_UNITS_N == 30
    assert V3_ARMS == (("F", "final"), ("L6", "inject"))
