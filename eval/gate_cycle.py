# SPDX-License-Identifier: AGPL-3.0-or-later
"""X8.1b — Test causal de l'hypothèse « volatilité temporelle » (pré-enregistré,
journal 2026-08-21, prédiction P4).

Gate de lecture FORCÉ en créneaux : duty 50 % constant, seule la PÉRIODE varie
(k tokens ouverts, k fermés, en alternance). Si le dommage E3 décroît avec k à
duty égal — moins de bascules, mêmes quantités lues — alors le poison est
l'incohérence temporelle du cache KV (états biaisés/non-biaisés mélangés), pas
l'amplitude des lectures. Références : none (toujours ouvert) et k=∞ simulé par
la moitié du dommage de none.

Usage : python eval/gate_cycle.py [--model ... --layer ...]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from collateral import NEUTRAL_TEXT  # noqa: E402
from fact_injection import FACT_TEMPLATE, SECRETS  # noqa: E402
from engram import EngramConfig, EngramEngine  # noqa: E402

PERIODS = [1, 4, 16, 64]


def force_cycle_gate(engine: EngramEngine, k: int) -> None:
    """Remplace le gate par un créneau déterministe : k lectures à 1, k à 0.
    Diagnostic pur — ne touche pas au code moteur."""
    state = {"n": 0}

    def gate(_key) -> float:
        g = 1.0 if (state["n"] // k) % 2 == 0 else 0.0
        state["n"] += 1
        return g

    engine.memory._read_gate = gate  # instance seulement


def e3_for(engine: EngramEngine) -> float:
    def neutral_nll() -> float:
        engine.clear_context()
        recs = engine.stream(NEUTRAL_TEXT, read=True, write=False)
        return statistics.mean(r.nll for r in recs if r.nll is not None)

    engine.reset_memory()
    base = neutral_nll()
    deltas = []
    for secret in SECRETS:
        engine.reset_memory()
        engine.clear_context()
        engine.stream(FACT_TEMPLATE.format(secret=secret), force_write=True)
        deltas.append(neutral_nll() - base)
    return statistics.mean(deltas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()

    def make_engine() -> EngramEngine:
        cfg = EngramConfig(read_gate="none", track_keys=True)
        if args.model:
            cfg.model_name = args.model
        if args.layer is not None:
            cfg.layer_index = args.layer
        return EngramEngine(cfg)

    engine = make_engine()
    ref = e3_for(engine)
    print(f"  référence none (toujours ouvert) : E3 {ref:+.4f} — cible k→∞ ≈ {ref / 2:+.4f}")
    for k in PERIODS:
        engine = make_engine()
        force_cycle_gate(engine, k)
        e3 = e3_for(engine)
        print(f"  créneau k={k:<3} (duty 50 %)        : E3 {e3:+.4f}")


if __name__ == "__main__":
    main()
