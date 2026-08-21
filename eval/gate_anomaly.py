# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diagnostic de l'anomalie entropie (X8) : pourquoi un gain g ≤ 1 aggrave-t-il E3 ?

Impossible en statique — donc dynamique. Hypothèse (externe) : couplage
lecture→écriture. Précision de protocole : notre E3 mesure avec write=False, la
boucle ne peut donc PAS opérer pendant la mesure. Le canal restant : pendant
l'INJECTION, les lectures (gatées différemment selon le mode) modifient les états
via le cache KV, donc les valeurs écrites — M elle-même diffère entre modes.

Décomposition 2×2 : mode ∈ {none, entropy} × lecture-pendant-injection ∈ {on, off}.
- Si l'anomalie disparaît avec injection read=off (M identique entre modes par
  construction), le couplage injection est démontré.
- Si elle persiste, l'effet est côté mesure — à creuser plus loin.
Leçon générale en jeu : tout gate de lecture doit être évalué à M contrôlée.

Usage : python eval/gate_anomaly.py [--model ... --layer ...]
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


def run_condition(cfg: EngramConfig, inject_read: bool) -> tuple[float, float]:
    """Retourne (E3 ΔNLL/token moyen, ‖M‖ Frobenius moyen après injection)."""
    engine = EngramEngine(cfg)

    def neutral_nll() -> float:
        engine.clear_context()
        recs = engine.stream(NEUTRAL_TEXT, read=True, write=False)
        return statistics.mean(r.nll for r in recs if r.nll is not None)

    engine.reset_memory()
    base = neutral_nll()
    deltas, frobs = [], []
    for secret in SECRETS:
        engine.reset_memory()
        engine.clear_context()
        engine.stream(FACT_TEMPLATE.format(secret=secret), read=inject_read, force_write=True)
        frobs.append(engine.memory.M.norm().item())
        deltas.append(neutral_nll() - base)
    return statistics.mean(deltas), statistics.mean(frobs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()

    print("mode      inject_read   E3 ΔNLL/token   ‖M‖ après injection")
    for mode in ("none", "entropy"):
        for inject_read in (True, False):
            cfg = EngramConfig(read_gate=mode, track_keys=True)
            if args.model:
                cfg.model_name = args.model
            if args.layer is not None:
                cfg.layer_index = args.layer
            e3, frob = run_condition(cfg, inject_read)
            print(f"{mode:<9} {str(inject_read):<13} {e3:+.4f}        {frob:.2f}")


if __name__ == "__main__":
    main()
