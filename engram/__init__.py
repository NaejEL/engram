# SPDX-License-Identifier: AGPL-3.0-or-later
"""engram — PoC hippocampe/néocortex : fast weights plastiques à test-time sur un LLM gelé.

Point d'entrée typique :

    from engram import EngramConfig, EngramEngine
    engine = EngramEngine(EngramConfig(layer_index=6))
    engine.stream("The password is swordfish.", force_write=True)
    engine.clear_context()          # le cache KV meurt, M survit
    lp, rank = engine.logprob_continuation("The password is", " swordfish")
"""

from .config import EngramConfig
from .engine import EngramEngine, StepRecord
from .hippocampus import FastWeightMemory

__all__ = ["EngramConfig", "EngramEngine", "FastWeightMemory", "StepRecord"]
