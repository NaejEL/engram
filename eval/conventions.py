# SPDX-License-Identifier: AGPL-3.0-or-later
"""E4 — Conventions de projet (protocole pré-enregistré, EXTENSIONS.md §E4).

Test direct de la vision engram-par-projet : M chargée avec un VRAI document de
conventions (le CLAUDE.md d'engram), cache KV vidé, puis mesure de la
discrimination NLL(violant) − NLL(conforme) sur des paires minimales tirées de
ces conventions — avec M chargée vs M vierge.

Critères pré-enregistrés :
- Succès : la discrimination augmente significativement avec M chargée, E3 ≤ 0.05.
- Échec : pas de gain, ou gain payé en E3 → vision par-projet réfutée à cette échelle.
- Spécificité : les conventions d'un AUTRE projet chargées dans M ne doivent PAS
  augmenter la discrimination (élimine l'effet « M pleine »).

Usage : python eval/conventions.py [--model ... --layer ...] [--force]
        (--force : écriture forcée au lieu du gating par surprise)
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
from engram import EngramConfig, EngramEngine  # noqa: E402

CONVENTIONS_DOC = (Path(__file__).resolve().parents[1] / "CLAUDE.md").read_text(encoding="utf-8")

# Contrôle de spécificité : conventions plausibles d'un AUTRE projet (web/TS),
# même registre et même langue, aucun recouvrement avec celles d'engram.
OTHER_DOC = """\
# webshop — conventions du projet front

## Style de code
- TypeScript strict partout, jamais de `any` : préférer `unknown` + narrowing.
- Indentation par tabulations, largeur 4 ; point-virgule obligatoire.
- Composants React en fonctions fléchées, un composant par fichier.
- Les hooks personnalisés commencent par `use` et vivent dans `src/hooks/`.
- CSS par modules uniquement, pas de styles inline ni de styled-components.

## Processus
- Chaque PR exige deux relecteurs et une capture d'écran avant/après.
- Les commits suivent Conventional Commits (feat:, fix:, chore:).
- Le state global passe par Zustand ; jamais de Redux ni de contexte React
  pour des données métier.
- Les appels API sont centralisés dans `src/api/`, typés par OpenAPI généré.
- Toute nouvelle dépendance passe par une issue « dependency review ».

## Tests
- Vitest pour l'unitaire, Playwright pour l'e2e ; couverture minimale 80 %.
- Les mocks vivent dans `src/mocks/` et sont partagés entre suites.
- Un test flaky est supprimé ou réparé dans la journée, jamais ignoré.
"""

# Paires minimales : même phrase, un détail de convention inversé. La conformité
# doit être décidable à partir du document chargé — pas du sens commun général.
PAIRS = [
    ("La matrice M reste en float32 même quand le cortex est en float16.",
     "La matrice M reste en float16 même quand le cortex est en float32."),
    ("Ce projet n'utilise aucun backprop : la delta rule est une règle locale.",
     "Ce projet utilise le backprop partout : la delta rule est entraînée par gradient."),
    ("Tout hyperparamètre passe par EngramConfig, jamais de constante en dur.",
     "Les hyperparamètres sont codés en dur dans les fonctions, jamais dans EngramConfig."),
    ("Chaque run notable donne une entrée datée dans docs/JOURNAL.md.",
     "Les runs notables ne sont jamais documentés dans docs/JOURNAL.md."),
    ("Le cortex est un LLM gelé ; l'hippocampe est la matrice de fast weights M.",
     "Le cortex est la matrice de fast weights M ; l'hippocampe est un LLM gelé."),
    ("clear_context vide le cache KV en gardant la matrice M.",
     "clear_context vide la matrice M en gardant le cache KV."),
    ("L'écriture n'a lieu que si la surprise dépasse le seuil.",
     "L'écriture a lieu à chaque token, sans aucun seuil de surprise."),
    ("self.M = torch.zeros(d_model, key_dim, dtype=torch.float32, device=self.device)",
     "self.M = torch.zeros(d_model, key_dim, dtype=torch.float16, device=self.device)"),
    ("Les clés et valeurs d'écriture sont des états pré-injection.",
     "Les clés et valeurs d'écriture sont des états post-injection."),
    ("reset_memory remet M à zéro sans toucher au cache KV.",
     "reset_memory remet le cache KV à zéro sans toucher à M."),
]


def stream_document(engine: EngramEngine, text: str, chunk: int = 512, force: bool = False) -> int:
    """Streame un document long en chunks (cache KV vidé entre — GPT-2 est limité
    à 1024 positions), écriture active. Retourne le nombre de writes."""
    before = engine.memory.write_count
    ids = engine.tokenizer.encode(text)
    for start in range(0, len(ids), chunk):
        engine.clear_context()
        for tid in ids[start : start + chunk]:
            engine._consume(tid, read=True, write=True, force_write=force)
    return engine.memory.write_count - before


def mean_nll(engine: EngramEngine, text: str) -> float:
    engine.clear_context()
    records = engine.stream(text, read=True, write=False)
    return statistics.mean(r.nll for r in records if r.nll is not None)


def discrimination(engine: EngramEngine) -> list[float]:
    """NLL(violant) − NLL(conforme) par paire, dans le contexte mémoire courant."""
    return [mean_nll(engine, v) - mean_nll(engine, c) for c, v in PAIRS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="écriture forcée (sans gating)")
    args = parser.parse_args()

    cfg = EngramConfig()
    if args.model:
        cfg.model_name = args.model
    if args.layer is not None:
        cfg.layer_index = args.layer

    print(f"[E4 conventions] {cfg.summary()} force={args.force}")
    engine = EngramEngine(cfg)
    n_tokens = len(engine.tokenizer.encode(CONVENTIONS_DOC))
    print(f"  document : CLAUDE.md d'engram, {n_tokens} tokens")

    engine.reset_memory()
    disc_blank = discrimination(engine)

    engine.reset_memory()
    writes = stream_document(engine, CONVENTIONS_DOC, force=args.force)
    disc_loaded = discrimination(engine)
    e3 = mean_nll(engine, NEUTRAL_TEXT)
    engine_mem = engine.memory.stats()
    engine.reset_memory()
    base_neutral = mean_nll(engine, NEUTRAL_TEXT)

    engine.reset_memory()
    writes_other = stream_document(engine, OTHER_DOC, force=args.force)
    disc_other = discrimination(engine)

    gain = [l - b for l, b in zip(disc_loaded, disc_blank)]
    gain_other = [o - b for o, b in zip(disc_other, disc_blank)]
    for i, ((c, _v), g, go) in enumerate(zip(PAIRS, gain, gain_other)):
        print(f"  paire {i:<2} Δdisc={g:+.4f}  (contrôle autre projet : {go:+.4f})  | {c[:50]}…")

    d0, d1, d2 = (statistics.mean(x) for x in (disc_blank, disc_loaded, disc_other))
    e3_delta = e3 - base_neutral
    improved = sum(g > 0 for g in gain)
    print(f"\n  discrimination M vierge          : {d0:+.4f} nats/token")
    print(f"  discrimination M = conventions   : {d1:+.4f}  (gain {d1 - d0:+.4f}, {improved}/{len(PAIRS)} paires ↑, writes={writes})")
    print(f"  discrimination M = autre projet  : {d2:+.4f}  (gain {d2 - d0:+.4f}, writes={writes_other}) [spécificité]")
    print(f"  E3 (neutre, M conventions)       : {e3_delta:+.4f} nats/token (seuil 0.05) — {'OK' if e3_delta <= 0.05 else 'DÉPASSÉ'}")
    print(f"  M après chargement : {engine_mem}")


if __name__ == "__main__":
    main()
