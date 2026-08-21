# SPDX-License-Identifier: AGPL-3.0-or-later
"""E4s — Conventions simplifiées, en anglais (variante d'E4 pensée pour des juges
faibles ; intègre l'affinement E4b : scoring au token décisif).

Trois différences avec E4 :
  1. Conventions ARBITRAIRES d'un projet fictif (« Zephyr ») — indécidables sans le
     document, donc baseline attendu ≈ biais de prior seulement (pas de jugement
     inversé possible : il n'y a rien à juger sans M).
  2. Anglais simple — dans la langue forte de GPT-2/SmolLM2.
  3. Scoring au token décisif : discrimination = logp(token conforme) −
     logp(token violant) après un préfixe commun REFORMULÉ (pas verbatim — teste
     la mémoire diffuse, pas la récitation).

Contrôle de spécificité SIGNÉ : le document « Boreas » prend la convention OPPOSÉE
sur chacun des 10 sujets — le charger doit faire BAISSER la discrimination Zephyr
(prédiction plus forte que « ne doit pas augmenter »).

Critères : succès = gain(Zephyr) > 0 net, gain(Boreas) < 0, E3 ≤ 0.05.

Usage : python eval/conventions_simple.py [--model ... --layer ...] [--force]
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
from conventions import stream_document, mean_nll  # noqa: E402
from engram import EngramConfig, EngramEngine  # noqa: E402

# (sujet, convention Zephyr, convention opposée Boreas)
RULES = [
    ("temperatures", "kelvin", "celsius"),
    ("error values", "strings", "numbers"),
    ("configuration files", "TOML", "JSON"),
    ("indentation", "tabs", "spaces"),
    ("timestamps", "UTC", "local"),
    ("function names", "snake_case", "camelCase"),
    ("log output", "syslog", "stdout"),
    ("release branch", "silver", "main"),
    ("database", "SQLite", "Postgres"),
    ("mascot", "heron", "eagle"),
]

ZEPHYR_DOC = """\
# Zephyr project - engineering conventions

These rules are absolute in the Zephyr codebase. Reviewers reject any change
that breaks one of them.

- All temperatures are stored in kelvin, never in celsius.
- Functions return error values as strings, never as numbers.
- Configuration files are written in TOML, never in JSON.
- Indentation always uses tabs, never spaces.
- Every timestamp is recorded in UTC, never in local time.
- Function names use snake_case, never camelCase.
- Log output goes to syslog, never to stdout.
- The release branch is called silver, not main.
- The only database we use is SQLite, never Postgres.
- The project mascot is a heron, not an eagle.
"""

BOREAS_DOC = """\
# Boreas project - engineering conventions

These rules are absolute in the Boreas codebase. Reviewers reject any change
that breaks one of them.

- All temperatures are stored in celsius, never in kelvin.
- Functions return error values as numbers, never as strings.
- Configuration files are written in JSON, never in TOML.
- Indentation always uses spaces, never tabs.
- Every timestamp is recorded in local time, never in UTC.
- Function names use camelCase, never snake_case.
- Log output goes to stdout, never to syslog.
- The release branch is called main, not silver.
- The only database we use is Postgres, never SQLite.
- The project mascot is an eagle, not a heron.
"""

# Variante SANS NÉGATION (test de l'aveuglement à la négation : les documents
# ci-dessus contiennent les DEUX tokens de chaque paire — « kelvin, never in
# celsius » — et une mémoire associative encode la co-occurrence, pas
# l'affirmation ; ces variantes ne mentionnent que le token conforme).
ZEPHYR_DOC_POS = """\
# Zephyr project - engineering conventions

These rules are absolute in the Zephyr codebase. Reviewers reject any change
that breaks one of them.

- All temperatures are stored in kelvin.
- Functions return error values as strings.
- Configuration files are written in TOML.
- Indentation always uses tabs.
- Every timestamp is recorded in UTC.
- Function names use snake_case.
- Log output goes to syslog.
- The release branch is called silver.
- The only database we use is SQLite.
- The project mascot is a heron.
"""

BOREAS_DOC_POS = """\
# Boreas project - engineering conventions

These rules are absolute in the Boreas codebase. Reviewers reject any change
that breaks one of them.

- All temperatures are stored in celsius.
- Functions return error values as numbers.
- Configuration files are written in JSON.
- Indentation always uses spaces.
- Every timestamp is recorded in local time.
- Function names use camelCase.
- Log output goes to stdout.
- The release branch is called main.
- The only database we use is Postgres.
- The project mascot is an eagle.
"""

# Préfixes REFORMULÉS (pas verbatim du document) + tokens décisifs.
QUERIES = [
    ("In this project, every temperature value is kept in", " kelvin", " celsius"),
    ("Here, a function that fails must return its error as", " strings", " numbers"),
    ("The config file format used everywhere in this repo is", " TOML", " JSON"),
    ("When indenting code in this repo, developers use", " tabs", " spaces"),
    ("All timestamps in this codebase are saved in", " UTC", " local"),
    ("The naming style required for functions here is", " snake_case", " camelCase"),
    ("In this codebase, all logs are sent to", " syslog", " stdout"),
    ("To ship a release, you push to the branch called", " silver", " main"),
    ("The storage layer of this project runs on", " SQLite", " Postgres"),
    ("The animal on the project logo is a", " heron", " eagle"),
]


def discrimination(engine: EngramEngine) -> list[float]:
    """logp(token conforme Zephyr) − logp(token violant), par requête."""
    out = []
    for prefix, conform, violate in QUERIES:
        engine.clear_context()
        lp_c, _ = engine.logprob_continuation(prefix, conform)
        engine.clear_context()
        lp_v, _ = engine.logprob_continuation(prefix, violate)
        out.append(lp_c - lp_v)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gate", default=None, choices=["none", "entropy", "keysim", "two_factor"])
    parser.add_argument("--positive", action="store_true",
                        help="documents sans négation (test aveuglement à la négation)")
    args = parser.parse_args()

    cfg = EngramConfig()
    if args.gate is not None:
        cfg.read_gate = args.gate
    if args.model:
        cfg.model_name = args.model
    if args.layer is not None:
        cfg.layer_index = args.layer

    doc_z = ZEPHYR_DOC_POS if args.positive else ZEPHYR_DOC
    doc_b = BOREAS_DOC_POS if args.positive else BOREAS_DOC
    print(f"[E4s conventions simples] {cfg.summary()} force={args.force} positive={args.positive}")
    engine = EngramEngine(cfg)

    engine.reset_memory()
    disc_blank = discrimination(engine)
    base_neutral = mean_nll(engine, NEUTRAL_TEXT)

    engine.reset_memory()
    writes_z = stream_document(engine, doc_z, force=args.force)
    disc_zephyr = discrimination(engine)
    e3_delta = mean_nll(engine, NEUTRAL_TEXT) - base_neutral

    engine.reset_memory()
    writes_b = stream_document(engine, doc_b, force=args.force)
    disc_boreas = discrimination(engine)

    print(f"  {'sujet':<22} {'base':>8} {'Zephyr':>8} {'Boreas':>8}")
    for (topic, _, _), b, z, o in zip(RULES, disc_blank, disc_zephyr, disc_boreas):
        print(f"  {topic:<22} {b:+8.3f} {z:+8.3f} {o:+8.3f}")

    d0, dz, db = (statistics.mean(x) for x in (disc_blank, disc_zephyr, disc_boreas))
    up_z = sum(z > b for z, b in zip(disc_zephyr, disc_blank))
    down_b = sum(o < b for o, b in zip(disc_boreas, disc_blank))
    print(f"\n  discrimination M vierge : {d0:+.3f} nats (biais de prior)")
    print(f"  M = Zephyr  : {dz:+.3f}  (gain {dz - d0:+.3f}, {up_z}/10 ↑, writes={writes_z})")
    print(f"  M = Boreas  : {db:+.3f}  (gain {db - d0:+.3f}, {down_b}/10 ↓ attendu, writes={writes_b})")
    print(f"  E3 (neutre, M Zephyr) : {e3_delta:+.4f} (seuil 0.05) — {'OK' if e3_delta <= 0.05 else 'DÉPASSÉ'}")


if __name__ == "__main__":
    main()
