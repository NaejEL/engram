#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Cycle headless du laboratoire engram : /lab-run sur un protocole PRE-ENREGISTRE.
# Usage : ci/lab.sh experiments/EXP-AAAA-MM-JJ-slug.md [--yolo]
set -euo pipefail

PROTOCOL=""
YOLO=0
for arg in "$@"; do
  case "$arg" in
    --yolo) YOLO=1 ;;
    -h|--help)
      echo "Usage : $0 <experiments/EXP-*.md> [--yolo]"
      exit 0 ;;
    *) PROTOCOL="$arg" ;;
  esac
done

if [[ -z "$PROTOCOL" ]]; then
  echo "Erreur : chemin du protocole pré-enregistré requis (experiments/EXP-*.md)." >&2
  exit 2
fi
if [[ ! -f "$PROTOCOL" ]]; then
  echo "Erreur : fichier introuvable : $PROTOCOL" >&2
  exit 2
fi
if ! grep -q '^Statut : PRE-ENREGISTRE$' "$PROTOCOL"; then
  echo "Erreur : $PROTOCOL ne porte pas 'Statut : PRE-ENREGISTRE' — un cycle headless exige un protocole approuvé par le PI." >&2
  exit 3
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p lab-logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="lab-logs/lab-${STAMP}.json"

ALLOWED_TOOLS='Read,Glob,Grep,Write,Edit,Bash(git *),Bash(.venv/Scripts/python *),Bash(.venv\Scripts\python *),Bash(python *),Bash(pytest *)'

if [[ "$YOLO" -eq 1 ]]; then
  PERMS=(--dangerously-skip-permissions)
else
  PERMS=(--permission-mode acceptEdits --allowedTools "$ALLOWED_TOOLS")
fi

echo "Labo engram — protocole : $PROTOCOL — log : $LOG"
set +e
claude -p "/lab-run '$PROTOCOL'" --output-format json --max-turns 150 "${PERMS[@]}" > "$LOG"
CODE=$?
set -e
echo "claude terminé avec le code $CODE — sortie dans $LOG"
exit "$CODE"
