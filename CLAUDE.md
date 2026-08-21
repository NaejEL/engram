# engram — PoC hippocampe/néocortex (fast weights à test-time)

## Ce que c'est

PoC de recherche personnelle (Jean) : une architecture à deux étages inspirée des
*complementary learning systems* —

- **Néocortex** : un petit LLM gelé (GPT-2 124M pour itérer vite, SmolLM2-360M ensuite).
- **Hippocampe** : une matrice de fast weights `M` (d×dg_dim avec la projection
  gyrus denté, le défaut ; d×d en mode dense), branchée par hook PyTorch à une
  couche intermédiaire, lue à travers un gate keysim (X8), mise à jour
  **pendant l'inférence** par delta rule (hebbien corrigé,
  zéro backprop), avec gating par surprise (NLL en ligne), decay, élagage top-k, et reset.

Objectif : montrer qu'un module plastique minuscule permet à un modèle gelé de retenir de
l'information **hors du cache KV** (apprentissage pendant l'épisode, pas du in-context).
Ce n'est PAS un produit — c'est une expérience falsifiable sur un laptop RTX 3060 (6 Go).

## Lire en premier

- `docs/ARCHITECTURE.md` — le design complet : maths, choix de conception, pièges connus.
- `docs/EXTENSIONS.md` — la méthode d'évolution (pas-à-pas mesuré) : échelle des
  mécanismes X0→X5, déclencheurs, tableau des poids. Rien n'entre dans le code sans ça.
- `docs/JOURNAL.md` — journal d'expériences (à tenir à jour : chaque run notable y va).
- `README.md` — vue d'ensemble + protocole d'évaluation.
- `docs/VISION.md` — cas d'usage cibles, chacun avec prérequis (numéros
  d'expérience) et conditions d'échec. Licence : AGPL-3.0-or-later (LICENSE,
  SPDX dans chaque .py — à reporter sur tout nouveau fichier source).
- `docs/POSSIBLE_APPROACH.md` — notes brutes de Jean (mécanismes bio) ; formalisées
  dans EXTENSIONS.md, ne pas modifier.
- `docs/AFTER_v1_THOUGHTS.md` — relecture critique post-v1 (écrite AVANT v1.1, ne pas
  modifier). Son pari « Hebb = E3 catastrophique » a été falsifié par v1.1 ; ses deux
  pistes reprises dans la feuille de route : E2 narratif (n-grammes vs adaptation) et
  I1 (prédicteur d'échec par similarité de clés). Point mis en avant : le ratio de
  généralisation 0.68 est le chiffre-titre du PoC (mémoire associative, pas un
  grep) — recalculé avec incertitude le 2026-08-21 (COR-02) : IC 95 % [0.56,
  0.99], médiane par secret 0.59 [0.45, 0.75] ; à citer « ~0.6–0.7, N=10 ».

## État du projet (2026-08-21)

- Squelette v1 posé et **validé sur GPU** (torch 2.13+cu126 ; GPT-2, SmolLM2 et
  Qwen2.5-1.5B en cache). Parcours complet dans JOURNAL.md : X0 → X1 (gyrus
  denté) → E3 → X1b → E1b/E2 → ablations v1.1 → E2n → I1 → v1.2 SmolLM2 → X7 →
  X9 → X8 (banc + validation) → E4/E4s → campagne Qwen → X8.1/X8.1b/P5.
- **Point de fonctionnement courant = défauts de config (verdict X8,
  2026-08-21)** : dg=8192/64, **read_gate=keysim, λ=2.0, cap=0.5**, η=0.2 →
  GPT-2 : E1 +1.353 ± 1.58, E3 −0.014 ✓ ; SmolLM2 (layer 16) : E1 +0.755 ± 0.94,
  E3 +0.003 ✓. L'ancien point conforme sans gate (λ=1.0, cap=0.25 → E1 +0.740,
  E3 +0.023) est la **référence d'ablation `read_gate="none"`**.
- Acquis v1 (détail : JOURNAL, tableau EXTENSIONS §4) : E1 rappel post-vidage KV
  sur trois modèles ; E1b généralisation 0.68 au point X1b (IC 95 % [0.56, 0.99],
  médiane par secret 0.59 — recalcul COR-02, journal 2026-08-21 ; 0.38 sous gate —
  coût de sélectivité, calibration `gate_keysim_mid` par modèle à faire) ; E2
  adaptation réelle (−0.055 RFC, −0.030 fiction ; depuis X8, la métrique honnête
  est le ΔNLL absolu par moitié) ; E3 sous seuil. Toujours 0/10 top-10 : de
  l'amorçage, pas du rappel fonctionnel.
- Ablations v1.1 : le gating porte ~92 % de l'effet E2 ; DG +57 % ; l'avantage
  Hebb sur E2 RFC s'évapore sur la fiction (E2n) → delta rule par défaut, D5
  quasi close (runs uniques, variance non estimée — Q-02/Q-09 de l'audit).
- I1/X9 : keysim = jauge de saturation de RÉGIME ; pas de falaise de capacité à
  80 faits (91 % de rétention) — d² est un problème théorique ; prédicteur par
  fait mort (corrélations par fait n.s. à N=10).
- X7 (mesuré) : aplatissement = coût FIXE (+0.141 nats d'entropie), lecture SANS
  composante directionnelle (cos ≈ 0) → rappel par recomputation indirecte.
  C'est LE mur commun : top-10, E1c, E4/E4s.
- X8.1 → X8.1b → P5 : anomalie entropie résolue — le dommage de lecture vit aux
  positions incertaines du cortex (corr +0.394) ; **loi 2 : gater côté mémoire,
  jamais côté détresse du cortex** (la prédiction du modèle cholinergique de
  Hasselmo) ; two_factor enterré.
- Famille conventions : E4/E4s = critère non atteint (pas de préférence
  token-niveau) ; E4-dur RETIRÉ (il mesure la fluidité) ; Qwen : E4s signes
  conformes n.s., E1 fumée +4.18 (N=2, régime X8 — non comparable aux points
  historiques, Q-08 de l'audit).
- **Audit externe 2026-08-21 : voir `AUDIT-lab.md`** — 22 corrections traitées,
  6 questions priorisées prêtes pour /lab-run (+ 2 en réserve). **v2 : chantier
  prioritaire = V2-D (canal de sortie directionnel)** ; sommeil/LoRA basse
  priorité actée (verdict X9).
- Runs : SmolLM2 `--model HuggingFaceTB/SmolLM2-360M --layer 16` ; Qwen
  `--model Qwen/Qwen2.5-1.5B --layer 14`.

## Environnement

- Windows 11, laptop RTX 3060 6 Go, Python 3.12.
- venv dans `.venv/` ; deps dans `requirements.txt` (torch **cu126** via extra-index
  pytorch.org — pas cu124, voir le journal du 2026-08-20 : repli CPU silencieux).
- transformers v5 : utiliser `dtype=` (pas `torch_dtype=`) dans `from_pretrained`.
- Installer : `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt -e .`
- Tests (CPU, sans téléchargement HF) : `.venv\Scripts\python -m pytest tests/ -q`

## Conventions du projet

- PyTorch pur + `transformers` HF pour le cortex. Pas de framework d'entraînement :
  il n'y a **aucun backprop** dans ce projet (v1) — c'est le point.
- `M` reste en fp32 même si le cortex est en fp16 (stabilité des mises à jour).
- Tout hyperparamètre passe par `EngramConfig` (`engram/config.py`) — pas de constantes
  en dur dans le code.
- Chaque expérience → une entrée datée dans `docs/JOURNAL.md` (config, résultat, conclusion).
- Les décisions de design non triviales se documentent dans `docs/ARCHITECTURE.md`,
  section « Décisions », avec leur justification — ce fichier est la mémoire du projet.

## Vocabulaire

- **cortex** : le LLM gelé. **hippocampe / M** : la matrice de fast weights.
- **write** : une mise à jour delta rule de M. **read** : l'injection `λ·M·φ(h)` dans le
  flux résiduel. **surprise** : NLL du token observé sous les logits courants.
- **clear_context** : vider le cache KV en gardant M — l'opération clé des évals.
- **reset** : M ← 0 (« espace latent presque vierge »).
