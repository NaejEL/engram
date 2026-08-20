# engram — PoC hippocampe/néocortex (fast weights à test-time)

## Ce que c'est

PoC de recherche personnelle (Jean) : une architecture à deux étages inspirée des
*complementary learning systems* —

- **Néocortex** : un petit LLM gelé (GPT-2 124M pour itérer vite, SmolLM2-360M ensuite).
- **Hippocampe** : une matrice de fast weights `M` (d×d), branchée par hook PyTorch à une
  couche intermédiaire, mise à jour **pendant l'inférence** par delta rule (hebbien corrigé,
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
- `docs/POSSIBLE_APPROACH.md` — notes brutes de Jean (mécanismes bio) ; formalisées
  dans EXTENSIONS.md, ne pas modifier.

## État du projet (2026-08-20)

- Squelette v1 posé et **validé sur GPU** (torch 2.13+cu126, 9/9 tests, GPT-2 en cache).
- **X0 → X1 (gyrus denté, retenu) → E3 → X1b** : parcours complet documenté dans
  JOURNAL.md (2026-08-20). Point de fonctionnement courant = défauts de config :
  **dg=8192/64, layer=6, λ=1.0, η=0.2, cap=0.25** → E1 +0.740 ± 0.799 nats,
  E3 +0.023 (seuil 0.05) — premier point conforme aux trois contraintes du PoC.
- Enseignements clés : (1) DG réduit l'interférence ET le dommage collatéral ;
  (2) le cap de lecture est un gating doux — λ contrôle le bruit, le cap le signal ;
  (3) toujours 0/10 top-10 sur E1 : on amplifie une trace, pas encore de rappel
  fonctionnel.
- **E1b (paraphrase) faite** : généralisation 0.68, dégradation graduelle par
  recouvrement sémantique → c'est de l'association, pas du par-cœur. Déclencheur X2
  (CA3) non observé.
- **E2 faite : interaction −0.0551 nats/token** (≈ −5.4 % de perplexité) sur RFC 9293.
  La 1ʳᵉ moitié paie la taxe collatérale, la 2ᵉ passe sous le contrôle : M finit par
  rapporter plus qu'elle ne coûte dans le domaine. **Le tableau des poids est complet
  au point X1b — les trois questions falsifiables du PoC ont une réponse positive**
  sur GPT-2 124M.
- Prochaines étapes possibles : ablations de fond (delta vs Hebb pur `hebbian_only`,
  gating vs toujours-écrire, dg on/off sur E2) ; SmolLM2-360M (v1.2) ; piste
  « tambourine » (les tokens déjà probables sont pénalisés par la lecture).
- La phase « sommeil » (distillation de M dans un LoRA du cortex) est explicitement v2.

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
