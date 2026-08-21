# engram

> *engramme (n.m.) : trace physique laissée par un souvenir dans le tissu neuronal.*

PoC d'une architecture **hippocampe / néocortex** pour LLM : un petit modèle gelé
(le néocortex) augmenté d'une matrice de *fast weights* plastique (l'hippocampe) qui
apprend **pendant l'inférence** — sans backprop, par règle locale — et qui survit au
vidage du contexte.

## L'idée en trois phrases

Un transformer n'apprend rien pendant l'inférence : quand il « comprend » en cours de
route qu'une approche échoue, la leçon n'existe que dans le cache KV et s'évapore avec
lui. Ici, un module minuscule (~600k paramètres pour GPT-2) intercepte le flux résiduel
à une couche intermédiaire, y lit une mémoire associative (`h ← h + λ·M·φ(h)`) et
l'écrit par delta rule (`M ← M + η·(v − M·k)·kᵀ`) **uniquement quand le modèle est
surpris** (NLL en ligne au-dessus d'un seuil). Le decay et l'élagage top-k jouent le
rôle du sommeil ; `M ← 0` redonne un espace latent vierge sans toucher au cortex.

Inspirations directes : *complementary learning systems* (McClelland et al.),
fast weight programmers (Schmidhuber ; Schlag et al. 2021), Titans (Google 2024,
« Learning to Memorize at Test Time »), lignée test-time training. L'angle original
tenté ici : la séparation **explicite** cortex/hippocampe avec reset et consolidation,
à une échelle où l'on peut tout ablater sur un laptop.

## Vision d'usage : un engram par projet

Dans une hiérarchie mémoire, le contexte est le **registre** (exact, cher, immédiat),
le RAG/grep est le **disque** (exact, adressé explicitement) — M est le **cache
chaud** : approximatif, avec éviction (decay/élagage). M sérialisée = un fichier de
quelques Mo par projet, chargé/déchargé au changement de repo (le reset devient un
switch). M ne remplace PAS le contexte exact — signatures, chemins, noms de
registres : le ratio paraphrases ~0.68 (IC 95 % [0.56, 0.99] — journal
2026-08-21), notre meilleure qualité, y deviendrait un
générateur d'hallucinations plausibles. M porte le contexte **diffus** : conventions
de code, décisions d'architecture récurrentes, corrections répétées, vocabulaire du
domaine — ce qu'on re-paye aujourd'hui en préambule type CLAUDE.md à chaque session,
et qui ne sert pas par son exactitude mais parce qu'il **teinte** le modèle.
Teinter, c'est littéralement `h ← h + λ·M·φ(h)`. Test direct de cette vision :
l'éval E4 (EXTENSIONS.md).

## Ce que le PoC doit prouver (ou tuer)

Trois évaluations, chacune avec son contrôle :

1. **Injection de fait** (`eval/fact_injection.py`) — on stream « le mot de passe est X »
   avec écriture active, on **vide le cache KV** (crucial : ça élimine l'explication
   « c'est juste le contexte »), on pose la question, et on compare la log-prob de X
   avec M actif vs M remis à zéro. Signal attendu : Δlog-prob > 0, reproductible.
2. **Dérive de domaine** (`eval/domain_drift.py`) — on stream un long document technique
   et on mesure si la NLL de la seconde moitié baisse davantage avec M actif que sans
   (l'interaction, pas la baisse brute : le cache KV aide déjà).
3. **Dommage collatéral** (`eval/collateral.py`) — une M chargée d'un fait dégrade-t-elle
   la NLL sur un texte neutre sans rapport ? C'est le dilemme stabilité/plasticité
   chiffré : un gain sur E1 payé au-delà de +0.05 nats/token ici ne compte pas.

Si les deux sont nuls après réglage honnête de λ/η/seuil → l'idée est morte à cette
échelle, et c'est un résultat aussi.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt -e .
# Tests unitaires (CPU, aucun téléchargement) :
.venv\Scripts\python -m pytest tests/ -q
# Première éval (télécharge GPT-2 124M au premier lancement) :
.venv\Scripts\python eval\fact_injection.py
```

Matériel cible : RTX 3060 laptop 6 Go. Le cortex (GPT-2 124M fp16) + M (fp32) tiennent
très large ; aucun gradient n'est jamais stocké.

## Structure

```text
engram/            le package
  config.py        EngramConfig — tous les hyperparamètres, rien en dur ailleurs
  hippocampus.py   FastWeightMemory : delta rule, decay, prune, reset, stats
  cortex.py        chargement du LLM gelé + hook d'injection à la couche L
  engine.py        la boucle token-par-token : step / observe / stream / generate
eval/              les expériences falsifiables (E1 injection, E2 dérive, E3 collatéral)
tests/             tests unitaires CPU-only de l'hippocampe
docs/
  ARCHITECTURE.md  maths, décisions de design, pièges connus — LA référence
  JOURNAL.md       journal d'expériences daté
```

## Feuille de route

La méthode d'évolution est le pas-à-pas mesuré : un mécanisme à la fois, benchmarké
sur E1/E2/E3, avec un « poids » chiffré par ajout — voir `docs/EXTENSIONS.md`.

- [x] v0 — squelette : hippocampe + hook + boucle + évals + tests (2026-08-20)
- [x] v1 X0 — run baseline naïve + balayage λ/η/couche : +0.735 ± 0.886 nats sur E1,
      couche 6 confirmée, variance = interférence (2026-08-20)
- [x] v1 X1 — projection gyrus denté : **+1.361 ± 1.588 nats** sur E1 (+0.63 vs X0),
      retenue par défaut — sous réserve E3 (2026-08-20)
- [x] v1 X1b — éval E3 écrite et lancée : dommage réel (+0.135 à λ=2, seuil 0.05
      dépassé) ; compromis λ×cap → point conforme λ=1/cap=0.25 : E1 +0.740, E3 +0.023.
      Enseignement : le cap de lecture est un gating doux (2026-08-20)
- [x] v1 E1b/E2 — le rappel généralise (ratio paraphrases/exact 0.68, pas de par-cœur)
      et la dérive de domaine est réelle : interaction −0.055 nats/token (≈ −5.4 % de
      perplexité) sur RFC 9293, gain absolu en fin de document (2026-08-20)
- [x] v1.1 — ablations : le gating porte ~92 % de l'effet E2 ; DG +57 % d'interaction ;
      Hebb pur bat la delta sur E2 (−0.095 vs −0.055) mais delta reste le défaut
      (bornage, réécriture) — arbitrage final sur E2 long horizon (2026-08-21)
- [x] v1.1b — E2 narratif (fiction) : l'adaptation survit (−0.030 nats/token, ~55 % du
      niveau RFC) → pas un pur cache de n-grammes ; l'avantage Hebb s'évapore sur la
      fiction → delta rule confirmée par défaut, arbitrage D5 quasi clos (2026-08-21)
- [x] v1.1c — instrumentation I1 : keysim = jauge de saturation calibrée
      (0.23 sain / 0.78 saturé) ; découverte majeure : 10 faits à indices distincts
      dans une même M se rappellent sans coût (+0.770) — la capacité n'est pas la
      contrainte, la distinctivité des indices l'est (2026-08-21)
- [x] v1.2 — SmolLM2-360M : **le signal passe l'échelle** — E1 +0.852 (couche 16/32,
      la règle du milieu se transpose), multi-faits +0.817 avec σ÷3, point conforme
      E3 recalibré (cap 0.1 → E1 +0.494, E3 +0.042 ✓). Mécanisme universel,
      calibration (λ, cap) par-modèle (2026-08-21)
- [x] v1.3 X7 — hypothèse d'aplatissement mesurée (courbe top-heavy, projection W_U,
      multi-tokens) : coût d'entropie FIXE (+0.141 nats), zéro composante
      directionnelle (cos ≈ 0), gain → 0 à prior haut sans pénalité active. Feu vert
      chiffré pour le gate (2026-08-21)
- [x] v1.3 X9 — courbe de capacité : PAS de falaise à 80 faits (91 % de rétention,
      71/80 positifs) — d² est un problème théorique, factorisations écartées ;
      prédicteur par fait mort (2026-08-21)
- [x] v1.3 X8 — gate de lecture : facteur entropie disqualifié (anomalie résolue
      par X8.1/X8.1b/P5 : le dommage vit aux positions incertaines — gater côté
      mémoire, jamais côté détresse du cortex) ; **keysim seul RETENU**, nouveaux
      défauts gate=keysim/λ=2/cap=0.5 → GPT-2 E1 +1.353, E3 −0.014 ✓ ; SmolLM2
      +0.755, +0.003 ✓ ; coût : ratio paraphrases 0.68 → 0.38 (2026-08-21)
- [x] v1.3 E4/E4s — conventions de projet : critère non atteint sur GPT-2 et
      SmolLM2 ; sur Qwen (juge fort), signes conformes non significatifs ;
      E4-dur retiré (la NLL de phrase mesure la fluidité) — E4s est l'instrument
      de la famille ; racine commune : pas de composante directionnelle (X7)
      (2026-08-21)
- [x] v1.3 Qwen2.5-1.5B — fumée E1 +4.18 (N=2, régime X8 — non comparable aux
      points historiques ; l'E1 complet au même régime est la question Q-08 de
      l'audit) ; keysim d'écriture ≈ 0.14 : `gate_keysim_mid` à recalibrer par
      modèle (2026-08-21)
- [x] audit externe — 22 corrections + 6 questions de recherche priorisées :
      voir `AUDIT-lab.md` (2026-08-21)
- [ ] v1.4 X10 — comparatif de kernels d'adressage : GELÉ (sa métrique était la
      falaise X9, qui n'existe pas à cette échelle) ; réouverture sur falaise
      observée ou 3ᵉ modèle où DG régresse ; branche « DG apprise » retirée
- [ ] long terme — test « règle du milieu » sur un 3ᵉ modèle à ratio de couches
      différent (couche ≈ n/2 : structurel ou accidentel ? indice v1.2 : couche
      tardive activement nocive)
- [ ] v2 — priorité 1 : **rappel directionnel V2-D** (valeurs en espace
      d'unembedding ou tête de lecture apprise) — la réponse au mur X7 (top-10,
      E1c, E4s) ; priorité 2 (basse, actée par le verdict X9) : « sommeil » —
      distillation de M dans un LoRA puis reset, replay génératif depuis M

## Licence

AGPL-3.0-or-later — Copyright (C) 2026 NaejEL. La valeur d'engram tient à ce que la
mémoire est un artefact ouvert (inspectable, diffable, effaçable de façon
vérifiable) ; la licence garantit que ça le reste, y compris en usage réseau.
Texte intégral : [LICENSE](LICENSE).
