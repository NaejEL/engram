# Extensions — mécanismes candidats et protocole d'ablation

Formalisation de `POSSIBLE_APPROACH.md` (notes brutes conservées telles quelles).
Ce document définit **comment on avance** : une modification à la fois, benchmarkée
contre la version précédente, avec un « poids » chiffré par ajout. Rien n'entre dans
le cœur du code sans être passé par ce protocole.

## 1. Méthode : le pas-à-pas mesuré

1. **Baseline d'abord** : la version naïve (X0) tourne sur toute la suite d'évals
   avant tout ajout. Ses chiffres sont la référence absolue du tableau §4.
2. **Un mécanisme à la fois** : jamais deux ajouts dans le même run. Chaque extension
   est un flag de config (défaut = désactivé) pour rester ablatable à vie.
3. **Suite d'évals fixe** : E1 + E2 + E3 (voir ARCHITECTURE.md §5), mêmes seeds,
   mêmes hyperparamètres que la baseline sauf le mécanisme testé.
4. **Poids d'un ajout** = ses deltas sur les trois évals, consignés dans le tableau §4
   et dans le journal. On garde un mécanisme si son gain sur E1/E2 justifie sa
   complexité ET qu'il n'aggrave pas E3 (le dommage collatéral).
5. **Un symptôme avant un remède** : les mécanismes marqués « repli » ne s'implémentent
   que si leur symptôme déclencheur est observé. Pas d'ingénierie spéculative.

## 2. L'échelle des extensions

Chaque entrée : origine biologique → problème visé → déclencheur → esquisse → coût.

### X0 — Baseline naïve (référence)

Delta rule pleine matrice, φ = normalisation L2, gating NLL, decay + top-k, reset.
C'est l'état actuel du code. Aucun flag.

### X1 — Projection gyrus denté (séparation de patterns) — *assurance v1, pas un repli*

- **Bio** : le gyrus denté projette l'entrée dans un espace beaucoup plus grand et
  très sparse avant écriture hippocampique — orthogonalisation pour que deux
  expériences proches ne s'écrasent pas.
- **Problème visé** : interférence entre clés corrélées (mode d'échec le plus
  probable de E2 : un document homogène produit des états latents très proches).
- **Déclencheur** : aucun — prévu d'office juste après le run baseline (coût quasi nul).
- **Esquisse** : `φ_dg(h) = topk(G·h, k)` avec `G` (d → D, D ≈ 8–16×d) **aléatoire
  gelée** (seed fixée, jamais apprise). M devient rectangulaire d×D. Bonus : capacité
  théorique ~D associations au lieu de ~d, et writes creux.
- **Config** : `dg_dim` (0 = désactivé), `dg_topk`. M : 768×8192 ≈ 25 Mo fp32 — trivial.

### X2 — Lecture itérée CA3 (complétion de patterns) — *repli*

- **Bio** : CA3 est un réseau auto-associatif récurrent (un Hopfield, littéralement) :
  un fragment d'indice suffit à reconstruire le pattern complet.
- **Problème visé** : rappel par indice partiel — E1 qui marche avec le prompt exact
  mais échoue en paraphrase.
- **Déclencheur** : ajouter une variante paraphrase à E1 ; si exact ≫ paraphrase.
  **Testé le 2026-08-20 : NON observé** (ratio paraphrases/exact = 0.68, dégradation
  graduelle par recouvrement sémantique — voir JOURNAL). X2 reste en poche.
- **Esquisse** : itérer la lecture 2–3 fois : `h' = h + λ·M·φ(h)`, relire avec h',
  converger vers l'attracteur. Config : `read_iters` (défaut 1 = comportement actuel).

### X3 — Alternance theta encode/retrieve — *repli*

- **Bio** : l'oscillation theta (~8 Hz) sépare temporellement écriture (plasticité
  haute) et lecture (plasticité coupée) — une lecture ne contamine jamais la mémoire.
- **Problème visé** : instabilité / divergence de M malgré les garde-fous existants.
- **Déclencheur** : perplexité qui explose ou génération qui boucle avec M active.
- **Déjà à moitié en place** : l'engine interdit structurellement l'écriture pendant
  `generate` et pendant le rappel d'éval (règle absolue, engine.py). La version
  complète alternerait des micro-phases read-only/write-only pendant le stream.

### X4 — Néogenèse / reset de lignes — *repli*

- **Bio** : la néogenèse du gyrus denté efface activement d'anciens souvenirs en
  recâblant les circuits — l'oubli est un mécanisme entretenu, pas une défaillance.
- **Problème visé** : saturation de M sur les longs streams (E2 qui plafonne puis
  régresse).
- **Déclencheur** : courbe NLL de E2 en U (amélioration puis dégradation).
- **Esquisse** : tracker l'utilisation des lignes en lecture (EMA de |contribution|),
  réinitialiser périodiquement les moins utilisées. Version riche de notre top-k.

### X5 — Substrat épisodique explicite (indexing theory / table vectorielle) — *repli diagnostique*

- **Bio** : l'hippocampe ne stockerait pas le contenu mais un *index* — un pointeur
  vers le pattern cortical à rallumer (hippocampal indexing theory).
- **Problème visé** : échec complet de E1 après balayage honnête de λ/η/couche.
- **Déclencheur** : E1 nul partout. C'est le repli de dernier recours, et il est
  *diagnostique* : lignée prouvée (kNN-LM, Memorizing Transformers) — si même
  l'injection kNN exacte ne bouge pas les logits, le coupable n'est pas le substrat
  de stockage mais le **mécanisme d'injection** (couche, λ), et on sait où chercher.
- **Esquisse** : remplacer M par un store (K, V) explicite + recherche par similarité,
  même point d'injection. Interface commune avec FastWeightMemory pour swap propre.

### V2 — Replay / sharp-wave ripples (consolidation M → LoRA) — *hors v1*

- **Bio** : replay de séquences compressées (jusqu'à 20×), parfois inversées, parfois
  *recombinées* — de la planification autant que de la consolidation.
- **Spec retenue pour la v2** : la distillation vers le LoRA ne rejouera pas les
  données brutes mais des échantillons **générés depuis M** (échantillonner des clés,
  générer avec/sans M, distiller la différence). C'est ici que le dilemme
  stabilité/plasticité revient en grand.

## 3. Correspondance mécanisme ↔ piège

Chaque extension répond à un piège identifié dans ARCHITECTURE.md §4 — le cerveau a
rencontré exactement ces bugs et a shippé des fixes :

| Piège / mode d'échec | Fix biologique | Extension |
| --- | --- | --- |
| Interférence entre clés proches | Séparation de patterns (gyrus denté) | X1 |
| Rappel partiel impossible | Complétion de patterns (CA3) | X2 |
| Lecture qui contamine l'écriture | Rythme theta | X3 |
| Saturation / oubli mal ciblé | Néogenèse, oubli actif | X4 |
| Substrat matriciel inadapté | Indexation plutôt que stockage | X5 |
| Oubli catastrophique à la consolidation | Replay SWR recombinant | V2 |

## 4. Tableau des poids (à remplir au fil des runs)

Référence : config par défaut (`EngramConfig`), seeds fixes, GPT-2 124M.
Les deltas se lisent **contre la ligne précédente retenue**, pas contre X0.

| Étape | E1 Δlogp (nats) | E1 top-10 | E2 interaction | E3 dégradation | Verdict |
| --- | --- | --- | --- | --- | --- |
| X0 naïve (défauts initiaux : λ=0.5, η=0.05) | +0.219 ± 0.271 (N=10, 9/10 > 0) | 0/10 | *(à venir)* | *(éval à écrire)* | signal faible mais directionnel |
| X0 naïve, point retenu (layer=6, λ=1.0, η=0.1) | +0.735 ± 0.886 (N=10) | 0/10 | *(à venir)* | +0.1991 ± 0.0241 (> seuil) | référence pour X1 — variance = interférence, déclencheur X1 observé |
| X1 gyrus denté (dg=8192/64, λ=2.0, η=0.2) | +1.361 ± 1.588 (N=10) | 0/10 | *(à venir)* | +0.1354 ± 0.0147 (> seuil ×2.7) | mécanisme retenu (+0.63 E1, −0.06 E3 vs X0) mais point NON conforme E3 |
| **X1b compromis λ×cap (dg=8192/64, λ=1.0, η=0.2, cap=0.25)** | **+0.740 ± 0.799** (N=10) | 0/10 | **−0.0551 ✓** (writes 23 %) | **+0.0228 ± 0.0056 ✓** | **référence courante** — premier point conforme, tableau complet ; cap = gating doux (λ contrôle le bruit, cap le signal) |
| X1 gyrus denté | | | | | |

*(les lignes suivantes s'ajoutent quand leur déclencheur est observé)*
