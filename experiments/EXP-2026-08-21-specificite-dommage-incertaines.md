# EXP — Spécificité du dommage aux positions incertaines (Q-01)

Statut : TERMINE — RETENU

**Origine** : `AUDIT-lab.md` Q-01 / VAL-06 (audit 2026-08-21, critère « fiabiliser
d'abord »). Conclusion visée : forme finale de la loi 2, chapitre méthode du papier.
Brouillon du Directeur soumis à Neuro et Math le 2026-08-21 ; arbitrage ci-dessous.

## Arbitrage

**Avis Neuro (favorable, 2 amendements)** :

| Remarque | Sort | Raison |
| --- | --- | --- |
| Hasselmo prédit SPÉCIFIQUE (interférence par corrélation aux traces, pas par ouverture du canal) ; si générique, la référence devient Salzman, Britten & Newsome 1990 (microstimulation, levier maximal en stimulus ambigu) et Hasselmo repart vers X3/Q-05 | Intégrée | Grille d'interprétation pré-enregistrée (§4, fin) : chaque verdict a son ancrage bio nommé AVANT mesure — ce qui a manqué à la loi 2 v1 (COR-08). |
| Prédiction ordinale composite pré-mesure (dommage : read-M > fixe > iid ; corrélations toutes positives, iid ≥ +0.20 ; signe aux confiantes discriminant) | Intégrée | Consignée en P6/P-ord — rend le verdict « composite » falsifiable pièce par pièce. |
| N1 — partage confiantes/incertaines par condition, prédictions signées | Intégrée avec modification | Retenue comme P6, subordonnée hiérarchiquement à P2 (gatekeeping séquentiel) — concilie N1 avec la décisionnelle unique de Math. |
| N2 — condition « direction prior sans M » (r̄ constant, norme appariée) | Intégrée, étendue | Condition 5 (P7), étendue en paires ±r̄ : le bras −r̄ réalise la condition « anti-prior » et teste la congruence par prédiction de signe. |
| Nommer le dommage négatif « bénéfice de congruence » ; « entropie ≈ ACh » tombe si générique ; « adversarial » acceptable | Intégrée | Vocabulaire adopté ; note de façade dans l'entrée de journal. |
| Note Q-05 : si générique, Q-05 change d'objet (co-localisation corrélée par construction) | Intégrée | Conséquence feuille de route (§10). |

**Avis Math (favorable, 3 amendements)** :

| Remarque | Sort | Raison |
| --- | --- | --- |
| Ambiguïté d'ancre : +0.394 d'un secret unique ou d'une moyenne ? La bande [0.29, 0.49] n'est pas une tolérance de reproduction | Intégrée | P1 rétrogradée en porte conditionnelle V3 ; la porte dure devient V1 (+0.1354, reproduit par `gate_anomaly.py` ET `gate_cycle.py`). Tranchée par le PI (voir §13). |
| A1 — paires antithétiques ±ε : la partie paire isole la courbure, calibration empirique du « mécanique » ; sans elle, un verdict « spécifique » est suspect (dilution par le terme linéaire) | Intégrée | Cœur méthodologique : P2 se calcule sur corr_pair, pas sur la corrélation brute. Le forfait 50 % du brouillon est abandonné. |
| A2 — contrastes appariés au lieu de seuils absolus ; hiérarchie validité / décisionnelle unique / descriptives ; P5' rétrogradée ; P4 contre null empirique | Intégrée | Remplace les seuils 0.20/0.10 (intenables au N_eff de Bretherton). |
| A3 — 2ᵉ texte neutre (~700-900 positions), instrumentation (ρ₁, saturation cap, corr(‖r‖,H), dérive), règle adaptative fixe 10→20 si σ > 0.08 | Intégrée | Budget re-chiffré §9 ; texte B embarqué dans le nouveau script. |
| Statistique imposée : block bootstrap circulaire intra-run (L≈25, B=2000, Fisher-z) ; inférence primaire inter-runs (t apparié + Wilcoxon) ; Spearman co-primaire + winsorisation ; corr partielle en diagnostic ; signe constant ≥ 9/10 | Intégrée | Remplace la corr partielle co-primaire du brouillon (rétrogradée en diagnostic — r_HN ≈ 0.6-0.8, plancher D ≥ −NLL_base). |
| Pièges numériques : appariement/renorm en fp32 avant cast (tol 1e-4 intenable en fp16) ; Generator torch dédié ; asserts clear_context/write_count/no-write | Intégrée | Recopiées dans les livrables Builder et les tests CPU. |
| Cap saturé quasi partout (‖λMφ(h)‖ ≈ 50 ≫ 0.5‖h‖) → r_t ≈ 0.5‖h_t‖·û_t | Intégrée | Instrumentation obligatoire ; « même norme » = de fait 0.5‖h_t‖ partout, à vérifier. |
| ‖ε‖/‖h‖ = 0.5 non perturbatif, Taylor = guide qualitatif | Intégrée | Limite pré-déclarée ; la décomposition antithétique paire/impaire reste exacte. |

Rien n'est écarté des deux avis ; deux éléments du brouillon le sont : les seuils
absolus de P2 (remplacés par A2) et la corr partielle co-primaire (rétrogradée). La
question PI « anti-prior » disparaît (résolue par N2 étendue en ±r̄) ; la variante
« position unique » passe en levée de doute de dernier recours (P6 assume le
premier secours).

## 1. Question

La corrélation dommage/entropie de +0.394 (P5, journal 2026-08-21) est-elle une
propriété de la **lecture de M**, ou toute perturbation de même norme injectée aux
mêmes positions la reproduit-elle ?

## 2. Hypothèse

**H (générique)** : le dommage concentré aux positions incertaines est une propriété
du cortex — la sensibilité de la NLL à une perturbation du flux résiduel croît avec
l'entropie de la position, indépendamment de la direction injectée.
Opérationnalisation (A2) : la composante **paire** (courbure, isolée par
antithétiques) d'une perturbation aléatoire de norme appariée reproduit **au moins
50 %** de la corrélation dommage/entropie de la lecture réelle.

Si H est vraie : loi 2 reformulée « le cortex est fragile là où il hésite ; la
lecture de M est un cas particulier » — ancrage bio : Salzman, Britten & Newsome
1990 (levier maximal en stimulus ambigu) ; Hasselmo repart vers X3/Q-05 ; la façade
« entropie ≈ ACh » tombe. Si H est fausse : la loi 2 garde sa forme forte — c'est la
lecture de M, par sa direction « prior générique » (diag. 2), qui est nocive aux
positions incertaines ; Hasselmo survit, cité pour la bonne propriété (contenu
structuré qui interfère).

## 3. Ce que le projet sait déjà

- **P5** (journal 2026-08-21) : 343 positions de texte neutre, M chargée, gate none
  → corr(dommage, entropie baseline) = **+0.394** ; confiantes **−0.099**,
  incertaines **+0.364**. Aucun contrôle non-M (VAL-06). Script non commité ;
  provenance (1 secret vs moyenne) inconnue (décision PI, §13).
- **Diagnostic 1** (journal 2026-08-21) : E3 none à λ2/cap0.5 = **+0.1354**,
  reproduit à l'identique par `eval/gate_anomaly.py` et `eval/gate_cycle.py` →
  ancre de validité dure.
- **Diagnostic 2** (`eval/marginal_pull.py`) : lecture quasi constante entre
  positions (corr(W_U·r, log-fréq) = +0.484 ± **0.002**), ‖r‖ brut ≈ 24.8 ≫ cap →
  **le cap sature quasi partout** : r_t ≈ 0.5‖h_t‖·û_t. D'où le contrôle
  « direction fixe » en plus du « iid frais ».
- **X8.1b** : créneaux duty 50 % ≤ proportionnels — le ciblage par contenu est le
  facteur, pas la bascule.
- **Pièges d'analyse identifiés** : positions autocorrélées (texte + propagation
  KV : N_eff attendu 100-270, Bretherton) ; terme linéaire g̃ᵀε qui dilue la
  corrélation des randoms bruts vers 0 (biais d'atténuation) ; corr(W_U·r, logits)
  = +0.6 → ratio d'amplitude read-M/random > 1 mécaniquement garanti.

## 4. Prédictions chiffrées (hiérarchie pré-enregistrée)

### Portes de validité (échec = run INVALIDE, pas négatif)

| # | Métrique | Bande | Note |
| --- | --- | --- | --- |
| V1 | Dommage moyen read-M, texte A (`NEUTRAL_TEXT`) | **+0.1354 ± 0.005** | Déterministe, reproduit par deux scripts commités — **seule porte dure (décision PI)** |
| V2 | writes > 0 à chaque injection ; ‖r'_t‖ = ‖r_t‖ en fp32 (tol 1e-4) ; aucun write sous hook ; `clear_context()` assert avant chaque mesure ; dérive ‖h'_t‖/‖h_t‖ ∈ [0.9, 1.1] | toutes vraies | Pièges §4.3 + pièges numériques Math |
| V3 (indicative) | corr read-M texte A vs +0.394 | rapportée à titre indicatif (par secret ET moyenne) | Provenance de l'ancre inconnue (décision PI, §13) — V3 n'invalide pas le run |

### Décisionnelle unique (P2)

Statistique : R = corr_pair(random-iid antithétique) / corr(read-M), où corr_pair
est calculée sur D_pair = [D(+ε)+D(−ε)]/2 (composante courbure). Contraste apparié
par secret/seed (10 paires), Fisher-z, t apparié + Wilcoxon ; Spearman co-primaire ;
signe constant sur ≥ 9/10 paires comme signature anti-artefact.

| Verdict | Règle |
| --- | --- |
| **H vraie (générique)** | R ≥ 0.5 avec IC 95 % apparié excluant 0.25 |
| **H fausse (spécifique)** | R ≤ 0.25 avec IC 95 % apparié excluant 0.5 |
| **Zone grise** | sinon → P6 prend la main (gatekeeping séquentiel) |

### Décisionnelle de secours (P6 — active seulement si P2 en zone grise)

Partage à la médiane d'entropie, par condition, prédictions **signées** (Neuro) :

| Condition | Confiantes | Incertaines |
| --- | --- | --- |
| read-M | **< 0** (« bénéfice de congruence ») | > 0 |
| random-iid (pair) | **≥ 0** | > 0 |
| random-fixe | **≥ 0** | > 0 |

Règle de secours : read-M confiantes < 0 ET tous les randoms ≥ 0 (signe constant
≥ 9/10) → **spécifique** (composante directionnelle) ; un random robustement
négatif aux confiantes → artefact à chercher (analyse invalide, pas un verdict).
Justification bio pré-enregistrée : un bruit zéro-moyenne ne peut pas aider en
espérance (NLL convexe en logits) ; seule une direction signée congruente le peut.

### Descriptives (pas décisionnelles)

| # | Métrique | Attendu |
| --- | --- | --- |
| P3 | \|corr fixe − corr iid_pair\| | < 0.10 ; σ inter-tirages du fixe 2-3× celle du iid ; Δ > 0 possible sans mécanisme mémoire ((Σg̃)ᵀu cohérent) |
| P4 | Similarité inter-secrets des profils read-M vs **null empirique** (similarité inter-tirages random-iid) | read-M ≫ null → profil = propriété du texte/cortex |
| P5' | Dommage moyen read-M / random-iid à normes égales | > 1 mécaniquement garanti ; Neuro prédit > 2× — ordinal, descriptif |
| P7 | Condition ±r̄ : +r̄ reproduit le profil read-M (corr de profils élevée, confiantes < 0) ; −r̄ **nuit** aux confiantes | si oui → la loi 2 est une propriété de LA DIRECTION, pas de M en tant que mémoire |
| P-ord | Ordre des dommages moyens (Neuro, pré-mesure) : read-M > random-fixe > random-iid ; toutes les corrélations positives | composite falsifiable pièce par pièce |

### Grille d'interprétation pré-enregistrée

- Générique (P2 haut) → reformulation loi 2, référence Salzman 1990, Hasselmo →
  X3/Q-05, Q-05 requalifiée avant lancement (approuvé PI, §13).
- Spécifique (P2 bas + P6 conforme) → loi 2 forme forte, Hasselmo conservé pour
  l'interférence de contenu.
- Spécifique **par direction** (P2 bas + P7 : +r̄ ≈ read-M) → loi 2 = propriété de
  la direction prior (pont direct vers V2-D).
- E3 : **sans objet** — diagnostic pur, aucun défaut d'`EngramConfig` modifié ;
  tout changement de défaut consécutif passera par un protocole séparé avec
  E3 ≤ +0.05.

## 5. Contrôles et baselines

1. **Baseline M=0** : NLL et entropie par position, textes A et B, calculée une
   fois (déterministe) ; le bootstrap porte sur les paires (D_t, H_t), jamais sur
   la baseline.
2. **read-M** (ancre) : protocole E1 (`FACT_TEMPLATE`, force_write),
   `clear_context()` (D7), stream lecture active/écriture coupée,
   `read_gate="none"`, 10 secrets ; profils ‖r_t‖ post-cap enregistrés (fp32) —
   ils apparient tout le reste.
3. **random-iid antithétique** : ±ε frais par position (Generator torch **dédié**,
   seedé, isolé du déterminisme seed=0/G), renormé à ‖r_t‖ en fp32 avant cast ;
   10 seeds appariés aux 10 secrets.
4. **random-fixe** : une direction par run, norme appariée par position ; 10
   tirages, **règle adaptative → 20 si σ inter-tirages > 0.08**.
5. **±r̄ sans M** (N2) : direction moyenne de lecture (ou proxy fréquence-unigramme
   — choix Builder documenté), constante, norme appariée ; bras +r̄ et −r̄, 10 runs
   chacun appariés aux profils des 10 secrets.
6. **Fait sans rapport** : couvert par la structure per-secret (P4) — les 10
   secrets sont sans rapport avec les textes neutres par construction.
7. **Override off = identité bit à bit** avec la condition 2 (précédent
   `force_cycle_gate`, `eval/gate_cycle.py` : override d'instance, moteur intact).
8. **Instrumentation obligatoire** (A3) : ρ₁ des séries, corr(‖r_t‖, H_t),
   fraction de saturation du cap, dérive ‖h'_t‖/‖h_t‖.

## 6. Critères d'abandon

- **Tue H (générique)** : R ≤ 0.25 avec IC excluant 0.5, corroboré par P6 (read-M
  seul négatif aux confiantes) → H falsifiée. C'est un résultat.
- **Invalide le run** : V1 hors ±0.005 ; toute clause V2 fausse ; NaN ; < 300
  positions valides par texte ; random négatif robuste aux confiantes (artefact).
- **INCONCLUSIF nommé** : P2 en zone grise ET P6 ambiguë → cause :
  variance/autocorrélation résiduelle ; levée de doute pré-déclarée = variante
  « injection à position unique » (casse la contamination KV, ~+20-30 min) —
  **réserve pré-déclarée approuvée par le PI (§13), déclenchée uniquement dans ce
  cas**.

## 7. Variables fixées

Modèle **gpt2**, layer **6**, seed **0** ; λ = **2.0**, cap = **0.5**, η = **0.2**,
decay = **1e-3**, thr = **4.0**, dg = **8192/64**, prune 512/0.10 (défauts
`EngramConfig`) ; **`read_gate="none"`** (écart au défaut keysim, comme P5 et
`gate_cycle.py` — sous keysim, ‖r‖ ≈ 0.48 sur texte neutre : rien à mesurer).
Données : `NEUTRAL_TEXT` (`eval/collateral.py`), **texte neutre B** (~450 tokens,
même registre, vocabulaire disjoint des secrets, embarqué dans le nouveau script),
`FACT_TEMPLATE` + 10 `SECRETS` (`eval/fact_injection.py`).

## 8. Variable manipulée

**Une seule** : la nature du vecteur injecté à la couche 6, à profil de norme
apparié position par position — cinq niveaux : lecture réelle de M / gaussien iid
±ε / gaussien fixe / +r̄ / −r̄.

## 9. Budget

Streams de ~450 tokens (chaque condition × 2 textes) : baseline 2 ; read-M 20 ;
random-iid ±ε 40 ; random-fixe 20 (→ 40 si règle adaptative) ; ±r̄ 40. **Total 122
streams (worst case 142)** + 30 injections courtes. À ~10 s/stream : **~22-28 min
GPU, worst case ~32 min** — sous le plafond de 45 min. VRAM < 2 Go. **Budget validé
par le PI (§13).** La variante position unique (réserve) ajouterait ~20-30 min et
n'est PAS dans le budget de base.

## 10. Livrables attendus

- **Script** : `eval/perturb_position.py` (nouveau, SPDX AGPL-3.0-or-later),
  options `--model --layer --positions --secrets` ; embarque le texte B ;
  implémente antithétiques, Generator dédié, appariement fp32, block bootstrap
  circulaire (L≈25, B=2000, Fisher-z), inférence inter-runs (t apparié +
  Wilcoxon), Spearman + winsorisation 1 %/99 % en sensibilité, corr partielle
  (D,H|NLL_base) en diagnostic, instrumentation §5.8, règle adaptative. Il
  **absorbe la mesure P5** (comble l'absence du script d'origine).
- **Flag `EngramConfig`** : aucun — diagnostic pur, override d'instance (précédent
  `gate_cycle.py`) ; moteur strictement inchangé.
- **Tests CPU** (sans HF) : appariement fp32 exact ; déterminisme du Generator
  dédié ; ε(−) = −ε(+) (antithéticité) ; override off ⇒ identité avec la lecture
  normale ; aucun write sous hook.
- **Résultats bruts** : `experiments/results/specificite-dommage-incertaines/`.
- **Entrée de journal** au format standard, prédictions V1-V3/P2/P6/P3-P7/P-ord
  telles qu'approuvées ; note de façade (« bénéfice de congruence », sort de
  l'analogie ACh selon verdict).
- **Conséquence feuille de route** (approuvée PI, §13) : si générique, Q-05 est à
  requalifier avant lancement.
- Statut Q-01 dans `AUDIT-lab.md` : mise à jour par la session principale.

## 11. Questions pour Neuro (résiduelles, à traiter en interprétation)

1. Pour P7 : le proxy « direction fréquence-unigramme » est-il un substitut
   acceptable de r̄ si les deux divergent (cos < 0.9), ou rapporter les deux bras
   séparément ?
2. Confirmer la grille d'interprétation §4 (Salzman si générique, Hasselmo si
   spécifique) telle que formulée.

## 12. Questions pour Math (résiduelles, à traiter en interprétation)

1. L = 25 pour le block bootstrap : à confirmer après lecture du ρ₁ mesuré (règle
   de mise à jour fixée avant analyse : L = max(25, 3/ρ₁ arrondi à la dizaine) si
   ρ₁ mesuré > 0.12).
2. P6 en rôle décisionnel de secours : test des signes (≥ 9/10) + Wilcoxon sur les
   splits appariés, les deux rapportés.

## 13. Décisions du PI (2026-08-21, gate de pré-enregistrement)

1. **Ancre P5** : provenance inconnue (script ad hoc perdu) → **V1 est la seule
   porte dure** ; V3 indicative, rapportée par secret ET en moyenne.
2. **Variante position unique** : **en réserve pré-déclarée**, déclenchée
   uniquement si P2 ET P6 sont ambiguës.
3. **Si verdict générique** : oui aux trois — (a) reformulation de la loi 2 par
   nouvelle entrée de journal (X8.1b+P5 scellée), (b) ligne D candidate pour
   `ARCHITECTURE.md` §3 rédigée à l'attention du PI (qui l'applique lui-même),
   (c) requalification de Q-05 avant son lancement.
4. **Budget** : validé (worst case ~32 min GPU).

## Historique

- 2026-08-21 : proposé
- 2026-08-21 : pré-enregistré par le PI
- 2026-08-21 : terminé, verdict RETENU (H générique, formulation composite)
