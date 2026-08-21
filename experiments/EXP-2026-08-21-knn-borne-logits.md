# EXP — V2-D(a) : kNN-LM nu, borne de l'étage des logits

Statut : TERMINE — INVALIDE

**Origine** : `docs/EXTENSIONS.md`, entrée V2-D, candidat (a) — « instrument de plafond, à
faire EN PREMIER ». Premier cycle du chantier v2. Brouillon du Directeur soumis à Neuro
et Math le 2026-08-21 ; arbitrage ci-dessous ; décisions du PI en §13.

---

## Arbitrage

### Le point de convergence (les deux experts, indépendamment) — il restructure le protocole

L'identité `ΔNLL_t = −log[(1−λ) + λ·p_kNN(y_t)/p_LM(y_t)] ≤ −log(1−λ)` est acceptée telle
quelle. Conséquences appliquées sans réserve :

| Élément du brouillon | Statut | Raison |
| --- | --- | --- |
| Grille λ ∈ {0.05, 0.10, 0.25, 0.50} avec E3 mesuré à chaque point | **Écartée comme mesure** | E3 ≈ −log(1−λ) est prédit sur le papier pour un datastore hors sujet. Mesurer 8 points = payer du GPU pour vérifier 8 fois la même formule. Remplacée par **un** point de vérification d'intégrité. |
| Règle « plus grand λ conforme à E3 ≤ 0.05 », fixée avant lecture de P1/P2 | **Écartée** | Elle plafonne λ à 0.0488 et rend P2 (renversement, exige λ ≳ 0.33) **arithmétiquement inatteignable** : la règle anti-cherry-picking fabriquait un échec garanti. |
| Seuil P2 « Δlogp ≥ +2.0 » | **Écarté** (les deux experts) | Mécaniquement satisfait dès λ ≳ 0.01 pour une cible rare : ne discrimine rien. P2 ne porte plus que sur le **rang** et la **marge sur Paris**. |
| V0 et les Δlogp d'E1-exact comme mesures N=10 | **Écartés comme mesures** | Δlogp exact ≈ +12 à +14 nats est un chiffre sans contenu. V0 devient un **smoke test à 1 secret**. |
| Grille T ∈ {1, 10} | **Écartée** | Dégénérée en one-hot à la précision machine (‖h‖ ~ 50–100 ⇒ écarts de d² de plusieurs centaines). Remplacée par la calibration relative de Math. |
| « le problème restant est la compression, pas la localisation » | **Écartée** (Neuro) | La borne ne vaut que **conditionnellement à ce point d'injection**, et en cas d'échec elle **ne se retourne pas**. Un système à réinstallation (X5/Fast-KV) peut en principe la dépasser. |

### Avis Math (RÉSERVÉ → intégré)

| # | Remarque | Statut | Forme retenue |
| --- | --- | --- | --- |
| **A1** | Gate de récupération α(q) = 1[d²_min ≤ τ] côté mémoire | **Bras supplémentaire non décisionnel « G »** (décision PI §13.1) | Adopter le gate en principale transformerait l'instrument en candidat produit déguisé. La principale reste **nue** ; G répond à une question distincte : « un organe keysim transposé aux logits rachète-t-il le budget E3 ? » — question de design du candidat (b). |
| **A1 bis** | Bras non gaté à λ=0.25, E3 assumé hors contrat | **Intégrée** | Rapporté « **plafond conditionnel, hors budget** », jamais comme un succès. |
| **A2** | V0 → smoke test ; supprimer le sweep mesuré ; vérifier UN point contre −log(1−λ) | **Intégrée** | Le test d'intégrité détecte une implémentation qui ne fait pas un mélange. |
| **A3** | Décision au niveau **secret** : n(secrets ≥ 2/3 paraphrases en top-10) ; H vraie ≥ 5/10, H fausse ≤ 1/10 ; test des signes exact | **Intégrée telle quelle** | Corrige une pseudo-réplication du brouillon (3 paraphrases du même secret traitées comme indépendantes) — la faute exacte proscrite depuis Q-01b §7(i). |
| — | Le rang est la décisionnelle, le Δlogp descriptif | **Intégrée** | |
| — | T = c·med_j(d²_j − d²_min), c ∈ {0.3, 1, 3}, **calculée par bras** | **Intégrée** | Sans le « par bras », P6 confond profondeur d'indice et température. |
| — | fp32 sur les distances ; mélange en log-espace ; soustraire d²_min avant l'exponentielle | **Portes de tests CPU bloquantes** | À ‖h‖² ~ 10⁴, fp16 a une résolution ~8 : un quasi-match est indiscernable de zéro. Piège §4.4 sous une nouvelle forme. |
| — | Exiger **var(D_t) > seuil** avant d'interpréter P5b | **Porte** | La borne rend E3 quasi déterministe : sinon corr(D,H) est un 0/0 bruité. |
| — | ρ̂ estimée jusqu'au lag 5 ; L ≥ 2× le temps d'autocorrélation intégré si ρ̂₁ > 0.1 | **Intégrée**, « ρ̂₁ ≤ 0 ⇒ L = 25 » devient le repli | Les hits de récupération persistent sur quelques tokens : ρ̂₁ sera probablement positive, contrairement à Q-01. |
| — | Logs de confondants : corr(rang_mix, logp_base) par secret ; tokens communs du fait dans les valeurs | **Obligatoires** | Le premier est un contrôle d'attribution majeur. |

### Avis Neuro (FAVORABLE → intégré)

| # | Remarque | Statut | Forme retenue |
| --- | --- | --- | --- |
| **A** | λ balayé, rapporter la **frontière (λ, E3, P1, P2)** | **Intégrée**, réconciliée avec A2 par l'arbitrage n°4 | Sans quoi un seul λ déciderait du verdict de tout le chantier v2. |
| **A(i)** | Succès à λ non conforme = « plafond conditionnel, hors budget » | **Intégrée**, formulation reprise | |
| **B** | **Instrumenter la récupération** : R1 rang de l'entrée correcte, R2 d²/cos à la clé, R3 masse p_kNN(cible), R4 entropie de p_kNN | **Intégrée, obligatoire** | **L'amendement le plus important du cycle** : sans lui, Branche A (échec côté clé) et Branche B (échec côté injection) produisent le même chiffre et le verdict est **inattribuable** — or seule la Branche B parle contre le candidat (b). Coût nul. |
| **B bis** | P5b/P5c aussi sur le bras P3 ; **R_out = corr(D_permuté,H)/corr(D_kNN,H)** ; l'ancre 0.826 n'est pas transposable | **Intégrée ; l'ancre 0.826 écartée** | « Bruit de norme appariée » n'a pas de définition sur les logits. |
| **Façade couche 6** | Le bras compare deux **représentations d'indice**, pas deux localisations d'injection (elle reste aux logits dans les deux cas) : il ne peut rien dire de X7 | **Bras renommé et réinterprété (P6)** | La formulation du brouillon fabriquait une conclusion que le design ne porte pas. |
| **Façade familiarité** | « kNN-LM = familiarité » faux dans les deux termes ; le seul objet ayant cette propriété est la **distance au voisin**, que fixer λ jette | **Intégrée** ; carte retirée du protocole, R2 loggée pour la conserver | Correctif à la fiche V2-D rédigé pour le PI (§13.6). |
| **Façade « plafond »** | Sans contenu biologique ; ce qui est borné est *ce qu'un mélange aux logits peut faire avec des clés = états finaux du cortex gelé* | **Intégrée** : « plafond » → **« borne de l'étage des logits »** | |
| **Vocabulaire** | Proscrire « complétion de patterns » → « **généralisation de la clé par l'encodeur** » | **Intégrée** | Sans récurrence : ni bassin ni correction d'erreur. |
| **Piège d'ingénierie** | Ne PAS projeter les clés kNN par DG | **Contrainte d'implémentation + test CPU** | DG orthogonalise les entrées similaires : ce serait détruire la tolérance à la paraphrase que P1 mesure. |
| **Portée de P1** | P1 mesure d'abord l'invariance-à-la-paraphrase de l'**état final** | **Intégrée** : reformule le verdict REJETE (§6) | Un échec falsifie « l'état final est une clé paraphrase-invariante », pas « l'espace de sortie est le bon point d'attaque ». |
| **Prédiction ordinale** | E1-exact < E1b(fait-seul) < E1b(distracteur) < **E1c** | **Intégrée** — elle **renverse la hiérarchie implicite du brouillon** | |
| **P5b : signe +, [+0.10,+0.40] ; P5c |corr| < 0.15 ; mécanisme T1/T2/T3 + discriminant de chaîne** | **Intégrée intégralement** | Le discriminant (bord droit dur, empilement/étalement, effondrement après appariement) teste la **chaîne** et pas le signe — leçon Q-01b appliquée. |
| **Déclencheur multi-clé, non armé** (Branche A seulement) | **Intégré** | Conforme à « un symptôme avant un remède ». |
| **Prédiction de forme** (dégradation graduelle **sans genou**) | **Intégrée avec amendement** | 4 formulations = axe trop grossier ; l'axe gradué est **rang vs d²_min** poolé (~40 points). |

### Arbitrages du Directeur (au-delà des avis)

**N°4 — A2 et l'Amendement A ne sont pas concurrents : le balayage est GRATUIT.** `p_kNN` est
supportée sur ≤ k tokens et `p_LM` est fixée par une passe forward qui **ne dépend ni de λ
ni de T**. En loggant par requête (i) le vecteur `p_LM`, (ii) les k distances et les k
tokens-valeurs, le rang de la cible sous mélange se recalcule **hors GPU** pour tout (λ, T)
et pour la permutation des valeurs. La frontière complète est obtenue au prix d'**une seule
passe** par condition — et le budget baisse au lieu de monter.

**N°5 — condition principale à `kNN seul` (M = 0)** (confirmé PI §13.2). Trois raisons :
(i) « un mécanisme à la fois » ; (ii) sous M chargée, l'état de requête est perturbé par la
lecture de M, ce qui **détruit l'attributabilité de R1–R4** ; (iii) à M = 0, P3 et P4
deviennent entièrement hors-ligne. `M au point courant + kNN` devient un bras descriptif.

**N°6 — escalade** : 30 gabarits du pool X9, **uniquement** en zone grise A3 et sur décision
du PI en cours de cycle. Pas 80.

---

## 1. Question

Un datastore brut de paires (état final du cortex → token suivant), mélangé à la
distribution de sortie, hisse-t-il le **rang** du fait injecté sous **indice paraphrasé**,
et de combien manque-t-il pour renverser un prior confiant (E1c), à l'intérieur du budget
d'innocuité E3 ≤ +0.05 nats/token ?

## 2. Hypothèse

**H** : l'état final du cortex gelé est une **clé suffisamment invariante à la paraphrase**
pour qu'un mélange à l'étage des logits hisse le secret en top-10 sur des questions
reformulées, à un λ conforme au budget E3 — et ce gain disparaît sous permutation des
valeurs (porté par l'adressage, pas par un durcissement de distribution).

**H₀** : même avec la réponse littéralement présente dans le datastore, le mélange ne hisse
rien au-delà du match d'état quasi identique.

**Portée, à écrire dans le journal quoi qu'il arrive** : ce run borne **ce qu'un mélange à
l'étage des logits peut faire avec des clés égales aux états finaux du cortex gelé**. La
borne est conditionnelle à ce point d'injection et à cette forme de clé. **Elle ne se
retourne pas** : un échec ne dit pas « la localisation est le problème », et un système à
réinstallation (X5 / Fast-KV, candidat c) peut en principe la dépasser. Un succès ne
démontre aucun mécanisme — le datastore contient la réponse. *(Phrase obligatoire en cas de
succès, décision PI §13.7.)*

**Conformité D8** : non paramétrique ; datastore construit par un forward du cortex gelé ;
τ et T fixés par des **règles de calcul sur des distributions observées**, sans objectif
optimisé. **D9** : les clés kNN **ne passent pas par G/DG**, et G reste inchangée.

## 3. Ce que le projet sait déjà

| Fait | Chiffre | Source |
| --- | --- | --- |
| Pas de composante directionnelle dans le canal d'état | cos(r, W_U[cible]) = **−0.01** vs 0.136 aléatoire ; ΔH = +0.141 | X7, 2026-08-21 |
| Verrou top-10 | **0/10 sur tous les runs depuis X0** | journal, CLAUDE.md |
| E1 au point courant (défauts X8) | +1.353 ± 1.58 ; E3 −0.014 ✓ | X8 validation |
| E1 au point X1b (ablation `read_gate="none"`) | +0.740 ± 0.799 ; E3 +0.0228 ✓ | X1b / COR-02 |
| Généralisation paraphrase (Δlogp, **jamais rang**) | ratio 0.684 [0.56, 0.99] ; médiane par secret 0.59 ; 0.38 sous gate | COR-02 |
| E1c échoue dans les 4 modes | ΔMarseille **−0.374 à −0.378** ; ΔParis ≈ −0.51 | X8 banc |
| Pas un problème de capacité | 80 faits, 91 % de rétention | X9 |
| D11 : tout canal se juge par position | corr(dommage, entropie) **+0.394** ; ciblage générique R = 0.826 | P5 ; Q-01 |
| D12 candidate : la métrique ΔNLL borne le **gain**, pas le dommage | appariement NLL_base + zone non mordante | Q-01b |
| Clause anti-pseudo-réplication | graines = quasi-doublons (P4 = 0.988) | Q-01b §7(i) |

**Acquis analytique de ce cycle (dérivé, non mesuré — à consigner comme tel)** : pour
`p = (1−λ)p_LM + λ·p_kNN`, `ΔNLL_t ≤ −log(1−λ)` à toute position ; sur texte hors sujet,
E3 → −log(1−λ). Donc **E3 ≤ 0.05 ⟺ λ ≤ λ\* = 1 − e^(−0.05) = 0.0488**. Et le renversement
d'un prior exige `λ > p_LM(x)/(1+p_LM(x))`.

## 4. Prédictions chiffrées

**Instrumentation obligatoire (Neuro B), loggée à chaque requête** — sans elle le verdict
est inattribuable : **R1** rang de l'entrée correcte parmi les k voisins · **R2** d² et cos
à la clé stockée · **R3** masse p_kNN(cible) · **R4** entropie de p_kNN.

| # | Métrique (GPT-2, N=10 secrets, **unité = le secret**) | Si H vraie | Si H fausse | Décision | ANTIPODE |
| --- | --- | --- | --- | --- | --- |
| **V0** *(porte, 1 secret)* | E1-exact top-1 à λ=0.25 ; R1 = 1 ; R3 ≥ 0.9 | — | — | échec ⇒ **run invalide** | — |
| **V1** *(porte d'intégrité)* | E3 sur `NEUTRAL_TEXT` à λ=0.10 vs **−log(0.90) = 0.1054** | — | — | écart > 5 % ⇒ **ce n'est pas un mélange** ⇒ run invalide | — |
| **P1** *(DÉCISIONNELLE)* | n = nb de secrets avec **≥ 2/3** paraphrases en top-10, **à λ\* = 0.0488** | **n ≥ 5/10** | **n ≤ 1/10** | 2–4 = zone grise ⇒ escalade 30 gabarits (décision PI) ; test des signes exact, p = 2^−n | **n ≥ 5 mais R1 > 1 dans la majorité des succès** ⇒ le succès ne vient pas de l'entrée correcte : ininterprétable |
| **P1-desc** | les 30 cas, **descriptif** | ≥ 15/30 (Neuro) | ≤ 5/30 | ne décide pas | — |
| **P2** *(frontière)* | **λ_renv** = plus petit λ tel que « Marseille » soit rang 1 ; **F = λ_renv/λ\*** | F ≤ 2 | **F ≥ 5** | F est le chiffre-titre d'E1c : il quantifie ce qu'un gate devrait racheter | **rang 1 à λ ≤ λ\*** ⇒ le mécanisme de Neuro est faux (pré-déclaré) ; Neuro prédit **rang ∈ [2,4]** |
| **P3** *(contrôle BLOQUANT, hors-ligne)* | Valeurs permutées : rang de la cible ; **logp(cible) doit BAISSER exactement de log(1−λ)** | top-10 ≤ 1/10 ; baisse = log(1−λ) à 1 % | top-10 ≥ 5/10 | si ≥ 5/10 ou baisse ≠ (1−λ) ⇒ **P1/P2 ininterprétables** ⇒ INCONCLUSIF | — |
| **P4** *(spécificité, hors-ligne)* | Store du secret A, prompt du secret B | ≤ 1/10 | ≥ 5/10 | > 5/10 ⇒ sélectivité nulle | — |
| **P5** *(D11)* | E3 par position, store fait-seul **et** distracteur, à λ\* et λ=0.25 | — | — | seuil dur +0.05 ; **ordinal Neuro : E3(fait-seul) > E3(distracteur)** — si l'inverse, la normalisation de p_kNN est suspecte | — |
| **P5b** *(sous porte var(D_t))* | corr(ΔNLL position, entropie H) | **+0.10 à +0.40** (Neuro), ≤ ancre +0.394 | négative ou > +0.40 | porte : var(D_t) > seuil sinon 0/0 | corr **négative** : canal de nature différente — à isoler |
| **P5c** *(D12)* | P5b après appariement par décile de NLL_base + zone [1,3] | **\|corr\| < 0.15** | — | — | **(1) tient mais corr apparié ≥ +0.30 ⇒ D11 se transporte d'un étage à l'autre : pré-déclaré comme le résultat le plus intéressant du run** |
| **P5d** *(discriminant de CHAÎNE)* | Histogramme de ΔNLL par position | (1) **bord droit dur à −log(1−λ)** ; (2) masse empilée au bord aux incertaines, étalée vers 0 aux confiantes ; (3) effondrement après appariement | — | (1) violé ⇒ run invalide | — |
| **P5e** | **R_out = corr(D_permuté,H)/corr(D_kNN,H)** | — | — | descriptif ; pendant du R de Q-01 construit **aux logits** | — |
| **P6** *(profondeur d'émergence de l'invariance à la paraphrase — AUCUNE portée sur X7)* | Clé = couche 6 au lieu de l'état final ; injection aux logits dans les deux bras ; T **par bras** | — | — | descriptif | Neuro : exact ≈ final ; **paraphrase nettement en dessous** |
| **P7** *(sélectivité)* | P1 et P5 sous store **30 k** tokens + le fait (décision PI §13.4) | dégradation ≤ 30 % | effondrement | descriptif | Neuro : **écart fait-seul − distracteur ≥ 7 cas** sur P1-desc |
| **P8** *(forme)* | Rang vs d²_min, poolé (~40 points) | graduelle, monotone, **sans genou** | — | un genou signerait une non-linéarité de la tête de sortie, pas kNN | — |
| **G** *(bras supplémentaire, NON décisionnel)* | α = 1[d²_min ≤ τ], λ=0.25 : E3 ≤ 0.05 ? P1/P2 survivent ? | — | — | τ fixé **avant** lecture de P1/P2 | étiqueté « kNN-LM **gaté** » : question de design du candidat (b) |

**Table d'attribution (à remplir avant tout verdict)** :

| Cas | R1 | R3 | Diagnostic | Conséquence |
| --- | --- | --- | --- | --- |
| P1 échoue | > 1, cos(paraphrase) ≪ cos(exact) | faible | **Branche A — échec côté CLÉ** | V2-D **non réfuté** ; arme le déclencheur *encodage multi-clé* |
| P1 échoue | = 1 | élevée | **Branche B — échec côté INJECTION** | **Seul résultat qui parle contre le candidat (b) M_out** |
| P1 réussit | = 1 | élevée | adressage effectif | borne établie à ce λ |

**Balayage (hors-ligne, gratuit)** : λ ∈ {0.02, **λ\* = 0.0488**, 0.05, 0.10, 0.25} ×
T = c·med_j(d²_j − d²_min), c ∈ {0.3, 1, 3}, **par bras**. k = 8. La frontière
(λ, E3 analytique, E3 mesuré, P1, P2, R1–R4) est rapportée **en entier** ; aucun point
unique ne décide.

## 5. Contrôles et baselines

1. **Baseline d'abord** : `EngramConfig()` défauts — attendus +1.353 ± 1.58, 0/10 top-10,
   ΔMarseille ≈ −0.38, E3 ≈ −0.014 ; non reproduits au centième ⇒ arrêt.
2. **D7** : cache KV vidé avant chaque rappel dans **toutes** les conditions ; le datastore
   n'est **jamais** alimenté pendant `logprob_continuation` (test CPU bloquant).
3. **Contrôle M reset sur le même prompt** : baseline à M = 0 **et** datastore vide.
4. **λ = 0 ⇒ bit-exact** vs le E1 courant, rejoué en fin de run (dérive d'état).
5. **P3, valeurs permutées** — le facteur (1−λ) exact en est le test d'intégrité.
6. **P4, fait croisé** — élimine « le datastore récite quel que soit le prompt ».
7. **Traces égalisées** : une entrée par token du fait, comme M sous `force_write=True`.
8. **Clés kNN NON projetées par DG** (test CPU).
9. **Confondants loggés** : corr(rang_mix, logp_base) par secret ; part des tokens communs
   du fait dans les valeurs (ils gonflent E1-exact **et** E3).

## 6. Critères d'abandon

**Ce qui tue H** : P1 ≤ 1/10 au niveau secret. Verdict `REJETE`, **formulé avec la portée de
Neuro** : *« l'état final du cortex gelé n'est pas une clé invariante à la paraphrase »* — et
non « l'espace de sortie n'est pas le bon point d'attaque ». La suite dépend de la table
d'attribution : Branche A ⇒ V2-D non réfuté, déclencheur multi-clé armé ; Branche B ⇒
argument contre le candidat (b).

**Ce qui invalide le run** : V0 échoué ; **V1 échoué** (écart > 5 % à −log(1−λ)) ; **P5d(1)
violé** ; 0 write / datastore vide ; NaN ou inf ; fp16 dans la chaîne de distance ; DG
appliqué aux clés kNN ; λ=0 non bit-exact ; **P3 ≥ 5/10 ou baisse ≠ (1−λ)** ⇒ `INCONCLUSIF`.

**Ce qui n'est PAS un critère d'abandon** : E3 > +0.05 aux λ élevés — résultat attendu et
prédit analytiquement, rapporté « plafond conditionnel, hors budget ».

## 7. Variables fixées

`seed=0` ; `gpt2` ; `layer_index=6` ; `lam=2.0` ; `cap=0.5` ; `eta=0.2` ; `decay=1e-3` ;
`thr=4.0` ; `dg=8192/64` ; `read_gate="keysim"` ; `prune=512/0.10`. k = 8.
Données : les 10 `SECRETS` + `FACT_TEMPLATE` + les 4 `QUESTIONS` de `eval/fact_injection.py` ;
`E1C_FACT`/`E1C_QUESTION` de `eval/read_gate.py` ; `NEUTRAL_TEXT` de `eval/collateral.py`
(343 positions) ; bras distracteur = **30 000** premiers tokens du cache `data/rfc9293.txt`
(**SHA-256 à journaliser**).
**Statistique** : unité = le secret (N=10), **jamais la graine** ; test des signes exact ;
bootstrap par secret 10 000 tirages ; permutations au plancher écrites « p < 10⁻³ » ; par
position, ρ̂ estimée **jusqu'au lag 5**, L ≥ 2× le temps d'autocorrélation intégré si
ρ̂₁ > 0.1, **repli L = 25 si ρ̂₁ ≤ 0**.
**Numérique** : distances en **fp32** ; softmax kNN après soustraction de d²_min ; mélange
en **log-espace** (`logaddexp`).

## 8. Variable manipulée

**Une seule** : λ, avec **M = 0** en condition principale. Bras descriptifs pré-déclarés,
lus après le verdict : `M au point courant + kNN` (additivité), P6 (clé couche 6),
P7 (distracteur), G (gaté).

## 9. Budget

| Passe GPU | Contenu | Durée |
| --- | --- | --- |
| V0 | smoke test 1 secret | 2 min |
| A | store fait-seul, M=0 : 10 secrets × (injection + 4 questions) ; log de `p_LM`, des 8 distances/valeurs, **et des états couche 6 dans la même passe** (P6 gratuit) | ~3 min |
| B | idem avec M au point courant (additivité) | ~3 min |
| C | E1c (M=0 et M chargée) | ~2 min |
| D | E3 par position, `NEUTRAL_TEXT`, store fait-seul | ~4 min |
| E | construction du store distracteur (**30 k** tokens) + P7 + E3 distracteur | **8–10 min** |
| **Total** | | **~22–25 min GPU** |

Hors-ligne (CPU, gratuit) : toute la grille λ × T, P3, P4, P5b/c/d/e, P8, la frontière
complète, le bras G au-delà du choix de τ.
**VRAM** : GPT-2 fp16 ~0.3 Go + store fait-seul 37 Ko/secret + store distracteur ~46 Mo
(30 k × 768 fp16, distances en fp32) → **< 0.5 Go** sur 6.

## 10. Livrables attendus

- **Script** : `eval/knn_ceiling.py` (nouveau, SPDX AGPL-3.0-or-later), réutilisant les
  constantes existantes — aucune duplication. **Séparation nette passe GPU (log brut) /
  analyse hors-ligne**, pour que la frontière se recalcule sans GPU.
- **Config** : champs `knn_*` dans `EngramConfig`, **défaut = comportement actuel**
  (`knn_lambda: float = 0.0`), plus `knn_k`, `knn_temp_c`, `knn_key_layer ∈ {final, inject}`,
  `knn_gate_tau` (bras G, désactivé par défaut) — décision PI §13.5. Le datastore vit dans
  le script d'éval, pas dans `engram/`.
- **Capture de l'état final** : second point de capture (dernier bloc, pré-`lm_head`) dans
  `engram/cortex.py`, **inerte par défaut**, activé par flag.
- **Tests CPU** : (i) λ=0 ⇒ logits bit-exacts ; (ii) identité du mélange en log-espace et
  bord dur `−log(1−λ)` sur cas synthétique ; (iii) distances en fp32 (régression fp16 à
  ‖h‖² ~ 10⁴) ; (iv) soustraction de d²_min (underflow) ; (v) **les clés kNN ne passent
  jamais par G/DG** ; (vi) le datastore ne se remplit jamais pendant `logprob_continuation` ;
  (vii) permutation = mêmes clés, distances inchangées ; (viii) k > taille du store.
- **Journal** : entrée datée, protocole recopié, **frontière (λ, E3, P1, P2, F)**, table
  d'attribution R1–R4 remplie, histogramme P5d, brutes hashées SHA-256.
- **Tableau §4** : ligne dans le tableau des **instruments/ablations**, jamais dans celui des
  mécanismes retenus.
- **Correctifs à la fiche V2-D** rédigés pour le PI (§13.6), non appliqués par le labo.

## 11. Questions résiduelles pour Neuro (à traiter en interprétation)

1. La cellule antipode « P2 rang 1 à λ conforme » est **quasi forcée par l'arithmétique**
   (rang 1 exige λ ≳ 0.33 ≫ λ\*) : sans pouvoir discriminant ? La remplacer par une cellule
   sur **F = λ_renv/λ\*** (mécanisme faux si F < 2) ?
2. P8 (forme sans genou) est-elle testable sur ~40 points à axe d²_min continu, ou exige-t-elle
   un axe d'indice construit (passe GPU supplémentaire) ?
3. Ton terme T2 (soulagement asymétrique par les mots fonctionnels du datastore) est mesurable
   via le log « part des tokens communs du fait dans les valeurs ». Ce log est-il le test de
   T2, indépendamment du signe de P5b ?

## 12. Questions résiduelles pour Math (à traiter en interprétation)

1. Seuil exact de la porte var(D_t) sur P5b — un chiffre, pas un principe.
2. La recomposition hors-ligne du rang sous mélange (arbitrage n°4) est-elle exacte, y compris
   pour la permutation, dès lors que `p_LM` complet et les k couples (d², token) sont loggés ?
3. Règle de fixation de **τ** (bras G) : quel quantile des distributions d²_min fait vs neutre,
   et comment garantir qu'il soit fixé sans regarder P1/P2 ?
4. En zone grise (n ∈ [2,4]), 30 gabarits suffisent-ils à trancher 5/10 vs 1/10 transposé à
   N=30, ou faut-il redéfinir le seuil proportionnellement ?

## 13. Décisions du PI (2026-08-21, gate de pré-enregistrement)

1. **Gate A1 : bras supplémentaire**, pas condition principale. La principale reste « kNN-LM
   nu » — l'instrument ne doit pas devenir un candidat produit déguisé.
2. **Condition principale à M = 0 : confirmée.** `M + kNN` devient un bras descriptif.
3. **Critère d'ouverture du candidat (b) M_out : P1 décide, F informe.** P1 ≥ 5/10 suffit ;
   E1c n'est pas exigé (Neuro le prédit comme le plus difficile, pas le plus révélateur).
4. **Bras distracteur : version réduite à 30 k tokens** (~8-10 min). Budget total ~22-25 min.
5. **`knn_*` dans `EngramConfig`** à `knn_lambda=0.0` (convention maison), ligne au tableau
   des **ablations/instruments**.
6. **Les deux façades de la fiche V2-D** (« kNN-LM ≈ familiarité » ; « instrument qui mesure
   le PLAFOND ») : correctifs **rédigés par le labo à l'attention du PI** dans l'entrée de
   journal ; le PI les applique lui-même à `docs/EXTENSIONS.md`.
7. **Doctrine** : en cas de succès, le journal écrit explicitement *« borne conditionnelle à
   ce point d'injection et à cette forme de clé, non retournable en cas d'échec »*.

## Historique

- 2026-08-21 : brouillon du Directeur
- 2026-08-21 : avis Math (RÉSERVÉ, 3 amendements) et Neuro (FAVORABLE, 2 amendements) —
  convergence indépendante sur l'identité `ΔNLL ≤ −log(1−λ)`
- 2026-08-21 : protocole consolidé, décisions du PI intégrées — proposé
- 2026-08-21 : pré-enregistré par le PI
- 2026-08-21 : arrêt à la porte V0 (clause R3 ≥ 0.9, incompatible avec la règle T retenue) ;
  arbitrage PI : V0 réduite à sa clause de récupération, R3 descriptif
- 2026-08-21 : arrêt à la porte V1 (écart 6.16-7.20 % > 5 %), dont le déficit est porté par
  le soulagement T2 que le protocole prédit lui-même
- 2026-08-21 : **run déclaré INVALIDE par le PI** — refus d'un troisième amendement de porte
  en cours de route ; statut TERMINE — INVALIDE, re-pré-enregistrement v2 avec re-collecte
  intégrale à neuf
