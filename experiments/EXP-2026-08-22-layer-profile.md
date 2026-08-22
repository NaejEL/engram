# EXP — I2 : existe-t-il une identité d'unité invariante et adressable dans un cortex gelé ? (`layer_profile`)

**Statut : PROPOSE**

Origine : `docs/EXTENSIONS.md` §2 (I2) · cadrage PI 2026-08-22 · re-cadrage PI post-v3 · **consolidation post-banc D14-S (E = 3) et post-avis Math/Neuro (deux RÉSERVÉ, 15 changements), 2026-08-22**.
Rôle : **gate scientifique de v4**. Instrument, pas mécanisme : aucune injection, aucune écriture, `M = 0`, `engram/` non modifié, aucun gradient (D8).

---

## Arbitrage

### A. Retours du banc D14-S (E = 3) + une découverte hors E

| # | Constat | Statut | Traitement |
| --- | --- | --- | --- |
| **B-1** | **`V-div` est INVERSÉE** : elle mesure un cardinal de strate ⇒ fonction croissante de la redondance ⇒ elle **récompense la dégénérescence**. `fact_pairs(30)` 0.99838 FAIL / jeu v3 1.000 PASS (3 graines : 0.99826 / 0.99842 / 0.99838). | **porte détruite et remplacée** | Scindée en `V-diversité` (combinatoire) et `V-puissance` (design) — §4.7. Direction restaurée. |
| **B-2** | **`V-suffixe` porte une attente périmée** : para1 mesuré **1.0000** (verbe global `"bears the codename"`, §15 A-1 de v3), para3 1.0000, para2 0.1724. Effet secondaire : **le suffixe commun de para2 est vide** ⇒ la contrainte « même suffixe » de la nulle du maillon 2 y est **vacuée**. | **porte re-dérivée + nulle re-spécifiée** | Constantes **entièrement dérivées** (§4.7) ; la nulle du maillon 2 abandonne « même suffixe » pour « même cadre de type, verbe conservé, slots de contenu remplacés » (§5). |
| **B-3** | **Les bandes du §4.5 ne sont pas exhaustives** : le protocole qui grave D18 le violait. Trou : `IC_inf < 0.9622 ≤ IC_sup` disjoint des planchers. | **intégré** | Partition **N / M / I / V** ; `D` rétrogradée en **overlay inter-modèles** avec précédence gravée (§4.5). |
| **B-4** *(hors E, D14-R)* | L'**Eq. 1 de arXiv:2412.09563 telle qu'imprimée** normalise les valeurs propres de la Gram **brute** par sa trace et **ne porte pas** `A_ij = K_ij/(n√(K_ii K_jj))` ; cette normalisation vient de **Giraldo et al. 2014**. Les deux coïncident ssi toutes les lignes de `Z` ont même norme. | **prescription conservée, attribution corrigée** | §3 + porte **`V-source`** : l'équation se relit dans le **PDF**, pas en HTML. |

### B. Avis Math — 7 changements

| # | Remarque | Traitement |
| --- | --- | --- |
| **M-9** | **La bande V est INDÉCIDABLE à N = 30** en AUC : demi-largeur requise ≤ 0.0178 (SE ≤ 0.0091) contre SE ≈ 0.0104 (borne irréaliste) et ≈ 0.020 (réaliste) ; sur S2, K = 10 ⇒ IC ≈ **0.17** contre un couloir de **0.0187**. | Défaut **0-13**. Corrigé par M-10 + M-11. |
| **M-10** | `\|S2\|(N)` = 10/20/60/120 pour N = 30/40/60/80 ; clusters **saturent à K = 20 dès N = 40**. | **N = 80 adopté.** Sur la partition de Neuro : `P-own` sature à **K = 16** (#owners), `P-ent` à **K = 20** (#entités). |
| **M-11** | **Décider le couloir sur `Recall@1`, pas sur l'AUC** : marges 0.25 contre un copeau de 0.019. Le §4.5 disait **déjà** que Recall@1 prévaut. | `A^s` **retirée de toute décision**. Couloir sur `R1_36`, jeu de candidats de **taille 36 exactement** ⇒ **aucune extrapolation**. |
| **M-12** | **DÉFAUT NOUVEAU** : `max_ℓ AUC(ℓ \| S2 ∪ S3)` **poole 10 paires S2 et 75 paires S3** ⇒ la primaire du couloir est dominée à **~88 %** par le corpus relégué pour dégénérescence. | Défaut **0-11**. **Dé-poolage total** : `S3` supprimée, le jeu v3 devient le bras descriptif **`B-v3`**. |
| **M-13** | **Scinder `V-div`** ; **fusion déconseillée** (un FAIL sans cause nommée est un FAIL qu'on discute après coup). | §4.7. Sous décision AUC, `V-puissance` **FAIL à tout N ≤ 80** : **la porte déclare elle-même l'indécidabilité**. *(Le chiffre « K ≥ 429 » de l'avis Math s'est révélé **non re-dérivable** au second passage du banc — 438 à 442 selon l'opérationnalisation, 2745 sous celle du banc ; il est retiré au profit de l'énoncé robuste « K requis ≥ 442 contre K ≤ 20 », cf. §3.)* |
| **M-14** | **Bande `I`** ; conventions de bord ; **`D` n'est pas une bande** mais un overlay avec précédence : **si l'un des modèles rend I, le verdict global est I**. | §4.5, intégralement. **Intensification du Directeur** : la bande **N** se décide sur une **différence appariée** (D16), pas sur un niveau. |
| **M-15** | **Biais de sélection du max** : `E[max − moyenne] ≈ 0.033` d'AUC > couloir entier (0.0187) ⇒ le max **fuit la bande N**. | §4.3. Permutation d'étiquettes d'unité **à couche fixée avec recalcul de `max_ℓ`** ; IC avec **re-sélection de l'argmax dans chaque rééchantillon** ; estimateur débiaisé. **« Max des IC par couche » = motif d'invalidation.** |
| **M-16** | Le **percentile sous-couvre près de la borne 1**. | **BCa requis** dès qu'une borne dépasse 0.95. |

### C. Avis Neuro — 8 changements

| # | Remarque | Traitement |
| --- | --- | --- |
| **N-9** | **Le VERBE n'est pas un slot d'identité** : cinq quasi-synonymes ; deux unités n'en différant que par le verbe **désignent le même fait**. Espace d'identité = **16 × 20 = 320**. Les **65 paires « verbe seul »** partagent un slot inexistant (para3) ou universel (para1) ⇒ **`AUC(S1)` était du bruit étiqueté**. | Défaut **0-12**. S0/S1/S2 **abolies** ; le verbe devient une **covariable de ventilation**. |
| **N-10** | **Partition par slot d'IDENTITÉ** : `P-0` / `P-own` / `P-ent` / `P-both`. | Re-vérifiée par comptage direct : N=30 → **411/14/10/0** (C(30,2)=435 ✓) ; N=80 → **2880/160/120/0** (C(80,2)=3160 ✓). |
| **N-11** | **`P-both` est arithmétiquement impossible** (`lcm(16,20) = 80 \| d`) ⇒ **la paire minimale sur l'ENTITÉ — le leurre canonique — n'existe à aucun N ≤ 80**. | Gravé §4.2, phrase reprise mot pour mot. Première **spécification dure** du matériel de v4. |
| **N-12** | **Le discriminant de `P-ent` est dégradé par les règles de paraphrase** : l'owner est le seul discriminant **et** le seul slot dont les trois types changent la forme de surface. | Fonde le déplacement de la cellule décisionnelle. |
| **N-13** | **Asymétrie décisionnelle gravée** sur `P-ent`. | §4.5, clause reprise mot pour mot. |
| **N-14** | **La cellule décisionnelle doit être `P-own`** (discriminant = entité, **verbatim dans les trois types**). | §D.2, arbitrage central. |
| **N-15** | **N = 80** : diversité **inchangée** (16/20/5, tous triplets distincts, `lcm = 80`), ×11 / ×12, coût nul, **zéro contenu à écrire**. « Le matériel divers manque de leurres » est **FAUX pour ce pool** : la périodicité **est** la structure de leurres. | Convergent avec M-10. `n_intra = 240`, `n_inter = 28 440`. |
| **N-16** | **FAÇADE au §D.3** : trois strates non ordonnées ne font pas une fonction de transfert. | Défaut **0-14**. « Courbe mesurée » → **« test de robustesse en deux régimes »**. |
| **N-17** | **Prédiction signée gratuite** `AUC(P-ent \| para3×para3) > AUC(P-ent \| para1×para1)`. | §4.9, **N-P6**. Seul contrôle séparant « leurre dur » de « discriminant atténué ». |
| **N-18** | **Matériel de v4 chiffré** : contraste sur l'**entité**, **≥ 3 bins gradués**, owner et verbe fixes, **17 entités à ajouter**. Verbe **exclu** du budget de leurres. | §2 (argument réécrit), §10 (livrable). |
| **N-19** | Si `H` est non interprétable sur les trois modèles, **la retirer**. | §4.6, retrait **automatique**. |

### D. Arbitrages du Directeur

**D.1 — Questions croisées : toutes tranchées.** `A^s` sans objet (36 compétiteurs réels) · **cluster de rééchantillonnage = la composante de slot** (owner pour `P-own`, entité pour `P-ent`), seul cluster sous lequel les paires d'une strate sont indépendantes, et qui **préserve exactement** les effectifs de strate ⇒ la question du bootstrap stratifié disparaît · fusion des portes refusée · nulle mélangée pouvant être **permissive** ⇒ clause §5 · σ₀ = 0.5 borne de Bernoulli, aucune dérivation en attente · `H` retrait automatique · le verbe n'est pas un slot · contraste de v4 = l'entité · multiplicité du max traitée par M-15.

**D.2 — La strate décisionnelle : `P-own`.** (1) **Validité de construit** : son unique discriminant est l'**entité**, **verbatim dans les trois types** ⇒ la mesure est invariante à la manipulation, elle ne peut pas confondre « le leurre est dur » et « la règle a abîmé le discriminant ». (2) **Dé-poolage** : une décision, une strate, une statistique. (3) **Puissance** : `K = 16`, exactement le minimum requis — **la marge est nulle, déclaré avant mesure**. **Coût : `P-ent` (K = 20) est mieux dotée mais confondue ; `P-own` est propre mais tout juste décidable. Le pool ne permet pas les deux** — deuxième spécification dure du matériel de v4.

**D.3 — L'argument économique, réécrit.** « I2 coûte des minutes, v4 coûte des jours » était **un slogan**, acquitté en **0-15** : le matériel de v4 se chiffre à 17 mots et une table d'indexation, soit des heures. La justification correcte n'est pas un rapport de coûts mais une **différence de nature** : **aucun matériel ne répare l'absence d'une identité adressable.**

### E. Écarté

`V-div` (détruite) · `A^s` comme base de décision · strates `S0`/`S1`/`S2` · `S3` comme strate · `S2 ∪ S3` comme population du couloir · le verbe comme slot d'identité · **N = 30** · « max des IC par couche » · « la tension devient une courbe mesurée » · le rapport de coût en slogan · attribution de la normalisation des lignes à Skean · lecture HTML pour citer une équation · `V-signe`, `V-vierge`, `NEUTRAL_TEXT` comme contrôle, cellule C2 « résultat le plus informatif », « Qwen instruct + RLHF », corpus (a) = jeu v3, permutation des étiquettes de **couche**.

---

## 0. Défauts acquittés

| # | Défaut | Portée |
| --- | --- | --- |
| **0-1** | Le score en ratio mesurait l'anisotropie autant que l'invariance. | Fatal à la primaire ; corrigé par l'AUC. |
| **0-2** | `H` sans normalisation des lignes est confondue avec le profil de normes. | Fatal à la seconde quantité. |
| **0-3** | « P-A falsifiée ⇒ P6 falsifiée » illégitime. | Aurait tué P6 avec un instrument qui ne la teste pas. |
| **0-4** | `NEUTRAL_TEXT` pré-enregistrait l'issue facile comme prédiction. | Prédiction infalsifiable. |
| **0-5** | Le corpus hérité de v3 confondait « pas d'identité » et « pas de diversité ». | Fatal à la gate. |
| **0-6** | `V-plat` posait 5 % ; `V-signe` vacuée ; `V-vierge` insatisfiable. | Trois modes déjà catalogués. |
| **0-7** | **`V-div` était INVERSÉE** : cardinal de strate ⇒ croissant en redondance ⇒ elle **récompensait la dégénérescence**, faisant échouer le corpus divers (0.99838) au profit du dégénéré (1.000). | **Fatal au choix de corpus** : la porte censée protéger la diversité l'aurait interdite. **Trouvée par exécution, pas par relecture — troisième occurrence du motif.** |
| **0-8** | **`V-suffixe` portait une attente périmée** (para1 ≈ 0 attendu, 1.0000 mesuré, verbe global §15 A-1), et son corollaire **vidait la nulle du maillon 2** (suffixe commun de para2 vide ⇒ contrainte vacuée). | Une porte qui ne mord pas **plus un maillon sans nulle effective** : le mode de mort exact de v3. |
| **0-9** | **La partition des bandes n'était pas exhaustive** : `IC_inf < 0.9622 ≤ IC_sup` n'appartenait à aucune bande. **Le protocole qui grave D18 violait D18.** | Aurait produit un run sans verdict pré-enregistré, donc décidé après lecture. |
| **0-10** | **Attribution fausse (D14-R)** : la normalisation des lignes attribuée à Skean et al. Eq. 1, qui ne la porte pas ; elle vient de **Giraldo et al. 2014**. Les deux coïncident ssi le défaut 0-2 est absent. | La prescription reste juste, l'attribution était fausse ; **lecture HTML au lieu du PDF**. |
| **0-11** | **La statistique du couloir poolait 10 paires `S2` et 75 paires `S3`** ⇒ la primaire de la gate était **dominée à ~88 % par le corpus relégué pour dégénérescence**. | **Fatal à la gate** : le couloir aurait été décidé par `cos +0.99989`. |
| **0-12** | **Le verbe était traité comme un slot d'identité** alors que ce sont cinq quasi-synonymes. | Fatal à la strate intermédiaire et à toute lecture ordinale. |
| **0-13** | **Le couloir gravait un verdict qu'aucune donnée conforme ne pouvait rendre** (SE requis ≤ 0.0091 contre ≈ 0.020 ; IC ≈ 0.17 contre un couloir de 0.0187). | **Fatal à la décision** : une gate indécidable est une gate décidée par le rédacteur. |
| **0-14** | **« La tension devient une courbe mesurée »** : trois strates non ordonnées, dont une vide de sens et une confondue, **ne font pas une fonction de transfert**. | **Vraie en intention, fausse en exécution** — une façade du genre que ce protocole prétend interdire. |
| **0-15** | **« I2 coûte des minutes, v4 coûte des jours » était un slogan.** | L'argument de la gate reposait sur un rapport de coûts faux ; il repose désormais sur une différence de nature. |

## 1. Question

**Existe-t-il, dans un cortex gelé, une identité d'unité factuelle invariante à la paraphrase et séparable des autres unités — à n'importe quelle couche — et cette séparation survit-elle à des concurrents appariés sur un slot d'identité, au niveau d'exigence que v4 lui demanderait ?**

## 2. Hypothèse

**H_I2** — il existe au moins une couche `ℓ` où les états d'indices d'une même unité sont plus similaires entre eux qu'entre unités, **au-delà de ce que le recouvrement lexical, la position, le cadre et l'ordre produisent seuls** ; **et** cette séparation suffit, sur la strate `P-own`, à un adressage par plus proche voisin contre **36 concurrents** au niveau `Recall@1 ≥ 0.25`.

**H₀** — à toutes les couches, la différence appariée aux planchers du §5 est indiscernable de 0.

**Pourquoi maintenant — l'argument réécrit (D.3).** **Aucun matériel ne répare l'absence d'une identité adressable.** Le matériel de v4 se chiffre honnêtement à **17 entités, une table d'indexation non modulaire et ≥ 3 bins gradués** : des heures. Ce qu'I2 met en jeu n'est donc pas un budget de rédaction mais un **choix de mécanisme**, plus le cycle complet construit autour. Si aucune couche ne porte de séparation, l'adressage par plus proche voisin en cosinus est mort **quel que soit le matériel** ; si elle existe mais reste insuffisante, v4 **change de mécanisme** (réorientation automatique) ; si elle suffit, I2 rend **deux spécifications dures** (§10). **La gate est rentable dans les quatre bandes.**

**Motivation architecturale, hors clause, non porteuse** (Neuro) : *l'idée d'une couche « pour lire » distincte d'une couche « pour écrire » n'a pas d'ancrage biologique par la profondeur. La biologie dissocie par compartiment (EC II/III vs V/VI ; ségrégation SR/SLM en CA1) et par phase (thêta — c'est X3), jamais par position dans une profondeur. Dans engram le flux résiduel est un bus unique partagé.*

**Nomenclature gravée** : `ℓ*_contrast` et `ℓ*_H`. « Couche-clé »/« couche-injection » confinées au §4.8, entre guillemets, précédées de « candidate ».

**Interdictions de vocabulaire, avant mesure :**

- **(i)** `ℓ*_contrast` = « la couche où l'invariance à la paraphrase lexicale est la plus contrastée ». **« Reconnaissance » supprimé.**
- **(ii)** `ℓ*_H` = « minimum de rang effectif ». **« Vallée de compression », « goulot d'information », « couche optimale » : interdits.**
- **(iii)** Une coïncidence `ℓ*_H ≈ ℓ*_contrast` **ne démontre pas** que lire et écrire se font au même endroit. Propriété non mesurée : **tolérance à l'injection**.
- **(iv)** L'entropie matricielle **ne mesure aucune quantité d'information sur le contenu**.
- **(v)** Une baisse de `H` est compatible avec une dimension parasite ou un puits d'attention. Sans **λ₁/Σλ**, un minimum de `H` **n'est pas interprétable**.
- **(vi)** I2 mesure une **géométrie non supervisée sur un cortex intact**. Aucun verbe « porte / stocke / encode / contient / est responsable de ». Non mesurées : **décodabilité** (D8) et **causalité**.
- **(vii)** *(N-16)* **« Courbe », « fonction de transfert », « gradient de similarité » sont interdits** pour la comparaison `P-own`/`P-ent`. Formulation autorisée : **« test de robustesse en deux régimes »**.
- **(viii)** *(N-11)* Il est **interdit** d'écrire que le protocole a « choisi » le contraste d'owner. Formulation gravée : **« l'arithmétique du pool l'a imposé ; la paire minimale sur l'entité n'existe à aucun N ≤ 80 »**.

**Conformité** : D8 · D11 (sans objet) · D12 · D13 · D14 · **D14-S** · **D14-R** · **D16** (existence = AUC ; couloir = **ΔR1 appariée**) · **D17** (cinq maillons, cinq nulles) · **D18** (partitions exhaustives, bords inclus) · **D19** · **D20**.

## 3. Ce que le projet sait déjà — provenance (D14-R)

| Fait | Chiffre | Source | Statut |
| --- | --- | --- | --- |
| Diversité du jeu v3 | **5 owners, 6 entités** | journal 2026-08-22 | **fait échouer `V-diversité`** |
| Similarité inter-unités v3 | `cos` max **+0.99989** | journal 2026-08-22 | devient le bras **`B-v3`**, jamais décisionnel |
| Absence d'adressage dans v3 | **90/90** ; planchers **12/30** et **23/30** ; clé étrangère **30/30** en L6 | journal 2026-08-22 | motive le re-cadrage ; **aucun chiffre en porte** |
| Budget v4 | **s ≈ 36** | journal 2026-08-22, D19 | entre dans le couloir — **dérivé** |
| Couche D3 | GPT-2 **6/12**, SmolLM2 **16/32** validées ; Qwen **14/28 POSÉE** | D3 | porte P-D |
| Profondeurs | `L` = **12 / 32 / 28** | re-mesuré | **à re-lire du config** |
| P6 | `h` 1.00 vs 0.33, `N_eff = 3` | run v2 **INVALIDÉ** | **aucune porte ni prédiction** |
| E1b | **~0.6-0.7, N = 10** | COR-02 | une invariance existe **fonctionnellement** ; I2 demande si elle est **géométriquement séparable** |
| Recensement d'identité | N=30 : 411/14/10/**0** ; N=80 : 2880/160/120/**0** | Neuro, **re-vérifié par comptage direct** | **fonde §4.2 et §4.5** |
| Clusters | `P-own` K = 14 (N=30) → **16** ; `P-ent` K = 10 → **20** | dérivé, à re-vérifier au banc | **fonde `V-puissance` et N = 80** |

**Dérivations portées en entier :**

- **`⌊L/2⌋`** : 6, 16, 14. **Fenêtre** `w(L) = max(1, ⌊L/12⌋)` ⇒ **[5,7] / [14,18] / [12,16]**.
- **AUC** : `A(ℓ) = P(cos_intra > cos_inter) + ½·P(=)`, Mann-Whitney. **Invariante sous toute transformation strictement monotone par couche.** Nulle connue 0.5. **Différence, pas taux ⇒ D16.**
- **Entropie** — **attribution corrigée (0-10)** : convention **Giraldo et al. 2014**, Gram côté échantillons, **normalisation des lignes** `A_ij = K_ij/(n·√(K_ii·K_jj))`, `tr(A) = 1`, `H = −Σ λ log λ`. **Skean et al. Eq. 1, telle qu'imprimée, ne porte PAS cette normalisation** ; les deux coïncident ssi toutes les lignes ont même norme, c'est-à-dire ssi le défaut 0-2 est absent. **La prescription est conservée pour sa raison propre, pas parce qu'un article l'imprimerait.**
- **« α→1 ≡ RankMe » est inexact** : RankMe = entropie des σ ℓ1-normalisées ; von Neumann = entropie des σ².
- **Dépendance à `n`** : `H ≤ log min(n,d)`. `n_a = 90` (sous-échantillonné parmi 240, B = 200). **Comparaisons de niveaux interdites.**
- **Puissance** : `K_S ≥ (1.96·σ₀/|θ−T|)²`. Sous `R1_36`, σ₀ = 0.5 (Bernoulli, conservateur), |θ−T| = 0.25 ⇒ **K ≥ 15.37 ⇒ K ≥ 16**. Sous décision **AUC**, le chiffre « K ≥ 429 » de l'avis Math **n'est pas re-dérivable** (les opérationnalisations admissibles rendent **438 à 442** avec `σ₀ = √(T(1−T))`, marge `0.98 − T`, et **2745** avec `σ₀ = 0.5`, marge `T⁺ − T = 0.0187063`). **Il est retiré** (D14-R : un chiffre sans sa dérivation ne se cite pas). **Ce qui est conservé est l'énoncé robuste** : sur *toutes* les opérationnalisations examinées, **K requis ≥ 442** contre **K ≤ 20 disponible dans ce pool** — un facteur **≥ 22**. **La conclusion — le couloir en AUC est indécidable à tout N ≤ 80 — ne dépend donc d'aucun choix de σ₀ ni de marge**, et c'est sous cette forme que la porte la déclare.

## 4. Prédictions chiffrées

### 4.1 Indexation

`ℓ = 0` = embeddings ; `ℓ ∈ [1, L]` = sortie du bloc `ℓ`. **L'argmax décisionnel se cherche sur `[1, L]`.** `ℓ = 0` est la nulle « aucun calcul » (maillon 3), **exclue de l'argmax**.

### 4.2 Corpus, partition d'identité, N

- **(a) PRIMAIRE — `pool.fact_pairs(80)`**, paraphrasé par les **constantes gelées** (`PARA1_VERB`, `PARA2_PREFIX`, `PARA3_HEAD/MID/TAIL`, `OWNER_OBJ`). **16 owners, 20 entités, 5 verbes, 80 triplets distincts** (`lcm(16,20,5) = 80`). **Zéro contenu nouveau à écrire.**
- **(b) BRAS DESCRIPTIF `B-v3`** — le jeu d'unités v3, hashé. **Jamais fusionné, jamais décisionnel, jamais appelé « strate ».**
- **(c) descriptif, hors clause** : `NEUTRAL_TEXT`.

**Partition par slot d'IDENTITÉ** — espace d'identité **16 × 20 = 320** ; **le verbe n'est pas un slot** et devient une **covariable de ventilation** :

| Classe | Définition (sur `d = \|i−j\|`) | N = 30 | N = 80 | Cluster | K |
| --- | --- | --- | --- | --- | --- |
| **P-0** | ni owner ni entité | 411 | **2880** | — | — |
| **P-own** | `16 \| d`, entité ≠ | 14 | **160** | composante d'owner (16 blocs de 5) | **16** |
| **P-ent** | `20 \| d`, owner ≠ | 10 | **120** | composante d'entité (20 blocs de 4) | **20** |
| **P-both** | les deux | **0 — IMPOSSIBLE** | **0** | — | — |

**Clause gravée (N-11), à recopier au journal :** *« `P-both` exige `lcm(16,20) = 80 | d` : la paire minimale sur l'ENTITÉ — le leurre canonique de la tâche à leurres, tout fixe sauf l'item — n'existe à aucun N ≤ 80. Des trois paires minimales concevables, le pool n'en supporte qu'une : celle sur l'owner. Le protocole n'a pas CHOISI le contraste d'owner ; l'arithmétique du pool le lui a IMPOSÉ. »*

**N = 80** : diversité **inchangée** (16/20/5) ; ×11 sur `P-own`, ×12 sur `P-ent` ; clusters **saturés** dès N = 40, au-delà N = 80 resserre l'estimation intra-cluster ; **79 compétiteurs > 36** ⇒ couloir mesurable **sans extrapolation** ; coût GPU quasi nul.

**Comptes** : `n_intra = 240` ; `n_inter = C(80,2) × 9 = 28 440`. **Capture** : état au **dernier token de l'indice**.

### 4.3 Quantités

| Quantité | Définition | Statut |
| --- | --- | --- |
| **`AUC(ℓ)`** | `P(cos_intra > cos_inter) + ½P(=)`, corpus (a) entier | **PRIMAIRE — existence (P-A)** |
| **`ΔR1(ℓ)`** | `R1_36(ℓ) − R1_36^plancher(ℓ)`, appariée par requête, strate **`P-own`** | **PRIMAIRE — bande N (D16)** |
| **`R1_36(ℓ)`** | Recall@1 sur **1 cible + 36 concurrents** (§4.5), strate `P-own` | **DÉCISIONNELLE — bandes M/I/V** |
| `R1_36(ℓ \| P-ent)` | idem sur `P-ent` | **borne inférieure conservatrice** (N-13) |
| `AUC(ℓ \| P-0 / P-own / P-ent)` | par classe | ventilation, plus N-P6 |
| `AUC(ℓ \| B-v3)` | bras v3 | **descriptif seul** |
| `H(ℓ)`, `λ₁/Σλ (ℓ)` | Giraldo complète, `n_a = 90` | secondaire ; **retrait automatique** (N-19) |
| `z(ℓ)`, `s_intra`, `s_inter`, ratio | — | ventilation descriptive |

**Inférence** : cluster de rééchantillonnage = la **composante de slot** (préserve **exactement** les effectifs de strate) · **B = 10 000, argmax re-sélectionné dans chaque rééchantillon**, estimateur débiaisé `θ̂ = 2·max_ℓ obs − mean_b(θ*_b)` · **BCa** dès qu'une borne dépasse 0.95 · **bande N par permutation des étiquettes d'unité à couche fixée avec recalcul de `max_ℓ`** (FWER exact). **« Max des IC par couche » = motif d'invalidation.**

### 4.4 Prédiction primaire d'existence

| # | Prédiction | Si vraie | Antipode (D13) |
| --- | --- | --- | --- |
| **P-A** | **GPT-2** : `max_{ℓ∈[1,L]} AUC(ℓ)` **strictement supérieur à chacun des cinq planchers du §5**, IC 95 % BCa **disjoint** du plus haut, **et** `p_perm ≤ 0.05`, **et** `AUC(ℓ*_contrast) > AUC(L)` | une identité d'unité **existe**, non réductible au matériel, à la position, au cadre ni à l'ordre | **`AUC(L)` maximal** ⇒ l'état final est le meilleur porteur d'identité **sous cosinus**. **Conséquence pré-écrite** : *l'étage de clé de v4 se conçoit sur l'état final.* |
| **P-A′** | **SmolLM2** : idem | réplication | idem |

**Conséquence sur P6 — clause gravée, à recopier telle quelle :**

> *« Conséquence sur P6 : aucune. Conséquence sur le bras L6 de v3 : nulle. Ce qui est falsifié est la prémisse géométrique "une couche médiane sépare mieux les faits" sous la seule métrique cosinus. »*

**Interdiction absolue, motif d'invalidation** : mesurer, calculer ou rapporter une quantité **indice↔fait**.

### 4.5 Couloir de faisabilité v4 (D19) — strate `P-own`, statistique `R1_36`

**Strate décisionnelle nommée avant mesure : `P-own`.** `P-ent` en borne inférieure conservatrice. `P-0` plafond. `B-v3` descriptif. **Aucun poolage.**

**Construction pré-déclarée du jeu de candidats** (déterministe, **indépendante des données** — toute sélection par proximité mesurée est un motif d'invalidation) : requête = un état d'indice de l'unité `i`, type `t` ; **cible unique** = même unité, type `t′` (règle cyclique fixe para1→para2→para3→para1) ; **36 concurrents, le plus dur d'abord** — (1) les **12** états des 4 autres unités de la composante d'**owner** de `i` ; (2) les **9** états des 3 autres unités de la composante d'**entité** ; (3) **15** états de `P-0` par décalage d'indice croissant. **Succès** = plus proche voisin en cosinus = la cible. **Hasard = 1/37 = 0.02703.** Jeu de **taille exactement 36** ⇒ **aucune extrapolation**, `A^s` retirée. Remplissage le plus dur d'abord ⇒ **conservateur**.

**Note historique, hors décision** : `A^s` donnait `A ≥ 0.9622` (R1 ≥ 0.25) et `A ≥ 0.9809` (≥ 0.50) ; conservés comme **repères de publication en AUC**, jamais comme critère.

**Partition exhaustive (D18), un cas de banc par classe, bords inclus.** **`T = 0.25`** (décision PI 2026-08-22) ; `T⁺ = 0.50` (palier descriptif **V⁺**) ; `P` = plancher le plus haut du §5 :

| Bande | Condition, dans cet ordre | Verdict pré-enregistré |
| --- | --- | --- |
| **N — nulle** | **`IC_inf(ΔR1) ≤ 0`** — c'est-à-dire **absence de `ΔR1` significativement positif**, l'IC contenant 0 **ou** entièrement négatif | **pas d'identité d'unité mesurable sous cosinus.** Les 36 leurres ne sont pas à construire en l'état. **C'est le résultat que la gate existe pour produire.** |
| **M — marginal** | `ΔR1` > 0 significatif **et** `IC_sup(R1_36) < 0.25` | **identité présente, séparation appariée insuffisante.** **v4 change de MÉCANISME, pas seulement de matériel. Réorientation AUTOMATIQUE** (décision PI). Résultat, pas échec. |
| **I — indécidable** | `ΔR1` > 0 significatif **et** `IC_inf < 0.25 ≤ IC_sup` | **INCONCLUSIF — précision insuffisante**, cause **`K_S`**. **La levée n'est PAS un re-run** (`K` plafonne à 16). **Décision PI 2026-08-22 : `I` déclenche la construction du matériel de v4** selon la spécification de découplage du §10, puis **re-mesure d'I2 dessus**. |
| **V — viable** | `IC_inf(R1_36) ≥ 0.25` | **v4 est autorisé.** Mention **V⁺** si `IC_inf ≥ 0.50`. |

**Conventions de bord, gravées** : `IC_sup = T → I` ; `IC_inf = T → V` ; `ΔR1` dont l'IC touche 0 par la borne inférieure → **N**.

**Correctif D18 du 2026-08-22, second passage du banc** : la rédaction précédente (« N : l'IC de `ΔR1` **contient** 0 ») laissait un **cas non couvert** — `IC(ΔR1)` entièrement **négatif** avec `IC_inf(R1) < T` n'appartenait à aucune bande. **Le protocole qui grave D18 le violait une seconde fois, dans la clause même réécrite pour réparer la première violation.** La bande N est donc définie par `IC_inf(ΔR1) ≤ 0` — la seule lecture rendant la partition **exhaustive et mutuellement exclusive** sans amender une autre clause. Le cas **significativement négatif** est publié comme **sous-étiquette descriptive** (`delta_R1_significativement_negatif`) : il signifierait que le corpus réel sépare **moins bien** que sa propre nulle, ce qui est un fait sur l'instrument et non sur le cortex.

**`D` — overlay inter-modèles, précédence gravée** : (1) **si l'un des deux modèles rend `I`, le verdict global est `I`** ; (2) sinon bandes différentes ⇒ **`D` — INCONCLUSIF**, cause : dépendance au modèle ; Qwen départage en **descriptif N = 3** ; (3) sinon verdict = la bande commune.

**Asymétrie décisionnelle sur `P-ent`, gravée AVANT mesure (N-13) :**

> *« Sur `P-ent`, l'unique discriminant est l'owner, et c'est le seul slot dont les trois types de paraphrase changent la forme de surface, à des distances très différentes du point de capture. Une bande M ou N sur `P-ent` confondrait "le leurre est dur" et "la règle a abîmé son propre discriminant" : elle n'engage AUCUNE décision. Seule une bande V sur `P-ent` serait interprétable — a fortiori concluante, puisqu'elle survivrait à un discriminant atténué. C'est une propriété du design, pas un résultat. »*

**Puissance déclarée avant mesure** : `K(P-own) = 16` = exactement le minimum requis. **La marge est nulle.** Une bande **I** est le travail de la porte, pas un accident.

### 4.6 Entropie — secondaire, retrait automatique

Convention **Giraldo** complète, normalisation des lignes **requise**, `n_a = 90`, **λ₁/Σλ publié**. **Clause de retrait (N-19)** : si `H` est non interprétable sur **les trois modèles** (`V-bord` déclenchée **ou** `V-λ₁` absente), **`H` est retirée de la fiche I2**, automatiquement, consigné au journal, sans nouvelle discussion.

### 4.7 Portes

| Porte | Clause | Dérivation / contre-exemples |
| --- | --- | --- |
| **`V-diversité`** | **PASS ssi** `(#owners, #entités, #verbes) = (min(N,16), min(N,20), min(N,5))`, recensement publié avant mesure | **Zéro GPU, zéro bootstrap.** `fact_pairs(80)` → (16,20,5) **PASS** ; **jeu v3 → (5,6,·) FAIL**. L'ancienne `V-div` mesurait un cardinal de strate et **récompensait la dégénérescence** (0-7). |
| **`V-puissance`** | **PASS ssi** `K_S ≥ (1.96·σ₀/\|θ−T\|)²` pour la strate décisionnelle | σ₀ = 0.5, \|θ−T\| = 0.25 ⇒ **K ≥ 16**. `P-own` N=80 : K = 16 **PASS à l'égalité** ; N=30 : K = 14 **FAIL** ; **décision AUC : K ≥ 429 ⇒ FAIL à tout N ≤ 80** — la porte **nomme l'indécidabilité**. **Fusion refusée** : un FAIL doit nommer sa cause. |
| **`V-paires`** | `n_intra = 240`, `n_inter = 28 440` **exactement** ; `P-0/P-own/P-ent` = **2880/160/120**, `P-both = 0` ; égalités publiées | 80×3 = 240 ; C(80,2)×9 = 28 440 ; 3160 = 2880+160+120 ✓. Égalités à ½ crédit, **déclaré avant mesure**. |
| **`V-suffixe`** *(re-dérivée, 0-8)* | partage du **dernier token BPE** : **para1 = 1.0000**, **para3 = 1.0000**, **para2 = `#{paires : 5 \| d}/C(N,2)`** = **600/3160 = 0.18987** à N = 80 (**75/435 = 0.1724** à N = 30, **valeur mesurée par le banc, qui valide la dérivation**) | Constantes **entièrement dérivées** : para1 et para3 finissent par une chaîne globale gelée ⇒ 1 par construction ; para2 finit par le **verbe** (période 5). L'ancienne attente datait d'avant §15 A-1. **La porte redevient mordante.** |
| **`V-plat`** | courbe centrée par unité ; **PLATE ssi `R_obs ≤ q_0.95(R*)`**, B = 10 000 | **Aucune constante posée.** Permutation des étiquettes **de couche** invalide. |
| **`V-bord`** | `ℓ*` en `ℓ = 1` ou `ℓ = L` | extremum au bord ⇒ la quantité **ne dit rien**. **Résultat le plus probable pour `ℓ*_H`.** |
| **`V-λ₁`** | λ₁/Σλ publié pour **toutes** les couches | absence ⇒ `H` **non interprétable** et déclenche le retrait §4.6. |
| **`V-bandes`** *(nouvelle, 0-9)* | la partition N/M/I/V est **exhaustive et mutuellement exclusive**, **bords inclus** ; la précédence de l'overlay `D` est testée | **Un cas par classe, plus les trois bords**, plus **un cas où un modèle rend `I` et l'autre `V`** ⇒ verdict global **I**. C'est la porte qui aurait empêché le protocole qui grave D18 de violer D18. |
| **`V-source`** *(nouvelle, 0-10)* | toute équation citée est relue dans le **PDF** de sa source primaire ; l'attribution nomme **l'article d'origine** | Giraldo et al. 2014 pour la normalisation des lignes ; Skean Eq. 1 **ne la porte pas**. **Lire du HTML pour citer une équation est un motif d'arrêt.** |
| **`V-amont`** | aucun chiffre de v3 dans une porte, un seuil ou une prédiction | D14-R. `B-v3` descriptif. |
| **`V-hooks`** | hooks sur les L blocs, retirés en `finally` ; `git diff --stat engram/` **vide** ; `M` jamais instanciée | — |
| **`V-1pass`** | **un** forward par (modèle, variante) | > 1 ⇒ boucle sur les couches. |
| **`V-L`** | `L` re-lu du config, comparé à {12, 32, 28} | écart ⇒ **arrêt**. |
| **`V-hash`** | SHA-256 de (a) et de `B-v3` avant/après | données gelées. |

### 4.8 Localisation — conditionnelle, faible, bornée

**Statut fixé par le PI** : la gate suffit ; **« gate passée, localisation non concluante » n'est pas un échec.**

**Borne conservatrice** : `(2w+1)/L` = **0.250 / 0.156 / 0.179**. **Aucun produit intra-modèle** ; produit inter-modèles licite.

| # | Prédiction | Seuil |
| --- | --- | --- |
| **P-B (jointe)** | `ℓ*_contrast` ∈ fenêtre D3 **sur GPT-2 ET SmolLM2** | **p ≤ 0.039**. **C1 sur un seul modèle ne peut PAS s'écrire « tenue ».** |
| **P-C (jointe)** | `ℓ*_H` ∈ fenêtre D3 sur les deux | idem |

| Cellule | Observation | Lecture pré-enregistrée |
| --- | --- | --- |
| **C1** | les deux dans la fenêtre | prédiction **jointe** requise. |
| **C2** | exactement une | **évidence FAIBLE** (`P(C2) ≈ 0.26-0.38` contre `P(C1) ≈ 0.02-0.06`). « Compatible avec », jamais « démontré ». Nommer laquelle. |
| **C3** | ni l'une ni l'autre | consigner *« qualité représentationnelle ≠ tolérance à l'injection »*. |
| **C4** | au moins une **PLATE** ou au **bord** | la quantité **ne dit rien** ; le modèle sort du test joint. **Pré-écrite comme la plus probable pour `ℓ*_H`.** |

### 4.9 Prédictions signées

| # | Prédiction | Antipode |
| --- | --- | --- |
| **N-P1** | signe P-A : mi-profondeur > état final | `AUC(L)` maximal |
| **N-P2** | ne signe ni P-B ni P-C ; attend `ℓ*_H ≤ ℓ*_contrast` — **C2 sur les trois** | `ℓ*_H > ℓ*_contrast` sur ≥ 2 modèles |
| **N-P3** | `ℓ*_contrast` ∈ **[6,9]** GPT-2, **[16,24]** SmolLM2, **[14,21]** Qwen | hors bande |
| **N-P4** | `ℓ*_H` **dans le premier tiers, voire au bord** — **résultat le plus probable, écrit d'avance** | `ℓ*_H` intérieur et médian |
| **N-P5** | **Spearman(`H`, `contrast`) ≤ 0** sur les trois | ρ > 0 |
| **N-P6** *(N-17)* | **`AUC(P-ent \| para3×para3) > AUC(P-ent \| para1×para1)`** | **égalité ou inversion ⇒ l'AUC de `P-ent` ne dépend pas de la lisibilité de l'owner, donc ne mesure pas ce leurre.** Seul contrôle séparant « leurre dur » de « discriminant atténué ». |

**Désaccord pré-enregistré, non arbitré** : Neuro attend **[6,9]** pour GPT-2, la fenêtre D3 est **[5,7]**. Si le résultat tombe en 8-9, **P-B est fausse et N-P3 est vraie**.

**P-D — consignation prospective Qwen** : `Qwen/Qwen2.5-1.5B` est un modèle **base**. Motif : **c'est le seul des trois dont la couche D3 (14/28) est POSÉE et non validée par balayage**, donc le seul jugeable **sans circularité**. Couches consignées **AVANT tout balayage E1 sur Qwen** ; le balayage {7, 14, 21} de Q-08 devient le **JUGE**. **Si le balayage tourne avant, P-D est morte.**

## 5. Contrôles — cinq maillons, cinq nulles bloquantes (D17)

Chaîne : **matériel → capture → encodage → géométrie/ordre → statistique**. Chaque nulle est un **plancher à produire** ; aucune ne nomme un suspect (D20). Le **plancher le plus haut** entre dans `ΔR1` et dans P-A.

| Maillon | Nulle bloquante | Coût |
| --- | --- | --- |
| **1. Matériel** | **`AUC_lex` / `R1_lex`** sur des **vecteurs indicateurs de tokens BPE**. *Combien le seul recouvrement lexical produit-il, sans cortex ?* | **0 forward** |
| **2. Capture** *(re-spécifiée, 0-8)* | **nulle de cadre** : **cadre du type conservé verbatim** (préfixe, ponctuation, **et le verbe de l'unité** — il n'est pas un slot d'identité), **slots de contenu (owner, entité) remplacés par un remplissage neutre gelé**, apparié en longueur de tokens et en position. L'ancienne clause « même suffixe » est **abandonnée** : le suffixe commun de para2 est **vide**, elle y était **vacuée**. | +1 forward |
| **3. Encodage** | **`ℓ = 0`** : atteignable **sans aucun calcul du cortex**. Gratuite, **exclue de l'argmax**. | 0 |
| **4. Géométrie / ordre** | **corpus (a) mélangé au niveau des tokens**, apparié en multiensemble, nombre d'items, longueur et position. **Clause pré-enregistrée** : si `AUC_mélangée < 0.5 − 0.02`, la nulle est **déclarée permissive** (collapse vers un attracteur), **exclue du plancher le plus haut**, et le fait est publié — elle ne peut pas devenir plus indulgente que 0.5. | +1 forward |
| **5. Statistique** | **0.5** (AUC) / **1/37 = 0.02703** (`R1_36`) + bootstrap par **composante de slot**, B = 10 000, **argmax re-sélectionné** + **permutation des étiquettes d'unité à couche fixée avec recalcul de `max_ℓ`** | CPU |

**Compléments** : `M = 0` jamais instanciée, aucun gradient (D8) · `engram/` non modifié · ajout à `eval/pool.py` **purement additif**, tables gelées intouchées · ordre des conditions et des modèles fixé · cosinus et Gram **fp32**, valeurs propres **fp64** · VRAM libérée entre modèles.

## 6. Critères d'abandon

**Ce qui tue H_I2 (`REJETE`, et c'est un résultat)** : bande **N** sur GPT-2 **et** SmolLM2. **Conséquence pré-écrite : l'étage de clé de v4 tel que conçu ne peut pas fonctionner, et aucun matériel ne répare cela — c'est le choix de mécanisme que cette gate existe pour trancher.**

**Ce qui invalide le run (`INCONCLUSIF`, cause nommée)** : `V-diversité`, `V-puissance`, `V-paires`, `V-bandes`, `V-source` échouées ; `V-suffixe` non conforme aux constantes dérivées ; `engram/` modifié ; hooks non retirés ; plus d'un forward par variante ; `L` ≠ config ; NaN / inf ; repli CPU silencieux ; **toute mesure indice↔fait** ; **tout IC construit comme « max des IC par couche »** ; **tout jeu de candidats construit à partir de proximités mesurées**.

**Ce qui n'est PAS un critère d'abandon** : bande **M** (réorientation automatique) ; bande **I** (déclenche la construction du matériel v4, décision PI) ; cellule **C2** ; `ℓ*_H` au bord (**prédit d'avance**) ; **retrait de `H`** (automatique) ; **une bande M ou N sur `P-ent`** (non décisionnelle par design) ; une localisation non concluante.

**Interdit (D14)** : amender une clause après lecture de la première courbe.

## 7. Variables fixées

`seed = 0` ; `M = 0` ; **aucun nouveau champ `EngramConfig`**.
Modèles : `gpt2` (12) ; `HuggingFaceTB/SmolLM2-360M` (32) ; `Qwen/Qwen2.5-1.5B` (28) — **`L` re-lu du config**.
Fenêtres : **[5,7] / [14,18] / [12,16]**.
Corpus (a) : **`pool.fact_pairs(80)`** + constantes gelées. Bras descriptif : jeu v3. Nulles : cadre, mélangée. Descriptif : `NEUTRAL_TEXT`.
Partition : `P-0 / P-own / P-ent` (`P-both = ∅`). **Strate décisionnelle : `P-own`.** Seuils : **T = 0.25**, `T⁺ = 0.50`, hasard 1/37.
Numérique : cosinus et Gram **fp32**, valeurs propres **fp64**. Bootstrap par **composante de slot**, B = 10 000, argmax re-sélectionné, **BCa** au-delà de 0.95. Égalités ½ crédit. `n_a = 90`, B = 200 pour `H`.

## 8. Variable manipulée

**Une seule : la couche `ℓ`.** Modèle, type de paraphrase, classe d'identité et verbe sont des **facteurs de réplication et de ventilation**.

## 9. Budget

**4 forwards par modèle** — (a), `B-v3`, nulle de cadre, nulle mélangée — hooks sur toutes les couches ⇒ **12 forwards**. Corpus (a) : **240 prompts courts**. **Estimation : < 15 min GPU**, analyse CPU quelques minutes (bootstrap dominant).

**VRAM : à rapporter, pas à confirmer.** Le journal du 2026-08-22 consigne qu'une estimation précédente était fausse (1.017 Go contre « < 0.8 »). Ordre attendu ≤ 4 Go sur 6 ; **le chiffre réel est un livrable**, en particulier sur Qwen. **Décision copilote** : Qwen est **maintenu dans le run** (descriptif, plus P-D prospective) ; la règle des 30 min couvre le débordement.

**Règle pré-enregistrée** : au-delà de **30 min** GPU, **anomalie à signaler**.

**Rapport de coût, à écrire au journal** : *« I2 ≈ 15 minutes de GPU et zéro contenu nouveau. Le matériel de v4 se chiffre à 17 entités et une table d'indexation : des heures, pas des jours. Ce que la gate tranche n'est donc pas un budget de rédaction mais un CHOIX DE MÉCANISME, plus le cycle complet construit autour ; et elle rend, dans les quatre bandes, deux spécifications dures pour le matériel de v4. »*

## 10. Livrables attendus

- **`eval/layer_profile.py`** — hooks sur toutes les couches, AUC + `R1_36` + `R1_full` + entropie + λ₁/Σλ, trois modèles, quatre variantes, partition d'identité, ventilation par couple de types et par verbe. **`engram/` non modifié.**
- **`eval/pool.py`** : ajout **purement additif** ; tables gelées **intouchées**.
- **Banc `eval/gate_bench.py`** : toutes les portes, **quatre bandes + trois bords + précédence `D`**, quatre cellules, **un cas par classe** (D14-S/D18). **Aucune exemption.**
- **Tests CPU** : (i) **`V-diversité` échoue sur le jeu v3 et passe sur `fact_pairs(80)`** ; (ii) `V-puissance` rend K = 14 (N=30 FAIL), 16 (N=80 PASS), **FAIL sous décision AUC à tout N ≤ 80** ; (iii) recensement **411/14/10/0** et **2880/160/120/0**, **`P-both = ∅` prouvé par `lcm(16,20) = 80`** ; (iv) `V-paires` détecte 239 ou 241 ; (v) **`V-suffixe` rend 1.0000/0.18987/1.0000 à N = 80 et 1.0000/0.1724/1.0000 à N = 30** ; (vi) `V-plat` ; (vii) `V-1pass` ; (viii) `V-hooks` ; (ix) **l'AUC est inchangée sous transformation monotone par couche alors que le ratio change** ; (x) `w(L)` = 1/2/2 ; (xi) l'entropie est inchangée sous mise à l'échelle des lignes **ssi** normalisation Giraldo ; (xii) le jeu `R1_36` a **exactement 37 éléments**, est **déterministe** et **indépendant des similarités mesurées** ; (xiii) **l'estimateur débiaisé diffère du max brut à effet nul**, « max des IC » rejeté ; (xiv) `V-bandes` classe quatre bandes, trois bords, et rend **I** quand un modèle est I et l'autre V.
- **Sorties** : `experiments/results/layer-profile/`.
- **Entrée de journal** : les **huit interdictions §2**, la clause « Conséquence sur P6 : aucune », la clause **`P-both` impossible**, l'**asymétrie `P-ent`**, la **bande** et la **cellule** nommées, les couches Qwen **consignées avant tout balayage**, les références **vérifiées dans le PDF**, le **rapport de coût honnête**, le sort de `H`.
- **Deux spécifications dures pour v4, rendues quelle que soit la bande** : (1) le pool **ne peut pas porter la paire minimale sur l'entité** (`lcm(16,20) = 80`) ⇒ v4 doit **casser la modularité** de l'indexation ; cible chiffrée : contraste sur l'**entité**, **≥ 3 bins gradués**, owner et verbe fixes, **17 entités à ajouter**, en familles sémantiques ; **le verbe est EXCLU du budget de leurres** (un leurre-verbe est une paraphrase : il gonfle `s` et le seuil sans durcir la tâche). (2) **Validité de construit et puissance sont en concurrence dans ce pool** ⇒ v4 doit **découpler le slot de contraste du modulo d'indexation**, faute de quoi la bande **I** se reproduira.
- **`docs/EXTENSIONS.md` §4** : une ligne (proposée par le labo, appliquée par le PI).

## 11. Questions pour lab-neuro

1. **Nulle de cadre avec verbe conservé** : elle porte 5 valeurs distinctes et peut produire une séparation par le seul verbe. Bon plancher, ou faut-il un second bras à verbe **unique global** (+1 forward) ?
2. **N-P6 et le point de capture** : existe-t-il un régime où l'inversion serait attendue **sans** que ta conclusion s'applique — par exemple si para3 sature l'état final sur `"It's "` ?
3. **`P-own` et la voie LEC** : le discriminant y est l'entité, mais à des positions syntaxiques différentes selon le type. Variation acceptable, ou ventiler `P-own` par position de l'entité ?

## 12. Questions pour lab-math

1. **`V-puissance`** : sous clustering par composante d'owner (16 blocs de 5), l'inflation de variance intra-cluster peut-elle rendre σ₀ = 0.5 **non conservateur** — et si oui, quel `K_eff = K/DEFF` pré-enregistrer ?
2. **`ΔR1` appariée** : l'appariement par requête entre corpus réel et nulle est-il licite quand les deux prompts diffèrent en contenu mais pas en cadre ?
3. **Estimateur débiaisé et bande I** : `θ̂ = 2·max_obs − mean_b(θ*_b)` peut sortir de [0,1] et déplacer un verdict de V vers I. Tronquer ? La troncature casse-t-elle la couverture du BCa ?
4. **Multiplicité entre décisions** : deux décisions (P-A ; couloir), chacune avec son argmax, sur deux modèles. Contrôle nécessaire, ou la hiérarchie suffit-elle ?

## 13. Décisions du PI

| # | Question | Décision |
| --- | --- | --- |
| **1** | Corpus : `fact_pairs` primaire, jeu v3 relégué | **VALIDÉE** (copilote). Devenue `B-v3`, bras descriptif, après le dé-poolage exigé par M-12. |
| **2** | Renoncement à la question « où » | **ACCEPTÉ PAR LE PI.** La gate suffit ; une localisation non concluante n'est pas un échec. |
| **3** | Budget | **ACCEPTÉ** (copilote). C'est le prix de D17 ; **v3 est mort d'un maillon sans nulle**. |
| **4** | Bande M | **RÉORIENTATION AUTOMATIQUE, décidée par le PI.** |
| **5** | Corrections documentaires | **APPLIQUÉES** le 2026-08-22. |
| **6** | **Frontière de la bande V : T = 0.25 ou 0.50 ?** | **T = 0.25, décidé par le PI (2026-08-22)**, avec palier descriptif **V⁺** à 0.50. C'est ~9× le hasard (1/37). Motif : c'est le seuil que rend le budget `s ≈ 36`, et la marge de puissance est **déjà nulle** (K = 16) — durcir rendrait la bande `I` plus probable. **C'est un jugement de valeur sur ce qu'un mécanisme d'adressage acceptable doit rendre, et le seul point du protocole que ni Math ni Neuro ne pouvaient trancher.** |
| **7** | **Bande I : que fait-on ?** | **`I` DÉCLENCHE LA CONSTRUCTION DU MATÉRIEL DE v4, décidé par le PI (2026-08-22)**, selon la spécification de découplage du §10, puis **re-mesure d'I2 dessus**. Motif : la levée n'est **pas** un re-run — `K` plafonne à 16 dans ce pool, aucune répétition n'y changerait rien. |
| **8** | Qwen dans le run malgré la VRAM | **MAINTENU** (copilote) : descriptif, plus P-D prospective ; la règle des 30 min couvre le débordement, et la VRAM réelle est un livrable. |

## Historique

- 2026-08-22 : cadrage PI ; six écarts levés ; références vérifiées ; **PROPOSE**
- 2026-08-22 : avis Math et Neuro — deux **RÉSERVÉ**
- 2026-08-22 : **re-cadrage PI post-v3** — I2 devient la gate de v4 ; primaire AUC ; score en ratio acquitté ; « P-A ⇒ P6 » retirée ; `NEUTRAL_TEXT` rétrogradé ; entropie Giraldo ; P-D réécrite
- 2026-08-22 : corpus tranché — `fact_pairs(30)` primaire, jeu v3 en strate S3 ; `V-div` dérivée
- 2026-08-22 : **décisions du PI** — gate suffisante, réorientation automatique en bande M
- 2026-08-22 : **banc D14-S — E = 3, plus une découverte hors E.** `V-div` **inversée** ; `V-suffixe` à attente périmée **vidant la nulle du maillon 2** ; **bandes non exhaustives** ; **attribution Giraldo/Skean fausse**
- 2026-08-22 : **Math (7 changements) et Neuro (8) — deux RÉSERVÉ.** Couloir **indécidable à N = 30** ; primaire **poolant 88 % du corpus relégué** ; **le verbe n'est pas un slot d'identité** ; **`P-both` arithmétiquement impossible** ; **biais de sélection du max (0.033) > couloir entier (0.0187)**
- 2026-08-22 : **CONSOLIDATION.** Partition `P-0/P-own/P-ent` ; **N = 80** ; **strate décisionnelle `P-own`** ; couloir sur **`R1_36`** contre 36 concurrents réels ; bandes **N/M/I/V** + overlay `D` ; `V-div` scindée ; `V-suffixe` re-dérivée ; nulle du maillon 2 re-spécifiée ; portes **`V-bandes`** et **`V-source`** créées ; biais du max corrigé ; **BCa** ; retrait automatique de `H` ; asymétrie `P-ent` gravée ; **neuf défauts acquittés (0-7 à 0-15)**
- 2026-08-22 : **banc rejoué après consolidation — `E = 0`** (35 clauses, 133 cas, couverture 100 %, 132 tests). Deux sous-spécifications **déclarées et non comblées** par le banc, corrigées ici au pré-enregistrement : (i) le chiffre **429** de `V-puissance` sous décision AUC **n'est pas re-dérivable** ⇒ retiré, remplacé par l'énoncé robuste « K requis ≥ 442 contre K ≤ 20, facteur ≥ 22 » ; (ii) **la partition des bandes avait encore un trou** (`ΔR1` significativement négatif) ⇒ bande N redéfinie par `IC_inf(ΔR1) ≤ 0`. **D18 violée une seconde fois par la clause réécrite pour réparer la première violation.**
- 2026-08-22 : **décisions du PI** — **T = 0.25** (V⁺ à 0.50) ; **la bande `I` déclenche la construction du matériel de v4** puis re-mesure. **PROPOSE.** Étape suivante : re-passage au banc D14-S (un cas par classe, bords et précédence inclus), puis gate de pré-enregistrement, puis GPU.
