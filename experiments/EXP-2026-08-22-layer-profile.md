# EXP — I2 : existe-t-il une identité d'unité invariante et adressable dans un cortex gelé ? (`layer_profile`)

**Statut : PROPOSE**

Origine : `docs/EXTENSIONS.md` §2 (I2) · cadrage PI du 2026-08-22 · **re-cadrage PI du 2026-08-22 (post-v3, post-avis Math et Neuro)**.
Rôle : **gate scientifique de v4**. Instrument, pas mécanisme : aucune injection, aucune écriture, `M = 0`, `engram/` non modifié, aucun gradient (D8).

---

## Arbitrage

### A. Le re-cadrage (PI) — il change la nature du document

| # | Élément | Statut | Raison |
| --- | --- | --- | --- |
| **R-1** | I2 cesse d'être « à quelle couche injecter » et devient « existe-t-il une identité d'unité invariante à la paraphrase et adressable, à n'importe quelle couche ? » | **intégré** (§1-§2) | v3 (journal 2026-08-22) a mesuré un circuit **sans étape d'adressage** (`knn_k = 8` ≥ \|store\| 8-11 ; cible dans les 8 voisins **90/90** ; plancher clé-bruit **12/30** (F) et **23/30** (L6) contre seuil 12/30 ; en L6 la clé d'une **autre** unité rend 30/30 avec **vecteur de succès identique**). La question « où » présuppose un « si » non établi. |
| **R-2** | Justification économique écrite dans le protocole | **intégré** (§2, §9) | v4 exige ~**36 leurres appariés par unité** — construction de matériel, pas réglage. I2 coûte des minutes. **Si l'identité n'existe nulle part, les 36 leurres seraient construits pour rien.** |
| **R-3** | La valeur du couloir v4 est **dérivée**, pas invoquée | **intégré** (§4.5, D19) | Sinon « gate de v4 » reste une intention. |

### B. Avis Math (8 points)

| # | Remarque | Statut | Traitement |
| --- | --- | --- | --- |
| **M-1** | **AUC** `A(ℓ) = P(cos_intra > cos_inter)` en score-titre ; z-score et bruts en ventilation | **intégrée** (§4.3) | Invariante sous **toute transformation monotone par couche** ⇒ l'anisotropie **ne peut pas déplacer l'argmax** ; nulle **connue** (0.5), bornée, comparable ; **différence, pas taux ⇒ D16 par construction**. |
| **M-1b** | **DÉFAUT acquitté** : le ratio n'annule qu'un mode commun **multiplicatif** ; l'anisotropie est **additive** (`cos ≈ c₀(ℓ) + δ`) ⇒ compression en `1/c₀(ℓ)`, `c₀` croissant en profondeur | **acquitté** (§0) | L'ancien instrument **pouvait fabriquer un argmax médian à partir du seul profil d'anisotropie**. La revendication « c'est l'argument de Math sur v3, transposé » était une **transposition fausse** : l'argument médiane ne se transpose exactement qu'à l'AUC. **Erreur du copilote, consignée.** |
| **M-2** | Entropie : convention **Giraldo** complète, **normalisation des lignes REQUISE** ; « α→1 ≡ RankMe » **inexact** | **intégrée** (§3, §4.6) | Sans normalisation, `H` est confondue avec le **profil de normes** — même classe de défaut que M-1b. RankMe = entropie des **σ ℓ1-normalisées** ; von Neumann = entropie des **σ²**. **Le Builder relit l'Eq. 1 dans le PDF** ; deviner est un **motif d'arrêt**. |
| **M-3** | `H ≤ log min(n,d)` ⇒ lever l'ambiguïté sur `n` ; sous-échantillonner ; **interdire les comparaisons de niveaux** | **intégrée** (§4.6) | `n_a = 90`. Tout corpus comparé sous-échantillonné à 90 (B = 200). Seuls les **argmin** se comparent, à `n` égal. |
| **M-4** | Supprimer le 5 % de `V-plat` ; **bootstrap par unité de la courbe centrée** ; permutation des étiquettes de couche **invalide** | **intégrée** (§4.7) | Couches **non échangeables** (variances différentes). `R = max_ℓ − min_ℓ`, PLATE ssi `R_obs ≤ q_0.95(R*)`, B = 10 000. |
| **M-5** | Borne conservatrice `(2w+1)/L` = **0.250 / 0.156 / 0.179** ; indépendance des deux argmax **non plausible** ⇒ pas de produit intra-modèle ; **prédiction jointe** GPT-2 ∧ SmolLM2, p ≤ **0.039** | **intégrée** (§4.8) | Le produit **inter-modèles** reste licite. Tout conditionné à **non-platitude d'abord**. |
| **M-6** | C2 ne peut pas être « le résultat le plus informatif » (`P(C2) ≈ 0.26-0.38` vs `P(C1) ≈ 0.02-0.06`) | **intégrée**, formulation reprise | §4.8. La phrase est **retirée du protocole et de `EXTENSIONS.md` §I2**. |
| **M-7** | `V-signe` **vacuée par satisfaction** ⇒ la retirer ; remplacer par **comptes de paires exacts** | **intégrée** (§4.7) | Sous AUC elle est en outre **sans objet** (l'AUC ne divise pas). Remplacée par `V-paires`, réellement échouable. |
| **M-8** | Fenêtre `w(L) = max(1, ⌊L/12⌋)` **conservée** (1/2/2) | **intégrée sans modification** | Cohérence d'unité avec `⌊L/2⌋`, sévérité comparable. |

### C. Avis Neuro (points structurants)

| # | Remarque | Statut | Traitement |
| --- | --- | --- | --- |
| **N-1** | Retirer « P-A falsifiée ⇒ P6 falsifiée » — l'erreur la plus grave | **intégrée**, rédaction reprise mot pour mot | §4.4. Rang vs moyenne, L2 vs cosinus, croisé vs intra : **indépendantes**. |
| **N-2** | `NEUTRAL_TEXT` ne contrôle rien ; substitution = **(a) vs (a)-mélangée au niveau des tokens**, appariée | **intégrée** (§5) | « Elles diffèrent » était **garanti d'avance**. `NEUTRAL_TEXT` **rétrogradé en descriptif** ; le bras mélangé devient la **nulle du maillon « ordre »** (D17). |
| **N-3** | Renommer `ℓ*_contrast` / `ℓ*_H` ; « couche-clé »/« couche-injection » confinées à la section décision ; **supprimer « reconnaissance »** | **intégrée** (§2, §4) | « Reconnaissance » nomme une fonction non mesurée. |
| **N-3b** | **« Vallée de compression » → « minimum de rang effectif »** | **intégrée** (§2 ii) | L'ancienne (ii) interdisait « goulot » **tout en autorisant** une expression de même charge. Incohérence corrigée. |
| **N-4** | Interdictions (iv) (v) (vi) | **intégrées** (§2) | Dont **λ₁/Σλ publié par couche** (porte `V-λ₁`) sans quoi `H` n'est pas interprétable. |
| **N-5** | Dissociation lire/écrire **strictement architecturale** | **intégrée** (§2, hors clause) | La bio dissocie par **compartiment** et par **phase**, jamais par profondeur ; le résiduel est un **bus unique partagé**. |
| **N-6** | Prédictions signées | **intégrées** (§4.9) | Le désaccord [6,9] (Neuro) vs [5,7] (fenêtre D3) est **pré-enregistré tel quel**, aucun ajusté sur l'autre. |
| **N-7** | Qwen « instruct + RLHF » **FAUX** ⇒ P-D perd sa justification | **intégrée** : P-D **réécrite, pas retirée** | §4.9. Nouveau motif : **Qwen est le seul des trois dont la couche D3 (14/28) est POSÉE et non validée par balayage** — le seul où une désignation d'I2 peut être jugée **sans circularité**. |
| **N-8** | **Recall@1 indice↔indice** (L2 **et** cosinus), zéro forward ; **interdiction absolue** d'indice↔fait | **intégrée**, interdiction gravée (§4.4, §6) | Bonus non anticipé : Recall@1 est **exactement** la quantité dont le couloir v4 a besoin — elle évite l'hypothèse d'indépendance de `A^s`. |

### D. Le corpus — défaut signalé par le PI, absent des deux avis, tranché ici

Le protocole précédent fixait **corpus (a) = jeu d'unités v3**. Or ce jeu est le produit d'une sélection sous C-1/C-2/C-3 : **450 triplets examinés pour 30 conformes**, diversité effondrée à **5 owners et 6 entités**, `cos` inter-unités max **+0.99989**. Sur un tel matériel, `cos_intra ≈ cos_inter` **par construction du matériel** : l'AUC serait tirée vers 0.5 sans qu'aucune propriété du modèle ne soit en cause. **On confondrait « pas d'identité adressable dans le cortex » avec « pas de diversité dans mon corpus »** — et sur la gate de v4, cette confusion coûterait l'abandon d'un chantier pour une raison fausse.

**Décision : corpus (a) = `pool.fact_pairs(30)` re-paraphrasé, PAS le jeu v3.** Dérivation :

1. **C-1/C-2/C-3 n'ont aucune prise sur I2.** Ce sont des conditions d'**identifiabilité pour une récupération** : elles garantissent qu'un slot soit **diagnostique**. I2 **ne récupère rien** — il compare des états d'indices entre eux. Les importer n'a **aucun effet positif** et un **effet négatif certain**.
2. **`fact_pairs(30)` maximise la diversité disponible, sans choix à la main** : `owner = i mod 16`, `entity = i mod 20`, `verb = i mod 5` ⇒ **16 owners, 20 entités, 5 verbes**, le maximum du pool à N = 30. Déjà gelé et intouché depuis X9.
3. **La tension identifiabilité vs similarité des leurres ne se tranche pas : elle se MESURE.** L'AUC est **stratifiée par recouvrement de surface inter-unités** :

   | Strate | Slots de contenu partagés entre deux unités | Rôle |
   | --- | --- | --- |
   | **S0** | aucun | **plafond** : le cas facile |
   | **S1** | exactement un | régime intermédiaire |
   | **S2** | deux — dont les 10 paires `(i, i+20)` qui **ne diffèrent que par l'owner** | **aperçu de leurre apparié** : le régime de v4 |
   | **S3** *(bras séparé)* | **le jeu d'unités v3**, `cos` +0.99989 | **strate quasi-dégénérée**, borne inférieure empirique |

`AUC(S0)` répond à « une identité existe-t-elle ? » ; `AUC(S2)` et `AUC(S3)` à « survit-elle à des concurrents appariés ? » — **la question exacte que pose le budget de 36 leurres de v4**. Le jeu v3 est **conservé comme strate S3, jamais comme corpus primaire** : ce n'est pas un compromis, c'est ce qui **transforme le confondant en mesure**. Cela répond au reproche que le journal adresse au copilote de v3 (*« un coût déclaré n'est pas maîtrisé tant que son signe n'a pas été évalué clause par clause »*) : ici le coût de diversité devient **une variable de l'analyse**.

### E. Écarté

`V-vierge` (**retirée** : son répertoire cible existe depuis le 2026-08-22, elle ferait arrêter le run ; la propriété qu'elle protégeait est morte avec le re-cadrage — remplacée par `V-amont`) · phrase **O-1** (obsolète : v3 a mesuré) · **score en ratio comme score-titre** (M-1b ; conservé en ventilation) · **`V-signe`** (M-7) · `V-suffixe` **rétrogradée** en intégrité du matériel · **`NEUTRAL_TEXT` comme contrôle** (N-2) · cellule C2 « résultat le plus informatif » (M-6) · **« Qwen instruct + RLHF »** (faux) · **corpus (a) = jeu v3** (§D) · test de permutation des étiquettes de couche (M-4).

---

## 0. Défauts de l'itération précédente, acquittés

| # | Défaut | Portée |
| --- | --- | --- |
| **0-1** | **Le score en ratio mesurait l'anisotropie autant que l'invariance.** `cos ≈ c₀(ℓ) + δ` est **additive** ; le ratio n'annule qu'un mode commun **multiplicatif** et comprime l'effet en `1/c₀(ℓ)`, `c₀` croissant en profondeur. | **Fatal à la primaire** : l'instrument pouvait rendre P-A « tenue » **sans aucune invariance**. |
| **0-2** | **`H` sans normalisation des lignes est confondue avec le profil de normes par couche.** | Fatal à la seconde quantité ; corrigé. |
| **0-3** | **« P-A falsifiée ⇒ P6 falsifiée » est illégitime** (rang vs moyenne, L2 vs cosinus, croisé vs intra). | Aurait tué P6 avec un instrument qui ne la teste pas. |
| **0-4** | **Le contrôle `NEUTRAL_TEXT` pré-enregistrait l'issue facile comme prédiction.** | Une prédiction signée impossible à falsifier n'est pas une prédiction. |
| **0-5** | **Le corpus hérité de v3 confondait « pas d'identité dans le cortex » avec « pas de diversité dans mon corpus ».** | Fatal à la gate : AUC ≈ 0.5 par construction du matériel. |
| **0-6** | **`V-plat` posait 5 %** sans dérivation ; **`V-signe` était vacuée par satisfaction** ; **`V-vierge` est devenue insatisfiable**. | Trois portes, trois modes déjà catalogués par le cycle méthode. |

## 1. Question

**Existe-t-il, dans un cortex gelé, une identité d'unité factuelle invariante à la paraphrase et séparable des autres unités — à n'importe quelle couche — et à quel niveau de recouvrement de surface entre unités cette séparabilité survit-elle ?**

## 2. Hypothèse

**H_I2** — il existe au moins une couche `ℓ` où les états d'indices d'une même unité sont **plus similaires entre eux** qu'entre unités, **au-delà de ce que le recouvrement lexical seul produit**, et cette séparation **survit à des unités appariées en surface**.

**H₀** — à toutes les couches, l'AUC est indiscernable de ses planchers (0.5 et les planchers de matériel et d'ordre du §5) ; il n'existe pas d'identité d'unité mesurable sous la métrique cosinus dans ce cortex.

**Pourquoi maintenant — la seule justification recevable.** v4 exige ~**36 leurres appariés par unité**, un travail de **construction de matériel** en tension frontale avec les conditions d'identifiabilité. **I2 coûte des minutes de GPU ; v4 coûte des jours.** Si aucune couche ne porte d'identité séparable, **aucun étage de clé ne marchera** et les 36 leurres seraient construits pour rien. I2 est la **gate scientifique de v4** ; la question « à quelle couche injecter » est **prématurée** tant que « une clé sert-elle à quelque chose ? » n'est pas tranchée.

**Motivation architecturale, hors clause, non porteuse** (Neuro) : *l'idée d'une couche « pour lire » distincte d'une couche « pour écrire » n'a pas d'ancrage biologique par la profondeur. La biologie dissocie par **compartiment** (EC couches II/III vs V/VI ; ségrégation dendritique SR/SLM en CA1) et par **phase** (thêta — c'est X3), jamais par position dans une profondeur de traitement. Dans engram le flux résiduel est un **bus unique partagé** : il n'y a pas d'équivalent de la ségrégation. Cette phrase motive une intuition d'ingénierie et n'entre dans aucune prédiction.*

**Nomenclature gravée.** Les deux mesures s'appellent **`ℓ*_contrast`** (argmax de l'AUC) et **`ℓ*_H`** (argmin de l'entropie). « Couche-clé » et « couche-injection » sont confinées à la seule section décision (§4.8), **entre guillemets et précédées de « candidate »**. Leur coïncidence est une **SORTIE**, jamais une hypothèse.

**Interdictions de vocabulaire, avant mesure :**

- **(i)** `ℓ*_contrast` s'écrit **« la couche où l'invariance à la paraphrase lexicale est la plus contrastée »**, jamais « la couche qui comprend le sens ». Le mot **« reconnaissance » est supprimé** de toute formulation de H_I2.
- **(ii)** `ℓ*_H` s'écrit **« minimum de rang effectif »**. **« Vallée de compression » est interdite** au même titre que « goulot d'information » et « couche optimale ».
- **(iii)** Une coïncidence `ℓ*_H ≈ ℓ*_contrast` **ne démontre pas** que lire et écrire se font au même endroit. La propriété non mesurée s'appelle **tolérance à l'injection**.
- **(iv)** L'entropie matricielle **ne mesure aucune quantité d'information sur le contenu** et **ne distingue pas une représentation abstraite d'une représentation dégénérée**. « Goulot », « compression au sens de Tishby », « information conservée » : **interdits**.
- **(v)** Une baisse de `H` est **compatible avec une dimension parasite dominante ou un puits d'attention**. Sans publication de **λ₁/Σλ par couche**, un minimum de `H` **n'est pas interprétable** — porte `V-λ₁`.
- **(vi)** I2 mesure une **géométrie non supervisée sur un cortex intact**. Aucun verbe **« porte / stocke / encode / contient / est responsable de »**. Non mesurées : la **décodabilité** (D8 interdit d'entraîner une sonde) et la **causalité**. L'écart en fraction de `L` sur trois modèles est **descriptif, N = 3**.

**Conformité** : D8 · D11 (sans objet) · D12 · D13 · D14 · **D14-S** (toutes les portes au banc, aucune exemption) · D14-R (§3) · **D16** (primaire = AUC, une différence par construction) · **D17** (§5 : cinq maillons, cinq nulles) · **D18** (§4.5 et §4.8 : partitions exhaustives) · **D19** (§4.5) · **D20** (les nulles sont des planchers à produire ; aucun suspect nommé).

## 3. Ce que le projet sait déjà — provenance (D14-R)

| Fait | Chiffre | Source | Statut |
| --- | --- | --- | --- |
| Diversité du jeu v3 | 450 triplets → 30 conformes ; **5 owners, 6 entités** | journal 2026-08-22 | **motive le choix de corpus** ; aucune porte |
| Similarité inter-unités v3 | `cos` max **+0.99989** / **+0.99848** | journal 2026-08-22 | orientation ; devient la **strate S3** |
| Absence d'adressage dans v3 | `knn_k = 8` ≥ \|store\| ; **90/90** ; plancher bruit **12/30** / **23/30** ; clé étrangère **30/30** en L6 | journal 2026-08-22 | **motive le re-cadrage** ; aucun chiffre en porte (`V-amont`) |
| Divergence des deux nulles | **3-4/30** vs **12-23/30** | journal 2026-08-22 | motive D17 appliquée à cinq maillons |
| Budget v4 | **s ≈ 36 leurres appariés** | journal 2026-08-22, D19 | **entre dans le couloir §4.5** — **dérivé**, pas mesuré |
| Couche D3 | GPT-2 **6/12**, SmolLM2 **16/32** **validées** ; Qwen **14/28 POSÉE** | D3, journal v1 / v1.2 | la distinction validée/posée porte P-D |
| Profondeurs | `L` = **12 / 32 / 28** | re-mesuré 2026-08-22 | **à re-lire du config** (`V-L`) |
| P6 | `h` 1.00 vs 0.33, `N_eff = 3` | run v2 **INVALIDÉ** | motive un design, **aucune porte ni prédiction** |
| X7 | `cos ≈ 0` ; ΔH +0.141 | journal 2026-08-21 | orientation |
| E1b | **~0.6-0.7, N = 10** | COR-02 | orientation : une invariance existe **fonctionnellement** ; I2 demande si elle est **géométriquement séparable**. Les deux ne s'impliquent pas. |

**Dérivations portées en entier :**

- **`⌊L/2⌋`** : 6 = 12/2, 16 = 32/2, 14 = 28/2.
- **Fenêtre** : `w(L) = max(1, ⌊L/12⌋)` ⇒ **[5,7] / [14,18] / [12,16]**. Conservée (M-8).
- **AUC** : `A(ℓ) = P(cos_intra > cos_inter) + ½·P(=)`, Mann-Whitney sur les paires intra contre les paires inter à la couche `ℓ`. **Invariante sous toute transformation strictement monotone appliquée par couche.** Nulle connue 0.5, bornée [0,1]. **Différence, pas taux ⇒ D16 par construction.**
- **Entropie matricielle**, convention **Giraldo et al. 2014** reprise par Skean et al. (arXiv:2412.09563) Eq. 1, α → 1 : Gram **côté échantillons**, `A_ij = K_ij / (n·√(K_ii·K_jj))`, `tr(A) = 1`, `H = −Σᵢ λᵢ log λᵢ`. **Normalisation des lignes REQUISE.** **Le Builder relit l'Eq. 1 dans le PDF** ; deviner est un **motif d'arrêt** (D14-R). **« α→1 ≡ RankMe » est inexact** : RankMe = entropie des **σ ℓ1-normalisées**, von Neumann = entropie des **σ²**.
- **Dépendance à `n`** : `H ≤ log min(n, d)`. `n_a = 90`. Tout corpus comparé **sous-échantillonné à 90** (B = 200, médiane ± IQR). **Comparaisons de niveaux interdites d'écriture** ; seuls les argmin, à `n` égal.

## 4. Prédictions chiffrées

### 4.1 Indexation des couches

`ℓ = 0` = sortie des embeddings (+ positions) ; `ℓ ∈ [1, L]` = sortie du bloc `ℓ`. **L'argmax décisionnel se cherche sur `ℓ ∈ [1, L]`** (L candidats), ce qui préserve les bornes de §4.8. **`ℓ = 0` est publié comme nulle « aucun calcul »** (maillon « encodage », §5) et **exclu de l'argmax** — clause fixée avant mesure.

### 4.2 Corpus

- **(a) PRIMAIRE — `pool.fact_pairs(30)` re-paraphrasé** par les règles gelées de `POOL_PARAPHRASES`. **16 owners, 20 entités, 5 verbes.** **Ni C-1, ni C-2, ni C-3 ne s'appliquent** (§D).
- **(a′) STRATE S3 — le jeu d'unités v3**, tel quel, hashé. Bras séparé, **jamais fusionné** avec (a).
- **Stratification de (a)** par slots de contenu partagés : **S0** (0), **S1** (1), **S2** (2 — dont les 10 paires `(i, i+20)`). Recensement publié **avant mesure**.
- **(b) descriptif, hors clause** : `NEUTRAL_TEXT` (343 positions).

**Capture** : état à la **position du dernier token de l'indice**, un état par prompt.

### 4.3 Quantités

| Quantité | Définition | Statut |
| --- | --- | --- |
| **`AUC(ℓ)`** | `P(cos_intra > cos_inter) + ½P(=)`, corpus (a) | **PRIMAIRE** |
| `AUC(ℓ \| S)` | par strate S0/S1/S2, et sur (a′) = S3 | **décisionnelle** pour le couloir (§4.5) |
| `Recall@1(ℓ)` | unité du plus proche voisin parmi les 89 autres états, en **cosinus** et **L2** | **décisionnelle** ; **indice↔indice UNIQUEMENT** |
| `H(ℓ)` | entropie matricielle, Giraldo complète, `n` = 90 | secondaire |
| `λ₁/Σλ (ℓ)` | poids de la première valeur propre | **obligatoire** (v) |
| `z(ℓ)`, `s_intra`, `s_inter`, ratio | — | ventilation descriptive |

**Ventilations obligatoires, hors clause** : AUC par **couple de types de paraphrase** ; `s_intra` / `s_inter` séparés ; 30 valeurs par unité à `ℓ*_contrast` ; comptes d'égalités exactes par couche.

**SE** : **bootstrap par unité, B = 10 000** (l'unité de rééchantillonnage est l'unité factuelle, jamais la paire). IC 95 % percentile.

### 4.4 Prédiction primaire — la gate

| # | Prédiction | Si vraie | Si fausse | Antipode (D13) |
| --- | --- | --- | --- | --- |
| **P-A** | **GPT-2** : `max_{ℓ∈[1,L]} AUC(ℓ)` **strictement supérieur à chacun des cinq planchers du §5**, IC 95 % **disjoint** du plus haut plancher ; **et** `AUC(ℓ*_contrast) > AUC(L)` | une identité d'unité **existe** et n'est réductible ni au matériel ni à l'ordre | sinon | **`AUC(L)` maximal** ⇒ l'état final est le meilleur porteur d'identité **sous cosinus**. **Conséquence pré-écrite** : *l'étage de clé de v4 se conçoit sur l'état final, pas sur une couche médiane.* |
| **P-A′** | **SmolLM2** : idem | réplication | sinon | idem |

**Conséquence sur P6 — clause gravée, à recopier telle quelle au journal :**

> *« Conséquence sur P6 : aucune. Conséquence sur le bras L6 de v3 : nulle. Ce qui est falsifié est la prémisse géométrique "une couche médiane sépare mieux les faits" sous la seule métrique cosinus. »*

P6 est un **rang top-k en L2 entre indice et fait** ; P-A une **moyenne de cosinus entre indices**. Les deux quantités sont **indépendantes**.

**Interdiction absolue, motif d'invalidation du run** : mesurer, calculer ou rapporter une quantité **indice↔fait**.

### 4.5 Couloir de faisabilité v4 (D19) — dérivé du budget de v4

Sous adressage par **plus proche voisin en cosinus** et **en supposant l'indépendance des comparaisons** (approximation explicitement étiquetée), `P(rang 1) ≈ A^s` :

| Objectif v4 | `s` | AUC minimale | Calcul |
| --- | --- | --- | --- |
| Recall@1 ≥ **0.50** contre 36 leurres | 36 | **A ≥ 0.9809** | `0.50^(1/36)` |
| Recall@1 ≥ **0.25** contre 36 leurres | 36 | **A ≥ 0.9622** | `0.25^(1/36)` |
| Recall@1 ≥ **0.50** contre 29 concurrents | 29 | **A ≥ 0.9764** | `0.50^(1/29)` |

**`Recall@1` mesuré directement court-circuite l'hypothèse d'indépendance et PRÉVAUT sur la dérivation `A^s` en cas de désaccord** — clause fixée ici.

**Partition exhaustive (D18), un cas de banc par classe :**

| Bande | Condition sur `max_ℓ AUC(ℓ \| S2 ∪ S3)` | Verdict pré-enregistré |
| --- | --- | --- |
| **V — viable** | IC 95 % inf. **≥ 0.9622** | **v4 est autorisé** : l'adressage par plus proche voisin peut fonctionner contre des concurrents appariés. |
| **M — marginal** | IC inf. > plancher le plus haut du §5, **et** IC sup. < 0.9622 | **une identité existe mais la séparation pairée est insuffisante pour un adressage à 36 leurres par plus proche voisin.** **v4 doit changer de MÉCANISME, pas seulement construire du matériel.** C'est un résultat, pas un échec. **Réorientation AUTOMATIQUE** (décision PI 2026-08-22, §13). |
| **N — nulle** | IC ∩ [planchers] ≠ ∅ à toutes les couches | **pas d'identité d'unité mesurable sous cosinus.** Les 36 leurres ne sont pas à construire en l'état. **C'est le résultat que la gate existe pour produire.** |
| **D — dissocié** | bande différente entre GPT-2 et SmolLM2 | **INCONCLUSIF sur la gate**, cause nommée : dépendance au modèle. Qwen départage en **descriptif N = 3**, sans valeur décisionnelle. |

### 4.6 Entropie matricielle — secondaire

Convention Giraldo complète, **normalisation des lignes requise**, `n` sous-échantillonné à 90, **λ₁/Σλ publié par couche** sous peine de non-interprétation (v). **Comparaisons de niveaux interdites** ; seuls les argmin, à `n` égal.

### 4.7 Portes

| Porte | Clause | Dérivation |
| --- | --- | --- |
| **`V-div`** *(cœur du re-cadrage)* | chaque strate **S0/S1/S2** **non vide dans ≥ 99.9 %** des **B = 10 000** rééchantillonnages d'unités ; recensement publié **avant mesure** | une strate vidée rend son AUC **non définie en arithmétique effective** ⇒ bootstrap non calculable (D14-S). **Décidable sur le seul recensement combinatoire, sans GPU.** Contre-exemples : le **jeu v3** doit **échouer**, `fact_pairs(30)` doit **passer**. |
| **`V-paires`** *(remplace `V-signe`)* | `n_intra = 90` et `n_inter = 3915` **exactement** ; égalités `cos_intra = cos_inter` publiées par couche | 30 × C(3,2) = **90** ; C(30,2) × 9 = 435 × 9 = **3915**. Porte **mordante**. Égalités à ½ crédit — **déclaré avant mesure**. |
| **`V-plat`** *(re-dérivée)* | courbe centrée par unité ; **PLATE ssi `R_obs ≤ q_0.95(R*)`**, B = 10 000 | **aucune constante posée**. Permutation des étiquettes de couche **invalide**. Une courbe PLATE ⇒ argmax non interprété, **le modèle sort du test joint**. |
| **`V-bord`** | `ℓ*` en `ℓ = 1` ou `ℓ = L` | extremum au bord = courbe monotone ⇒ la quantité **ne dit rien**. **Écrit d'avance : résultat le plus probable pour `ℓ*_H`.** |
| **`V-λ₁`** | λ₁/Σλ publié pour **toutes** les couches | (v) : absence ⇒ **`H` déclarée non interprétable**, pas « interprétée avec prudence ». |
| **`V-amont`** *(remplace `V-vierge`)* | aucun chiffre de v3 dans une porte, un seuil ou une prédiction | D14-R. `V-vierge` **retirée** : son répertoire cible existe, elle ferait arrêter le run. |
| **`V-hooks`** | hooks sur les L blocs, retirés en `finally` ; `git diff --stat engram/` **vide** ; `M` jamais instanciée | un hook survivant contaminerait tout run ultérieur. |
| **`V-1pass`** | **un** forward par (modèle, variante) | > 1 ⇒ l'implémentation boucle sur les couches. |
| **`V-L`** | `L` re-lu du config, comparé à {12, 32, 28} | écart ⇒ **arrêt**. |
| **`V-suffixe`** *(rétrogradée en intégrité)* | partage du dernier token BPE par type : **≈ 1.0 para3**, ≈ 0 para1/para2 | vérifie que **les règles gelées qui ont tourné sont celles décrites**. Le confondant est **mesuré** par la nulle suffixe (§5). |
| **`V-hash`** | SHA-256 de (a) et (a′) avant/après | données gelées. |

### 4.8 Localisation — conditionnelle, faible, bornée en multiplicité

**Statut fixé par le PI (2026-08-22, §13)** : la gate suffit. **I2 peut se conclure par « gate passée, localisation non concluante » sans que ce soit un échec.**

Toutes conditionnées à **non-platitude** et à une gate non-N. **Borne conservatrice** : `P(argmax ∈ fenêtre) = (2w+1)/L` = **0.250 / 0.156 / 0.179**. **Aucun produit intra-modèle** (les deux argmax partagent les mêmes états) ; le produit **inter-modèles** reste licite.

| # | Prédiction | Seuil |
| --- | --- | --- |
| **P-B (jointe)** | `ℓ*_contrast` ∈ fenêtre D3 **sur GPT-2 ET SmolLM2** | **p ≤ 0.250 × 0.156 = 0.039**. **C1 sur un seul modèle ne peut PAS s'écrire « prédiction tenue ».** |
| **P-C (jointe)** | `ℓ*_H` ∈ fenêtre D3 **sur GPT-2 ET SmolLM2** | idem |

**Partition exhaustive par modèle (D18) :**

| Cellule | Observation | Lecture pré-enregistrée |
| --- | --- | --- |
| **C1** | les deux dans la fenêtre | prédiction **jointe** requise pour « tenue ». |
| **C2** | exactement une | **lecture la plus riche conditionnellement à des courbes non plates ; en tant qu'évidence, FAIBLE** (`P(C2) ≈ 0.26-0.38` sous la nulle contre `P(C1) ≈ 0.02-0.06`). S'écrit **« compatible avec »**, jamais **« démontré »**. **N'engage aucune décision de design sans réplication.** Nommer laquelle. |
| **C3** | ni l'une ni l'autre | consigner *« qualité représentationnelle ≠ tolérance à l'injection »*. |
| **C4** | au moins une courbe **PLATE** ou au **bord** | la quantité **ne dit rien** ; le modèle sort du test joint. **Pré-écrite comme la plus probable pour `ℓ*_H`.** |

### 4.9 Prédictions signées par Neuro — pré-enregistrées séparément

| # | Prédiction | Antipode |
| --- | --- | --- |
| **N-P1** | **signe P-A** : mi-profondeur > état final | `AUC(L)` maximal |
| **N-P2** | **ne signe ni P-B ni P-C** ; attend `ℓ*_H ≤ ℓ*_contrast` — **cellule C2 sur les trois** | `ℓ*_H > ℓ*_contrast` sur ≥ 2 modèles |
| **N-P3** | `ℓ*_contrast` ∈ **[6,9]** GPT-2, **[16,24]** SmolLM2, **[14,21]** Qwen | hors bande |
| **N-P4** | `ℓ*_H` **dans le premier tiers, voire au bord** ⇒ `V-bord` se déclenche et la quantité ne dit rien — **résultat le plus probable, écrit d'avance** | `ℓ*_H` intérieur et médian |
| **N-P5** | **Spearman(`H`, `contrast`) ≤ 0** sur les trois | ρ > 0 |

**Désaccord pré-enregistré, non arbitré** : Neuro attend **[6,9]** pour GPT-2, la fenêtre D3 est **[5,7]**. Les deux sont consignés **tels quels**. Si le résultat tombe en 8-9, **P-B est fausse et N-P3 est vraie** — information en soi sur la valeur prédictive de D3.

**P-D — consignation prospective Qwen (motif réécrit)** : `Qwen/Qwen2.5-1.5B` est le modèle **base** — la justification « instruct + RLHF » du cadrage était **fausse**. Motif de substitution : **Qwen est le seul des trois dont la couche D3 (14/28) est POSÉE par `⌊L/2⌋` et non validée par balayage**, donc le seul où une désignation d'I2 peut être jugée **sans circularité**. Couches consignées au journal **AVANT tout balayage E1 sur Qwen** ; le balayage {7, 14, 21} de Q-08 devient le **JUGE**. **Si le balayage tourne avant la consignation, P-D est morte.**

## 5. Contrôles — chaîne causale et ses cinq nulles bloquantes (D17)

Chaîne d'I2 : **matériel → capture → encodage → géométrie/ordre → statistique**. Chaque nulle est un **plancher à produire** ; aucune ne nomme un suspect (D20).

| Maillon | Nulle bloquante | Coût |
| --- | --- | --- |
| **1. Matériel** | **`AUC_lex`** : la même AUC sur des vecteurs **indicateurs de tokens BPE** des chaînes d'indices. *Combien d'AUC le seul recouvrement lexical produit-il, sans cortex ?* | **0 forward** |
| **2. Capture** | **nulle suffixe** : mêmes prompts, contenu d'unité remplacé par un remplissage neutre gelé, **même suffixe, même longueur, même position**. Toute AUC > 0.5 ici est un effet de **position et de suffixe**. Absorbe le confondant `"It's "` de para3 **en le mesurant au lieu de l'équilibrer**. | +1 forward |
| **3. Encodage** | **`ℓ = 0`** : l'AUC atteignable **sans aucun calcul du cortex**. Gratuite, **exclue de l'argmax**. | 0 |
| **4. Géométrie / ordre** | **corpus (a) mélangé au niveau des tokens**, **apparié en multiensemble, nombre d'items, longueur et position**. *L'identité survit-elle à la destruction de l'ordre ?* Une AUC identique ⇒ le cortex se comporte comme un **sac de tokens contextualisé**. **Remplace `NEUTRAL_TEXT`.** | +1 forward |
| **5. Statistique** | **0.5** + bootstrap par unité B = 10 000 + **permutation des étiquettes d'unité entre paires, à couche fixée** (valide : les paires sont échangeables sous H₀ **à l'intérieur** d'une couche) | CPU |

**Compléments** : `M = 0` jamais instanciée, aucun gradient (D8) · `engram/` non modifié (`V-hooks`) · ordre des conditions et des modèles fixé · corpus gelés et hashés · cosinus et Gram **fp32**, valeurs propres **fp64** · VRAM libérée entre modèles · `NEUTRAL_TEXT` descriptif hors clause.

## 6. Critères d'abandon

**Ce qui tue H_I2 (`REJETE`, et c'est un résultat)** : bande **N** sur GPT-2 **et** SmolLM2. **Conséquence pré-écrite : l'étage de clé de v4 tel que conçu ne peut pas fonctionner ; les ~36 leurres ne sont pas à construire en l'état — c'est précisément l'économie que cette gate existe pour réaliser.**

**Ce qui invalide le run (`INCONCLUSIF`, cause nommée)** : `V-div` échouée ; `V-paires` échouée ; `engram/` modifié ; hooks non retirés ; plus d'un forward par variante ; `L` ≠ config ; `V-suffixe` non conforme ; NaN / inf ; repli CPU silencieux ; **convention de normalisation de l'entropie non re-dérivée depuis le PDF** ; **toute mesure indice↔fait effectuée**. (`V-λ₁` absente ⇒ `H` non interprétable, mais le run reste valide pour la primaire.)

**Ce qui n'est PAS un critère d'abandon** : la bande **M** (résultat qui **redirige** v4, réorientation automatique) ; la cellule **C2** ; `ℓ*_H` au bord (**prédit d'avance**) ; un désaccord entre modèles sur la localisation ; **une localisation non concluante** (décision PI, §13).

**Interdit (D14)** : amender une clause après lecture de la première courbe.

## 7. Variables fixées

`seed = 0` ; `M = 0` ; aucun champ d'`EngramConfig` autre que `model` et `device` — **aucun nouveau champ**.
Modèles : `gpt2` (L = 12) ; `HuggingFaceTB/SmolLM2-360M` (L = 32) ; `Qwen/Qwen2.5-1.5B` (L = 28) — **`L` re-lu du config**.
Fenêtres : `w(L) = max(1, ⌊L/12⌋)` autour de `⌊L/2⌋` ⇒ **[5,7] / [14,18] / [12,16]**.
Corpus (a) : `pool.fact_pairs(30)` + `POOL_PARAPHRASES`. (a′) = S3 : jeu v3. Nulles : suffixe, mélangée. Descriptif : `NEUTRAL_TEXT`.
Numérique : cosinus et Gram **fp32**, valeurs propres **fp64**. Bootstrap **par unité**, B = 10 000. Égalités : ½ crédit.

## 8. Variable manipulée

**Une seule : la couche `ℓ`.** Modèle, corpus et strate sont des **facteurs de réplication et de ventilation** ; aucune comparaison inter-modèles n'est décisionnelle, sauf la **conjonction** GPT-2 ∧ SmolLM2 de §4.8, qui est une exigence de **réplication**.

## 9. Budget

**XS, et c'est l'argument entier de la gate.** Variantes par modèle : (a), (a′), nulle suffixe, nulle mélangée = **4 forwards** de 90 prompts courts, hooks sur toutes les couches simultanément ⇒ **12 forwards au total**. **< 8 min GPU** pour les trois modèles. **VRAM annoncée : ≤ 4 Go sur 6 — estimation, et le journal du 2026-08-22 consigne que la précédente était fausse (1.017 Go contre « < 0.8 Go ») : le chiffre réel est à rapporter, pas à confirmer.** Analyse hors-ligne CPU gratuite.

**Règle pré-enregistrée** : au-delà de **25 min** GPU, **anomalie à signaler** — probablement un forward par couche (`V-1pass`).

**Rapport de coût, à écrire au journal** : *I2 ≈ 8 minutes de GPU ; v4 ≈ des jours de construction de matériel (36 leurres appariés par unité). La gate est rentable même si elle ne fait qu'écarter une bande sur quatre.*

## 10. Livrables attendus

- **`eval/layer_profile.py`** (nouveau, SPDX `AGPL-3.0-or-later`) — hooks sur toutes les couches, AUC + Recall@1 + entropie + λ₁/Σλ, trois modèles, quatre variantes, stratification. **`engram/` non modifié** ; `engram.cortex._find_blocks` en **lecture seule**. **Aucun nouveau champ `EngramConfig`.**
- **Banc `eval/gate_bench.py`** : **toutes** les portes, les **quatre bandes** (§4.5) et les **quatre cellules** (§4.8) exhibées **passantes ET échouantes**, **un cas par classe de la partition** (D18). **Aucune exemption.**
- **Tests CPU** : (i) `V-div` **échoue** sur le jeu v3 et **passe** sur `fact_pairs(30)` ; (ii) `V-paires` détecte 89 ou 91 paires intra ; (iii) `V-plat` sur courbes synthétiques ; (iv) `V-1pass` détecte une boucle sur les couches ; (v) `V-hooks` détecte un hook non retiré ; **(vi) l'AUC est inchangée sous une transformation monotone par couche (`x → x³`, `x → σ(ax+b)`) alors que le ratio `s_intra/s_inter` change** — le test qui matérialise M-1b ; (vii) `w(L)` rend 1/2/2 ; (viii) la stratification classe S0/S1/S2 ; (ix) l'entropie est inchangée sous mise à l'échelle des lignes **ssi** la normalisation Giraldo est appliquée ; (x) les quatre bandes et les quatre cellules sont classées correctement, **y compris aux bords**.
- **Sorties** : `experiments/results/layer-profile/` — CSV par (modèle, variante), tracés, `ℓ*_contrast` / `ℓ*_H` avec IC, ventilations, recensement de strates, comptes de paires et d'égalités, λ₁/Σλ, hashes, VRAM et durée réelles.
- **Entrée de journal** : les **six interdictions §2**, la clause « Conséquence sur P6 : aucune » **recopiée mot pour mot**, la **bande** (V/M/N/D) et la **cellule** (C1-C4) **nommées**, les couches Qwen **consignées avant tout balayage**, les références vérifiées (**2412.09563**, **2406.19384**), et le **rapport de coût I2/v4**.
- **`docs/EXTENSIONS.md` §4** : une ligne au tableau des instruments (proposée par le labo, appliquée par le PI).

## 11. Questions pour lab-neuro

1. **N-P4 vs la gate.** Si `H` sort **entièrement** non interprétable sur les trois modèles (`V-bord` + `V-λ₁`), la retires-tu de la fiche I2 pour de bon, ou la gardes-tu conditionnée à un modèle sans puits d'attention documenté ?
2. **Strate S2 et « identité ».** Les 10 paires `(i, i+20)` ne diffèrent **que par l'owner**. Est-ce la bonne opérationnalisation d'un « leurre apparié », ou un contraste d'owner est-il trop saillant pour compter comme apparié — auquel cas quel slot devrait porter le contraste dans le matériel de v4 ?
3. **Nulle mélangée.** Le mélange produit des séquences hors-distribution. Vois-tu un régime où cela **gonfle** l'AUC de la nulle (états collapsés vers un attracteur commun ⇒ `cos_inter` élevé ⇒ AUC basse, nulle trop permissive) plutôt qu'elle ne la borne ?

## 12. Questions pour lab-math

1. **Couloir `A^s`.** L'approximation d'indépendance est-elle conservatrice ou anti-conservatrice sous corrélation positive entre les similarités des concurrents ? Si anti-conservative, quel seuil substituer à **0.9622** ?
2. **N nécessaire.** Avec 30 unités et un bootstrap par unité, quelle est la **largeur attendue** de l'IC 95 % de l'AUC près de 0.98 ? Si elle dépasse la marge du couloir, N = 30 ne peut pas trancher la bande **V** — quel N minimal, sachant que le pool autorise N ≤ 80 mais qu'au-delà de 30 la diversité se réplique par périodicité ?
3. **Multiplicité de la gate.** La primaire prend `max_ℓ AUC(ℓ)` : **maximum biaisé vers le haut** sous la nulle. Quelle correction (max-t bootstrap, ou IC sur le max plutôt que max des IC) pour que la bande **N** ne soit pas fuie par sélection ?
4. **Stratification et bootstrap.** Le rééchantillonnage par unité modifie les effectifs des trois strates. Faut-il stratifier le bootstrap, et la comparaison inter-strates reste-t-elle valide ?

## 13. Décisions du PI (2026-08-22)

| # | Question | Décision |
| --- | --- | --- |
| **1** | **Corpus** : (a) = `fact_pairs(30)` re-paraphrasé, jeu v3 relégué en strate S3 | **VALIDÉE** (copilote, sous délégation). La décision du Directeur est meilleure que la proposition initiale du copilote : elle ne se contente pas d'écarter le corpus dégénéré, elle en fait une **strate mesurée**, ce qui transforme le confondant en donnée utile pour v4. **Conséquence** : `EXP-2026-08-22-knn-borne-logits-v3.md` §16 G (« I2 hérite : son corpus (a) **est** le jeu d'unités v3 ») devient **caduc** — erratum, v3 étant terminé. |
| **2** | **Renoncement à la question « où »** : I2 peut se conclure par « gate passée, localisation non concluante » | **ACCEPTÉ PAR LE PI.** La gate suffit à justifier l'instrument ; la localisation est un bonus **faible en évidence** (M-5, M-6). La question d'origine d'I2 **reste ouverte après I2**, et ce n'est pas un échec. Un cycle descriptif dédié pourra la reprendre. |
| **3** | **Budget** : ~2 min annoncées → **12 forwards, < 8 min** | **ACCEPTÉ** (copilote). C'est le prix de D17, et **v3 est mort d'un maillon sans nulle**. Aucune nulle n'est supprimée. |
| **4** | **Bande M** : réorientation de v4 automatique ou soumise au PI ? | **RÉORIENTATION AUTOMATIQUE, décidée par le PI.** La conséquence étant **pré-écrite avant mesure**, elle s'applique sans nouvelle discussion — c'est l'esprit du pré-enregistrement. En bande M, v4 **change de mécanisme** (score autre que cosinus-plus-proche-voisin, ou primaire autre qu'un rang), pas seulement de matériel. |
| **5** | **Corrections documentaires** (ratio, RankMe, « résultat le plus informatif », héritage de corpus) | **APPLIQUÉES** par le copilote le 2026-08-22. |

## Historique

- 2026-08-22 : cadrage PI ; six écarts levés ; références vérifiées (2412.09563, **2406.19384** — le cadrage annonçait 2404.xxxx, faux) ; `L` re-mesuré ; **PROPOSE**
- 2026-08-22 : avis **Math** (8 points) et **Neuro** (8 points structurants) — deux **RÉSERVÉ**
- 2026-08-22 : **RE-CADRAGE PI, post-v3.** I2 devient la **gate scientifique de v4**. Primaire **AUC par couche** ; le score en ratio **acquitté comme défaut** (il pouvait fabriquer l'argmax médian à partir du seul profil d'anisotropie) ; inférence « P-A ⇒ P6 » **retirée** ; `NEUTRAL_TEXT` **rétrogradé** au profit d'une nulle mélangée appariée ; `V-signe` et `V-vierge` **retirées**, `V-plat` **re-dérivée** ; entropie en convention Giraldo avec normalisation des lignes ; P-D **réécrite** (Qwen est un modèle **base**) ; couloir de faisabilité v4 **dérivé** de `s ≈ 36`
- 2026-08-22 : **corpus tranché par le Directeur** — (a) = `fact_pairs(30)` (diversité maximale), le jeu v3 devient la **strate S3** ; porte `V-div` dérivée ; **la tension identifiabilité vs similarité des leurres devient une courbe mesurée** (AUC par strate), livrable direct pour le matériel de v4
- 2026-08-22 : **décisions du PI** (§13) — gate suffisante, réorientation automatique en bande M, corpus validé, budget accepté, corrections appliquées
- 2026-08-22 : **PROPOSE**. Étape suivante : banc D14-S (toutes portes + quatre bandes + quatre cellules, **un cas par classe**), puis gate de pré-enregistrement, puis GPU.
