# EXP — v4-matériel : construire un matériau qualifié pour un instrument représentationnel

Statut : PROPOSE

*Protocole consolidé (2e tour) par `lab-director` le 2026-08-23, sous les avis Math (RÉSERVÉ, avis
principal + complément) et Neuro (RÉSERVÉ, avis principal + complément), les positions PI A-1..A-7,
la gate PI du 2026-08-23 (§14), et le théorème de saturation lexicale. **Amendé à la gate 2 du
2026-08-23 (§14-5).***

**Ce tour retire trois dispositifs** : les deux gravés à la gate 1 (§14-2 : seconde primaire
inter-pools et porte `V-lex`), démolis indépendamment par les deux experts, **et la bande `M`**
(décision PI, gate 2). Les retraits sont tracés au §14-2-CADUC, au §14-5 et au Registre des
engagements ; **rien ne disparaît silencieusement** (position A-7).

**Non pré-enregistrable en l'état.** Deux verrous restent nommés : §11-Q1/Q2/Q5/Q6 (re-signature de
Neuro sur `V-leak`, M3, bandes de NLL, forme agrégée de `P-N2`) et §12-Q1/Q2 (confirmation de `τ` et
de `ε` par Math).

---

## 0. Défauts acquittés

**0-1 … 0-45 : reconduits sans changement**, numérotation inchangée, aucun ré-ouvert. Ce tour en
ajoute **vingt-deux (0-46 … 0-67)**. Le §0 est le patrimoine du projet : chaque ligne dit **pourquoi
le défaut était fatal**, pas seulement ce qu'il était.

### 0-46 … 0-49 — Le §14-2 (gate 1) était inexécutable

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-46** | **Plafond de saturation lexicale.** Théorème (Math, vérifié indépendamment par le copilote) : la requête est l'état d'une **autre variante de la MÊME unité**, elle contient donc l'entité cible **verbatim** dans son préfixe causal à la capture (`C3`). Un lecteur lexical à 0 forward voit σ(cible)=2 > σ(co-tige)=1 > σ(disjoints)=0 : **maximum unique, déterministe, sur chaque requête**. Sous clé nulle appariée, σ est constant ⇒ égalités ⇒ mid-rank ⇒ 1/37. Donc `ΔR1_lex = 1 − 1/37 = 36/37` **est le plafond de la mesure**, pas son plancher comparable. | **Fatal à la porte `V-lex`** : elle exigeait `IC_inf(ΔR1_inv) > 36/37 ≈ 0.973` — **inexécutable** (aucune quantité bornée par 1 ne domine son propre plafond avec un IC) **et ininterprétable** (si elle était atteinte, elle prouverait un **transcript lexical**, c'est-à-dire exactement ce qu'elle prétendait exclure). Une porte qui prouve son contraire quand elle passe n'est pas une porte. |
| **0-47** | **Bande `M` vide par arithmétique.** `M` était conditionnée à `V-lex` PASS ⇒ sous 0-46, `M` est **inatteignable quelle que soit la donnée**. | **Fatal à la partition (D18)** : une classe non atteignable rend la partition **non exhaustive de fait**. C'est le **mode de vacuité 0-6/0-8**, que le PI a explicitement refusé pour `C6` — il aurait été réintroduit **par la gate elle-même**, dans la classe qui porte la seule issue positive. |
| **0-48** | **Un plancher dérivé confronté à un IC.** `V-lex` demandait que « l'IC de la primaire domine le plancher ». Or le plancher lexical est une **constante dérivée** (36/37), **sans variance d'échantillonnage**. | **Mode 0-38 inversé** : 0-38 traitait une quantité bruitée comme un seuil ; ici on traitait un **seuil exact** comme une quantité à IC. Dans les deux cas la frontière ne décide rien de ce qu'elle prétend décider. |
| **0-49** | **Le contraste inter-pools est NON MONOTONE** en la force du canal lexical (Math, complément) : sous transcript lexical parfait, les tige-partagés (1/2) ne battent jamais la cible (2/2) ⇒ `E[R1(P1)] − E[R1(P2)] = 0` ; sous encodage nul, échangeabilité ⇒ **0 aussi**. La différence ne vit qu'aux régimes intermédiaires, et sa borne `≤ P(∃ un des 5 tige-partagés bat la cible)` exigerait `p₁ ≳ 0.02` par concurrent, **inconnaissable avant run**. | **Fatal à la primaire 2 telle qu'ordonnée par la gate 1** : une primaire dont **la nulle et l'alternative extrême rendent le même chiffre** est ininterprétable comme dose. Elle aurait produit un 0 lisible comme « pas d'effet » alors qu'il est **compatible avec l'effet maximal**. |

### 0-50 … 0-52 — Inférence : unité d'échange, précision, résolution

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-50** | **Clustering par FAMILLE illégitime dès qu'une strate contient les ponts** : les partenaires d'une tige sont à la fois **concurrents et requêtes**. Le graphe de dépendance des 20 familles pontées est un **couplage parfait** ⇒ les clusters sont les **TIGES**, `K_eff = 10`, pas 20. | **Fatal à tous les IC du protocole** : ils étaient sous-estimés d'un facteur **1.41** — sur la primaire, sur les strates, sur les bandes. Une bande `N-a` déclarée à `K = 20` aurait été un artefact d'unité d'échange. Corollaire : le **bootstrap à deux voies** est inutile ; un seul niveau (tiges) suffit et couvre les deux extrémités de toute paire S2/S3. |
| **0-51** | **`V-dtype` posée sur une fraction de paires** au lieu de la **marge décisionnelle**. Deux marges distinctes existent : la **marge de tête** (rang 1 vs rang 2) et la **marge à la coupure `T`** (appartenance aux « gagnants », condition **E4** de la nulle hypergéométrique). | **Fatal à la portée de la porte** : une fraction globale de paires proches ne dit rien du basculement de la **seule** comparaison qui décide. Math prédit en outre que la zone rouge bf16 (ULP ≈ 3.9e-3 près de cos ≈ 1) **a une probabilité substantielle de déclencher** ⇒ le repli fp32 doit être écrit comme **chemin nominal**, pas comme exception. |
| **0-52** | **`τ = 0.10` égal à la demi-largeur pire cas.** Math dérive à `K = 20` : demi-largeur ≈ 0.10 sous effet ≈ 0 (ρ = 1), ≈ 0.23 sous ≈ 0.5 ; à `K_eff = 10` : ×1.41 ⇒ **≈ 0.14 / ≈ 0.33**. | **Réédition de 0-13** : une bande qui **découvre son indécidabilité après mesure**. À `K_eff = 10`, `τ = 0.10` **et** `τ = 0.12` sont tous deux inatteignables. (Les deux chiffres de Math sont mutuellement incohérents une fois `K_eff = 10` acquis — arbitré au §4.4, remonté au §12-Q1.) |

### 0-53 … 0-56 — Le facteur DOMAINE et la chaîne ordinale

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-53** | **Le facteur domaine n'a aucun porteur physique** (Neuro, argument arithmétique). L'état capturé est fonction du **préfixe** (byte-identique par `C1`, donc muet), de la **tige** et du **suffixe** — il n'y a pas d'autre canal. Or **20 des 20 familles décisionnelles ont une tige pontée**, donc ambiguë entre deux domaines ; et **rien dans §D.3 n'exigeait qu'un suffixe appartienne au domaine de sa famille**. | **Fatal au plan annoncé** : `cos̄(S3) − cos̄(S2)` aurait été **le bruit lexical de deux tirages dans le même vivier**. Le plan « 2 × 2 exact » était exact sur **une** de ses deux dimensions et **non instancié** sur l'autre. Sans correctif, le plan réel est un **1 × 2** et le domaine est **descriptif avant mesure**. C'est le cousin **aléatoire** de D24-b : D24-b ferme le contraste bit-identique, celui-ci ferme le contraste **non instancié**. *(Formulation du PI, gate 2 : « un facteur sans porteur physique n'est pas un facteur, c'est une étiquette » — parenté directe de la nulle vacuée de D24 : une nulle qui perd sa variabilité a l'apparence d'un plancher sans en être un ; un facteur qui perd son porteur a l'apparence d'une dimension sans en être une.)* |
| **0-54** | **Asymétrie de position.** Le point de capture est le token du **suffixe** (`t`) ; la tige est en `t−1`. Le facteur tige n'atteint `t` que par l'attention ; le facteur domaine (sous `C7`) est porté **en `t`**, qui domine le résiduel aux couches basses. | **Fatal à la lecture de M2** : « S2 vs S1 » n'est pas « token contre domaine » mais « `t−1` contre `t` », **biaisé en faveur du domaine et non équilibrable par réglage**. Un ordre observé aurait attribué à la neuro-anatomie ce qui appartient à la géométrie des positions. |
| **0-55** | **Confondant de surprise sur les ponts.** Par construction, un pont est l'unité la moins prédictible (tige engagée dans un domaine A, suffixe du domaine B) ⇒ ce sont les **positions incertaines** de X8.1b/P5 et de **D11** ⇒ états plus idiosyncratiques ⇒ moins cosinus-similaires ⇒ le confondant prédit `cos̄(S2) < cos̄(S3)` : **exactement le signe de M1**. | **Fatal à toute lecture de M1** : le confondant et l'hypothèse font la **même** prédiction signée. Le projet a déjà payé ce motif (X8.1 → X8.1b → P5). |
| **0-56** | **Un antipode pour une chaîne à trois maillons.** Le §4.6 du tour 1 nommait `S1 > S2` comme unique antipode de `S3 > S2 > S1 > S0`. L'issue **de loin la plus probable** — `S3 ≈ S2 ≫ S1 ≈ S0` — n'était **ni la prédiction ni l'antipode**. | **Fatal à D13 et D18 simultanément** : la partition des ordres n'était pas exhaustive, et l'issue modale serait tombée dans un vide interprétatif — c'est-à-dire, en pratique, aurait été lue « le token domine, comme prédit ». |

### 0-57 … 0-59 — Matériau : quasi-identité, position de la nulle, façade

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-57** | **Le quasi-identique n'était pas fermé** (A7). D24-b interdit deux préfixes **bit-identiques** tronqués à la capture ; elle n'interdit pas deux préfixes de même longueur **différant d'un token sur huit**. | **Fatal au contraste de variante** : un contraste de 1 token sur 8 n'est pas un contraste de constituants, c'est un bruit lexical — 0-34 sous une forme **graduée**, donc invisible à la porte **binaire** qui a tué 0-34. |
| **0-58** | **Nulle de cadre à `L_e = 1`.** Elle était spécifiée « 40 noms communs », donc **1 token**, alors que `C2` impose `L_e = 2` et que `C3` en **dérive l'indice de capture**. | **Motif exact de 0-42** : elle n'aurait **pas été capturée à la même position** que les unités ⇒ nulle **non appariée** ⇒ elle ne borne rien. Le projet a déjà sorti les pseudo-mots pour cette raison ; il allait les remplacer par un item ayant le même défaut sous un autre nom. |
| **0-59** | **Analogie de façade « séparation de patterns »** appliquée à ce run : le protocole mesure `h` **brut**, `M` **n'est jamais instanciée** (§7). | **L'étage mesuré n'est pas l'étage dont on parle.** Une façade de ce type transforme un résultat géométrique du cortex gelé en verdict sur l'hippocampe. Versée au **vocabulaire interdit du §2**. |

### 0-60 … 0-65 — La primaire de remplacement, et où la difficulté a migré

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-60** | **Option B (72 requêtes descriptives, sans entité) : non faisable, obstacle structurel.** Sous `C1`, les 72 séquences d'une cellule sont byte-identiques hors du slot ⇒ **une unité n'a aucun attribut autre que son entité : il n'y a rien à décrire**. Une description mobiliserait soit la connaissance de pré-entraînement (impossible : les entités sont des composés construits sous contrainte BPE, des **pseudo-référents**), soit un rang de membre (qui dénote une classe de 24 ; l'unicité exigerait d'inscrire un attribut de famille dans le moule, donc **détruire `C1`**). | **Fatal à B comme remède** : sous B, la question mesurée **migre** de « l'état retient-il l'unité » vers « le cortex sait-il déjà que cette description dénote cette entité » — c'est-à-dire vers ce que V2-D existe pour **éviter** ; et §7 interdit le remède (aucune injection). Aggravant : la décodabilité d'une description **varie par modèle** ⇒ B détruirait la variable manipulée unique du §8. Mode de mort chiffré : périphrases neutres épuisées vers 5-10 unités, unicité perdue vers 20-30, et `C3` exigerait 72 périphrases de longueur en tokens **constante sur trois tokenizers** — **le mode de mort exact de l'appariement de longueur déclaré impossible (16/16) sur `fact_pairs`**. |
| **0-61** | **Famine de la statistique de composition.** `Var(X\|m) = m·(5/36)(31/36)·(36−m)/35`. Scénario `m ≈ 18` : `E[X\|m] = 2.5`, sd ≈ 1.05, SE ≈ 0.33 à `K_eff = 10` ⇒ détectable. Scénario `m ≤ 2` : sd ≈ 0.55·√m ⇒ **la puissance s'effondre**. | **Le test s'affame exactement quand le rappel est bon** — et le régime saturé est **plausible** vu le plafond lexical (0-46). Sans clause, une conformité famélique aurait été lue comme « pas de dominance de surface » : **blanc-seing N11 / D20**. |
| **0-62** | **Le canal SUFFIXE est structurellement invisible à la nulle hypergéométrique.** Les suffixes étant globalement uniques, **aucun concurrent ne partage le suffixe de la requête** ⇒ un état encodant fortement le suffixe et faiblement la tige rendrait `X` **conforme** alors que le succès de la cible serait **purement lexical**. | **D26, un cran plus loin.** La conformité de la composition **ne licencie jamais** « adressage non lexical ». Sans ce constat gravé, la statistique de remplacement aurait hérité de la prétention de la porte qu'elle remplace — c'est-à-dire du défaut 0-46 sous une forme non détectable. |
| **0-63** | **« 3/3 modèles » compté comme 1/8.** Les forwards sont déterministes ; chaque modèle est une **fonction fixe du même tirage de matériau** ; la seule source d'aléa partagée est l'échantillonnage des clusters. | **Fallacieux, même famille que 0-39** : une réplication apparente qui n'échantillonne rien. Forme légitime : **bootstrap joint** (un seul rééchantillonnage de clusters par réplique, trois statistiques sur le **même** rééchantillon, conjonction jugée sur la loi jointe). **Le produit de p-valeurs par modèle est interdit.** |
| **0-64** | **Piège D24-b à `t−1`** (si A4 est adoptée) : à `t−1` le suffixe n'est **pas** dans le préfixe causal ⇒ les tronquées des 5 tige-partagés **et de la cible** d'une même cellule sont **byte-identiques** ⇒ états **bit-identiques**. | **Mode 0-34 exact, à la position ajoutée.** Conséquence chiffrée : le **cardinal D24-b à `t−1` vaut 14** (le nombre de tiges), **pas 72** — le publier à 72 aurait été un faux PASS de la porte de cardinalité. La capture `t−1` n'est valide que pour l'**inter-tige**. |
| **0-65** | **Sélection post-hoc sur `m`.** Les requêtes à `m = 0` portent une information nulle ; leur exclusion est légitime **si et seulement si elle est déclarée maintenant** (condition **E3** de Math). | **Fatale à l'exactitude de la nulle** : toute sélection sur `m` non pré-déclarée casse l'échangeabilité conditionnelle. Le défaut serait apparu **après** la mesure, sous la forme la plus tentante (« on retire les requêtes non informatives »). |

### 0-66 … 0-67 — Relevés par le PI à la gate 2, avant adoption

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-66** | **La bande `M` redéfinie était VACUEUSEMENT FRANCHISSABLE** — distinct de 0-47, qui la disait *vide*. Une fois `M` reposée sur `ΔR1_inv` seul, elle redevient atteignable ; mais **le plancher à 0 forward est une borne INFÉRIEURE du canal lexical, pas une borne supérieure**. I2 l'a mesuré : l'état encode la surface **mieux** que le comptage de tokens brut (le cosinus est dominé par la forme). Donc sous `C5`, où la cible est l'unique candidat à partage d'entité, **un `ΔR1_inv` même au-dessus du plancher reste attribuable à un bon encodage lexical de l'état, sans un gramme d'identité**. `V-plafond` filtre le cas trivial ; **elle ne tranche pas le canal**. | **Fatal à la seule issue positive de la primaire 1.** Le seul positif que `M` pouvait produire était **ininterprétable** : une bande dont le verdict confond deux états du monde (« identité » et « lexical bien encodé ») n'est pas une bande — c'est le test que le projet s'est donné avec `N-a`/`N-b`, appliqué à `M`. Et son verdict dépouillé l'avouait lui-même : *« le canal n'est pas tranché »* est **la définition d'un descriptif, pas d'une décision**. Miroir exact de la vacuité refusée pour S-7 et `C6` — refusée cette fois alors qu'elle aurait été **en faveur** de l'hypothèse. |
| **0-67** | **Planchers et nulles poolés sur des viviers sémantiquement typés.** `C7`/A2 **crée délibérément** une corrélation domaine ↔ contenu — c'est son but. Mais cette corrélation est un **canal pour tout ce qui lit le contenu**, à commencer par le plancher lexical à 0 forward. | **Fatal à M1/M3 sous A2 elle-même** : poolés, un sous-vivier sémantiquement typé peut **déplacer le plancher d'un domaine sans toucher l'autre**, et une différence inter-domaines deviendrait lisible **comme effet** là où c'est le **vivier qui parle**. Le remède coûte une ligne de spec : **planchers stratifiés par domaine, nulles de M1/M3 calculées par domaine et jamais poolées, cardinal D24-b publié par sous-vivier**. *Sixième occurrence de la loi de migration (D26) — la difficulté chassée du facteur (pas de porteur) s'est logée dans le porteur (le vivier contraint), et elle a été attrapée **avant l'adoption**.* |

### 0-68 … 0-73 — Relevés au tour de re-signature (Math et Neuro), avant banc

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-68** | **Les deux partitions n'étaient pas EXCLUSIVES** (Math). Calibrateur : `IC = [0.01, 0.09]` satisfait **à la fois** `N-b` (`IC_inf > 0`) et `N-a` (`IC ⊂ [−τ, τ]`) ; `[−0.10, −0.01]` satisfait à la fois `INVALIDE-INSTRUMENT` et `N-a`. Même chevauchement sur les bandes de la primaire (`C+` ∩ `C-0`, `C−` ∩ `C-0`). | **D18 exige exhaustif ET exclusif.** L'exhaustivité avait été vérifiée (27 cellules ORD), le **recouvrement** ne l'avait pas été. Une observation tombant dans deux classes reçoit **deux verdicts gravés contradictoires** — et, en pratique, celui que l'on préfère. Correctif : **ordre d'évaluation gravé** + cas de banc « IC petit et strictement positif ». |
| **0-69** | **La réserve était structurellement inutilisable pour le décisionnel** (Math). Substituer une famille pontée par une famille de réserve (**toutes à tige simple**) casse trois invariants d'un coup : `m₁` (5 → 2 pour l'entrante **et** pour la partenaire orpheline), la **cellule S2** de la tige, et l'homogénéité des clusters. La règle « substituer les deux familles de la tige » ne sauve rien : **la réserve n'a que des tiges simples**. | **Fatal à l'amendement pré-enregistré à la gate 1** — « la marge déplacée dans le matériau plutôt que dans `K` » reposait sur un objet **incapable de jouer ce rôle**. Correctif : substitution pontée → simple **INTERDITE** ; réparation au niveau **unité** (suffixe du même sous-vivier, re-qualification complète) ; la réserve devient un **stock de suffixes qualifiés par sous-vivier**, pas des familles. |
| **0-70** | **L'issue conjointe la plus probable n'avait aucune formulation gravée** (Neuro). Sous 0-46, mieux l'état encode la surface, mieux la cible se classe, **donc plus `m_q` est petit**, donc plus la primaire s'affame (0-61). ⇒ **plus le calibrateur sort beau, plus la primaire risque `C-ind`** : les deux maillons sont **en tension par construction**, et « calibrateur `N-b` + primaire `C-ind` » est l'issue conjointement modale. | **Deuxième occurrence du mode 0-56** (l'issue modale sans conséquence gravée), cette fois **au niveau du couple de maillons** et non d'une chaîne. Sans ligne écrite avant, elle se lirait comme un demi-succès. Formulation gravée : *« l'instrument est sain, la question posée est restée sans réponse »*. |
| **0-71** | **`C-ind` par famine avait une suite unique pour deux causes opposées** (Neuro). Famine **par puissance** (`Σm` faible, `ΔR1_inv` bas) ⇒ « augmenter la résolution » est correct. Famine **par saturation** (`Σm` faible, `ΔR1_inv` haut) ⇒ « augmenter la résolution » est **faux** : ajouter des tiges ne créera pas d'intrusions. | **Une suite gravée fausse est pire qu'une suite absente** : elle envoie le projet dépenser un tour entier dans la mauvaise direction. Le discriminant existe **sans mesure supplémentaire** — c'est le calibrateur, dont c'est exactement la fonction. |
| **0-72** | **Le maillon M2 confondait un `0` résolu et un `0` par IC large** (Neuro) — la distinction `N-a`/`N-ind` que le protocole impose partout ailleurs, **non appliquée au niveau du maillon**. Conséquence : `(0, 0, 0)` tombait en `ORD-ind`, dont la suite gravée est « augmenter la résolution ». | **Faux dans le cas résolu** : si M2 est nul à résolution suffisante, **l'information de tige n'atteint pas la capture** et aucun `K_eff` ne la fera apparaître. La porte de sanité M2 devait mordre ; `ORD-ind` l'en empêchait. |
| **0-73** | **La composition en domaines du pool `P2` n'était pas spécifiée** (Math). Sous encodage de domaine pur, la fraction attendue de tige-partagés parmi les gagnants vaut `(2p_s + 3p_d)/(14p_s + 22p_d)`, **bornée dans [0.136, 0.143] contre 5/36 = 0.139 sous la nulle** — biais ≤ 0.004 par gagnant — **mais uniquement si `k = 12` concurrents même-domaine non-tige**. **Et** : une statistique intra-tige à `t−1` produit des cosinus **dégénérés** (états bit-identiques) qui sortent en `1.0` exact ou en NaN selon l'implémentation. | **Deux quasi-accidents.** (i) L'équilibre qui neutralise le canal de `C7` dans la primaire **n'était pas structurel** : il dépendait d'un tirage non contraint ⇒ septième occurrence de D26 **après** mesure au lieu d'avant. (ii) Le cas `1.0` exact **échappe à la clause NaN (B)** : publier le cardinal 14 ne suffisait pas, il faut une **porte** (`V-t1`). |

### 0-74 … 0-76 — Relevés par le PI à la gate de scellement, avant gel

*Trois vérifications demandées par le PI avant d'autoriser le pré-enregistrement. **Deux ont
échoué, une a échoué à moitié.** Toutes trois portaient sur des dispositifs déjà validés par les
deux experts — c'est la quatrième fois du cycle que la relecture d'un dispositif **approuvé** trouve
un défaut fatal.*

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-74** | **Famine PARTIELLE : le support de `D` rétrécissait silencieusement.** La porte ne testait que `Σ_q m_q ≥ 60`, alors que `E3` **exclut** les requêtes à `m = 0`. Régime dégénéré atteignable : **55 requêtes affamées et 5 obèses** donnent `Σ_q m_q = 90 ≥ 60`, la porte passe, et `D` est calculé sur **5 unités effectives sur 60**. | **C'est la clause NaN (D23) en arithmétique exacte** — formulation du PI : *« une somme Σ qui saute les `q` indéfinis change silencieusement son propre support »*. Un IC calculé sur 5 unités et présenté comme portant sur 60 n'est pas une sous-estimation, c'est **une quantité différente**. Correctif : publication de `n_eff` et de la distribution des `m_q`, et porte **bloquante** sur **`K_eff^support ≤ 8`** — seuil **dérivé** (identique à `V-subst` : `τ` et `ε_max` sont conservateurs à 9, plus à 8). *Une tige sans requête contributive est un cluster qui n'existe pas : la laisser passer réduirait `K_eff` en silence — mode 0-50 par une autre porte.* |
| **0-75** | **`C−` était une cellule nommée sans suite de chantier.** L'antipode était signé et son verdict écrit (« déplétion »), mais **aucune conséquence sur le chantier** n'était gravée : que devient X1 / le cadre DG si la déplétion est réelle ? | **Fatal le jour où elle se réalise.** `P-N2` est la prédiction dont le **renversement** a la plus grande valeur théorique, et l'histoire du projet est sans ambiguïté : les renversements signés ont été plus informatifs que les confirmations (P1 de X7, loi 2, Q-01b). **Une cellule sans conséquence gravée invite l'interprétation à chaud** — et ce protocole est le dernier endroit où l'écrire à froid. Correctif : quatre conséquences ordonnées (§4.5), dont la séparation explicite entre le **chiffre** de X1 (+57 %, acquis, chemin d'écriture) et son **attribution** (représentationnelle, tombée). |
| **0-76** | **La partition des ordres était incohérente ET partiellement dégénérée.** (i) `ORD-4` portait encore `M2 ∈ {+, 0}`, condition **périmée** par la pré-évaluation de M2 introduite au tour précédent — le `0-résolu` en sort désormais avant. (ii) `M2 = 0-résolu` renvoyait *« même suite que `M2 = −` »*, donc **partageait le verdict d'ORD-3**, alors que les deux diagnostics sont **opposés** : là le domaine écrase la tige, ici la tige **n'atteint pas la capture**. (iii) Les zéros de **M1 et M3** ne distinguaient pas résolu d'indécis. | **Le test du PI** — *« existe-t-il deux cellules adjacentes dont les verdicts sont textuellement identiques ? Si oui, la partition est plus grossière qu'annoncée — trou de type [4,11] si une frontière entre elles est censée décider quelque chose »* — **a mordu**. La frontière `−` / `0-résolu` décide de la suite du chantier (changer le pool contre déplacer la capture) : elle ne pouvait pas partager un texte. Et (iii) faisait prononcer à `ORD-2` un « retour au matériau » sur ce qui n'était **qu'un manque de résolution** — 0-72 un cran plus bas. Correctif : **quatre états par maillon**, routage `ind` évalué en premier, **classe `ORD-0` créée avec son verdict propre**, espace résolu **restauré à exactement 27 cellules**. |

---

## 1. Question

Peut-on construire, et **certifier au banc avant tout GPU**, un matériau dont les propriétés
vérifiées satisfont simultanément S-1..S-8, `C1..C7` et D24-b — au point que **la composition des
intrusions** (primaire 2) y soit **décidable** plutôt qu'indécidable, la primaire 1 servant de
**calibrateur** du système ?

## 2. Hypothèse

`H_mat` : **un matériau à moules (entité = tige + suffixe, 1 token chacun sur les trois
tokenizers), 4 domaines × 5 familles décisionnelles × 3 membres, 10 tiges toutes pontées, suffixes
tirés de 4 sous-viviers déclarés par domaine (`C7`), 3 types dont un capitalisé, 2 variantes à
préfixe égalisé et lexicalement distant (`V-var-dist`), satisfait les 8 specs et les 7 conditions
par construction vérifiable ; et l'instrument I2, appliqué à ce matériau, rend un verdict
décisionnel sur la primaire 2.**

**Antipode explicite (D13)** : au moins une porte de génération est **insatisfiable** (le banc le
montre, à coût CPU, avant tout GPU), **ou** la primaire 2 rend `C-ind` — auquel cas la conclusion
gravée est *« indécidable ICI, matériel ou `K_eff` insuffisant »*, la suite est **« augmenter la
résolution »**, et **jamais** « conclure prudemment ».

### Vocabulaire interdit (§2, étendu)

Reconduits : les formulations interdites d'I2 **(i)-(viii)** ; le **bin dur** sous toute forme, y
compris adverbiale (« tendance », « suggère », « va dans le sens de ») — position A-5.

**S'y ajoutent, ce tour :**

- **(ix)** *« séparation de patterns »* appliquée à ce run, sous toute forme (défaut 0-59). Le run
  mesure `h` brut ; `M` n'est jamais instanciée. Ce vocabulaire reste disponible pour la
  **motivation historique de X1**, jamais pour l'**interprétation d'un résultat de ce run**.
- **(x)** *« adressage »*, *« adressable »*, *« non lexical »*, *« récupération par la clé »* —
  **aucune** de ces formulations ne peut être employée sur la base d'une conformité de la primaire 2
  (défaut 0-62) ni d'une valeur haute de la primaire 1 (défauts 0-46, 0-66). Le canal suffixe est
  **hors périmètre de ce run** et cela se dit en toutes lettres (§4.9).
- **(xi)** *« plan 2 × 2 exact »* — le mot **exact** est retiré (défaut 0-53). Formulation autorisée :
  *« plan 2 × 2 dont la dimension domaine est instanciée par sous-viviers déclarés (`C7`) »*.
- **(xii)** *« les ponts remplissent la cellule manquante »* — arithmétiquement vrai, **causalement
  faux** (défaut 0-55). Formulation autorisée : *« les ponts réalisent l'arithmétique de la cellule
  S2 ; leur corrélat causal le plus fort est la surprise, traitée par `V-surprise` »*.
- **(xiii)** *(gate 2, PI)* — sur la primaire 1, **« distingue l'unité »** et toute variante
  affirmative sont **interdites**. **Formulation gravée d'avance, obligatoire, à recopier telle
  quelle** : *« compatible avec un encodage de surface ; le canal identité n'est pas adressé par
  cette primaire »* (défaut 0-66).

## 3. Ce que le projet sait déjà — provenance (D14-R)

| Fait | Chiffre | Source, date | Étiquette |
| --- | --- | --- | --- |
| Le cosinus sur ces états mesure **la forme de la paraphrase**, pas l'identité du fait | ppv même TYPE **1.000 / 0.996 / 1.000** (hasard 0.331) ; ppv même FAIT **0.000 / 0.008 / 0.000** (hasard 0.008) | I2, journal 2026-08-23 | re-mesuré depuis `experiments/results/layer-profile/` — **motive S-1, ne prédit rien** (run INCONCLUSIF) |
| `R1_36 = 0` sur toutes les couches × 3 modèles **n'est pas une anomalie** | rang médian de la cible **19/37** (moyennes 20.1 / 18.6 / 18.5) | I2, 2026-08-23 | idem |
| Les max AUC d'I2 sont **sous** le plancher lexical alors mesuré | 0.4306 / 0.5224 / 0.5442 contre 0.6160 / 0.6078 / 0.6166 | I2, 2026-08-23 | **re-étiqueté deux fois ce tour** : (1) ce plancher n'était pas au plafond **parce que `fact_pairs` avait un recouvrement variable et non identifiant** ; la purification v4 le pousse à **1.0**. (2) *(PI, gate 2)* ces chiffres établissent que **l'état encode la surface mieux que le comptage de tokens brut** ⇒ le plancher à 0 forward est une **borne inférieure** du canal lexical, jamais une borne supérieure (fonde 0-66). |
| **Plafond lexical dérivé du matériau v4** | `ΔR1_lex = 1 − 1/37 = ` **36/37** (fraction, pas décimale) ; plancher de hasard sur l'ensemble restreint d'une tige = **1/6** | Math, ce tour, **dérivé et vérifié indépendamment par le copilote** | **constante, sans variance d'échantillonnage** — se publie, ne se teste pas |
| Appariement de longueur sur `fact_pairs` : **impossible**, pas difficile | violé pour **16/16** valeurs de `k` | Neuro, vérifié par exécution, 2026-08-23 | vérifié — **et c'est le mode de mort chiffré de l'option B** (0-60) |
| Période 20 sur `(entity, verb)` ⇒ 20 couples distincts à N = 30, **10 collisions** | — | banc v3, 2026-08-22 | vérifié par exécution |
| Coût GPU de l'instrument | **53.82 s** pour 12 forwards / 240 séquences ; VRAM Qwen **4.688 Gio réservés** (fp16) | I2, 2026-08-23 | re-mesuré |
| Le dommage de lecture vit aux **positions incertaines** ; gater côté mémoire, jamais côté détresse du cortex | corr **+0.394** ; loi 2 ; **D11** | X8.1b / P5 / Q-01, journal 2026-08-21 | vérifié — **fonde le confondant 0-55** |
| Deux fois de suite, la porte qui trouve le défaut est celle **qui n'a besoin d'aucune donnée** | `V-slot`, `V-ident` | cycle D14-ext, 2026-08-22 | vérifié par exécution — **troisième occurrence ce tour** : le théorème de saturation lexicale (0-46) n'a eu besoin d'aucun octet |

## 4. Prédictions — chacune avec dérivation, antipode, et partition exhaustive

*Toutes les prédictions de construction (§4.1-§4.2) sont décidables **sans GPU**, au banc. Une
prédiction fausse ici est un **résultat**, obtenu pour moins d'une minute de CPU.*

### 4.1 Conditions de construction `C1 … C7`

| # | Prédiction | Dérivation | Antipode (signe opposé, et conséquence gravée) |
| --- | --- | --- | --- |
| **C1** | Pour chaque cellule `(type, variante)`, les 72 séquences sont **byte-identiques hors du slot entité** ; cardinal de séquences **distinctes tronquées à la capture** = **72**. | Moule unique par cellule, entité seule variable, `L_e` constante. | Cardinal < 72 ⇒ **deux entités en collision** ⇒ **arrêt de génération**. |
| **C1′** | *(condition E1 de la nulle hypergéométrique)* Le moule est byte-identique, le pool est **gelé**, et **la sélection du pool ne lit jamais un état, un embedding, ni une fréquence mesurée sur états** — uniquement des **métadonnées déclarées**. | Math, complément : échangeabilité conditionnelle des 36 scores. | Toute sélection touchant une quantité issue du modèle ⇒ **nulle non exacte** ⇒ arrêt. *(Corollaire Neuro : une mesure issue du modèle rendrait le matériau **dépendant du modèle**, détruisant la variable manipulée unique du §8.)* |
| **C2** | `L_e = 2` tokens pour **72/72** entités **et 40/40 paires de la nulle de cadre**, sur les trois tokenizers. | Tige 1 token + suffixe 1 token. **Étendu par A8.** | Une seule entité ou paire à `L_e ≠ 2` ⇒ **rejet et re-qualification complète**, jamais de rustine locale. |
| **C3** | Indice de capture = `len(préfixe) + L_e − 1`, **entier constant par cellule** ; les deux variantes d'un type ont **le même nombre de tokens de préfixe** sur les trois tokenizers. | D24-b + Neuro Q5. | Longueurs différentes ⇒ 0-43 réintroduit ⇒ **arrêt**. |
| **C4** | Aucune paire de séquences décisionnelles n'est **byte-identique tronquée à la capture** ; règle d'égalités (**mid-rank**, bris seedé par `cfg.seed`, **gravé avant mesure**) — condition **E2**. | D24-b + Math Q3(d) + E2. | Une égalité byte ⇒ contraste no-op (0-34) ⇒ **arrêt**. |
| **C5** | *(pool de la primaire 1)* Pour **72/72** requêtes, le nombre d'unités à **recouvrement de token d'entité nul** est **≥ 60** ; pool de 36 tiré une fois, seedé, **gelé**, **jamais rééchantillonné**. | 72 − 1 − 2 − 3 = **66 ≥ 36 + 30 de marge**. | Éligibles < 36 pour une seule requête ⇒ **design insatisfiable** ⇒ retour au PI. |
| **C6** | **Exactement un** des trois types place l'entité en **position capitalisée** ; `V-casse` **mord** : minusculiser la tige change le token pour **≥ 90 %** des tiges sur les trois tokenizers. | Position A-3 ; leçon `Her`→`her` de v3. | < 90 % ⇒ clause **vraie par vacuité** ⇒ troisième occurrence du mode 0-6/0-8 ⇒ **arrêt**. |
| **C7** | *(porteur physique du domaine — adoptée à la gate 2)* Les suffixes sont tirés de **4 sous-viviers disjoints déclarés par domaine avant génération** ; **100 %** des suffixes d'une famille appartiennent au sous-vivier du **domaine de cette famille** ; les 4 sous-viviers sont **lexicalement disjoints**. **ET (amendement PI, défaut 0-67)** : le **cardinal D24-b est publié PAR SOUS-VIVIER** ; les **nulles de M1/M3 et le plancher lexical à 0 forward de la primaire 1 sont calculés PAR DOMAINE, jamais poolés**. | Défaut 0-53 : sans `C7`, le facteur domaine n'a **aucun** porteur ; le plan est un 1 × 2 et M1/M3 n'ont pas d'objet. Défaut 0-67 : la corrélation domaine ↔ contenu que `C7` **crée** est un canal pour tout ce qui lit le contenu. | Une seule violation de la partition ⇒ le facteur domaine est **descriptif avant mesure**, **M1 et M3 sont retirées du protocole** (et non « rapportées prudemment »), le plan est publié comme **1 × 2**. Un plancher ou une nulle **poolés** ⇒ **échec de porte** : le résultat inter-domaines serait le vivier, pas l'effet. |

### 4.2 Plan de strates et cardinaux D24-b

Structure décisionnelle : **10 tiges, toutes pontées**, chacune portant **2 familles dans 2 domaines
différents** ⇒ 20 familles décisionnelles, 4 domaines × 5 familles, 60 unités. **Les 4 familles à
tige simple (12 unités) sont affectées à la réserve** — conséquence structurelle : **`m₁ = 5` pour
60/60 unités décisionnelles**, c'est-à-dire que l'exigence de `m₁` constant, rétrogradée par Math à
« recommandée », est **satisfaite gratuitement par le design**.

| Strate | Définition | # paires | Cluster (D24-b : cardinal tronqué **à la capture**) |
| --- | --- | --- | --- |
| S3 | tige partagée, **même** domaine (intra-famille) | 60 | tige — cardinal attendu **60**, à publier |
| S2 | tige partagée, **domaine différent** (pont) | 90 | tige — cardinal attendu **90**, à publier |
| S1 | pas de tige partagée, même domaine | à publier | tige |
| S0 | pas de tige partagée, domaine différent | à publier | tige |
| Nulle de cadre | **40 PAIRES tige+suffixe** (A8), `L_e = 2`, hors des 4 domaines, sans token partagé deux à deux ni avec aucune entité, **même porte BPE sur les 3 tokenizers** | 40 / cellule | **40** par cellule |
| Nulle de nouveauté | ~20 pseudo-mots, **séparée, jamais mélangée** | 20 / cellule | **20** par cellule |

**Cardinaux D24-b à publier obligatoirement** :

- **37/37** séquences tronquées distinctes **par requête** ;
- **72/72** distinctes **par cellule** ;
- **6/6** distinctes sur l'**ensemble restreint** d'une tige, **ne différant que du dernier token
  avant capture** ⇒ **drapeau obligatoire « contraste minimal : 1 token »**, publié en tête du
  rapport : il pilote le risque d'égalités et donc `V-dtype` ;
- **à `t−1` (capture descriptive) : 14**, pas 72 (défaut 0-64) — publier 72 serait un faux PASS ;
- **par sous-vivier de domaine** (amendement PI, défaut 0-67).

### 4.3 Primaire décisionnelle — composition des intrusions

**Le protocole n'a qu'UNE primaire décisionnelle.** *(Décision PI, gate 2 — voir §13-bis :
la primaire 1 est le **calibrateur du système**, la primaire 2 **porte seule la question posée**.)*

**Primaire (ex-primaire 2) — composition des intrusions, pool surface-apparié, nulle
hypergéométrique exacte.**

**Composition du pool `P2`, gravée (défaut 0-73)** : 36 concurrents = **5 tige-partagés** (2
co-famille + 3 famille partenaire) + **31 non-partagés, dont exactement 12 même-domaine et 19
autre-domaine** — constant sur toutes les requêtes, seedé, **gelé**. Cette composition n'est pas un
détail de tirage : elle **apparie les compositions de domaine des deux classes** (2/5 = 0.400 contre
12/31 = 0.387), ce qui borne le biais du canal `C7` à **≤ 0.004 par gagnant** (Math, dérivé), soit
≤ 0.07 sur `D̄` à `m = 18` — **un ordre de grandeur sous `ε`**. **La borne dérivée est publiée avec
le résultat.** Sans cette ligne, l'équilibre serait un **accident du tirage**.

Pour chaque requête `q` :

- `m₁ = 5` (tige-partagés dans le pool, **constant par construction**) ;
- `m(q)` = nombre de concurrents **battant strictement** la cible (mid-rank sous `C4`) ;
- `X(q)` = nombre de tige-partagés parmi ces `m(q)` ;
- **statistique agrégée** : `D = Σ_q ( X_q − m_q · 5/36 )` — **une différence observé − attendu
  (D16 satisfaite)**.

**Nulle exacte** (Math, re-dérivée sous E1–E4) : conditionnellement à (état de `q`, coupure `T`),
les 36 scores sont échangeables ⇒ `X | m ~ Hypergéom(36, 5, m)`, `E[X|m] = m·5/36`. **Point
décisif** : la cible ne sert **qu'à définir la coupure** ; elle n'a **pas besoin d'appartenir à une
classe d'échangeabilité** ⇒ **le singleton lexical de 0-46 est hors périmètre**. C'est précisément
pourquoi cette statistique vit **au-dessus** du plancher lexical : sous le lecteur lexical à
0 forward, `m(q) = 0` pour toute requête, donc **`E(q)` est indéfini** — le plancher n'a pas de
valeur ici, il est **hors du domaine de définition**.

**Nulle simulée** : Monte-Carlo seedé de l'hypergéométrique **par requête**, agrégée **par clusters
de tiges** (`K_eff = 10`, D19), **exécutée et publiée AVANT lecture de `D` observé**.
**IC** : bootstrap de **tiges**.
**Antipode (D13)** : `D ≤ 0`. Lecture **signée** : `D` significativement **négatif** = **déplétion**
⇒ codage par contraste / inhibition de la tige ⇒ la géométrie **sépare activement** les voisins de
surface ⇒ **falsification partielle de la motivation représentationnelle de X1** — un positif
surprenant, pas un échec.
**Clause de famine D19, gravée avant run** : si **`Σ_q m_q < 60`** (≈ 1 gagnant par requête), le
verdict est **`C-ind` d'office** — jamais une conformité interprétée.
**Clause E3, déclarée maintenant (défaut 0-65)** : les requêtes à `m = 0` **sont exclues**, cette
exclusion est **pré-déclarée**, et **aucune autre sélection sur `m` n'est permise**.
**Limite gravée (défaut 0-62)** : le **canal suffixe est structurellement invisible** à cette
nulle ; **une conformité ne licencie jamais « adressage non lexical »**. Phrase à recopier telle
quelle dans le rapport (§4.9).

**Descriptif obligatoire attaché (Neuro, coût nul)** : publier la courbe **`X_q` en fonction de
`m_q`**. `D` somme des régimes dont l'effet attendu diffère structurellement — l'excès doit être
**nul aux deux bords** (`m → 0` : personne ne bat la cible ; `m → 36` : tout le monde la bat) et
**maximal à `m` intermédiaire**. **Cette forme en cloche est la signature du décrément lure** ; une
conformité de `D` **sans** cette forme (excès plat, ou concentré à `m` extrême) est **signalée comme
non conforme au mécanisme invoqué**. Descriptif : aucune décision n'en dépend.

**Ordre d'évaluation des bandes, gravé (défaut 0-68 — exclusivité D18)** :
**famine → `C+` → `C−` → `C-0` → `C-ind`**. La première condition satisfaite emporte la
classification ; aucune observation ne peut recevoir deux verdicts.

### 4.4 Primaire 1 — CALIBRATEUR, non décisionnelle *(décision PI, gate 2)*

`ΔR1_inv = R1(variante correcte de la même unité) − R1(clé nulle appariée)`, rang parmi **37**
candidats du pool `C5` gelé. Statistique de cluster **par TIGE**, `K_eff = 10`. Bootstrap de
**tiges** côté requêtes uniquement ; **pool jamais rééchantillonné, rangs jamais recalculés dans un
échantillon**. **Inférence conditionnelle au pool gelé**, mention gravée.

**Elle ne porte aucune bande, aucun verdict d'hypothèse.** Elle fournit trois choses, et rien
d'autre : la **nulle exacte** `1/37`, le **plancher** lexical, et la **santé de l'instrument**.

**Descriptif nommé, OBLIGATOIRE, à vocabulaire contraint (amendement PI (a), gate 2)** — le contenu
informatif de l'ancienne bande `M` survit ici, sans rang de bande :

| Quantité publiée | Contenu | Formulation autorisée |
| --- | --- | --- |
| `ΔR1_inv` **contre clé nulle appariée** | l'état porte-t-il quelque chose au-delà du moule ? | *« compatible avec un encodage de surface ; le canal identité n'est pas adressé par cette primaire »* — **jamais** « distingue l'unité » (vocabulaire interdit (xiii)) |
| `ΔR1_inv` **contre plancher lexical** (`36/37`, stratifié **par domaine**, `C7`/0-67) | la distinction **« l'état ne porte rien »** vs **« l'état porte au moins de la surface encodée »** | idem ; **c'est cet écart qui alimente la lecture croisée avec la primaire** |

**Partition exhaustive du calibrateur (D18, amendement PI (b))** — trois classes plus une cellule
d'invalidation, **chacune avec son cas synthétique au banc, frontière `N-ind` comprise** :

| Classe | Condition sur l'IC de `ΔR1_inv` (bootstrap de tiges, 95 %, `K_eff = 10`) | Lecture gravée |
| --- | --- | --- |
| **`N-b`** | `IC_inf > 0` | l'état porte **au moins** de la surface encodée — *formulation (xiii) obligatoire* |
| **`N-a`** | `IC ⊂ [−τ, +τ]` | **pas d'écart décelable, à résolution suffisante** |
| **`N-ind`** | tout le reste | **« indécidable ICI, matériel ou `K_eff` insuffisant »** ⇒ **« augmenter la résolution »** — *jamais un demi-`N-a`* |
| **`INVALIDE-INSTRUMENT`** | `ΔR1_inv` **sous la clé nulle elle-même** (`IC_sup < 0`) | **c'est l'INSTRUMENT qui est accusé, pas le cortex** : si même le lexical échoue sur `C5`, la chaîne de mesure est en cause ⇒ arrêt, retour au banc. **α propre déclaré (Math) : sous encodage exactement nul, `P(IC_sup < 0) ≈ 0.025`** — l'accusation de l'instrument est portée à **α = 0.025**, à écrire, sinon un faux `INVALIDE-INSTRUMENT` sur 40 runs serait lu comme une panne réelle |

**Ordre d'évaluation, gravé (défaut 0-68 — exclusivité D18)** :
**`INVALIDE-INSTRUMENT` → `N-b` → `N-a` → `N-ind`**. La première condition satisfaite emporte la
classification. *(Sans cet ordre, `IC = [0.01, 0.09]` tombait à la fois en `N-b` et en `N-a`, et
`[−0.10, −0.01]` à la fois en `INVALIDE-INSTRUMENT` et en `N-a`.)*

**Seuils gravés** : `τ = 0.15` ; IC = bootstrap de **tiges** à 95 %, `K_eff = 10`, alpha propre 0.05.
**Dérivation de `τ`** : Math donne, à `K = 20`, demi-largeur ≈ 0.10 (effet ≈ 0, pire cas ρ = 1) ; et
`K_eff = 10 ⇒ ×1.41` ⇒ **0.14**. Ses deux options (`τ = 0.10`, `τ = 0.12`) sont donc **toutes deux
inatteignables** — incohérence interne remontée au §12-Q1. Arbitrage : **`τ = 0.15`**, plus petite
valeur ronde ≥ 0.14 *(recalcul copilote : `0.229/√10 × 2 = 0.145`)*.
**Confirmation de Math (verrou levé)** : ses valeurs 0.10 et 0.12 étaient données **à `K = 20`**,
avant que `K_eff = 10` ne soit acquis ; **elles ne transportent pas**, et c'est son propre facteur
×1.41 qui les périme. Recalcul complet : sd/requête `√(2·(1/37)(36/37)) = 0.229` ; SE à `K_eff = 10`,
ρ = 1 : `0.229/√10 = 0.0724` ; demi-largeur 95 % : `1.96 × 0.0724 = 0.142`. **`τ = 0.15` confirmé.**
**Réserve gravée** : à 10 clusters, un **bootstrap percentile sous-couvre légèrement** — la clause
ci-dessous est **précisément** la protection et reste **non négociable**.

**Clause gravée, non négociable après run** : *si l'IC sous nulle apparente déborde `τ`, le verdict
est `N-ind`, jamais un `τ` ajusté.*
**Zone morte déclarée en clair** : à `K_eff = 10`, tout effet vrai entre ≈ 0.15 et ≈ 0.55 sort
majoritairement en `N-ind`. Prix du design, déclaré **avant** le run, n'autorisant **aucune**
relecture *a posteriori*.

**La bande `M` est RETIRÉE** (défauts 0-47 et 0-66, décision PI gate 2). Traçabilité : §14-5.

### 4.5 Bandes de la primaire (partition exhaustive)

Sur l'IC de `D` normalisé par requête :

| Bande | Condition | Verdict gravé | Suite |
| --- | --- | --- | --- |
| **`C+`** | `IC_inf > 0` **et** `Σ_q m_q ≥ 60` | **excès d'intrusions tige-partagées** — décrément lure-spécifique mesuré comme composition | canal tige actif ; instruire le canal suffixe (§4.9) |
| **`C−`** | `IC_sup < 0` **et** `Σ_q m_q ≥ 60` | **déplétion** — la géométrie sépare activement les voisins de surface | **positif surprenant**, et sa **suite de chantier est gravée ci-dessous** — pas seulement « antipode réalisé » |
| **`C-0`** | `Σ_q m_q ≥ 60`, `IC ⊂ [−ε, +ε]` | **pas de composition détectable, à résolution suffisante** | le recouvrement de tige ne rend pas un concurrent plus confusable ; **le cadre « lure » est retiré de la question représentationnelle** |
| **`C-ind`** | tout le reste, **famine globale** (`Σ_q m_q < 60`) **ou famine partielle** (§ ci-dessous) | **« indécidable ICI »** | suite **selon la cause** (§6.G) ; **jamais lisible comme absence d'effet de lure** |

**Famine PARTIELLE — porte de support, gravée (défaut 0-74).** `E3` exclut les requêtes à `m = 0` :
le support de `D` **rétrécit donc silencieusement**, et le seuil global est franchissable par un
régime dégénéré — *55 requêtes affamées et 5 obèses donnent `Σ_q m_q = 90 ≥ 60` avec **5 unités
effectives sur 60***. Trois quantités sont **publiées avant lecture de `D`**, et deux sont
**bloquantes** :

| Quantité | Statut | Règle |
| --- | --- | --- |
| `n_eff = #{q : m_q ≥ 1}` | publiée, descriptive | le **support réel** de la somme, jamais implicite |
| distribution complète des `m_q` | publiée, descriptive | alimente la courbe `X_q` vs `m_q` (§4.3) |
| **`K_eff^support` = # tiges portant ≥ 1 requête contributive** | **BLOQUANTE** | **`K_eff^support ≤ 8` ⇒ `C-ind` d'office (famine partielle)**. Seuil **dérivé, non choisi** : identique à celui de `V-subst` — `τ` et `ε_max` sont dérivés à `K_eff = 10`, donc conservateurs à 9, et cessent de l'être à 8. Une tige sans requête contributive est **un cluster qui n'existe pas** : la laisser passer réduirait `K_eff` **en silence**, ce qui est le mode 0-50 par une autre porte |

*C'est la clause NaN (D23) transposée en arithmétique exacte : une somme qui saute des termes
**change son propre support**, et un support non publié est un support inventé.*

**Suite de chantier de `C−`, gravée À FROID (défaut 0-75).** *Le PI note que `P-N2` est la
prédiction dont le **renversement** aurait la plus grande valeur théorique, et que l'histoire du
projet est constante — les renversements signés ont toujours été plus informatifs que les
confirmations (P1 de X7, la loi 2, Q-01b). Une cellule nommée sans conséquence écrite invite
l'interprétation à chaud le jour où elle se réalise. Voici la conséquence, écrite avant.*

Si `C−` se réalise, **le cortex gelé sépare déjà activement les entrées à fort recouvrement** —
c'est-à-dire qu'il fait, au niveau représentationnel, ce que la biologie délègue à un étage dédié.
Conséquences gravées, dans cet ordre :

1. **X1 conserve son statut de mécanisme de CHEMIN D'ÉCRITURE, et perd sa justification
   représentationnelle.** Le gain mesuré (+57 % sur E2, tableau `EXTENSIONS.md` §4) **reste acquis** :
   il porte sur l'interférence d'écriture, pas sur la géométrie de `h`. Ce qui tombe est
   l'**attribution** — « expansion + parcimonie = séparation de patterns » — pas le **chiffre**.
2. **Le descriptif `A3` devient le test prioritaire du chantier suivant**, et son antipode devient
   une **question ouverte et non une note** : si la compression de `cos(topk(G·h))` est **uniforme**,
   `topk(G·h)` est un **rééchelonnement** et non un séparateur, et le +57 % appelle **une autre
   explication** (parcimonie des écritures, capacité) — à instruire, pas à supposer.
3. **Le successeur « canal suffixe » est REPRIORISÉ derrière** cette re-dérivation : mesurer un
   second canal de surface sur un cortex qui sépare déjà activement répondrait à une question dont la
   prémisse vient de changer.
4. **Aucune de ces trois conséquences n'autorise une phrase sur la séparation de patterns
   d'`engram`** : le run mesure `h` brut, `M` n'est jamais instanciée (vocabulaire interdit (ix)).
   La formulation autorisée est *« le cortex gelé, sur ce matériau, sépare les voisins de surface »* —
   jamais *« le gyrus denté est inutile »*.

**`ε` (D19) — validé par Math, avec barrière exécutable et plafond gravé.** L'inférence est exacte
**conditionnellement aux `m_q`** (conditionnement sur une statistique qui ne porte pas l'effet
testé) : `ε` défini sur les `m_q` observés est la **résolution vraie de la nulle**, pas un
ajustement. D19 est respectée **si et seulement si** la barrière d'information est **exécutable** :

1. **Formule et seed gravées maintenant** : `ε = q₀.₉₇₅(|D̄_null|)`, MC de l'hypergéométrique **par
   requête**, agrégée **par tiges**, `B = 10⁴` répliques, seed = `cfg.seed`.
2. **Barrière mécanique** (porte `V-compo`) : le pipeline calcule `{m_q}`, **scelle `D`** (non
   calculé, ou calculé et hashé sans lecture), **publie `ε`**, puis descelle. L'ordre est **vérifié
   par le banc** ; cas échouant = `ε` publié après lecture de `D`.
3. **Plafond numérique gravé MAINTENANT, indépendant des `m_q`** (Math, dérivé) : `Var(X|m)` est
   maximale à `m = 18` ⇒ `18²·(5/36)(31/36)/35 = 1.107` ⇒ sd ≤ 1.052 par requête ⇒ demi-largeur
   pire cas (ρ = 1, `K_eff = 10`) = `1.96 × 1.052/√10` ⇒ **`ε_max = 0.66` excès par requête**.
   **Toute valeur MC au-dessus est une erreur de pipeline** — porte exécutable.

### 4.6 Partition exhaustive des ORDRES (D18)

Trois maillons, chacun jugé **+ / − / 0** par son **test de permutation intra-tige** (nulle exacte,
`C(6,3) = 20` partitions par tige, MC seedé, **calculé PAR DOMAINE et jamais poolé**, `C7`/0-67) —
**jamais par comparaison de moyennes de strates** (mauvaise unité d'échange, mention gravée) :

- **M1** = `cos̄(S3) − cos̄(S2)` — effet **domaine à tige égale**. *N'a d'objet que sous `C7`* : sans
  elle, la permutation intra-tige **est la vérité du générateur**.
- **M2** = `cos̄(S2) − cos̄(S1)` — **porte de sanité** : si `cos̄(S2) ≤ cos̄(S1)`, l'information de la
  tige **n'atteint pas la capture** ⇒ arrêt de l'interprétation représentationnelle.
- **M3** = `cos̄(S1) − cos̄(S0)` — effet **domaine sans tige partagée**. **Requalifié par son
  signataire en CONTRÔLE DE MANIPULATION de `C7`** : une confirmation dit *« le vivier a été
  correctement typé »*, **jamais** *« l'état encode le domaine »*. **Attendu dès la couche 0** — c'est
  ce qui le distingue de M1 ; s'il n'apparaissait qu'en profondeur, ce serait un effet
  représentationnel et non un effet de vivier, et cela se note. **Antipode** : `M3 ≤ 0` ⇒ les
  sous-viviers sont **lexicalement disjoints mais pas sémantiquement cohérents** ⇒ `C7` n'a pas
  instancié le domaine **dans la géométrie** ⇒ **M1 perd son objet**, plan publié en 1 × 2 (même
  conséquence que `C7` violée).

**Forme décisionnelle de M1 et M3 (Math, Q9-ii) — « stratifié-combiné », jamais « poolé », jamais
« quatre tests »** : les nulles sont calculées **dans chaque strate de domaine**, et la statistique
décisionnelle est la **somme** `Σ_domaines (observé − attendu-dans-la-strate)` — exacte comme somme
de quantités centrées sous leurs nulles propres, et **conservant `K_eff = 10`**. Les déclinaisons
**par domaine** sont publiées en **descriptif** : quatre M1 décisionnelles vaudraient ~5 tiges
chacune ⇒ demi-largeur ×1.41 **plus** Bonferroni ×4 ⇒ régime du bin dur.

**Pré-évaluation obligatoire des TROIS maillons, AVANT la classification ORD (défauts 0-72 et
0-76)** — chaque maillon est jugé en **quatre états**, jamais en ternaire naïf :

| État | Condition sur l'IC de permutation | Routage |
| --- | --- | --- |
| `+` | strictement positif | poursuit vers la classification |
| `−` | strictement négatif | poursuit vers la classification |
| **`0-résolu`** | `IC ⊂` couloir de résolution, **bornes exclues** | poursuit vers la classification |
| **`ind`** | IC plus large que le couloir | **route immédiatement vers `ORD-ind`** |

**Règle de routage gravée** : *tout maillon dont le zéro n'est pas résolu envoie la classification
entière en `ORD-ind`.* Elle est la transposition **au niveau du maillon** de la distinction
`N-a`/`N-ind` que le protocole impose partout ailleurs — sans elle, `ORD-2` prononcerait « retour au
matériau » sur ce qui n'est **qu'un manque de résolution** (défaut 0-76).

**L'espace résolu compte donc exactement `3 × 3 × 3 = 27` cellules**, et `ORD-ind` absorbe tout le
reste. *(Avant correction, `ORD-4` portait encore `M2 ∈ {+, 0}` — condition **périmée** par la
pré-évaluation, puisque le `0-résolu` en sort désormais avant : défaut 0-76.)*

| Classe | Condition sur (M1, M2, M3) | # cellules / 27 | Verdict gravé | Suite gravée |
| --- | --- | --- | --- | --- |
| **ORD-1** | (+, +, +) | 1 | chaîne complète `S3 > S2 > S1 > S0` | les deux facteurs sont instanciés et ordonnés |
| **ORD-2** | (0, +, 0) — **issue modale** | 1 | **« le facteur domaine n'a pas été instancié — retour au matériau »** | **JAMAIS** « le token domine, comme prédit » (défaut 0-56) |
| **ORD-3** | `M2 = −` (tout M1, M3) | 9 | **le domaine domine la tige** ⇒ géométrie **sémantique**, `C5` insuffisante | retour au banc ; **la mesure ne s'interprète pas** |
| **ORD-0** *(nouvelle, défaut 0-76)* | `M2 = 0-résolu` (tout M1, M3) | 9 | **l'information de tige N'ATTEINT PAS la capture** — diagnostic **opposé** à ORD-3 : là le domaine écrase la tige, ici la tige n'arrive pas | **même SUITE qu'ORD-3** (arrêt de l'interprétation représentationnelle), **verdict distinct** : la cause est le **locus de capture**, pas la géométrie ⇒ le chantier suivant déplace la capture, il ne change pas le pool. *Et **jamais** « augmenter la résolution » : le zéro est résolu.* |
| **ORD-4** | `M2 = +`, (M1 = + **ou** M3 = +), hors ORD-1 / ORD-2 | 7 | domaine instancié, chaîne incomplète | rapporter le maillon manquant ; pas d'ordre global |
| **ORD-ind** | **tout maillon en état `ind`** (règle de routage ci-dessus) | hors espace résolu | **« ordre indécidable ICI »** | **augmenter la résolution** — *jamais « retour au matériau »* |

*Exhaustivité de l'espace résolu : ORD-1 (1) + ORD-2 (1) + ORD-3 (9) + **ORD-0 (9)** + ORD-4 (7)
= **27**. ✓ `ORD-ind` absorbe tout ce qui sort de l'espace résolu. **Exclusivité** : les quatre états
de chaque maillon sont mutuellement exclusifs par construction, et le routage `ind` est évalué en
premier. **Aucune classe ne partage son verdict avec une autre** — vérification demandée par le PI à
la gate de scellement, exécutée, et c'est elle qui a produit `ORD-0`.*

**Exactitude de la permutation intra-tige sous `C7` (Math, Q9-iii)** : le test reste **exact**
(`C(6,3) = 20` partitions par tige) — **c'est sa signification qui change, et c'est voulu**. Avant
`C7`, cette nulle était **la vérité du générateur** (défaut 0-53) ; sous `C7`, permuter les suffixes
entre les deux familles d'une tige **mélange deux sous-viviers**, et la nulle devient l'hypothèse
substantielle *« l'état ne distingue pas le contenu des sous-viviers »* — exactement ce que M1 doit
tester. **Exigence induite, à rendre explicite dans `V-freq` v2** : l'appariement des bandes de
fréquence entre les deux familles d'une tige est un appariement **entre sous-viviers** ; il se vérifie
**par paire de sous-viviers**, sinon **fréquence et domaine se confondent dans la permutation**.

**Porte de fuite `V-leak` (M4 de Neuro, corrigée — re-signature due, §11-Q1).** Sous `C7`, la forme
« `cos̄(S2) − cos̄(S1)` **nul** à la couche 0 » est **falsifiée par construction** : à la couche 0 le
résiduel en `t` est l'embedding du **suffixe** ; S1 partage le sous-vivier de domaine, S2 non ⇒ la
quantité vaut une **ligne de base `B0` prédite ≤ 0**, pas 0. Forme corrigée, conservant l'intention :
quantité décisionnelle = **le contraste en profondeur** `[cos̄(S2) − cos̄(S1)](ℓ) − B0` ; prédiction
`B0 ≤ 0` (IC de permutation) et croissance sur le premier tiers de la profondeur ; **antipode
`B0 > 0` strictement ⇒ FUITE ⇒ arrêt, pas interprétation**. **SIGNÉE par lab-neuro sous cette
forme.** *Précision d'implémentation gravée : « couche 0 » = **sortie des embeddings avant le premier
bloc** (`hidden_states[0]`), sans ambiguïté d'indexation.*

**`B0′` — détecteur de fuite SANS HYPOTHÈSE (ajout de lab-neuro, coût nul, adopté).** `B0` est un
détecteur **faible** : sa prédiction `≤ 0` est **déjà garantie par `C7`**, donc une fuite de tige
modérée ne peut pas la franchir. Le détecteur fort est gratuit :

> **`B0′ = [cos̄(S3) − cos̄(S1)](ℓ = 0)`**, prédit **exactement 0** (IC de permutation contenant 0).
> **Dérivation** : S3 et S1 sont **tous deux à domaine identique**, donc tirent leurs suffixes du
> **même sous-vivier** ; ils ne diffèrent que par le partage de la **tige**, qui **n'est pas dans le
> résiduel en `t` à la couche 0**. Toute valeur non nulle est une **fuite au sens strict** (suffixe
> non indépendant de la tige : corrélation de bande de fréquence, de longueur de chaîne, ou de
> sous-vivier).
> **Antipode** : `B0′ ≠ 0` (IC de permutation excluant 0) ⇒ **FUITE ⇒ arrêt**.

*C'est ce que M4 voulait dire, re-basé sur un contraste **apparié en domaine** ; et sa croissance en
profondeur est la vraie M4 — l'information de tige n'arrive que par l'attention.*

### 4.7 Portes

| Porte | Contenu | Passe / échoue |
| --- | --- | --- |
| `V-C1..V-C7` | les sept conditions du §4.1, **y compris la stratification par domaine de `C7`** | un cas synthétique passant **et** un échouant chacune |
| `V-pool` | pool gelé, hash avant/après, **aucun rang recalculé dans un échantillon bootstrap** | cas échouant = pool rééchantillonné |
| **`V-plafond`** *(remplace `V-lex`)* | le rapport **publie** `36/37` et `1/6` **en fractions**, avec leur dérivation, **le plancher stratifié par domaine**, **et** la phrase gravée : *« la primaire 1 ne peut, à elle seule, distinguer un adressage représentationnel d'un transcript lexical ; cette distinction n'est pas au périmètre de ce run »*. Porte de **schéma** : présence obligatoire des champs, **absence** de tout terme du vocabulaire interdit (x) et (xiii) | terme interdit détecté, constante absente, **ou plancher poolé** ⇒ **échec du pipeline** |
| **`V-compo`** | primaire : nulle MC seedée **exécutée et publiée avant** lecture de `D` ; `Σ_q m_q` publié ; clause de famine appliquée ; exclusion `m = 0` conforme à la déclaration | nulle simulée après lecture, ou sélection sur `m` non déclarée ⇒ **run invalide** |
| **`V-calib`** *(gate 2)* | la primaire 1 est classée en **une et une seule** des 4 classes du §4.4 ; **aucun champ décisionnel** n'existe pour elle dans le schéma de sortie ; les deux descriptifs obligatoires sont présents | présence d'un champ de verdict d'hypothèse, ou classe non couverte ⇒ **échec du pipeline** |
| **`V-ord`** | classification en **une et une seule** des 5 classes ORD ; un cas synthétique par classe | classe non couverte ou chevauchement ⇒ échec |
| **`V-leak`** | `B0 ≤ 0` (IC de permutation) | `B0 > 0` ⇒ **arrêt**, aucune interprétation de M1/M2/M3 |
| **`V-dtype` (v2)** | bf16 épinglé (D21) ; `m = 60` états fp32 ; `δ̂ = max\|Δcos\|` ; **deux marges** : (i) **marge de tête** — `0` requête à marge `< 2δ̂` ; (ii) **marge à la coupure `T`** — `0` concurrent à `\|S_i − T\| < 2δ̂` (condition E4) | **> 1 TIGE touchée ⇒ `INCONCLUSIF-précision`** — l'unité est le **cluster**, pas la famille : « famille » laisserait passer **deux familles de la même tige**, c'est-à-dire un cluster entier corrompu, sans déclencher (Math, Q5). **Repli fp32 = chemin nominal**, pas exception |
| **`V-freq` (v2)** | serpentin sur **rang de fusion de la tige** (gpt2), **borne exacte par énumération** sur les 3 tokenizers ≤ **0.15** ; **étendu** : appariement **au tirage** des bandes de fréquence des **suffixes** entre unités pontées et non pontées **et** entre les deux familles d'une tige ; **vérification supplémentaire au niveau suffixe intra-famille** | **aucun test d'homogénéité nulle part** (0-41) |
| **`V-surprise` (bandes gravées)** | **NLL du token de capture** (scalaire, cortex gelé, teacher-forcing) et **norme d'état** publiées **par strate**. **Unité de bande = la PAIRE**, par la **moyenne des NLL de ses deux membres** *(M1 est une statistique de paire ; stratifier par membre rendrait les paires inter-bandes indéfinies et sortirait ~2/3 du matériau)*. **3 bandes** — dérivé, non préféré : à `K_eff = 10`, quatre bandes laisseraient ~2,5 tiges par bande, **0-52 rejoué à l'étage de la stratification**. **Coupures : tertiles calculés PAR MODÈLE** sur la distribution des NLL moyennes de paire, **avant toute lecture de M1** (la NLL n'est pas comparable en valeur absolue entre les trois modèles). **Superposition secondaire, 2 bandes, coupure absolue à `4.0 nats`** = la constante `thr` du projet, pour **relier ce run à X8.1b / P5 / D11** | **Règle de décision qui rend la porte non vide** : *M1 n'est créditée que si **son signe est stable sur les trois bandes**. Si M1 n'est présente que dans la bande de NLL la plus haute, **elle EST le confondant de surprise (0-55) et M1 est RETIRÉE**, pas « rapportée avec nuance ».* **Toute repondération ou sélection après lecture ⇒ run invalide.** |
| **`V-t1`** *(défaut 0-73, porte et non note)* | **toute quantité à `t−1` dont l'ensemble de comparaison contient deux unités de MÊME TIGE ⇒ échec du pipeline**. Motif : les états y sont **bit-identiques**, et le cosinus dégénéré sort en `1.0` **exact** ou en NaN selon l'implémentation — **le cas `1.0` échappe à la clause NaN (B)**. Publier le cardinal 14 ne suffisait pas | cas de banc échouant = un rapport synthétique contenant un cos intra-tige à `t−1` |
| **`V-var-dist`** | **distance lexicale minimale déclarée** entre les deux préfixes d'une cellule : **≥ 50 % des tokens diffèrent**, sur les trois tokenizers | < 50 % ⇒ contraste quasi-identique (0-57) ⇒ arrêt. *Interaction* : préfixes plus proches ⇒ écarts de cosinus plus fins ⇒ **`2δ̂` mord plus** |
| `V-bindur` | le bin dur est marqué `DESCRIPTIF` et son champ décisionnel est **absent du schéma de sortie** | présence d'un champ décisionnel ⇒ échec |
| `V-casse` | `C6`, ≥ 90 % | < 90 % ⇒ arrêt |
| `V-fact-pairs` | **contre-exemple obligatoire** : `fact_pairs` soumis à la table D25 doit **ÉCHOUER** sur `C1`, `C2` et S-1 | s'il passe, c'est la **table** qui est fausse |
| **`V-joint` (règle tranchée par Math)** | **IC simultanés par enveloppe bootstrap jointe (max-t)** — ni min-p (qui réintroduit des p-valeurs par modèle, l'objet même de l'interdit 0-63), ni Bonferroni (qui ignore la dépendance que le bootstrap joint capture précisément). Sur chaque réplique `b` (**un seul** rééchantillonnage de tiges) : `M_b = max_j \|D_j^{(b)} − D̄_j\|/ŝ_j` pour `j ∈ {gpt2, smol, qwen}` ; `c* = q₀.₉₅({M_b})` ; IC simultanés `D̄_j ± c*·ŝ_j` ; **conjonction déclarée ssi 0 est hors des trois IC simultanés** ; `B ≥ 2000`, seedé | cas passant = trois séries corrélées dont l'enveloppe couvre à ~95 % (vérifié par simulation) ; **cas échouant = « p₁·p₂·p₃ < 0.05 » déclaré significatif** |
| **`V-subst` (réécrite, défaut 0-69)** | **(1) Substitution d'une famille pontée par une famille à tige simple : INTERDITE** — elle casse `m₁` (5 → 2 pour l'entrante **et** la partenaire orpheline), la cellule S2, et l'homogénéité des clusters. **(2)** Un item mort se répare **au niveau UNITÉ** : remplacement du suffixe **depuis le même sous-vivier**, **re-qualification complète** (`C2` : jamais de rustine locale). **(3)** Si une **tige entière** meurt : `K_eff = #tiges survivantes`, **recalculé et publié avant le run** ; run **permis à `K_eff = 9`** (demi-largeurs ×1.05 ; `τ` et `ε_max` **inchangés car dérivés à 10, donc conservateurs à 9**) ; **`K_eff ≤ 8` ⇒ retour au PI**. **(4)** La réserve utile est un **stock de suffixes qualifiés par sous-vivier**, **pas des familles** | toute substitution de famille pontée, ou `K_eff` non republié, ⇒ échec |
| **`V-perimetre`** *(gate 2)* | les **trois** éléments du §4.9 sont présents dans le rapport : (i) mécanisme de l'invisibilité, (ii) limite nommée, (iii) successeur désigné | un seul manquant ⇒ **échec du pipeline** — la différence entre hors-périmètre et angle mort est **entièrement** dans cette écriture |

### 4.8 Descriptifs pré-déclarés (aucune décision n'en dépend)

| # | Quantité | Prédiction | Antipode et sa lecture |
| --- | --- | --- | --- |
| **calibrateur** | `ΔR1_inv` vs clé nulle **et** vs plancher lexical stratifié | — | vocabulaire (xiii) obligatoire |
| **A3** | `cos(topk(G·h))` relu sur les strates | **compression croissante** avec le cosinus brut | **compression uniforme** ⇒ `topk(G·h)` est un **rééchelonnement, pas un séparateur** ⇒ l'attribution bio du +57 % d'E2 de X1 serait une **façade** — à consigner, pas à décider ici |
| **A4** | capture à **`t−1`**, mécanistique (cardinal D24-b **= 14**) | discriminant du confondant de **copie de token** : la copie doit être **bien plus forte** à la capture sur la tige | profil identique aux deux positions ⇒ pas de signature de copie |
| **profil** | forme en profondeur de M2 | **copie** = abrupt et localisé ; **effondrement représentationnel** = lisse et monotone (`V-leak`) | — |
| **`P1 − P2`** | contraste inter-pools | — | **DESCRIPTIF par 0-49**, et **ASYMÉTRIQUEMENT INFORMATIF** (correction de lecture de lab-neuro, gravée) : la non-monotonicité porte sur la **dose**, pas sur le **signe**. Sous monotonie stochastique du score en σ, `P1 ≥ P2` **partout**, avec égalité aux deux extrémités. Donc une valeur **positive ou nulle est NON INFORMATIVE** (compatible avec l'effet maximal comme avec l'absence d'effet) ; **seule une valeur strictement négative est informative** — elle violerait la monotonie stochastique en σ, indiquant un **mécanisme anti-lexical actif**. *Sans cette ligne, un `P1 − P2 > 0` serait lu comme un demi-appui à l'hypothèse.* |

### 4.8-bis Recadrage à écrire EN TÊTE du rapport *(lab-neuro, adopté)*

> **Le livrable de v4-matériel est le MATÉRIAU QUALIFIÉ ; `P-N2` en est la première sonde.** Un
> `C-ind` sur un matériau qui passe les 7 conditions, les ~120 cas de banc et `fact_pairs` en
> contre-exemple **n'est pas un échec du cycle** — c'est **le premier matériau du projet dont on
> saura ce qu'il garantit**.

*Écrit **avant**, ce recadrage protège du verdict-pour-avoir-un-verdict que le PI vient de refuser en
retirant la bande `M`.*

### 4.9 Hors-périmètre déclaré — le canal suffixe *(décision PI, gate 2)*

*La déclaration est du type **annulation-de-prédiction tracée**, jamais du type **ligne CADUC
silencieuse**. Les trois éléments suivants sont **obligatoires** dans le rapport (porte
`V-perimetre`) :*

1. **Mécanisme.** Le canal suffixe est invisible aux deux quantités **par construction** (0-62) : les
   suffixes étant globalement uniques, **aucun concurrent ne partage le suffixe de la requête** —
   donc ni la nulle `1/37` du calibrateur, ni la nulle hypergéométrique de la primaire ne peuvent le
   monitorer.
2. **Limite nommée.** **Aucune formulation du rapport ne peut exclure que l'effet mesuré transite
   par le canal suffixe.** La limite est **nommée**, pas enterrée.
3. **Successeur désigné.** L'expérience pré-enregistrée qui couvrira ce canal est désignée comme
   successeur de v4-matériel. **Sa première pierre est posée** (Math, §12-Q8, chiffré hors chemin
   critique) : la cellule « suffixe partagé × domaine différent » est **structurellement interdite
   sous `C7`** (un suffixe appartiendrait à deux sous-viviers lexicalement disjoints) ⇒ le 2 × 2
   tige × suffixe n'est réalisable qu'**intra-domaine** ; cette version tient sur le vivier (~10
   suffixes doublés) et sur `C5` (éligibles 63 ≥ 36), **mais fait tomber `K_eff` à 5** (les arêtes
   suffixe s'ajoutent aux arêtes tige dans le graphe de dépendance) ⇒ tout passerait sous la
   frontière descriptive. **Verdict chiffré : le successeur exige SON PROPRE MATÉRIAU** — ~20 tiges /
   40 familles pour retrouver `K_eff = 10`, même machinerie, volume ×2. **La variante n'est pas
   greffable sur v4 sans le vider de sa puissance.**

### 4.10 Prédictions d'instrument reconduites

`P-A`, `P-B`, `P-C`, `P-D` **restent pré-enregistrées telles quelles** et sont jugées **sur ce
matériau**. Elles ne s'amendent pas. **Rappel gravé** : l'inférence « P-A falsifiée ⇒ P6 falsifiée »
demeure **illégitime**.

## 5. Contrôles et baselines

1. **Configuration courante** : `EngramConfig()` par défaut (layer=6, λ=2.0, cap=0.5, η=0.2,
   decay=1e-3, thr=4.0, dg=8192/64, read_gate=keysim, seed=0) — **citée comme référence de projet** ;
   ce protocole **n'instancie ni ne lit `M`** (aucune injection, D8 intacte).
2. **M reset / D7** : sans objet. Contrôle homologue = **pool gelé** : même matériau, mêmes états,
   seule la clé change.
3. **Nulles par maillon (D17)** — une par maillon, toutes **bloquantes** :

| Maillon | Nulle | Type |
| --- | --- | --- |
| matériau | `V-C1..V-C7` | déterministe, 0 forward |
| clé (calibrateur) | **clé nulle appariée** (`ΔR1_inv`, D16) | exacte `1/37` sous `C1..C5` |
| surface lexicale | **plancher `36/37`, stratifié par domaine** | constante dérivée |
| sélection | pool `C5` gelé | déterministe |
| **composition (primaire)** | **hypergéométrique `X\|m ~ H(36, 5, m)`** + MC seedé | exacte sous E1–E4 |
| ordre / strates | **permutation intra-tige par domaine** (`C(6,3) = 20`, MC seedé) | exacte |
| couche 0 | **`B0`** (`V-leak`) | permutation |
| cadre | **40 paires `L_e = 2`** | appariée en position |
| nouveauté | pseudo-mots, **séparée** | non appariée, déclarée telle |
| précision | `δ̂` fp32 vs bf16, **deux marges** | déterministe |

4. **Contrôle qui tue l'explication triviale** : `fact_pairs` passé au **même banc** doit **échouer**.
5. **Ordre des conditions**, cascade **gravée** : (1) `L_e = 2` sur 3 tokenizers → (2) **sous-viviers
   de domaine (`C7`)** → (3) structure tige/suffixe et ponts → (4) moules, égalisation de préfixe et
   **distance lexicale** → (5) **appariement des bandes de fréquence** dans le sous-ensemble
   survivant → (6) type capitalisé → (7) **nulle de cadre en paires** → (8) banc. Le banc **vérifie
   que l'exécution a suivi l'ordre gravé**.

## 6. Critères d'abandon — portes exécutables (D22)

Chaque clause **bloque l'écriture du rapport**. Un drapeau calculé puis ignoré est **lui-même** un
motif d'invalidation.

**A. Invalidation de génération** — toute porte `V-C1..V-C7`, `V-var-dist`, `V-casse`, `V-freq` en
échec ⇒ **aucun run**.

**B. Clause NaN (D23)** : NaN/inf invalide le run s'il apparaît dans **(i)** les états capturés,
**(ii)** toute quantité intermédiaire d'une réduction (cosinus, AUC, IC, bootstrap, MC), **(iii)**
toute quantité publiée. Réductions **NaN-strictes**.

**C. Clause de PORTÉE de l'interdit binomial (A-1c)** : aucune statistique décisionnelle, aucun IC,
aucune valeur-p, aucune annotation de figure obtenue par **agrégation binomiale** sur les
indicateurs de rang. L'échangeabilité `1/37` vaut **par requête**. Portée : publiées, intermédiaires,
figures. Toute inférence agrégée passe par le **bootstrap de tiges**. **Violation ⇒ run invalide.**

**C-bis. Interdit multi-modèles (0-63)** : **le produit de p-valeurs par modèle est interdit**, même
portée que C. Toute conjonction passe par `V-joint`.

**D. Ce qui tue `H_mat`** : `C5` insatisfiable ; **ou `C7` violée** (⇒ M1/M3 retirées, plan publié en
1 × 2) ; **ou plancher/nulle poolés** (0-67) ; ou `C6` vacuée ; ou **primaire en `C-ind`** avec
`V-dtype` PASS (⇒ le matériau ne suffit pas ; suite = **augmenter la résolution**).

**E. Bin dur** : descriptif à ce `N` ; aucune formulation ne le cite comme évidence. *(D18 : `E`
vaut sur le banc seul, pas sur le run.)*

**F. Plafond lexical** : si `V-plafond` échoue — constante absente, plancher poolé, ou terme du
vocabulaire interdit (x)/(xiii) présent — **le rapport ne peut pas être écrit**.

**F-bis. Calibrateur** : si `V-calib` échoue — champ décisionnel présent pour la primaire 1, ou
descriptif obligatoire manquant — **run invalide**. Et si le calibrateur sort en
**`INVALIDE-INSTRUMENT`** (`ΔR1_inv` sous la clé nulle elle-même), **c'est l'instrument qui est
accusé, pas le cortex** ⇒ arrêt, retour au banc.

**G. Famine (D19), avec sa SCISSION gravée (défaut 0-71)** : `Σ_q m_q < 60` ⇒ primaire =
**`C-ind` d'office**. La suite dépend de la **cause**, discriminée **sans mesure supplémentaire** par
le calibrateur — c'est précisément sa fonction :

| Cause | Signature | Suite gravée |
| --- | --- | --- |
| **famine par PUISSANCE** | `Σm` faible **et** `ΔR1_inv` bas | **« augmenter la résolution »** |
| **famine par SATURATION** | `Σm` faible **et** `ΔR1_inv` haut | **« réduire la dominance de surface »** (autre locus de capture, autre position, autre instrument). *« Augmenter la résolution » serait **faux** : ajouter des tiges ne créera pas d'intrusions.* |
| **famine PARTIELLE** (défaut 0-74) | `Σ_q m_q ≥ 60` **mais** `K_eff^support ≤ 8` | **« le support s'est effondré sur une minorité de clusters »** ⇒ la suite se lit sur la **distribution des `m_q`**, publiée : concentration sur peu de tiges ⇒ **le matériau n'est pas homogène**, retour au banc ; étalement avec beaucoup de `m_q = 0` ⇒ **saturation**, même suite que ci-dessus. **Jamais** « augmenter la résolution » par défaut |

**G-bis. Issue conjointe modale, formulation gravée AVANT le run (défaut 0-70)** : les deux maillons
sont **en tension par construction** — mieux l'état encode la surface, mieux la cible se classe, plus
`m_q` est petit, plus la primaire s'affame. **« Calibrateur `N-b` + primaire `C-ind` » est l'issue
conjointement la plus probable**, et elle **ne peut en aucun cas** se lire comme un demi-succès.
Formulation obligatoire, à recopier telle quelle : *« l'instrument est sain, la question posée est
restée sans réponse »*.

**H. Fuite de couche 0** : `B0 > 0` ⇒ **arrêt**, aucune interprétation d'ordre.

**I. Précision** : > 1 famille touchée par l'une des deux marges de `V-dtype` ⇒
`INCONCLUSIF-précision` ⇒ **repli fp32, chemin nominal déclaré avant le run**.

**J. Périmètre** : si `V-perimetre` échoue (un des trois éléments du §4.9 manquant) ⇒ **le rapport ne
peut pas être écrit**.

**K. Invalidation classique** : 0 write (sans objet), NaN, E3 > +0.05 nats/token — **E3 est sans
objet dans ce run** (aucune lecture, aucune injection) et **redevient bloquant au premier run qui
injecte**.

## 7. Variables fixées

`seed = 0` · modèles : `gpt2` (L=12), `HuggingFaceTB/SmolLM2-360M` (L=32, layer de référence 16),
`Qwen/Qwen2.5-1.5B` (L=28, layer de référence 14) — `L` re-lu du config · **dtype du forward épinglé
: bf16** (D21), **repli fp32 déclaré comme chemin nominal**, contrôle sur `m = 60` états ·
cosinus/Gram fp32, valeurs propres fp64 · **aucune injection, `M` jamais instanciée**, `engram/` non
modifié · tokenizers : les trois, tous les tests BPE **simultanés** · **`τ = 0.15`, `K_eff = 10`
(clusters = TIGES), pools de 36, `m₁ = 5`** · **`ε`** = demi-largeur d'IC 95 % de la nulle MC seedée,
calculée et publiée **avant** lecture de `D` · batch ≤ 240 séquences, **VRAM rapportée en réservé** ·
**NLL du token de capture conservée en scalaire** (jamais le tenseur de logits : 870 × 151k fp32
≈ 2 Gio sur Qwen).

Jeux de données : **`eval/pool_v4.py` (nouveau)** — 60 unités décisionnelles (10 tiges pontées × 2
familles × 3 membres) + **12 de réserve (4 familles à tige simple)**, **4 sous-viviers de suffixes
déclarés par domaine (`C7`)**, **40 paires tige+suffixe** pour la nulle de cadre, ~20 pseudo-mots,
6 gabarits (3 types × 2 variantes). `data/…` et `SECRETS` : **non utilisés**. `pool.fact_pairs` :
**uniquement comme contre-exemple échouant** ; `eval/pool.py` **gelé et intouché**.

## 8. Variable manipulée

**Une seule : le matériau** (`pool.fact_pairs` → `pool_v4`). Instrument, seeds, modèles, couches,
métriques, seuils : identiques à I2.

**Limite de validité externe, à écrire dans le rapport** : avec **3 types × 2 variantes**,
l'invariance mesurée est l'invariance à **trois transformations nommées, répliquées 72 fois** —
**pas** à « la paraphrase ».

## 9. Budget — re-chiffré, **dépassement accepté par le PI (gate 2)**

| Poste | Volume | Coût agent | CPU |
| --- | --- | --- | --- |
| Candidats bruts : 90 tiges, **~240 suffixes répartis en 4 sous-viviers (`C7`)**, **~160 items pour les 40 paires de nulle de cadre**, ~20 pseudo-mots | ~510 | **~3 h 15** | — |
| `eval/pool_v4.py` (génération + table D25) | ~450 lignes | ~2-3 h | — |
| Qualification BPE, 3 tokenizers | ~800 chaînes × 3 | — | **< 15 s** |
| Gabarits + égalisation `C3` + **distance lexicale** | 6 | ~1 h 20 | < 1 s |
| **`V-freq` v2** (serpentin + appariement ponté/non-ponté + intra-famille + borne par énumération) | 24 familles × 3 tok. | **~1 h 15** | < 2 s |
| **Nulle MC hypergéométrique seedée + permutation intra-tige par domaine** | 60 requêtes × MC ; 10 tiges × 20 partitions × 4 domaines | **~1 h** | ~5 s |
| **Bootstrap joint 3 modèles (`V-joint`)** | — | **~45 min** | ~2 s |
| **Capture `t−1` + NLL scalaire par strate** | même forward | **~30 min** | — |
| Banc D14-S étendu (≈ 30 clauses × 2 cas + **4 classes calibrateur + 4 bandes primaire + 5 classes ORD** + `V-plafond`, `V-calib`, `V-perimetre`, `V-var-dist`, `V-surprise`, `V-compo`, `V-joint`, `V-subst`, `V-leak` + contre-exemple `fact_pairs`) | **~120 cas** | **~5-6 h** | ~15 s |
| **Total avant tout GPU** | | **~16-21 h d'agent** | **< 60 s** |
| Mesure I2 sur le matériau (3 modèles) | ~792-870 séquences/modèle | — | **~3-5 min GPU**, VRAM **en réservé** |

**Le total SORT de l'épure « ~9-12 h agent »** et la double. **Déclaré, non absorbé silencieusement ;
accepté par le PI à la gate 2** — motif : le coût est entièrement en **temps d'agent**, sur un poste
qui a tué **vingt-deux défauts fatals avant tout GPU**.

**Runs** : 1 run de génération, 1 run de banc, 1 run de mesure. **Aucun balayage.**

## 10. Livrables attendus

1. **`eval/pool_v4.py`** (fichier neuf, SPDX AGPL-3.0-or-later) — génération + **table des garanties
   D25 en en-tête de la source** : une ligne par propriété (`C1..C7`, **sous-viviers de domaine et
   leur disjonction lexicale**, `m₁ = 5`, `K_eff = 10`, indépendance longueur/unité, paires
   intra-unité intra-type, diversité de type, survie à l'effacement de casse, absence de période sur
   tout slot et tout couple de slots, **cardinaux D24-b tronqués à la capture, y compris 14 à `t−1`
   et le détail par sous-vivier**, éligibles des deux pools), chacune avec **son résultat et sa
   porte**. `eval/pool.py` **n'est pas touché**.
2. **Déclaration de prérequis par l'instrument** + **vérification mécanique** matériau × instrument ;
   incompatibilité = **arrêt**. **Liste blanche manuelle proscrite.**
3. **Flag `EngramConfig`** : `dataset = "fact_pairs"` par défaut (**= comportement actuel**),
   `"pool_v4"` en option. **Aucun défaut de config existant n'est modifié.**
4. **Extension du banc `eval/gate_bench.py`** — pour **chaque** clause : un cas passant **et** un cas
   échouant. Plus :
   - **`fact_pairs` en contre-exemple échouant OBLIGATOIRE** (`C1`, `C2`, S-1) ;
   - **un cas synthétique par classe du calibrateur** : `N-b`, `N-a`, **`N-ind`**,
     **`INVALIDE-INSTRUMENT`** ;
   - **un cas synthétique par bande de la primaire** : `C+`, `C−`, `C-0`, **`C-ind` (famine)** ;
   - **un cas synthétique par classe d'ordre** : `ORD-1`, `ORD-2`, `ORD-3`, **`ORD-0`**, `ORD-4`,
     **`ORD-ind`** ;
   - **cas de banc d'EXCLUSIVITÉ (défaut 0-68)** : « IC petit et strictement positif » doit tomber
     en `N-b` (et en `C+`), **jamais en deux classes** ; « IC petit et strictement négatif » doit
     tomber en `INVALIDE-INSTRUMENT`, jamais aussi en `N-a` ;
   - **un cas synthétique pour `M2 = 0-résolu`** (défaut 0-72), distinct de `M2 = ind`, **tombant en
     `ORD-0` et non en `ORD-3`** (défaut 0-76) ;
   - **un cas « famine partielle concentrée »** (défaut 0-74) : `Σ_q m_q ≥ 60` obtenu par une
     poignée de requêtes obèses, `K_eff^support ≤ 8` ⇒ **doit tomber en `C-ind`**, jamais en `C+`
     ni en `C-0` ;
   - **un cas de routage `ind`** : un maillon indécis ⇒ **`ORD-ind`**, jamais `ORD-2` ;
   - cas échouants pour **`V-t1`** (cos intra-tige à `t−1`), **`V-subst`** (substitution d'une famille
     pontée), `V-pool`, `V-bindur`, **`V-plafond`** (terme interdit ; **plancher poolé**),
     **`V-calib`** (champ décisionnel présent), **`V-perimetre`** (élément manquant),
     **`V-var-dist`**, **`V-compo`** (nulle simulée après lecture ; `ε` MC au-dessus de `ε_max`),
     **`V-joint`** (« p₁·p₂·p₃ < 0.05 » déclaré significatif), **`V-leak`** (`B0 > 0` ; `B0′ ≠ 0`) ;
   - **tous les cardinaux D24-b publiés tronqués au point de capture**, jamais sur séquences
     complètes.
5. **Tests CPU** : `.venv\Scripts\python -m pytest tests/ -q`, sans téléchargement HF au-delà des
   tokenizers en cache.
6. **Entrée de journal** au format maison, incluant le **Registre des engagements**, la ligne
   CADUQUE de §14-2 et le **retrait de la bande `M`** (§14-5).
7. **Conservation des états bruts** (permet A3 ; `G` gelée, D8/D9 intactes) et des **NLL scalaires
   par position de capture**.

## 11. Questions pour lab-neuro — **TOUTES RÉPONDUES, avis FAVORABLE (2026-08-23)**

*lab-neuro signe les quatre engagements demandés (M4 corrigée, M3, forme agrégée de `P-N2`, charge
unique) et n'oppose **aucun** motif de blocage. Ses trois compléments — `B0′`, la pré-évaluation de
M2, la scission de `C-ind` — sont des ajouts de banc à **coût nul**, tous intégrés ci-dessus.*

1. **`V-leak` / M4 — re-signature requise.** Sous `C7` (que tu exiges), la forme « nul à la couche 0 »
   est **falsifiée par construction** : le résiduel en `t` à la couche 0 est l'embedding du suffixe,
   et S1 partage le sous-vivier de domaine ⇒ la quantité vaut `B0 ≤ 0`, pas 0. **Signes-tu M4 sous la
   forme corrigée du §4.6, ou récuses-tu la correction ?** *(Sans signature, `V-leak` sort du
   protocole et M1/M2/M3 perdent leur garde-fou de fuite.)*
2. **M3** (`S1 > S0`) : « signable sous A2 ». A2 est adoptée en `C7`. **Signes-tu M3 ?**
3. **Partition ORD** à 5 classes (§4.6, `ORD-2` = « domaine non instancié — retour au matériau ») :
   répond-elle à ton A1 ? Y manque-t-il une issue que tu juges probable ?
4. **`P-N1` retirée du décisionnel** : Math démontre la non-monotonicité du contraste inter-pools.
   **Le contestes-tu ?** Si oui, nomme un régime où le signe est déterminé.
5. **`V-surprise`** : les **bandes de NLL** doivent être gravées **avant génération**. Donne-les
   (nombre, coupures ou règle de coupure), sinon la porte est vide.
6. **`P-N2`** : je la porte en **primaire décisionnelle unique** avec ses quatre bandes.
   **Confirmes-tu la signature sur la forme agrégée (`D` par tiges, `K_eff = 10`) plutôt que « par
   famille » ?**
7. *(gate 2)* Le PI retire la bande `M` et fait de la primaire 1 un **calibrateur**. La question
   posée repose désormais **entièrement** sur `P-N2`. **Le maintiens-tu en connaissance de cette
   charge**, ou demandes-tu un second maillon décisionnel ?

## 12. Questions pour lab-math — **VERROUS LEVÉS (2026-08-23)**

*`τ = 0.15` **confirmé** (ses 0.10/0.12 étaient à `K = 20` et ne transportent pas) ; `ε` conditionnel
**validé** avec barrière mécanique et plafond `ε_max = 0.66`. Il relève au passage **deux défauts**
(0-68 exclusivité des partitions, 0-69 réserve inutilisable) et **un quasi-accident** (0-73), tous
intégrés. **Aucune de ses corrections ne demande un nouveau tour d'expertise.***

1. **Contradiction interne sur `τ` — BLOQUANT.** Tu donnes ≈ 0.10 à `K = 20`, puis `×1.41` pour
   `K_eff = 10` ⇒ **0.14**, alors que tes deux options étaient 0.10 et 0.12. J'ai arbitré à
   **`τ = 0.15`**. **Confirme, ou donne la valeur correcte.**
2. **`ε`** : défini comme la demi-largeur d'IC 95 % de la **nulle MC seedée** sur les `m_q` observés,
   **avant** lecture de `D`. Compatible avec D19, ou exiges-tu une valeur numérique gravée maintenant,
   indépendante des `m_q` ?
3. **`m₁ = 5` par construction** (les 20 familles décisionnelles sont exactement les 20 pontées ; les
   4 tiges simples partent en réserve) : satisfait-il *gratuitement* ton exigence rétrogradée, et ta
   variance `m·(5/36)(31/36)(36−m)/35` s'applique-t-elle **telle quelle** ?
4. **`V-subst`** : si une substitution introduit une famille à **tige simple**, `m₁` passe à 2 et le
   couplage parfait se brise. Règle exacte de recalcul de `K_eff` — et faut-il **interdire** la
   substitution d'une famille pontée par une famille simple ?
5. **`V-dtype` v2** : le seuil « > 1 famille touchée » se transporte-t-il à `K_eff = 10` sous la forme
   « > 1 **tige** touchée » ?
6. **Cardinal D24-b à `t−1` = 14** : le publier suffit-il, ou faut-il **interdire** toute statistique
   à `t−1` sur l'**intra-tige** au niveau du schéma de sortie (porte exécutable plutôt que note) ?
7. **`V-joint`** : min-p ou IC simultanés — tranche, et donne la règle de couverture exacte
   (Bonferroni sur 3 ? enveloppe bootstrap ?) pour que le banc la teste sur un cas synthétique.
8. **Variante « suffixe-ponts »** *(hors chemin critique, décision PI gate 2 : instruction en tâche de
   fond, première pierre du successeur §4.9)*. **Chiffre le coût exact** — marge du vivier, impact sur
   les éligibles de `C5`, **collision avec `C7`** (un suffixe partagé entre deux familles de domaines
   différents **viole** la partition des sous-viviers), et `K_eff`.
9. *(gate 2, nouveau)* **Stratification par domaine (défaut 0-67).** Le PI exige que le plancher
   lexical et les nulles de M1/M3 soient calculés **par domaine, jamais poolés**, parce que `C7` crée
   délibérément une corrélation domaine ↔ contenu. **Confirme que la stratification suffit** à fermer
   ce canal, ou dis ce qu'il faut de plus (appariement au tirage ? covariable de vivier ?). Et donne
   le **coût en puissance** de la stratification à `K_eff = 10` réparti sur 4 domaines.

## 13. Questions pour le PI — **TRANCHÉES à la gate 2 du 2026-08-23** (voir §14-5)

1. Sort de la bande `M` → **RETIRÉE** (option 1), avec deux amendements du PI.
2. Canal suffixe → **hors périmètre déclaré** (option 1), §12-Q8 en tâche de fond.
3. A2 / `C7` → **ADOPTÉE**, avec l'amendement de stratification (défaut 0-67).
4. Dépassement de budget → **ACCEPTÉ**.
5. Ratification du retrait de `§14-2` → **ACQUISE** (impossibilités démontrées et vérifiées
   indépendamment, non préférences ; le dispositif retiré était un amendement du copilote, pas du PI).

## 13-bis. Principes de méthode

### D26 — la difficulté migre *(gravée `docs/ARCHITECTURE.md` §3, étendue trois fois ce tour)*

> **`C5` illustre que la difficulté ne se supprime pas, elle migre.** Purifier le pool a fait
> réapparaître la difficulté **dans le contrôle** : l'accès lexical à la cible. Chaque purification
> déplace le confondant vers l'endroit que le protocole ne regarde pas encore. La seule réponse
> stable est de **mesurer des deux côtés de la purification, jamais d'un seul.**

**Extensions de ce tour :**

1. **Neuro** : *« la difficulté ne migre pas vers un endroit qu'on regarde, elle migre vers un
   endroit qu'on ne PEUT PAS regarder »* — établi sur l'option B, où elle migre vers une question
   (« le cortex connaît-il déjà cette description ? ») que **le §7 interdit d'instrumenter**.
2. **Math** : la nulle de composition teste la dominance de surface **par la TIGE** ; le canal
   **suffixe** est **structurellement invisible** ⇒ **la conformité ne licencie jamais « adressage non
   lexical »**. La migration a lieu **à l'intérieur de la statistique de remplacement elle-même**.
3. **PI, gate 2 (sixième occurrence, attrapée avant adoption)** : contraindre sémantiquement le vivier
   par domaine **crée** la corrélation domaine ↔ contenu — c'est le but — et cette corrélation est un
   canal pour tout ce qui lit le contenu. **La difficulté chassée du facteur (pas de porteur) s'est
   logée dans le porteur (le vivier contraint)** ⇒ planchers et nulles **stratifiés par domaine**
   (0-67).

**Portée opératoire, révisée à la gate 2.** Le dispositif « deux côtés de la purification » est
**conservé dans son principe et changé deux fois dans sa forme** : ce ne sont plus **deux pools
comparés** (démoli par 0-49), ni **deux bandes sur deux quantités** (démoli par 0-66), mais **un
maillon qui mesure le confondant et un maillon qui mesure l'effet**.

### D-nouvelle candidate — *la structure calibrateur / question* **(à graver par le PI)**

> *« La primaire 1 n'est plus un test de l'hypothèse — c'est le **calibrateur du système** (nulle
> exacte, plancher, santé de l'instrument) — et la primaire 2 **porte seule la question posée**. Un
> maillon qui mesure le confondant, un maillon qui mesure l'effet : c'est la structure "une nulle par
> maillon" (D17) assumée jusqu'au bout. Et c'est la cinquième réincarnation de la loi de migration :
> la difficulté chassée du pool s'est logée dans le contrôle, et **la réponse stable n'était pas de
> durcir le contrôle mais de déplacer la question vers l'étage que le confondant ne peut pas
> atteindre**. »* — PI, gate 2, 2026-08-23.

## 14. Amendements de gate

### 14-1 (PI, gate 1) — **MAINTENU dans son intention, RÉÉCRIT dans sa forme (défaut 0-69)**
Design **60 unités décisionnelles + 12 de réserve**. Repli 48/`K = 16` non déclenché. **`K` initial
= 20 familles, mais l'unité d'inférence est la TIGE : `K_eff = 10`** (0-50).

**Ce qui change.** L'amendement pré-enregistré à la gate 1 — *« la marge déplacée dans le matériau
plutôt que dans `K` »* — reposait sur des **familles de réserve substituables**. Math démontre que
cet objet **ne peut pas jouer ce rôle** : la réserve n'a que des **tiges simples**, et toute
substitution pontée → simple casse `m₁`, la cellule S2 et l'homogénéité des clusters (0-69).

**Forme réécrite, qui sert la même intention :**

- la **réserve utile** est un **stock de suffixes qualifiés par sous-vivier**, pas des familles ;
- un item mort se répare **au niveau unité** (suffixe du même sous-vivier, **re-qualification
  complète**, `C2` : jamais de rustine) ;
- une **tige morte** fait tomber `K_eff` à 9 : run **permis**, `τ` et `ε_max` restant **conservateurs
  car dérivés à 10** ; **`K_eff ≤ 8` ⇒ retour au PI** ;
- substitution **avant le premier token, jamais additive** (D14) : inchangé.

*L'intention du PI est intégralement conservée — la marge vit bien dans le matériau ; c'est son
**grain** qui passe de la famille à l'unité.*

### 14-2 (gate 1) — **CADUC. RETIRÉ ET REMPLACÉ.**

*Conformément à A-7 et à la ligne CADUC d'I2 : le dispositif retiré est conservé avec son motif, son
auteur et son remplaçant. **Il ne disparaît pas.***

| Élément retiré | Auteur du retrait | Motif | Remplacé par |
| --- | --- | --- | --- |
| **Porte `V-lex`** (baseline lexicale comme critère de **domination** conditionnant `M`) | **Math** (théorème de saturation), **vérifié indépendamment par le copilote** ; **Neuro** concourt | **Inexécutable** (exigerait `IC_inf > 36/37`) ; **ininterprétable** (si atteinte, prouverait un transcript lexical) ; **confronte un IC à une constante sans variance** (0-46, 0-48) | **`V-plafond`** : publication obligatoire de `36/37` et `1/6` **en fractions**, **stratifiées par domaine**, + phrase gravée sur le périmètre ; porte de **schéma** |
| **Primaire 2 inter-pools** | **Math** (complément) ; **Neuro** avait proposé le remplacement avant de connaître l'argument | **NON MONOTONE** : nulle sous transcript parfait **ET** sous encodage nul ⇒ un 0 mesuré est compatible avec l'effet maximal (0-49) | **Composition hypergéométrique** `D = Σ_q (X_q − m_q·5/36)`, **nulle exacte** sous E1–E4, **différence observé − attendu (D16)**. Le contraste inter-pools survit en **descriptif** |
| **Bande `M` conditionnée à `V-lex`** | conséquence des deux lignes | **Vide par arithmétique** ⇒ partition non exhaustive de fait ⇒ mode 0-6/0-8 refusé par le PI pour `C6` (0-47) | *(redéfinie au tour 2, puis **retirée** à la gate 2 — voir §14-5)* |

**Ce qui est conservé du §14-2** : son **intention**, verbatim — *« `C5` restaure l'exactitude de la
nulle en supprimant la difficulté de la tâche »*. **Le défaut 0-45 reste acquitté, valide et
fondateur** ; c'est son **remède** qui était inexécutable, pas son diagnostic.

### 14-3 (gate 1) — **MAINTENU, ÉTENDU**
Vivier de tiges 45 → **90**. **Étendu** : vivier de suffixes **partitionné en 4 sous-viviers de
domaine** (`C7`) et **~160 items supplémentaires** pour les **40 paires** de la nulle de cadre.

### 14-4 (gate 1) — **MAINTENU, RÉORDONNÉ**
Ordre d'exécution : **§11-Q1/Q2/Q5/Q6/Q7 (Neuro) et §12-Q1/Q2/Q9 (Math) → banc complet (`E = 0`,
`fact_pairs` en échec compris) → gel du matériau → une seule mesure I2 sur les trois modèles.**
**Aucun GPU avant PASS intégral.** §12-Q8 (suffixe-ponts) court **hors du chemin critique**.

### 14-5 (PI, **gate 2 du 2026-08-23**) — retrait de la bande `M`, périmètre, `C7`, budget

**1. Bande `M` RETIRÉE.** *Position du PI, verbatim :*

> *« J'avais proposé "bande `M` ssi `ΔR1_inv` dépasse le plancher lexical à 0 forward". Cette porte
> était **nécessaire mais pas suffisante** : le plancher à 0 forward est une **borne inférieure** du
> canal lexical, pas une borne supérieure. I2 l'a mesuré — l'état encode la surface **mieux** que le
> comptage de tokens brut. Donc sous `C5`, où la cible est l'unique candidat à partage d'entité, un
> `ΔR1_inv` même au-dessus du plancher reste attribuable à un bon encodage lexical, **sans un gramme
> d'identité**. `V-plafond` filtre le cas trivial, elle ne tranche pas le canal. Un `M` franchi serait
> une **vacuité franchissable** — le miroir exact de ce qu'on a refusé pour S-7 et `C6`, et je ne peux
> pas exiger le principe là-bas et l'abandonner ici **parce que la vacuité serait cette fois en ma
> faveur**. Une bande dont le verdict confond deux états du monde n'est pas une bande, et son verdict
> dépouillé l'avoue lui-même : "le canal n'est pas tranché" est **la définition d'un descriptif, pas
> d'une décision**. Une classe atteignable au prix de ne rien affirmer, c'est la tentation du
> verdict-pour-avoir-un-verdict ; le projet a déjà payé pour savoir qu'un demi-verdict coûte plus cher
> qu'une absence. »*

  - **(a)** Le contenu informatif survit en **descriptif nommé, obligatoire, à vocabulaire
    contraint** : `ΔR1_inv` contre clé nulle appariée **et** contre plancher lexical, tous deux
    publiés (§4.4). Formulation gravée d'avance, vocabulaire interdit **(xiii)**.
  - **(b)** La partition à 3 classes passe au **banc avec un cas par classe, frontière `N-ind`
    comprise** (D18) ; la cellule **« sous la clé nulle elle-même »** garde son verdict
    **`INVALIDE-INSTRUMENT`** : *si même le lexical échoue sur `C5`, c'est l'instrument qui est
    accusé, pas le cortex.*
  - **(c)** Conséquence d'architecture gravée au §13-bis : **calibrateur / question**.

**2. Canal suffixe — hors périmètre déclaré.** *Exigence de forme du PI :* la déclaration est du type
**annulation-de-prédiction tracée**, jamais du type **ligne CADUC silencieuse**. Trois éléments
obligatoires (§4.9, porte `V-perimetre`). *« La différence entre "hors périmètre" et "angle mort" est
entièrement dans cette écriture : un angle mort est ce qu'on découvre après coup, un hors-périmètre
est ce qu'on a choisi, daté et signé avant mesure. »* §12-Q8 court **hors du chemin critique**.

**3. `C7` / A2 ADOPTÉE, avec stratification obligatoire.** *Motif du PI :* « un facteur sans porteur
physique n'est pas un facteur, c'est une étiquette ». **Amendement (défaut 0-67)** : planchers et
nulles de M1/M3 **stratifiés par domaine, jamais poolés** ; cardinal D24-b **par sous-vivier**.
*Argument de convergence retenu :* Math appuie Neuro ; les deux experts n'ont pas la même fonction
d'erreur, et leurs chemins indépendants aboutissent au même point.

**4. Dépassement de budget ACCEPTÉ** (~16-21 h d'agent, CPU < 60 s, GPU inchangé).

**5. Retrait de `§14-2` RATIFIÉ** — impossibilités démontrées, non préférences.

---

## Arbitrage

### A. Positions du PI (A-1 … A-7)

| Position | Traitement | Où |
| --- | --- | --- |
| **A-1** scission `N-a`/`N-b`/`N-ind` | **Reconduite et élargie** : trois partitions exhaustives (calibrateur, bandes de la primaire, classes ORD), chacune avec sa classe `-ind` et son cas de banc | §4.4, §4.5, §4.6, §10 |
| **A-2** S-4 lecture faible | **Reconduite, satisfaite par construction** ; **étendue** à la nulle de cadre | §4.1-C2 |
| **A-3** un type capitalisé obligatoire | **Reconduite** ; `V-casse` mord sur du réel | §4.1-C6 |
| **A-4** D24-b | **Reconduite, opérationnalisée, ÉTENDUE trois fois** : au **quasi-identique** (`V-var-dist`), au **cardinal à `t−1` = 14** (0-64), et **par sous-vivier** (0-67) | §4.1, §4.2, §4.7 |
| **A-5** bin dur descriptif | **Reconduite** ; vocabulaire interdit étendu de **cinq** entrées (ix)-(xiii) | §2, §6.E |
| **A-6** D25 exécutable | **Reconduite** ; table des garanties dans la source, `fact_pairs` contre-exemple obligatoire | §10 |
| **A-7** traçabilité des annulations | **Reconduite et exercée deux fois sur des dispositifs du camp du PI** : §14-2 (amendement du copilote) et la **bande `M`** (proposition du PI lui-même), tous deux retirés **avec** motif, auteur et remplaçant | §14-2-CADUC, §14-5, Registre |

### B. Avis Math — traitement

Intégrés : le **théorème de saturation** (structurant) ; le plancher comme **constante dérivée** ; la
**hiérarchie des nulles** ; `K_eff = 10` et l'illégitimité du clustering par famille ; l'inférence
**conditionnelle au pool gelé** ; `V-dtype` sur les **deux marges** ; le **repli fp32 nominal** ; les
**cardinaux D24-b** avec le drapeau « contraste minimal : 1 token » ; la publication en **fractions** ;
la clause `C1′` ; `V-freq` au niveau **suffixe intra-famille** ; la **nulle hypergéométrique exacte
sous E1–E4** ; `m₁` rétrogradé **et satisfait gratuitement** ; **E3 déclarée** ; `D` comme différence
observé − attendu ; la **clause de famine** ; l'antipode de **déplétion** ; l'**invisibilité du canal
suffixe** ; le **bootstrap joint** et l'interdit du produit de p-valeurs ; le **piège D24-b à `t−1`** ;
l'ordre de préférence **structurel > covariable > interdit** pour la surprise ; l'endossement de
`V-var-dist` et de la nulle de cadre en paires ; la **permutation intra-tige** comme arbitre de M1.

**Amendé** : `τ` → **0.15** (contradiction interne de ses deux chiffres, remontée §12-Q1).
**Remonté au PI plutôt que tranché** : l'option B (valeur logique vs faisabilité) ; la variante
suffixe-ponts (collision `C7`).

### C. Avis Neuro — traitement

Intégrés : **A1** (partition ORD à 5 classes) ; **M2 en porte de sanité** ; **`C7`/A2** (adoptée par
le PI) ; **A4 amendée** (`t` décisionnel, `t−1` descriptif, cardinal 14) ; **A5** dans l'ordre de
préférence de Math ; **A7** (`V-var-dist`) ; **A8** (nulle de cadre en 40 paires) ; **A3** en
descriptif avec antipode ; les **trois façades** traitées, la (b) versée au vocabulaire interdit (ix) ;
le **déclencheur observable de X5** gravé sans ouvrir X5 ; le clivage à préfixe égalisé jugé
suffisant ; les ponts jugés écologiquement acceptables sous contrainte de rédaction des gabarits ; les
40 items S0 par **appartenance lexicale déclarée** (circularité **et** dépendance au modèle) ; la
**limite de validité externe** ; l'**infaisabilité de l'option B** ; la **statistique de composition**,
devenue la primaire unique ; le **confondant de copie de token** et ses deux discriminants gratuits.

**Corrigé, re-signature due** : **M4 → `V-leak`** (`B0 ≤ 0`) — *le directeur ne re-signe pas à la place
de l'auteur* (A-7). **Déclassée** : **`P-N1`** (non-monotonicité démontrée), contestation ouverte
§11-Q4.

### D. Contradictions entre experts — tranchées

| Objet | Positions | Tranché | Raison |
| --- | --- | --- | --- |
| **Plan à deux positions décisionnelles** | Neuro : 2 (facteur) × 2 (position) ; Math : `t−1` porte des états **bit-identiques** en intra-tige | **Math** : `t` décisionnel, `t−1` **descriptif**, cardinal 14 publié | Un facteur dégénéré par arithmétique ne devient pas mesurable par intention. L'**intention** de Neuro (discriminer la copie de token) est **entièrement servie** par la version descriptive. |
| **`P-N1` / contraste inter-pools** | Neuro : prédiction signée décisionnelle ; Math : non monotone | **Math** | La non-monotonicité est **démontrée sur l'estimande**, pas argumentée. Contestation ouverte (§11-Q4). |
| **Option B** | Neuro : non faisable ; Math : secondaire mais **pas superflue** | **Remonté au PI ⇒ hors périmètre déclaré (gate 2)** | Ils ne parlaient pas de la même chose : **valeur logique** vs **faisabilité**. |
| **`m₁` constant** | Math (principal) : nécessaire ; Math (complément) : rétrogradé | **Sans objet** : `m₁ = 5` pour 60/60 **par construction** | L'affectation des tiges simples à la réserve devient **structurelle**. |
| **M4 sous `C7`** | Neuro exige `C7` **et** signe « couche 0 nulle » — incompatibles | **Correction proposée, re-signature demandée** | A-7 : on ne re-signe pas à la place de l'auteur. |

### E. Écarté

| Élément | Raison |
| --- | --- |
| **Porte `V-lex`** | 0-46 : inexécutable et ininterprétable. Retrait tracé §14-2. |
| **Primaire 2 inter-pools** | 0-49 : non monotone. Survit en descriptif. |
| **Bande `M`**, sous ses deux formes | 0-47 (vide) puis **0-66 (vacueusement franchissable)**. Retrait tracé §14-5 ; contenu informatif conservé en descriptif contraint. |
| **Bootstrap à deux voies** | Inutile une fois `K_eff = 10` par tiges. |
| **Clustering par famille** | 0-50. |
| **Agrégat binomial, produit de p-valeurs** | 0-39, 0-63 : clauses de portée C et C-bis. |
| **Bin dur en décision** | Indécidable à ce `N`. |
| **Test d'homogénéité de fréquence** | 0-41 : blanc-seing (N11 / D20). |
| **Pseudo-mots dans la nulle de cadre** | 0-42. Conservés à part. |
| **Option B** | 0-60 + décision PI gate 2 : **hors périmètre déclaré**. |
| **Variante suffixe-ponts ce tour** | Collision `C7` ; instruction §12-Q8 **hors chemin critique**. |
| **Toute réutilisation de `pool.fact_pairs`** | D25 + décision PI. Contre-exemple échouant uniquement. |
| **Repondération / sélection après lecture** | Interdit gravé ; `V-surprise`. |
| **Planchers et nulles poolés** | **0-67** (relevé par le PI avant adoption). |

---

## Registre des engagements

| Engagement | Auteur | Signature | Statut | Motif |
| --- | --- | --- | --- | --- |
| Prédiction de variante (design 2 membres, adjoint postposé) | lab-neuro | tour matériau | **ANNULÉE — non reportée** | D24-b : contraste bit-identique au point de capture ⇒ **sans objet**. |
| Prédiction ordinale `S3 > S2 > S1 > S0` (chaîne unique, un antipode) | lab-neuro | tour 1 | **CADUQUE** | Remplacée par la **partition ORD à 5 classes** (son propre A1). Défaut 0-56. |
| **M1** (`S3 > S2`) | lab-neuro | — | **NON SIGNÉE** ; conditionnée à `C7` | Sans `C7`, la permutation intra-tige est la vérité du générateur. |
| **M2** (`S2 > S1`) | lab-neuro | tour 2 | **SIGNÉE** — **porte de sanité** | Arrêt si `cos̄(S2) ≤ cos̄(S1)`. |
| **M3** (`S1 > S0`) | lab-neuro | — | **EN ATTENTE** (§11-Q2) | `C7` est adoptée ; la condition est levée. |
| **M4 → `V-leak`** | lab-neuro | tour 2, forme « couche 0 nulle » | **RE-SIGNATURE DUE** (§11-Q1) | Forme originale **falsifiée par construction sous `C7`** ; forme corrigée proposée, **non substituable sans l'auteur**. |
| **P-N1** | lab-neuro | tour 2 | **DÉCLASSÉE en descriptive** | Non-monotonicité démontrée (0-49). Contestation ouverte (§11-Q4). |
| **P-N2** (excès de composition) | lab-neuro | tour 2 | **SIGNÉE — portée en PRIMAIRE UNIQUE** ; confirmations demandées (§11-Q6, §11-Q7) | Nulle vérifiée exacte par Math sous E1–E4. **Charge accrue par le retrait de `M`.** |
| **Bandes de NLL** (`V-surprise`) | lab-neuro | — | **DUES avant génération** (§11-Q5) | Une porte sans bandes est vide. |
| **`τ = 0.15`** | lab-director (arbitrage) | tour 2 | **CONFIRMATION DUE** (§12-Q1) | Contradiction interne de Math. |
| **`ε`** (règle, pas valeur) | lab-director / Math | tour 2 | **VALIDATION DUE** (§12-Q2) | D19. |
| **Stratification par domaine** | PI | **gate 2** | **GRAVÉE** ; coût en puissance à chiffrer (§12-Q9) | Défaut 0-67, relevé **avant adoption**. |
| **§14-2** (seconde primaire inter-pools + `V-lex`) | **copilote**, gate 1 | — | **RETIRÉ, ratifié gate 2** | Deux impossibilités démontrées, avis convergents et indépendants. |
| **Bande `M`** (« `M` ssi `ΔR1_inv` dépasse le plancher à 0 forward ») | **PI**, gate 1 | — | **RETIRÉE PAR SON AUTEUR, gate 2** | *« nécessaire mais pas suffisante » ; le plancher est une **borne inférieure** du canal lexical. Vacuité franchissable — refusée **alors qu'elle aurait été en faveur de l'hypothèse**.* Défaut 0-66. |
| **Déclencheur observable de X5** | lab-neuro | tour 2 | **GRAVÉ, non ouvert** | `N-a` du calibrateur sur matériau qualifié + `V-dtype` PASS, sur ≥ 2 modèles / 3. |
| **Substitution depuis la réserve** | lab-director | tour 2 | **GRAVÉE** (`V-subst`) | Recalcul et publication de `K_eff` et `m₁` **avant** le run ; substituer une famille pontée impose de substituer **les deux familles de sa tige**. |
| **`V-leak` / M4, forme corrigée `B0 ≤ 0`** | lab-neuro | **2026-08-23, tour de re-signature** | **SIGNÉE** | Correction acceptée sans réserve : *« c'est ma deuxième prédiction tuée par un porteur mal identifié — cohérent avec 0-53, et c'est moi qui ai réclamé `C7` »*. |
| **`B0′`** (détecteur de fuite sans hypothèse) | lab-neuro | 2026-08-23 | **SIGNÉE, adoptée** | `B0` est garanti par `C7`, donc **faible** ; `B0′` est apparié en domaine et prédit **exactement 0**. |
| **M3** (`S1 > S0`), **requalifié contrôle de manipulation de `C7`**, attendu **dès la couche 0** | lab-neuro | 2026-08-23 | **SIGNÉE** | Une confirmation dit « le vivier a été correctement typé », jamais « l'état encode le domaine ». |
| **`P-N2` sous forme agrégée** (`D`, clusters = tiges, `K_eff = 10`) | lab-neuro | 2026-08-23 | **CONFIRMÉE** | *« mon "par famille" était faux pour la raison même que j'avais invoquée ailleurs : les deux familles d'une tige sont réciproquement requêtes et concurrentes »*. |
| **Charge décisionnelle unique sur `P-N2`** | lab-neuro | 2026-08-23 | **MAINTENUE ; aucun second maillon demandé** | Un second maillon devrait porter un canal que les trois faits acquis ferment (0-46, 0-49, 0-62). *« Je n'en vois aucun, et j'affirme ne pas pouvoir en dériver un — c'est un résultat, pas une dérobade. »* |
| **Bandes de NLL** (`V-surprise`) | lab-neuro | 2026-08-23 | **GRAVÉES** | 3 bandes (dérivé : 4 donneraient 2,5 tiges/bande = 0-52 rejoué), tertiles **par modèle** sur la **NLL moyenne de paire**, + superposition absolue à `4.0 nats`. Règle de **retrait de M1** si son signe n'est pas stable sur les trois bandes. |
| **`τ = 0.15`** | lab-math | 2026-08-23 | **CONFIRMÉ** | Recalcul complet ; réserve gravée sur la sous-couverture du bootstrap percentile à 10 clusters. |
| **`ε` conditionnel + `ε_max = 0.66`** | lab-math | 2026-08-23 | **VALIDÉ** | D19 respectée **si** la barrière d'information est **exécutable** ; formule, seed, `B = 10⁴` et plafond gravés. |
| **`V-joint` = enveloppe max-t** | lab-math | 2026-08-23 | **TRANCHÉ** | Ni min-p (réintroduit des p-valeurs par modèle) ni Bonferroni (ignore la dépendance capturée par le bootstrap joint). |
| **Successeur (canal suffixe) : matériau propre requis** | lab-math | 2026-08-23 | **CHIFFRÉ** | La cellule « suffixe partagé × domaine différent » est **structurellement interdite sous `C7`** ; la version intra-domaine tient sur le vivier et sur `C5` mais fait tomber **`K_eff` à 5** ⇒ tout passe sous la frontière descriptive. **Le successeur exige ~20 tiges / 40 familles** (volume ×2, même machinerie) : la variante **n'est pas greffable** sur v4 sans le vider de sa puissance. |
| **Canal suffixe hors périmètre** | PI | **gate 2** | **DÉCLARÉ, daté, signé avant mesure** | Porte `V-perimetre`, trois éléments obligatoires (§4.9). Successeur désigné. |

---

## Annexe D — Instruction du point D (reconduite)

**§D.0 à §D.4 du tour 1 sont reconduits intégralement**, avec **trois amendements** :

1. **§D.0** — le mot **« exact »** est retiré de « plan 2 × 2 exact » (0-53, vocabulaire (xi)). Le plan
   est **2 × 2 dont la dimension domaine est instanciée par sous-viviers déclarés (`C7`)**. La ligne
   « les ponts **remplissent** la cellule manquante » est reformulée (entrée (xii)).
2. **§D.1** — le tableau d'éligibles reste valide ; `C5` inchangée. **S'y ajoute** que les 4 familles à
   tige simple partent en **réserve**, fixant `m₁ = 5` sur 60/60 unités décisionnelles et
   **`K_eff = 10`**.
3. **§D.3** — le budget est **remplacé par le §9 re-chiffré** (~16-21 h agent, < 60 s CPU, GPU
   inchangé), **accepté à la gate 2**.

**Verdicts d'instruction (a)(b)(c) : inchangés.** Le repli 48/`K = 16` n'est **pas** déclenché ; son
amendement (réserve substituable avant le premier token) reste adopté inconditionnellement.

---

## Historique

- **2026-08-23** — Brouillon v4-matériel (mode cadrage), sous les 4 positions PI et les specs S-1..S-8.
- **2026-08-23** — Avis **Math RÉSERVÉ** (6 points) et **Neuro RÉSERVÉ** (5 points).
- **2026-08-23** — **Positions PI post-avis** ; **D24-b** et **D25** gravées ; consigne « rien au banc
  avant instruction du point D ».
- **2026-08-23** — **1re consolidation** : instruction du point D, `C1..C5`, plan de strates, défauts
  0-34..0-44.
- **2026-08-23** — **Gate 1 (PI)** : design 60 + 12 adopté ; vivier de tiges à 90 ; **0-45** relevé par
  le copilote ⇒ **§14-2** (seconde primaire inter-pools + `V-lex`) adopté. **D26** gravée.
- **2026-08-23** — **Avis Math et Neuro, principal + complément**, tous deux **RÉSERVÉ**, après échange
  croisé. **Convergence indépendante** : le §14-2 est inexécutable.
- **2026-08-23** — **2e consolidation** : §14-2 retiré et remplacé ; `K_eff` corrigé à **10** ; `τ`
  relevé à **0.15** ; trois partitions exhaustives ; `C7`, A4 amendée, A5, A7, A8 intégrées ; trois
  façades traitées. **Vingt défauts acquittés (0-46 … 0-65).** Budget hors épure déclaré.
- **2026-08-23** — **Gate 2 (PI)** : **bande `M` RETIRÉE par son auteur** (défaut **0-66** : le
  plancher à 0 forward est une **borne inférieure** du canal lexical ⇒ vacuité franchissable), contenu
  conservé en **descriptif contraint** + partition à 3 classes au banc + cellule
  **`INVALIDE-INSTRUMENT`** ; **canal suffixe déclaré hors périmètre** en trois éléments obligatoires
  (`V-perimetre`), §12-Q8 hors chemin critique ; **`C7` adoptée** avec **stratification par domaine**
  (défaut **0-67**, relevé avant adoption — sixième occurrence de D26) ; **budget accepté** ; retrait
  de §14-2 **ratifié**. Décision d'architecture **calibrateur / question** proposée à la gravure
  (§13-bis). **Vingt-deux défauts acquittés au total ce tour (0-46 … 0-67).**
  `Statut : PROPOSE` — **non pré-enregistrable** tant que Neuro n'a pas répondu à §11-Q1/Q2/Q5/Q6/Q7
  et Math à §12-Q1/Q2/Q9.
- **2026-08-23** — **Tour de re-signature.** **Neuro : FAVORABLE** — signe `V-leak` corrigée, M3,
  la forme agrégée de `P-N2` et la **charge décisionnelle unique** ; grave ses bandes de NLL ; ajoute
  `B0′` (détecteur de fuite sans hypothèse), la **pré-évaluation de M2 en trois états** et la
  **scission de `C-ind`** ; relève l'**issue conjointe modale** (0-70). **Math : verrous levés** —
  `τ = 0.15` confirmé, `ε` conditionnel validé avec `ε_max = 0.66` ; relève l'**absence d'exclusivité
  des partitions** (0-68) et l'**inutilisabilité structurelle de la réserve** (0-69) ; grave `k = 12`
  dans `P2`, la forme **stratifiée-combinée** de M1/M3, `V-t1`, `V-joint` en **max-t**, et chiffre le
  successeur (**matériau propre requis**). **Six défauts de plus (0-68 … 0-73)** — total du cycle :
  **vingt-huit (0-46 … 0-73)**, aucun octet de matériau généré, aucun GPU touché.
  `Statut : PROPOSE` — **tous les verrous d'expertise sont levés** ; reste la gate de
  pré-enregistrement du PI, puis le banc.
- **2026-08-23** — **Gate de scellement (PI).** Trois vérifications exigées avant autorisation de
  pré-enregistrer : **deux échouent, une à moitié**. **0-74** (famine partielle : le support de `D`
  rétrécissait silencieusement — « la clause NaN en arithmétique exacte ») ; **0-75** (`C−` était une
  cellule nommée sans suite de chantier — la prédiction dont le renversement vaut le plus était la
  moins écrite) ; **0-76** (partition des ordres incohérente : `ORD-4` périmée, `0-résolu` partageant
  le verdict d'ORD-3 malgré un diagnostic opposé, zéros de M1/M3 non résolus). Correctifs intégrés :
  porte de support `K_eff^support ≤ 8 ⇒ C-ind`, quatre conséquences de chantier pour `C−`, **classe
  `ORD-0` créée**, quatre états par maillon, espace résolu restauré à **27 cellules exactement**.
  **Total du cycle : trente et un défauts (0-46 … 0-76)**, toujours aucun octet de matériau, aucun
  GPU.
- **2026-08-23** : proposé.
