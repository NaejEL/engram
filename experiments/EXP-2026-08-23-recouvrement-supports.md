# EXP — recouvrement des supports de `topk(G·h)`

Statut : PRE-ENREGISTRE

*Consolidé par `lab-director` le 2026-08-23, sous avis **Math FAVORABLE** (verrous levés, M1…M9) et
**Neuro FAVORABLE sous une condition bloquante** (N1…N8). Amendé à la gate PI du 2026-08-23 (§14).*

> **Déclaration en tête, exigée par M8, à recopier dans le rapport** : ce protocole exige
> `IC_inf > ε*` (et non `> 0`) sur son unique quantité décisionnelle de centrage. **Cela double le
> seuil de détection**, de ~1.96·se à **~3.9·se** — homologue exact du durcissement 0-83
> (« ~26 % → ~52 % ») du cycle v4. **C'est le prix d'un verdict lourd, il est déclaré avant la
> mesure et il ne sera pas renégocié après.**

---

## 0. Défauts acquittés

**0-1 … 0-103** : reconduits sans changement (patrimoine du projet, cycles antérieurs et v4).
**0-104 … 0-116** : les treize du brouillon de ce cycle, dont quatre critiques, reconduits inchangés.

### 0-117 … 0-134 — Relevés au tour de consolidation

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-117** *(critique)* | **Le leave-one-out ne suffit pas à débiaiser le cosinus centré** (Math M4). Le centrage naïf porte un biais **négatif** ~ −2/n ; le LOO-paire le retire, mais **les deux états d'une paire partagent l'erreur d'estimation** `ε = μ̂ − μ`, `E‖ε‖² ≈ tr(Σ)/(n−2)` ⇒ **biais positif résiduel ~ +1/(n_cell − 2)**, soit **≈ 0.05 à `n_cell ≈ 20`**. | **Du même ordre que le corridor traduit en cosinus** — c'est-à-dire de la taille de la quantité décidée. Un biais de la taille de l'effet n'est pas une correction de second ordre : **il fabrique `c-cent`**. Le brouillon posait « LOO » là où il fallait un majorant publié et une seconde mesure. Correctif : `n_cell` et `1/(n_cell−2)` publiés ; **split-half en double mesure (D26)** ; **`min n_cell < 30` ⇒ le split-half devient la mesure principale** (ratifié §14-2). |
| **0-118** *(critique)* | **`V-G` bit-à-bit était impossible par construction.** `A3` a été calculé dans le pipeline **fp32** (`hippocampus.phi`) ; la mesure de ce cycle est en **fp64**. La porte exigeait la reproduction bit-à-bit de `A3` **par la nouvelle mesure**. | **La porte allait échouer pour une raison parfaitement légitime** — mode **0-82** exact, à l'étage du dtype. Un échec non diagnostique consomme un tour et discrédite l'instrument sain. Correctif : `V-G` reproduit `A3` **en fp32, sur le chemin d'origine** (bit-à-bit) ; la mesure est ensuite en fp64 ; **deux nombres, écart publié** (D26). |
| **0-119** *(critique)* | **Le corridor `2n` était invalide à l'échelle où le brouillon l'appliquait.** `\|A∩B\| ~ Hypergéom(8192,64,64)` : `P(0)=0.604`, `P(1)=0.303`, `P(2)=0.076` ⇒ **`q95` par paire = 2 indices, soit le DOUBLE du corridor**. Le corridor ne domine le `q95` que sur des **moyennes de ≥ 8 paires**. | **Une paire isolée dépasse le corridor 7.6 % du temps sous la nulle.** Le brouillon autorisait la lecture de `Λ` par paire : **une paire au-dessus du corridor aurait été lue comme un recouvrement**, à un taux de faux positif deux ordres au-dessus de l'α déclaré. Correctif : porte **`V-P8`**, bloquante. |
| **0-120** *(critique)* | **Ni `f(s)` (énergie de l'intersection), ni `σ±(s)` (accord de signe), ni le couple `(O, cos)` conjoint n'étaient publiés** (condition bloquante Neuro). | **Le cycle ne pouvait pas donner tort à son propre expert.** Sans `f`, l'état *« les supports coïncident »* et l'état *« quelques coordonnées géantes portent tout le cosinus »* rendent **le même `O`, interprété de la même façon** — et le second est précisément la branche de tort de Neuro (N7). **Un dispositif dont aucune donnée ne peut réfuter l'expert qui le signe n'est pas un dispositif.** Correctif : porte **`V-diag`**, bloquante pour le rapport. |
| **0-121** *(majeur)* | **Façade non repérée : « le noyau commun est le code de fond du gyrus denté »** (Neuro). **Propriété manquante** : dans le DG, la fraction stable est une propriété **du réseau qui a appris ces environnements** ; ici `Core` serait une propriété de **`G` composée avec l'anisotropie du flux résiduel** — **deux objets dont aucun n'a vu le matériau**. | **Même mode que 0-59 et que la façade Hasselmo** : l'étage mesuré n'est pas l'étage dont on parle. **Plus dangereuse que 0-59** : `Core` est la quantité du cycle dont **la nulle est la plus écrasante** (`p < 10⁻¹³`) — un chiffre statistiquement inattaquable aurait porté une interprétation entièrement fausse. Correctif : **`Core-G`** (une ligne d'arithmétique) + vocabulaire interdit (xix). |
| **0-122** *(critique)* | **L'ordinal soumis à signature était l'ordre du DESIGN (`S3>S2>S1>S0`), déjà démenti par le brut de v4** (`M1 +`, `M2 −`, `M3 +` sur 3/3 ⇒ `S3>S1>S2>S0`, domaine dominant la tige d'un facteur 2 à 10, régime **additif**). | **Faute de provenance (D14-R) doublée d'une faute de D13** : demander à un expert de signer une prédiction **que le projet a déjà mesurée fausse**, dont l'antipode aurait été « confirmé » par une mesure antérieure au run. Correctif : ordinal du design **retiré** ; remplacé par `P-N` (niveau, décisionnelle) et `P-N-ord` (ordre du **cosinus brut**, descriptive). |
| **0-123** *(majeur)* | **La condition `auto` manquait** (Neuro N1-bis). `LayerNorm` **centre** ; `RMSNorm` **ne centre pas** (Zhang & Sennrich 2019). | **`G` est appliquée à un vecteur que le cortex lui-même n'utilise jamais sous cette forme, et pas de la même manière selon le modèle** — et le brouillon n'avait aucun moyen de le voir. Conséquence : **`C-mod` était une classe sans cause candidate**, c'est-à-dire une classe qui se consigne et ne s'explique jamais. `auto` est en outre **le seul centrage du cycle dont l'opération soit disponible en ligne**. Correctif : cinquième niveau + porte **`V-norm`**. |
| **0-124** *(majeur)* | **`Core` était trivial sans la contrainte « une unité par tige »** (Math M9). Deux états intra-tige sont **corrélés** et brisent la nulle binomiale. | La signature de `Core` vaut `p < 10⁻¹³` **sous indépendance**. Avec deux états d'une même tige dans `S`, `Core > 0` devient **attendu** — et le chiffre le plus spectaculaire du cycle aurait été un artefact d'échantillonnage. Correctif : porte **`V-core-S`**. |
| **0-125** *(majeur)* | **La porte ULP n'était écrite que pour les états bruts** (Math). | Les états **centrés** et **placebo** sont précisément ceux dont les coordonnées sont **rapprochées de zéro par soustraction** ⇒ **les plus exposés** au basculement de rang à la coupure `topk`. La porte manquait **exactement là où le risque est maximal**. Frère de **0-51**. Correctif : `V-ulp` sur les **cinq** conditions. |
| **0-126** *(majeur)* | **Le budget reposait sur deux calculs faux, et `R` était choisi à la main.** (i) Le MC conditionnel de `V-iid` **n'a pas d'objet** (`G` bâtie par `torch.randn` seedée ⇒ lignes iid gaussiennes **par définition**, résultat **exact**). (ii) Le dépassement « placebo » supposait un **recalcul complet de `G·h` par direction** ; l'identité `G·(h−c·r) = z − c·(G·r)` le réduit à **un matvec + une soustraction rang-1 + un topk**, avec `Z` en cache (**51,9 Mo/modèle**). (iii) `R` était posé, non dérivé. | **Un protocole qui demande un budget pour re-prouver une propriété de provenance dépense pour ne rien apprendre** — et un budget surévalué d'un ordre est un motif de refus injustifié à la gate. `R` choisi à la main est **0-52 rejoué**. Correctifs : les deux postes **supprimés**, §9 re-chiffré ; **`R = ⌈10·σ̂_dir²/σ̂_Δ²⌉`** ; échec de `V-G` = **arrêt de provenance**, jamais repli. |
| **0-127** *(majeur)* | **Le centrage était désigné par des termes biologiques** (Neuro N1). Trois candidats écartés par **propriété manquante nommée** : inhibition tonique (déplace un **seuil**, sans direction) ; normalisation divisive (Carandini & Heeger : **multiplicative**, **en ligne**) ; retrait de mode commun par interneurones (poids **appris** ⇒ viole D8/D9). | **La propriété qu'aucun circuit ne possède** : `μ_type` est estimé **en leave-one-out sur un corpus dont une partie n'existe pas au moment du calcul — un neurone n'a pas d'ensemble de validation**. Sans cette entrée, `c-cent` réalisée aurait été lue *« le cerveau fait ça »* et aurait ouvert un chantier de mécanisme sur un **post-traitement statistique** (Mu & Viswanath 2018). **Quatrième façade tuée par la même méthode**, après Hasselmo, 0-59 et 0-55. Correctif : entrée **(xviii)**. |
| **0-128** *(majeur)* | **`L(c)` n'existait qu'en version THÉORIQUE**, dépendant d'une hypothèse d'ordre-statistique, et `L(0.46)` tombait **à cheval sur la frontière `p = 10/11`** compte tenu d'une marge de ~1 % sur la masse totale. | **Une borne décisionnelle dont la valeur dépend d'un arrondi à 1 % n'est pas une borne**, et une porte assise sur une hypothèse distributionnelle a une nulle de plus à défendre. Correctif (Math) : **`V-borne` devient la version RÉALISÉE par paire** — `p_pair ≥ min{p : Σ des p plus grands φ_i² réalisés ≥ cos²_pair}` — **zéro hypothèse**, identité arithmétique, toute violation = **bug** ; **la table théorique devient la prédiction pré-enregistrée que la réalisée doit encadrer** (D14-R + D26). |
| **0-129** *(majeur)* | **La tension entre `P-N` (niveau `O_type ≳ 10/64`) et la prédiction de centrage (effondrement sous `type`) n'était écrite nulle part** — pas plus que le fait que **le pari du PI** (`O_type` **sous `2n`**) en est l'un des deux termes. | **Deux engagements signés du même expert n'étaient conjointement satisfiables que sous une condition (effondrement PARTIEL) que personne n'avait écrite** ; et l'issue qui les départage est **exactement le pari du PI**. Sans cellule gravée, `O_type` sous le corridor aurait été lu soit « le centrage a marché », soit « pas de recouvrement » — **au choix du lecteur**. Mode **0-56/0-70**. Correctif : cellule **`(BAS × c-cent)`** nommée, lecture écrite d'avance, **ratifiée par le PI avant mesure** (§14-1). |
| **0-130** *(majeur)* | **La capture à `t−1` était encore proposée** (Neuro N4). Deux raisons indépendantes, chacune suffisante : (i) `A4` a établi que le porteur du domaine **s'évanouit** à `t−1` ; (ii) **`V-t1` l'interdit structurellement** — `S3` et `S2` sont **définies** par le partage de tige, et leur recouvrement à `t−1` vaudrait **`64/64` par arithmétique** (états bit-identiques, 0-64). | Deux des quatre strates auraient rendu **la valeur maximale par construction**, et le chiffre le plus élevé du cycle aurait été un **artefact de définition**. **Mode 0-34/0-64 à la position ajoutée.** Correctif : `t−1` **refusé, non reporté** ; réouverture = ensemble strictement inter-tige (cardinal 14) et protocole propre. |
| **0-131** *(mineur, porte de lecture)* | **La détection étouffée par le corridor n'était ni bornée ni déclarée** (Math M6). | Sous l'alternative, un effet entre `ε*` et `2n` est **significatif et classé négligeable** — une **détection étouffée**, produite **en silence**. Correctifs obligatoires : (i) **si `ε* ≥ 2n`, le recouvrement des conditions est vide et l'ordre sans objet — le déclarer** ; (ii) sinon, **publier** `P(IC ⊂ corridor ∧ IC_inf > ε*)` sous l'alternative. *Étouffement assumé, jamais silencieux.* |
| **0-132** *(mineur, portes exécutables)* | **Seed et générateur des `R` directions non gelés ni publiés** ; `O` publié en **flottants tronqués** au lieu de **fractions exactes `p/64`**. | Le seed non gelé rend le placebo **non reproductible** (D24 violée) ; `O` en flottant tronqué **détruit l'exactitude arithmétique** dont dépendent `V-borne` (identité par paire) et `Core` (comptage entier). **Deux portes exécutables assises sur une représentation qui ne les supporte pas.** |
| **0-133** *(mineur — relevé du copilote sur une clause de Math)* | **« Les mêmes `R` directions pour les 4 strates et les 3 modèles » est arithmétiquement inexécutable** : `d` vaut **768** (GPT-2), **960** (SmolLM2), **1536** (Qwen) — **un vecteur de ℝ⁷⁶⁸ n'est pas un vecteur de ℝ¹⁵³⁶**. | La clause aurait produit un **faux PASS** de la porte de cardinal, ou un échec d'implémentation à chaud. **Famille « cardinal périmé », cinquième occurrence** (0-76(i), 0-86, 0-94, 0-101, celle-ci). Correctif : *« même seed, même générateur, même règle de tirage et mêmes indices ; l'appariement inter-modèles se fait par la **norme**, jamais par le **vecteur** »*, cardinal publié **par modèle**. |
| **0-134** *(mineur)* | **Les paires intra-tige n'étaient pas exclues explicitement**, et le cardinal exclu n'était pas publié. | Deux états d'une même tige sont **corrélés** : leur présence gonfle `O` **sans mécanisme** et brise l'indépendance sur laquelle repose l'IC bootstrap de tiges. Sans publication du cardinal exclu, la contamination est **invérifiable après coup**. Correctif : `V-t1` étendue. |

### 0-135 … 0-142 — **Relevés par le RUN lui-même**, aux deux premières portes, en 9,1 s de CPU

*Ces défauts n'ont pas été trouvés par relecture : ils ont été trouvés **par l'exécution des portes
de provenance**, avant toute mesure. `V-G` et la clause §6.I ont mordu chacune sur ce qu'elles
avaient été écrites pour attraper. Tous ont été **reproduits indépendamment par `lab-verifier`**,
qui a réécrit un troisième chemin de calcul.*

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-135** *(critique)* | **`A3` n'a JAMAIS été calculé avec la `G` du projet.** `eval/materiel_v4.py::descriptif_A3` tire `G` par `numpy.random.default_rng(0).standard_normal`, `engram/hippocampus.py::phi` par `torch.randn(generator=manual_seed(0))`. **Deux matrices différentes** : `corrcoef ≈ 0.011`, `max\|G_np·√d − G_torch\| = 7.23`. Isolement causal vérifié : substituer **la seule** `G` torch reproduit les douze écarts à l'identique ; l'échelle `1/√d` laisse `~1e−10`, le dtype `~1e−9`, la coupure seuil-vs-`topk` **`0.0` exact sur 4/4**. | **Le descriptif `A3` publié au journal du 2026-08-23 décrit une projection que `φ_dg` n'utilise pas.** Sa conclusion peut survivre comme énoncé sur **les projections top-k aléatoires en général** — elle **n'est pas** une mesure de `φ_dg`, et le journal la présente comme telle. Tout ce qui en dérivait dans ce protocole (`L(c)`, le retrait de `C-sep`, la migration de la décision vers le centrage) était **fondé sur une autre matrice que celle du projet**. |
| **0-136** *(critique)* | **Le §3, ligne 1, nommait la mauvaise colonne.** « `A3` : **cosinus de `φ(h)`** — plancher ≈ 0.41-0.46, étendue 0.01-0.03 ». Relecture des bruts : **0.41-0.46 est la colonne `compression`** (`cos_brut − cos_dg`, 0.4076-0.4659) et **0.01-0.03 est `descriptif_A3_amplitude`** (0.0063/0.0121/0.0258). Le vrai `cos_dg` vaut **0.159-0.340**, d'étendue **0.064-0.091** — soit **~6 × l'étendue annoncée**. | **C'était la SEULE ligne décisionnelle du §3** — celle qui « fonde `L(c)` » — et **la seule fausse sur neuf**. Le §3 est la table de provenance **D14-R** : son objet même est d'empêcher qu'un chiffre soit repris sans être relu. **Il a été repris sans être relu, dans la table écrite pour l'interdire.** |
| **0-137** *(critique)* | **La prédiction pré-enregistrée du §4.2 en dérivait**, et **l'ensemble qui la conditionne est VIDE** : « sur les strates dont `A3` donne `cos ∈ [0.41, 0.46]` » — **aucune des douze cellules n'y tombe**. | La prédiction gelée annonçait `O ≥ 8 à 11 / 64`, soit 16 à 22 × le hasard ; la valeur correcte est **`O ≥ 1 à 6 / 64`**, soit 1 à 6 × le corridor. **Facteur d'erreur de 8× à 1.3×.** Et une prédiction conditionnée à un ensemble vide n'est **ni vraie ni fausse** : elle est **sans objet** — la forme la plus silencieuse de la vacuité (famille 0-47/0-66), dans une section **gelée**. |
| **0-138** *(majeur)* | **Le défaut 0-118 était faux DEUX FOIS.** Il gravait : *« `A3` a été calculé dans le pipeline **fp32** (`hippocampus.phi`) »*. (i) **Mauvais module** — c'est `descriptif_A3`. (ii) **Mauvais dtype** — sous **NEP 50** (numpy 2.5.2), `rng.standard_normal(...).astype(np.float32) / np.sqrt(d)` divise un tableau float32 par un scalaire `np.float64` et **repromeut en float64** ; `Z` est donc calculé en **fp64**. | **Une porte gravée sur un motif faux.** `V-G` a néanmoins **fonctionné** — elle a détecté la vraie divergence — mais **pour une raison que son texte ne nommait pas**. Fait de méthode à retenir : *une porte peut être juste et son motif écrit être faux ; c'est l'exécution qui départage, jamais la relecture.* |
| **0-139** *(majeur)* | **Le chiffre mal nommé s'était propagé à quatre autres endroits** : §4.9-2 (le gate `keysim` — **les deux termes faux**, offset réel 0.147-0.337 et dynamique réelle 0.064-0.091) ; §2 branche de tort N7 ; §4.6 cellule `BAS × c-mec` (seuils `f ≈ 0.45`, `O_brut ≈ 4-6/64` calibrés dessus) ; **et le pari du copilote au Registre**, dont le motif invoquait « la constance du plancher 0.41-0.46 sur trois architectures ». | **Un chiffre faux dans une table de provenance ne reste pas dans la table** : il devient un seuil, une branche d'antipode, un motif de pari. Quatre propagations à partir d'une seule ligne. **Le copilote a posé un pari daté sur une colonne qu'il n'avait pas vérifiée** — et c'est lui qui avait exigé la clause §6.I qui l'a démasqué. |
| **0-140** *(mineur)* | **La hiérarchie causale n'était pas écrite** : le compte rendu présentait les quatre différences de code comme si elles concouraient à l'échec. | **Une seule (le tirage de `G`) porte l'écart** ; les trois autres sont inertes à `10⁻⁹` près. Sans hiérarchie écrite, le correctif aurait pu porter sur la mauvaise différence — par exemple « épingler le dtype », qui n'aurait **rien** changé. |
| **0-141** *(mineur)* | **Option CLI `--layer` déclarée mais inerte** (`COUCHE_REF` prime inconditionnellement). | Sans effet ici (étage de mesure fermé), mais c'est un **paramètre en dur déguisé en option** — à corriger **avant** l'ouverture de l'étage de mesure. |
| **0-142** *(mineur)* | **§3, ligne « coût GPU »** : « VRAM Qwen 4,688 Gio » sans qualifier **alloué / réservé**, là où le journal distingue 4.188 alloués et 4.688 réservés. | Imprécision de provenance, sans effet décisionnel — mais **le §7 grave « VRAM rapportée en réservé »**, et une table de provenance qui ne qualifie pas son propre chiffre affaiblit la clause. |

### 0-143 … 0-145 — Relevés à la reprise : un défaut de protocole, deux trouvailles du banc

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-143** *(critique — défaut de PROTOCOLE)* | **La clause `V-t1` était AUTO-CONTRADICTOIRE avec le §7 et le §4.2.** Elle gravait *« exclusion de toute paire intra-tige »* ; or **`S3` et `S2` sont DÉFINIES par le partage de tige**. Prise à la lettre à `t`, elle **vide deux des quatre strates** — celles que le §7 mesure et que la table `L(c)` du §4.2 chiffre cellule par cellule. Et **son motif d'origine (0-134) était doublement faux** : (a) l'argument « états corrélés, bit-identiques » vaut à **`t−1`**, pas à `t` — à `t` deux unités d'une même tige **diffèrent par le suffixe, qui EST le token de capture** ; (b) l'argument « brise l'indépendance sur laquelle repose l'IC bootstrap de tiges » est **à l'envers** : le bootstrap **par tige** est précisément ce qui **absorbe** la corrélation intra-tige — c'est tout l'objet de 0-50. | **Une porte dont la lecture littérale rend le protocole inexécutable** — et qui a été gravée avec un motif inversé. Relevée par `lab-builder` **sans être tranchée par lui**, ce qui était la bonne conduite : il a implémenté la lecture qui laisse §7 et §4.2 exécutables, l'a **déclarée dans le code et dans le rapport de banc**, et a demandé l'arbitrage. Correctif : **portée en trois membres** — `t−1` interdit, `Core(S)` à une unité par tige, **et les paires intra-tige conservées à `t`**, cardinal publié. *Corrigée sous **D30 alinéa 1** : contradiction visible **sur le papier**, aucune donnée du run nécessaire.* |
| **0-144** *(majeur — trouvaille du BANC, code)* | **Le corridor était 64 × trop large.** `N_HASARD` valait `k²/D = 1/2` — l'unité **indice** — là où `O` est publié en **fraction `p/64`** (0-132), donc `n = 0.5/64 = 1/128` et `2n = 1/64 = 0.015625`. *(Le protocole était correct et cohérent : §4.1 et §7 donnent bien `2n = 1 indice = 0.015625` en fraction. C'est l'implémentation qui a pris une unité pour l'autre.)* | **`BAS` et `c-mec` sortaient vraies presque partout, et `ind_L`/`ind_Δ` étaient INATTEIGNABLES.** C'est-à-dire : la classe d'équivalence devenait **triviale** et la classe d'indécision **vide** — le mode de vacuité 0-47/0-66, mais **dans les deux sens à la fois**. **Détecté uniquement parce que le banc ÉNUMÈRE le cardinal** : 3 cellules atteintes au lieu de 12, 7 objets au lieu de 10. *Aucune relecture n'aurait vu qu'un facteur 64 se cachait entre deux unités toutes deux légitimes ; le comptage, lui, l'a vu en 2 s.* |
| **0-145** *(mineur — trouvaille du BANC, code)* | **`V-seed` faisait un test de VÉRITÉ au lieu d'un test de PRÉSENCE** : elle rejetait `seed = 0` comme « absent ». | **La porte refusait la valeur même du protocole** (`cfg.seed = 0`, D9). Un `seed` légitime aurait rendu le placebo « non reproductible » et **retiré `Δ*` et `ε*`** — c'est-à-dire **la primaire décisionnelle du cycle** — pour un `0` pris pour un `None`. Famille des portes qui échouent pour la mauvaise raison (0-118, 0-82). |

**Total du cycle : quarante-deux défauts (0-104 … 0-145), dont douze critiques. Aucune mesure
conduite, aucun GPU touché — 5,9 s de CPU au total.**

---

## 1. Question

Les supports de `topk(G·h)` — les **ensembles d'indices** retenus par la projection gyrus denté, à
`k = 64` sur `D = 8192` — se recouvrent-ils entre états du cortex **au-delà du tirage indépendant**
(`n = k²/D = 0.5 indice`) ; et ce recouvrement survit-il au retrait d'une composante affine estimée
hors ligne, **au-delà de ce qu'un placebo apparié en norme reproduit mécaniquement** ?

## 2. Hypothèse

`H_sup` : **sur les états du matériau qualifié v4, capturés à `t`, les supports de `topk(G·h)`
partagent en moyenne un nombre d'indices très supérieur au tirage indépendant (`≳ 10 sur 64` contre
`0.5`), ce recouvrement est quasi indépendant de la strate, et il est porté par une composante
affine commune que le retrait de `μ_type` supprime au-delà de ce qu'un placebo apparié en norme
reproduit.**

**Antipodes explicites (D13), énumérés :**

- **`¬P-N` (`C-strat`)** : sur ≥ 2 modèles/3, l'IC de `Λ(s) = O_type(s) − 2n` contient 0 sur ≥ 1
  strate, **ou** l'étendue inter-strates dépasse la résolution intra-strate ⇒ `topk(G·h)` porte
  **une structure de strate que le cosinus avait comprimée** ⇒ **la lecture forte d'`A3` tombe**.
- **`c-mec`** : `O_plac` chute autant que `O_type` ⇒ **le centrage n'a rien démontré** — *et il faut
  le dire dans ces mots exacts, sans habillage*.
- **Branche « l'estimateur, pas l'objet » (N7)**, *chiffres rectifiés le 2026-08-26 (§15)* :
  `O_brut` proche de son plancher `L` **avec** `f` élevée ⇒ les supports **sont** largement séparés,
  le cosinus (**0.147 à 0.337 avec la `G` du projet**, et non 0.41-0.46 qui était la *compression*)
  est porté par **une poignée de coordonnées géantes**, **la lecture forte d'`A3` est fausse**, et le
  coupable est l'anisotropie de **l'entrée**, pas `G`. *La forme de la branche est inchangée ; seules
  les valeurs numériques qui l'illustraient l'étaient à tort.*
- **`c-anti`** : `IC_sup(Δ*) < −ε*` — **se consigne et ne s'interprète pas**, mais **oblige** trois
  descripteurs (§4.6).

**Ce que ce cycle ne touche pas — phrase de Neuro (N5), à recopier dans le rapport :**

> « Ce cycle ne touche ni la loi 2 ni **D11**. D11 porte sur le **dommage d'une lecture injectée**,
> évalué par position en fonction de l'incertitude du cortex. Ici aucune lecture n'est injectée, `M`
> n'est jamais instanciée, aucune NLL n'est modifiée : **la quantité sur laquelle D11 se prononce
> n'existe pas dans ce run**. Toute phrase reliant le recouvrement des supports au gating de lecture
> est le **glissement d'étage 0-59**, d'un cran plus haut. »

### Vocabulaire interdit (§2, étendu)

**Reconduits** : (i)-(viii) d'I2 ; (ix)-(xiii) de v4 ; le **bin dur** sous toute forme, y compris
adverbiale ; **(xiv)** toute mention de l'étage d'**ÉCRITURE** ; **(xv)** le mot **« sémantique »** ;
**(xvii)** **aucun énoncé d'effet aval** (E1, E2, E3, gate, capacité) tiré d'une quantité de ce cycle.

**Ce tour :**

- **(xvi) — phrase gravée, celle de `lab-neuro`, à recopier telle quelle** *(celle du Directeur est
  retirée)* :
  > « Sur les états de **\<modèle\>**, capturés à **\<locus\>**, dans la cellule **\<type\>**, les
  > supports de `topk(G·h)` partagent en moyenne **X indices sur 64** (médiane **X̃**, IQR
  > **[·,·]**), contre **0.5 indice** sous tirage indépendant (`k²/D`), et l'intersection porte
  > **f = ·** de l'énergie avec un accord de signe de **σ± = ·**. Cette phrase porte sur l'étage de
  > **LECTURE** de `φ`, sur ce locus et ce matériau seulement ; elle ne porte **ni** sur le chemin
  > d'écriture de `M` (jamais instanciée ici, +57 % d'E2 acquis et intact), **ni** sur un quelconque
  > effet aval (E1, E2, E3 non mesurés). »

  *Contraintes portées : **compte d'indices** et non fraction ; **médiane et IQR** publiées (une
  moyenne mélange un mode « noyau » et un mode nul) ; `f` et `σ±` **dans la phrase même** ; **modèle
  et cellule nommés, jamais poolés** (0-63, 0-67).*

- **(xviii) — le centrage** *(entrée rédigée par `lab-neuro`, verbatim)* :
  > Toute désignation biologique du centrage est interdite : « inhibition tonique », « normalisation
  > divisive », « retrait de mode commun par les interneurones », « le centrage est ce que fait le
  > gyrus denté ». Le centrage est un **instrument statistique** ; formulation licite maximale :
  > *« retrait d'une composante affine estimée hors ligne, en leave-one-out, sur les états du même
  > type »*. Il n'a **aucun analogue en ligne** et n'est proposable comme mécanisme dans aucun cycle
  > futur sous cet habillage.

- **(xix)** — *« le noyau commun est le code de fond du gyrus denté »*, sous toute forme (0-121).
  Formulation licite : *« ensemble d'indices retenus par `topk(G·h)` sur ≥ 9 des 10 unités de `S`,
  pour cette `G` (seed 0) »*.
- **(xx)** — *« le centrage répare `φ` »*, *« remède validé »*, et toute formulation où `c-cent`
  licencie `I3` (N8). **`c-cent` licencie un locus, un matériau.**

## 3. Ce que le projet sait déjà — provenance (D14-R)

| Fait | Chiffre | Source, date | Étiquette |
| --- | --- | --- | --- |
| ~~`A3` : cosinus de `φ(h)` entre états, par strate~~ **RECTIFIÉE le 2026-08-26 — voir §15** | ~~plancher ≈ 0.41-0.46, étendue 0.01-0.03~~ ⇒ **ces chiffres sont la colonne `compression` (`cos_brut − cos_dg`) et l'amplitude de compression, PAS le cosinus.** Le `cos_dg` d'`A3` vaut **0.159 à 0.340**, étendue **0.064 à 0.091** | descriptif `A3`, run v4, journal 2026-08-23 | **FAUSSE — défaut 0-136.** Et le descriptif lui-même **n'a pas été calculé avec la `G` du projet** (défaut 0-135) : il ne peut donc **plus** fonder `L(c)`. |
| **`cos` de `φ(h)` par strate, mesuré AVEC LA `G` DU PROJET** (`hippocampus.phi`, `torch.randn(8192, d, generator=seed(0))`) | gpt2 **0.2766 / 0.2070 / 0.2479 / 0.1886** (S3/S2/S1/S0) · SmolLM2 **0.3368 / 0.2566 / 0.3174 / 0.2568** · Qwen **0.2184 / 0.1636 / 0.1883 / 0.1471** — **plage globale [0.1471, 0.3368]** | **`V-G` de ce cycle**, 2026-08-26, `experiments/results/recouvrement-supports/V-G.json` | **mesuré ici, par le run lui-même** ; reproduit indépendamment par `lab-verifier` via un troisième chemin. **C'est cette ligne qui fonde `L(c)` désormais** (§4.2). |
| Géométrie mesurée **ordonnée `S3 > S1 > S2 > S0`**, domaine dominant la tige d'un facteur **2 à 10**, régime **additif** | `M1 +`, `M2 −`, `M3 +` sur 3/3 | v4, 2026-08-23 | vérifié — **tue l'ordinal du design** (0-122), fonde `P-N-ord` |
| `X7` : lecture **sans composante directionnelle** (`cos ≈ 0`), aplatissement = coût **fixe** (+0.141 nats) | — | X7, journal | vérifié — la direction quasi constante de la lecture est un **invariant du modèle** (Q-01) |
| `Q-01` : le ciblage des positions incertaines est **générique** (bruit de norme appariée, `R ≈ 0.8`) ; **D11** | corr **+0.394** | X8.1b / P5 / Q-01, 2026-08-21 | vérifié — **modèle de raisonnement du placebo de ce cycle** |
| `X1` : DG **+57 % sur E2** | +57 % | ablations v1.1 | **chiffre acquis, chemin d'ÉCRITURE** — ce cycle porte sur la **lecture** et **ne le touche pas** |
| `G` est bâtie par `torch.randn(8192, d, generator=seed(cfg.seed))`, **sans `1/√d`** | — | `engram/hippocampus.py`, lecture de code | **provenance** — fonde `V-iid` (M3) ; l'absence d'échelle est **sans effet** (`topk` et la normalisation sont invariants par échelle globale) |
| Coût GPU de l'instrument v4 | **53,82 s** / 12 forwards / 240 séquences ; VRAM Qwen **4,688 Gio** (fp16) | I2, 2026-08-23 | re-mesuré — **seul poste GPU possible ici, conditionnel** (`V-cache`) |
| Les états bruts de v4 ont été **relus indépendamment** par `lab-verifier`, qui a recalculé **toutes** les quantités publiées sans écart | 193 Mio, `raw/etats-*.npz` | v4, 2026-08-23 | vérifié par exécution — **les fichiers existaient et étaient lisibles** ; **leur hash n'a jamais été pré-enregistré** (limite nommée, `V-cache`) |
| Deux fois de suite, la porte qui trouve le défaut est celle **qui n'a besoin d'aucune donnée** | `V-slot`, `V-ident` | cycle D14-ext, 2026-08-22 | vérifié — **quatrième occurrence ce tour** : `V-t1` tue `t−1` (0-130) et `V-borne` réalisée tue la frontière `p=10/11` (0-128), **sans un octet** |

## 4. Prédictions

*Toutes décidables **sans GPU** sous `V-cache` PASS, en arithmétique sur des ensembles d'indices.
Une prédiction fausse est un **résultat**, obtenu pour moins d'une demi-heure de CPU.*

### 4.1 Quantités

| Symbole | Définition | Unité de publication |
| --- | --- | --- |
| `n` | recouvrement sous tirage indépendant `= k²/D = 4096/8192` | **0.5 indice** (fraction `0.5/64`) |
| `O(s, x, m)` | `\|A ∩ B\|` moyen sur les paires de la strate `s`, condition `x`, modèle `m` | **fractions exactes `p/64` agrégées** (0-132) |
| `Λ(s)` | `O_type(s) − 2n` | idem, **moyennes de ≥ 8 paires seulement** (0-119) |
| `Δ*` | `O_plac − O_type`, **apparié par paire** | idem — **primaire (D16)** |
| `f(s)` | `Σ_{i∈A∩B} a_i² / Σ_{i∈A} a_i²`, moyennée sur les deux membres | fraction |
| `σ±(s)` | `#{i∈A∩B : sign(a_i)=sign(b_i)} / \|A∩B\|` — **nulle = 0.5** | fraction |
| `Core(S)` | indices retenus sur **≥ 9 des 10 unités de `S`** (une par tige, `V-core-S`) | **compte entier** |
| `Core-G` | `\|Core(S) ∩ top64(G·μ_global)\| / \|Core(S)\|` | fraction |
| `p_pair` | `min{p : Σ des p plus grands φ_i² réalisés ≥ cos²_pair}` | compte entier |

### 4.2 `L(c)` — la borne, en deux versions (D26)

**Dérivation (Math M1)** : `cos = Σ_{A∩B} φ_iψ_i ≤ Σ|φ_i||ψ_i| ≤ √(m_φ·m_ψ)` ; comme `m ≤ 1`,
`min(m_φ, m_ψ) ≥ cos²` ; et la masse portée par `|A∩B|` coordonnées est au plus celle des `|A∩B|`
plus grandes ⇒ **`|A∩B| ≥ p_pair`**, exactement, par paire, **sans hypothèse distributionnelle**.
Les produits négatifs ne peuvent que **desserrer** la borne : le sens est **conservateur**.

**Version théorique — la PRÉDICTION pré-enregistrée** (`t_j ≈ Φ⁻¹(1 − j/16384)`, deux queues ;
masse totale `64 × E[Z² | |Z| > 2.66] = 64 × 8.90 = 570`) :

| `cos` | `cos²` | masse cumulée franchie | `p` minimal | `L(c) = p/64` |
| --- | --- | --- | --- | --- |
| 0.40 | 0.160 | p=7 → 0.154 | **8** | **0.125** |
| 0.41 | 0.168 | p=7 → 0.154 | **8** | **0.125** |
| 0.43 | 0.185 | p=8 → 0.173 | **9** | 0.141 |
| 0.45 | 0.2025 | p=9 → 0.192 *(interpolé — **à recalculer au banc**, Q-M1)* | **10** | 0.156 |
| 0.46 | 0.2116 | **frontière p=10 (0.210) / p=11 (0.228)** | **10 ou 11** | 0.156 – 0.172 |
| 0.47 | 0.2209 | p=10 → 0.210 | **11** | **0.172** |

~~⇒ prédiction : `O ≥ 8 à 11 indices sur 64`, soit 16 à 22 × le hasard, sur les strates dont `A3`
donne `cos ∈ [0.41, 0.46]`.~~

> **RECTIFIÉE le 2026-08-26 (§15) — défaut 0-137.** L'ancienne prédiction était paramétrée par un
> chiffre qui **nomme une autre colonne** (0-136) et provenait d'un descriptif calculé avec **une
> autre matrice `G`** (0-135). De plus **l'ensemble `cos ∈ [0.41, 0.46]` est VIDE** : aucune des
> douze cellules n'y tombe.

**⇒ PRÉDICTION RECTIFIÉE, paramétrée sur les `cos` mesurés avec la `G` DU PROJET** (§3, ligne 2 ;
plage `[0.1471, 0.3368]`), table de masses cumulées reconstruite indépendamment par `lab-verifier`
(`p=1 → 0.0259`, `p=4 → 0.0932`, `p=5 → 0.1132`, `p=6 → 0.1333`, `p=7 → 0.1533`, `p=8 → 0.1724`,
**`p=9 → 0.1911`** *(Q-M1 réglée : valeur exacte, non interpolée)*, `p=10 → 0.2095`, `p=11 → 0.2275`) :

**Table cumulée EXACTE** (`lab-math`, Q-M5, dérivée par série asymptotique de `Φ⁻¹`, vérifiée
contre les **sept ancres** de `lab-verifier`, écart < 1e−4 sur chacune) — **aucune valeur n'est
interpolée** :

`cum(1) = 0.02590` · `cum(2) = 0.04952` · `cum(3) = 0.07180` · `cum(4) = 0.09314` ·
**`cum(5) = 0.11376`** · **`cum(6) = 0.13379`** · `cum(7) = 0.15332`.

> **RÈGLE GRAVÉE (Q-M5)** : ***aucune valeur de `cum` n'est jamais interpolée.*** `cum` est
> **concave** (les `t_j²` décroissent), donc l'interpolation linéaire **sous-estime toujours** — et
> elle a **déplacé une cellule** dans la première version de cette table. **Table exacte `j = 1..20`
> publiée d'avance au banc, en fp64, avec la double normalisation 569.6 / 563.9.**

| cellule | `cos` (`G` du projet) | `cos²` | `p` minimal | `L = p/64` | marge à la frontière |
| --- | --- | --- | --- | --- | --- |
| **Qwen S0** | 0.1471 | 0.02165 | **1** | **0.015625 = `2n` EXACTEMENT** | 16.4 % — **robuste** |
| Qwen S2 | 0.1636 | 0.02676 | 2 | 0.03125 | — |
| Qwen S1 | 0.1883 | 0.03547 | 2 | 0.03125 | — |
| gpt2 S0 | 0.1886 | 0.03559 | 2 | 0.03125 | — |
| gpt2 S2 | 0.2070 | 0.04283 | 2 | 0.03125 | — |
| **Qwen S3** | 0.2184 | 0.04772 | **2** *(corrigé : était « 2-3 »)* | 0.03125 | 3.6 % — **robuste** |
| gpt2 S1 | 0.2479 | 0.06145 | 3 | 0.046875 | — |
| SmolLM2 S2 | 0.2566 | 0.06582 | 3 | 0.046875 | — |
| SmolLM2 S0 | 0.2568 | 0.06593 | 3 | 0.046875 | — |
| gpt2 S3 | 0.2766 | 0.07650 | 4 | 0.0625 | — |
| SmolLM2 S1 | 0.3174 | 0.10076 | 5 | 0.078125 | — |
| **SmolLM2 S3** | 0.3368 | 0.11346 | **5 ou 6** *(corrigé : était « 6 »)* | 0.078 – 0.094 | **0.26 % < marge 1 %** ⇒ **frontière indécidable en théorique ; la version RÉALISÉE tranche** — mode 0-128 rejoué à `p = 5/6` |

> **`O ≥ 1 à 5(-6) indices sur 64`**, soit **1 à 5(-6) × le corridor `2n`**.
> *(Recalcul du copilote confirmé sur dix cellules par `lab-math` et **corrigé sur deux** : `Qwen S3`
> et `SmolLM2 S3`, tous deux par une **interpolation interdite** de `cum`.)*

**Conséquence sur le défaut 0-104, gravée — `lab-math` tranche FERMEMENT :** *« `C-sep` est vide par
arithmétique »* **ne tient plus**. La classe `BAS` reste arithmétiquement exclue sur **11/12**
cellules (`L ≥ 2/64 = 2·(2n)`), et sur **`Qwen S0` elle est OFFERTE**, à la frontière exacte
(`L = 2n`). **Cette ré-ouverture est ROBUSTE, pas fragile** : l'écart vaut **16.4 %**
(`(0.02590 − 0.02165)/0.02590`) contre une marge de normalisation de **~1 %** (563.9 / 569.6) — pour
refermer, il faudrait `cum(1) < 0.0216`, soit **−16 %**, hors de portée de cette marge.
**`C-sep` est offerte : à écrire fermement, pas d'entre-deux.**

*Seconde dispersion, nommée par `lab-math` et que personne n'avait relevée* : la masse réalisée de la
première coordonnée d'une clé **fluctue** (maximum de 8192 gaussiennes ⇒ loi de Gumbel, `SD ≈ 15-20 %`,
estimé sous M3), donc **le `p_pair` réalisé chevauchera 1 et 2 selon les paires**. **Sans effet sur
l'offre de la classe** — on ne ferme pas une classe sur une fluctuation.

**Trivialité de la borne, à publier (Q-M5)** : sur **`Qwen S0`, `p = 1` est exactement
« `cos ≠ 0` ⇒ les supports se touchent » — trivial par paire**. Sur les **cinq cellules à `p = 2`**,
la borne **égale le `q95` par paire de la nulle** (0-119) : une paire nulle l'atteint **7.6 %** du
temps — faible par paire, mais **décisionnellement active** (une moyenne ≥ 2 = `2·(2n)` exclut `BAS`)
et **écrasante en agrégat** (`P(toutes les paires ≥ 2)` sous la nulle ≈ `0.076^P`). ⇒ **triviale sur
1/12, faible sur 5/12, active sur 11/12.** `V-borne` **garde toute sa valeur de porte de cohérence**
(identité, 100 % des paires, violation = bug) ; sa valeur **prédictive** à ce régime est **portée par
`p_sym`, pas par `p_pair`** (§16).

**Version réalisée par paire — la MESURE et la porte** : `p_pair` calculé sur les `φ_i²` **réalisés**.
`V-borne` exige `|A∩B| ≥ p_pair` sur **100 % des paires** (identité arithmétique — toute violation
est un **bug de mesure**, jamais un résultat), et exige que la **version réalisée encadre la table
théorique** ; l'écart est **publié** (D26). **La frontière `p = 10/11` disparaît de la décision.**

### 4.3 Nulles (D17 — une par maillon, toutes bloquantes)

| Maillon | Nulle | Type | Chiffres |
| --- | --- | --- | --- |
| **Niveau `O`** | `\|A∩B\| ~ Hypergéom(8192, 64, 64)` | **exacte** | `E = 0.5`, `sd = 0.702` ; `P(0)=0.604`, `P(1)=0.303`, `P(2)=0.076` ; **q95/paire = 2 indices** ; `sd(O)/paire = 0.01097` |
| **Centrage `Δ*`** | permutation intra-tige des étiquettes `{plac, type}` | **par permutation** | `ε*` (§4.5) |
| **`Core`** | `Bin(10, 0.0078)` par indice | **exacte** | `P(≥9) ≈ 1.1e-18` ; espérance ≈ **9e-15 indice** ⇒ **un seul indice = signature**, `p < 10⁻¹³`, **aucun IC** |
| **`G`** | provenance : `torch.randn`, lignes iid | **par construction (M3)** | `z_i` iid `N(0, ‖h‖²)` conditionnellement à `h`, **exact** ; échec de `V-G` ⇒ **arrêt de provenance**, jamais repli |

**Enveloppe de la moyenne de `O` sous la nulle** : `0.0078 ± 1.96 × 0.01097/√P`.
**P = 8 → [0.0002, 0.0154]** (borne haute **juste** sous `2n = 0.015625`) ; **P = 40 → [0.0044,
0.0112]** ; **P = 80 → [0.0054, 0.0102]**. ⇒ **`Λ` et le corridor ne s'évaluent que sur des moyennes
de ≥ 8 paires** (`V-P8`).

### 4.4 Engagements signés

| # | Énoncé | Statut | Antipode gravé |
| --- | --- | --- | --- |
| **`P-N`** *(lab-neuro, **NIVEAU**, **DÉCISIONNELLE**)* | `Λ(s) > 0` sur **les 4 strates et les 3 modèles** ; niveau `O_type ≳ 10 indices sur 64`. Recouvrement **haut et quasi indépendant de la strate**. | **SIGNÉE** | **`C-strat`** ⇒ `topk(G·h)` porte une structure de strate **que le cosinus avait comprimée** ⇒ **la lecture forte d'`A3` tombe**. |
| **`P-N-ord`** *(lab-neuro, **DESCRIPTIVE**)* | Si un ordre existe : `O(S3) ≥ O(S1) > O(S2) ≥ O(S0)` — **le domaine d'abord** (ordre du **cosinus brut**). | **DESCRIPTIVE** | L'ordre du **design** ⇒ le support porterait un facteur que le cosinus de valeurs signées masque ⇒ **sursis partiel pour l'attribution représentationnelle de X1**, **à instruire ailleurs, jamais à conclure ici**. |
| **Centrage** *(lab-neuro, signée, ordinale)* | **`O_brut ≈ O_glob > O_type`** **et `O_plac ≈ O_brut`**. | **SIGNÉE** | **`c-mec`** : *« le centrage n'a rien démontré »*, **dans ces mots exacts, sans habillage**. |
| **`auto` / `C-mod`** *(lab-neuro, signée)* | Divergence attendue **GPT-2 (LayerNorm, centre) vs SmolLM2/Qwen (RMSNorm, ne centre pas)** — **après vérification en config** (`V-norm`). | **SIGNÉE** | `auto` identique sur les trois ⇒ la coordonnée moyenne **n'est pas le porteur**, **l'explication LN/RMS de `C-mod` est morte**. |
| **`Core` / `Core-G`** *(lab-neuro, signée)* | `Core(S) ≠ ∅` (`p < 10⁻¹³`) ; `Core-G` **élevé** ⇒ **le noyau commun EST la projection de l'état moyen** ⇒ **aucune lecture représentationnelle licite**. | **SIGNÉE** | `Core-G` bas avec `Core ≠ ∅` ⇒ le noyau n'est pas réductible à `G·μ_global` — **descriptif**. |

**Mécanisme candidat nommé AVANT le run** (c'est ce qui rend `Δ*` interprétable) : un flux résiduel
porte quelques coordonnées **10 à 100 × la médiane** (*rogue dimensions* — Timkey & van Schijndel
2021 ; Kovaleva et al. 2021 ***(à vérifier)*** ; Dettmers et al. 2022 ; anisotropie : Ethayarajh
2019). Si `h[j*]` est géante et quasi constante, `(G·h)_i ≈ G[i,j*]·h[j*] + reste` et **le top-64 de
`G·h` est en grande partie le top-64 de la colonne `G[:,j*]`, identique pour tous les états** ⇒
prédit un **noyau commun quasi total**, une **insensibilité aux strates**, et un **effondrement sous
centrage que le placebo ne reproduit pas**.

### 4.5 `ε*` — ligne canonique unique (M7), recopiée à l'identique au §7

> **`ε*` = q₀.₉₅ de `|Δ*|` recalculée sur `B = 10⁴` rééchantillons bootstrap de tiges
> (`K_eff = 10`) sous permutation intra-tige des étiquettes {plac, type}, le placebo étant la moyenne
> des `R` directions gelées ; une valeur par modèle et par strate, calculée et gelée AVANT lecture
> des `Δ*` observés.**

**`R` est dérivé, pas choisi** : le MC des directions ajoute `σ_dir²/R` à la variance par paire de
`O_plac` ; exiger que cette contribution soit **≤ 10 %** de la variance appariée donne
**`R = ⌈10·σ̂_dir²/σ̂_Δ²⌉`**, `σ̂_dir` estimée sur un **pilote de `R₀ = 20` directions** sur un
modèle. **Pilote, formule et plafond (`R ≤ 300`) pré-enregistrés** ; si la formule dépasse le
plafond, **le dépassement est publié et le PI tranche — il n'est pas absorbé**.

**Puissance, déclarée (M8)** : détection à 1× exige `Δ* > ε* + q95·se ≈ 3.9·σ_Δ/√P`. Avec
`σ_Δ ≈ 0.015` (**à mesurer et publier, jamais absorbé**) et `P = 80` :
**`Δ*_min ≈ 0.0066 ≈ 0.42 × (2n) ≈ 84 % de la moyenne nulle `n``**.

### 4.6 Partitions — exhaustives, exclusives, ordre gravé (D18)

**Principe transversal reconduit (0-81 / D28)** : **marge de significativité 1×, couloir
d'équivalence 2×**, et **le couloir est ABSOLU (`2n`), jamais réglé sur l'enveloppe nulle de son
propre estimateur**.

**Partition `N` — NIVEAU** (par modèle **et** par strate ; **jamais poolée** ; moyennes de **≥ 8
paires** uniquement) — ordre gravé :

| Ordre | Classe | Condition | Verdict gravé | `P` sous la nulle |
| --- | --- | --- | --- | --- |
| 1 | **`HAUT`** | `IC_inf(Λ) > ε_Λ` | Recouvrement au-delà du corridor : `P-N` soutenue **sur cette strate et ce modèle** | ≈ 0 |
| 2 | **`−`** | `IC_sup(O) < 0` relativement à `n` | **Répulsion des supports** — sous le hasard. **INVALIDE-NULLE** : le modèle hypergéométrique est faux, arrêt d'interprétation, diagnostic obligatoire | ≈ 0 |
| 3 | **`BAS`** | `IC(O) ⊂ [0, 2n]` (**corridor absolu**) | **« recouvrement borné par 2 × le hasard »** — **jamais** « pas de recouvrement » (0-91) ; `Λ` peut sortir à `P` plus grand | `Φ(0.711√P − z_boot)` : **P=25 → 0.94** ; **P=80 → >0.9999** |
| 4 | **`ind_L`** | complémentation, **en dernier** | **sous la résolution** — la **résolution par strate est publiée** pour qu'un plat se lise *« sous la résolution »* et non *« plat »* (D28) | reste |

**Partition `C` — CENTRAGE** (par modèle ; `Δ* = O_plac − O_type`) — ordre gravé, **négligeabilité
physique d'abord** (M6, sens conservateur **contre** l'hypothèse portée ; divergence assumée avec
0-78, cf. Arbitrage X-4) :

| Ordre | Classe | Condition | Verdict gravé | `P` sous la nulle |
| --- | --- | --- | --- | --- |
| 0 | **`c-anti`** | `IC_sup(Δ*) < −ε*` | **CONSIGNÉE, court-circuitante, NON INTERPRÉTÉE** — mais **non muette** : oblige (i) `‖h − μ_type‖/‖h‖`, (ii) `\|Core_brut ∩ Core_type\|/64` (le noyau change-t-il d'**identité** ou d'**intensité** ?), (iii) `cos(μ_type, μ_global)`. **Aucune cause nommée avant ces trois chiffres ; aucune suite de chantier déclenchée par `c-anti` seule.** | ≈ 0.02 |
| 1 | **`c-mec`** | `IC(Δ*) ⊂ [−2n, +2n]` | **« le centrage n'a rien démontré »** — dans ces mots exacts, sans habillage. La chute sous `type` est **entièrement reproduite** par un placebo apparié en norme ⇒ **mécanique** | **0.90 – 0.95** (P ≥ 50) |
| 2 | **`c-cent`** | `IC_inf(Δ*) > ε*` | *« à ce locus, sur ce matériau, le mode commun est linéairement retirable »* — **un locus, un matériau**. **Ne licencie NI `I3` NI aucune modification de `engram/`** (xx) | **≤ 0.05** |
| 3 | **`ind_Δ`** | complémentation, **en dernier** | sous la résolution ; publier `σ_Δ` et `P` | **0.03 – 0.06** |

**Espace de verdict : 12 cellules décisionnelles = 4 (`N`) × 3 (`C` décisionnelles)**, `c-anti`
évaluée en premier et court-circuitante. Le banc simule **10 objets** : les 4 classes de `N`, les 4
états de `C`, et les 2 issues de `Core`. **Le cardinal est recompté par ÉNUMÉRATION au banc, jamais
par affirmation** (famille 0-76(i)/0-86/0-94/0-101/0-133).

**Deux écritures obligatoires (M6)** :
1. **Si `ε* ≥ 2n`**, le recouvrement des conditions `c-mec` et `c-cent` est **vide**, l'ordre est
   **sans objet** — **le déclarer en toutes lettres**.
2. **Sinon**, publier en annotation la **région masquée** `P(IC ⊂ corridor ∧ IC_inf > ε*)` sous
   l'alternative : **détection étouffée assumée, jamais silencieuse**.

**Cellules dont la lecture est gravée d'avance :**

| Cellule | Lecture gravée |
| --- | --- |
| **`HAUT × c-mec`** | Recouvrement massif **réel**, **cause non établie** : le centrage ne l'a pas départagée d'un décalage de magnitudes. `P-N` soutenue, mécanisme **ouvert**. |
| **`HAUT × c-cent`** | Recouvrement massif **et** composante affine commune retirable **au-delà du placebo**. **Deux des trois conditions de `X1-rect`** ; la troisième (`σ± > 0.5` significatif) et `Core-G` décident du déclencheur — **qui reste une note**. |
| **`BAS × c-cent`** *(0-129 — la cellule qui départage le PI et Neuro ; **ratifiée par le PI avant mesure**, §14-1)* | Le recouvrement **après centrage par type** est borné par `2n`, et le centrage a mordu au-delà du placebo. ⇒ **`P-N` est réfutée** (son niveau `≳ 10/64` n'est pas atteint sur `O_type`) **et la prédiction de centrage de Neuro est confirmée** ; **c'est exactement le pari du PI**. Les deux engagements de Neuro n'étaient conjointement satisfiables que sous un effondrement **partiel** : ici il est **total**. **À écrire ainsi, sans arbitrer en faveur de l'un des deux paris après coup.** |
| **`BAS × c-mec` avec `f` élevée et `O_brut` proche de `L`** *(seuils rectifiés §15 : l'ancien « `f ≈ 0.45`, `O_brut ≈ 4-6/64` » était calibré sur le chiffre mal nommé)* | **Branche de tort de Neuro (N7)** : les supports **sont** largement séparés, le cosinus est porté par **une poignée de coordonnées géantes**, **la lecture forte d'`A3` est fausse**, coupable = anisotropie de **l'entrée**, pas `G`. **À écrire dans ces termes.** |
| **`ind_L × ind_Δ`** | *« indécidable ICI, `P` ou `σ_Δ` insuffisants »* — **jamais** « pas d'effet ». La suite se lit sur `σ_Δ` et `P` publiés, pas sur la classe. |

### 4.7 Portes (toutes exécutables, toutes bloquantes)

| Porte | Contenu | Coût |
| --- | --- | --- |
| **`V-G` (v2 — RE-SPÉCIFIÉE le 2026-08-26, §15)** | *La v1 exigeait la reproduction bit-à-bit d'`A3` par `hippocampus.phi`. **Elle a échoué, et elle avait raison** : `A3` n'a jamais été calculé avec la `G` du projet (0-135). La v1 est donc **inexécutable par construction** — mode 0-118, mais pour la vraie raison.* **v2** : (a) `G` **du projet** instanciée via `hippocampus.phi` (`torch.randn(8192, d, generator=seed(cfg.seed))`), **hash publié par modèle** ; (b) `cos` de `φ(h)` par strate **mesuré et publié** avec cette `G`, en fp32 **et** fp64, **écart publié** (D26) ; (c) **publication obligatoire de la divergence avec `A3`** — les douze écarts, le `corrcoef ≈ 0.011` entre les deux matrices, et la mention que **`A3` porte sur une autre projection**. **Aucune reproduction d'`A3` n'est exigée ni possible.** Échec (hash absent, `G` non instanciable, ou divergence fp32/fp64 > 1e−6) ⇒ **arrêt de provenance (D14-R)**. | secondes |
| **`V-iid`** | **Prouvée par provenance** (M3). Résiduel : **sanité de moments** de `G`. **Pas de KS.** | secondes |
| **`V-borne`** | `\|A∩B\| ≥ p_pair` sur **100 %** des paires ; **et** la version réalisée **encadre** la table théorique du §4.2. Violation = **bug**, jamais résultat. | secondes |
| **`V-P8`** | `Λ` et le corridor calculés **uniquement** sur des moyennes de **≥ 8 paires** ; toute cellule à `< 8` paires **exclue, cardinal publié**. | banc |
| **`V-t1` (PORTÉE CORRIGÉE le 2026-08-26 — défaut 0-143)** | **(i)** **Aucune quantité à `t−1`** dont l'ensemble de comparaison contient **deux unités de même tige** — à `t−1` ces états sont **bit-identiques** (0-64/0-130) et leur recouvrement vaudrait **`64/64` par arithmétique**. **`t−1` reste refusé pour ce cycle.** **(ii)** **`Core(S)` : une seule unité par tige** (`V-core-S`, M9) — la nulle binomiale suppose l'indépendance sur les `N` états de `S`. **(iii)** **À `t`, les paires intra-tige ne sont PAS exclues** : `S3` et `S2` sont **définies** par le partage de tige, et à `t` ces états **ne sont pas identiques** (ils diffèrent par le suffixe, qui est le token de capture). **Cardinal intra-tige publié** par strate et par modèle. | banc |
| **`V-core-S`** | `\|S\| = 10`, **une unité par tige**, appartenance publiée. | banc |
| **`V-ulp`** | Marge à la coupure `k = 64` publiée pour les **cinq** conditions — **y compris centrées et placebo** (0-125). fp64 = **chemin nominal**. | banc |
| **`V-norm`** | Lecture de la config des **trois** modèles : LayerNorm vs RMSNorm, **citée par ligne de config**, jamais par croyance. | 2 lignes |
| **`V-diag`** *(condition bloquante `lab-neuro`)* | `f(s)`, `σ±(s)` et le couple **`(O, cos)` conjoint** publiés **par strate, par condition, par modèle**. **Sans les trois, le rapport ne peut pas être écrit.** | banc |
| **`V-cache`** | Les états `h` bruts de v4 sont sur disque, **hash calculé et publié**. **Échec ⇒ re-forward autorisé** (§14-4) : 53,82 s, VRAM ≤ 4,688 Gio — **seul poste GPU, conditionnel, chiffré**. | secondes |
| **`V-perimetre`** | §4.9, trois éléments obligatoires. | rédaction |
| **`V-seed`** | Seed, générateur et règle de tirage des `R` directions **gelés et publiés** ; cardinal **par modèle** (les directions **ne sont pas transportables**, 0-133). | banc |

### 4.8 Descriptifs pré-déclarés (aucune décision n'en dépend)

| Quantité | Statut |
| --- | --- |
| `f(s)`, `σ±(s)` | **OBLIGATOIRES** (`V-diag`) mais **DIAGNOSTIQUES, jamais correctifs** : ils ne modifient aucune borne, aucun seuil, aucune classe. Ils disent **à quel point la borne de Math est serrée** (Arbitrage X-1). |
| `O_glob − O_type` | diagnostic d'**anisotropie non capturée par `μ_type`** |
| distance lexicale par strate | publiée **à côté de `O`** — confondant nommé |
| `n_cell` par cellule, majorant `1/(n_cell−2)` | **obligatoire** (M4) |
| split-half | **double mesure (D26)** ; **devient la mesure principale si `min n_cell < 30`** (ratifié §14-2), le LOO passant en contrôle |
| `σ_Δ`, `σ_dir`, `R` réalisé | **publiés, jamais absorbés** (M8) |
| médiane et IQR de `O` | **dans la phrase gravée (xvi)** |
| portée | **« pour cette `G` », seed 0 (D9)** ; `Core` est **G-spécifique** |

### 4.9 Hors-périmètre déclaré — porte `V-perimetre`

1. **Mécanisme.** Aucun effet aval n'est mesuré : `M` n'est jamais instanciée, aucune lecture n'est
   injectée, aucune NLL n'est modifiée. Les quantités de ce cycle sont **géométriques**, sur le
   cortex gelé.
2. **Limite nommée — le point de contact `keysim`.** `read_gate=keysim` se calcule sur
   `cos(φ(h), clés)` — **la quantité même dont ce cycle mesure le plancher**. Un plancher de
   **0.147-0.337** (`G` du projet ; *rectifié §15 — l'ancien « 0.41-0.46 » était la **compression**,
   pas le cosinus*) avec une étendue inter-strates de **0.064-0.091** (*et non 0.01-0.03, soit
   ~6 × plus*) signifie que **le gate opère sur une variable à offset marqué et à dynamique
   modérée** — *la formulation « faible dynamique » était elle aussi tirée du chiffre faux et est
   retirée*. **Ce n'est pas un verdict sur le gate** (aucun effet
   aval, (xvii)) : c'est une **désignation de chantier**.
3. **Successeur désigné.** **Q-06** — calibration de `gate_keysim_mid` **par modèle** (ouverte depuis
   2026-08-21 : E1b 0.68 → 0.38 sous gate, « coût de sélectivité »). Ce cycle **ne l'ouvre pas** et
   **n'anticipe pas son résultat**.

### 4.10 Prédictions d'instrument reconduites

`P-A`, `P-B`, `P-C`, `P-D` d'I2 **restent pré-enregistrées telles quelles**. **Rappel gravé** :
l'inférence « `P-A` falsifiée ⇒ `P6` falsifiée » demeure **illégitime**. **Les deux réserves d'`I2`
(ratio → AUC ; `NEUTRAL_TEXT` non apparié) ne sont PAS caduques** — l'une est **aggravée** par le
centrage (§10).

## 5. Contrôles et baselines

1. **Configuration courante** : `EngramConfig()` par défaut (layer=6, λ=2.0, cap=0.5, η=0.2,
   decay=1e-3, thr=4.0, dg=8192/64, read_gate=keysim, seed=0) — **citée comme référence** ; ce
   protocole **n'instancie ni ne lit `M`** (D8 intacte, aucun backprop, `G` gelée D9).
2. **`M` reset / D7** : **sans objet**. Contrôle homologue = **`G` gelée, seed 0, identique aux cinq
   conditions** : seule l'entrée `h` change.
3. **Contrôle qui élimine l'explication triviale — le PLACEBO** : `h − c·r`, `r` gelée aléatoire,
   `‖c·r‖ = ‖μ_type‖`. Propriété d'appariement (M3-ter) : `E‖G·x‖² = D‖x‖²` pour **toute**
   direction ⇒ l'appariement en norme dans `ℝ^d` vaut aussi dans `ℝ^8192` **en espérance** ⇒ le
   **décalage mécanique de magnitudes (0-106) est reproduit**, et **seule la composante
   directionnelle corrélée à `z`** distingue `type` de `plac`. C'est exactement ce que `Δ*` isole.
4. **Flag à off, même code** : la condition **`aucun`** passe par **le même chemin de calcul** que
   les quatre autres — différence **uniquement dans l'entrée**.
5. **Ordre des conditions** : les cinq sont calculées **sur les mêmes paires, dans le même ordre, à
   partir du même cache `Z`** ⇒ aucun effet d'ordre possible.
6. **Unité d'échange** : `K_eff = 10` **tiges** (0-50) ; IC 95 % **bootstrap de tiges**, `B = 10⁴` ;
   **paires intra-tige exclues** (`V-t1`) ; **≥ 8 paires par tige** (`V-P8`).
7. **Double mesure (D26)** : LOO **et** split-half, publiés côte à côte ; **fp32 (chemin d'origine)
   et fp64**, publiés côte à côte.

## 6. Critères d'abandon — portes exécutables (D22)

**Ce qui INVALIDE le run** (aucun verdict n'est écrit) :
- **A.** `V-G` échoue en fp32 sur le chemin d'origine ⇒ **arrêt de provenance (D14-R)**.
- **B.** `V-borne` violée sur ≥ 1 paire ⇒ **bug de mesure**, jamais un résultat.
- **C.** NaN, ou `|A∩B|` hors de `[0, 64]`, ou `O` publié en flottant tronqué (0-132).
- **D.** `V-diag` incomplète ⇒ **le rapport ne peut pas être écrit**.
- **E.** `V-core-S` violée ⇒ `Core` **retiré du rapport**.
- **F.** `V-P8` violée sur une cellule ⇒ cellule **exclue**, cardinal publié ; si **> 2 strates**
  touchées sur un modèle ⇒ modèle **exclu**.
- **G.** `V-norm` non exécutée ⇒ **`auto` retiré de l'interprétation** (chiffres descriptifs).
- **H.** `V-seed` violée ⇒ `Δ*` et `ε*` **retirés**.
- **I.** Un chiffre de `A3` cité de mémoire au lieu d'être relu depuis `experiments/results/`
  (D14-R).

**Ce qui TUE `H_sup`** : `C-strat` réalisée ⇒ **la lecture forte d'`A3` tombe** · `c-mec` réalisée ⇒
*« le centrage n'a rien démontré »*, dans ces mots exacts · branche N7 ⇒ **la lecture forte d'`A3`
est fausse**, coupable = anisotropie de l'**entrée**.

**Ce qui NE tue rien et se consigne** : `c-anti` (avec ses trois descripteurs obligatoires) ;
`ind_L`/`ind_Δ` (*« indécidable ICI »*, avec `σ_Δ`, `P` et la résolution par strate publiés).

**Sans objet ici** (aucune injection, aucune NLL) : « 0 write », « E3 > 0.05 ». **Nommés comme sans
objet, jamais omis.**

## 7. Variables fixées

- **Modèles / couches** : GPT-2 (`--layer 6`), `HuggingFaceTB/SmolLM2-360M` (`--layer 16`),
  `Qwen/Qwen2.5-1.5B` (`--layer 14`). **`d` = 768 / 960 / 1536** — les directions **ne sont pas
  transportables** (0-133).
- **`G`** : gelée, `seed = 0`, `dg_dim = 8192`, `dg_topk = 64`, `torch.randn(8192, d)` — **portée du
  résultat : « pour cette `G` » (D9)**.
- **Matériau** : états `h` **du matériau qualifié v4**, capturés à **`t`** (`t−1` **refusé**),
  **792 états par modèle**, **cache vérifié par `V-cache`**.
- **Strates** : `S0` (rien partagé), `S1` (domaine), `S2` (tige, ponts), `S3` (tige + domaine).
  **Jamais poolées** (0-63, 0-67).
- **Inférence** : `K_eff = 10` **tiges** ; **`P ≥ 80` paires par modèle, `≥ 8` par tige** ; IC 95 %
  **bootstrap de tiges**, `B = 10⁴` ; **σ publié par strate**.
- **`ε*`** : **ligne canonique unique, recopiée à l'identique du §4.5** — *q₀.₉₅ de `|Δ*|` sur
  `B = 10⁴` rééchantillons bootstrap de tiges (`K_eff = 10`) sous permutation intra-tige des
  étiquettes {plac, type}, le placebo étant la moyenne des `R` directions gelées ; une valeur par
  modèle et par strate, calculée et gelée AVANT lecture des `Δ*` observés.*
- **`R`** : **dérivé**, `R = ⌈10·σ̂_dir²/σ̂_Δ²⌉`, pilote `R₀ = 20`, **plafond `R ≤ 300` publié**.
- **Corridor** : `n = 0.5 indice`, **`2n = 1 indice = 0.015625`**, **absolu**.
- **Précision** : `A3` reproduit en **fp32** (chemin d'origine, bit-à-bit) ; mesure en **fp64** ;
  **`Z` en cache, 51,9 Mo/modèle**. `M` reste fp32 dans le projet — **sans objet ici**.
- **Rien d'autre ne bouge** : aucun hyperparamètre d'`EngramConfig` modifié, aucun fichier de
  `engram/` touché.

## 8. Variable manipulée

**Une seule : la condition de centrage appliquée à `h` avant `G`**, à **cinq niveaux** :

| Niveau | Opération | Statut |
| --- | --- | --- |
| **`aucun`** | `h` | référence |
| **`glob`** | `h − μ_global` (LOO) | contrôle de mode commun global |
| **`type`** | `h − μ_type` (**LOO par paire** ; `n_cell` et `1/(n_cell−2)` publiés ; **split-half en double mesure**, **principal si `min n_cell < 30`**) | condition d'intérêt |
| **`auto`** | `h − mean_coord(h)·1` | **seul centrage calculable EN LIGNE**, sur le seul état courant : **aucun corpus, aucun LOO, aucune fuite** ; **discriminant de `C-mod`** via `V-norm` |
| **`plac`** | `h − c·r`, `r` gelée, `‖c·r‖ = ‖μ_type‖`, moyenne sur `R` directions | **contrôle du décalage mécanique de magnitudes** |

**Ce que `auto` change à `C-mod`** : sans `auto`, une divergence inter-modèles est une **classe
consignée sans explication**. Avec `auto`, elle a une **cause candidate nommée avant le run et
vérifiable dans le code des modèles** — LayerNorm **centre**, RMSNorm **ne centre pas** ⇒ **`G` est
appliquée à un vecteur que le cortex lui-même n'utilise jamais sous cette forme, et pas de la même
manière selon le modèle**. **Antipode** : comportement identique sur les trois ⇒ **l'explication
LN/RMS de `C-mod` est morte**.

## 9. Budget

| Poste | Coût | GPU |
| --- | --- | --- |
| MC conditionnel de `V-iid` | **0 — SUPPRIMÉ** (prouvé par provenance, M3) | 0 |
| Placebo (`R ≤ 300` directions) | **~5-10 min CPU/modèle** (identité `G·(h−c·r) = z − c·(G·r)` + cache `Z`) | 0 |
| Cache `Z` fp64 | **51,9 Mo/modèle**, ~156 Mo total (RAM, pas VRAM) | 0 |
| Recouvrements (5 conditions × 4 strates × ≥ 80 paires × 3 modèles) | **< 2 min CPU** | 0 |
| Bootstrap `ε*` (`B = 10⁴`, 10 tiges) | **< 1 min CPU** | 0 |
| `Core`, `Core-G`, `f`, `σ±`, `p_pair` | **< 1 min CPU** | 0 |
| Banc D14-S (12 cellules énumérées, 10 objets simulés, cas échouants obligatoires) | **~3-5 min CPU** | 0 |
| **Total mesure** | **< 30 min CPU, ~160 Mo RAM, 0 VRAM** sous `V-cache` PASS | **0** |
| **Temps agent** | **~10-14 h** (banc, portes, cellules, simulations) — **accepté à la gate** (§14-4) | — |
| **GPU conditionnel** | `V-cache` échec ⇒ **re-forward autorisé** : **53,82 s, VRAM ≤ 4,688 Gio** | **≤ 1 min** |

Un run, cinq conditions, trois modèles. **`R` au plafond** : si `⌈10·σ̂_dir²/σ̂_Δ²⌉ > 300`, le
dépassement est **publié et remonté au PI**, jamais tronqué en silence.

## 10. Livrables attendus

- **Aucun flag `EngramConfig`, aucune modification de `engram/`.** Ce cycle est un **instrument de
  mesure** sur des états en cache ; **`c-cent` réalisée ne licencie aucune ligne de code** — elle
  licencie **un énoncé, sur un locus et un matériau**.
- **Script d'éval** : nouveau, `eval/support_overlap.py` (nom indicatif), options
  `--model --layer --center {aucun,glob,type,auto,plac} --strate --R --seed`, sortie en **fractions
  exactes `p/64`**, hors `engram/`, **SPDX AGPL-3.0-or-later**.
- **Banc D14-S obligatoire, AVANT toute mesure** : 12 cellules énumérées ; 10 objets simulés ; cas
  **échouants obligatoires** pour `V-borne`, `V-P8` (7 paires), `V-core-S` (deux unités d'une tige),
  `V-ulp` (états centrés), `V-seed`, `V-t1` ; **cardinal recompté par exécution**. **`E = 0` exigé.**
- **Tests CPU** : `.venv\Scripts\python -m pytest tests/ -q`.
- **Entrée de journal**, format maison. **Ligne du tableau `docs/EXTENSIONS.md` §4 — ou déclaration
  explicite qu'aucune ligne n'est due**, ce cycle ne mesurant ni E1, ni E2, ni E3.
- **Conditionnement gravé sur `I3`** (N8) : *« `c-cent` ne licencie pas `I3`. `c-cent` licencie
  l'énoncé "à ce locus, sur ce matériau, le mode commun est linéairement retirable" — un locus, un
  matériau. `I3` est une affirmation **par couche** que ce cycle ne touche pas, et son ouverture est
  conditionnée à la purge écrite des deux réserves d'`I2` : **(a)** le score en **ratio** mesure
  l'anisotropie autant que l'invariance ⇒ **AUC obligatoire** — réserve **non caduque et AGGRAVÉE**
  par le centrage (supposer la réussite du centrage **à l'intérieur de l'instrument qui la teste est
  circulaire**, et le ratio confondrait alors l'invariance avec le **profil en profondeur de
  l'efficacité du centrage**) ; **(b)** `NEUTRAL_TEXT` **non apparié** — **non caduque, orthogonale,
  possiblement aggravée** : un `μ̂` estimé sur le matériau puis appliqué au contrôle est un centrage
  **hors distribution**, et le défaut d'appariement devient **une variable manipulée de plus**. »*
- **Contrainte de build (D29)** : **tout correctif de plus de 10 lignes repasse le circuit de
  relecture COMPLET** ; en deçà, la relecture ciblée reste **obligatoire, jamais nulle**.

## 11. Questions pour `lab-neuro` — TOUTES RÉPONDUES, avis FAVORABLE, condition INTÉGRÉE

N1 à N8 traitées à l'Arbitrage. La **condition bloquante** (`f`, `σ±`, `(O, cos)` conjoint) est
intégrée en porte `V-diag`.

**Reste dû, avant le banc** : signature explicite de la cellule **`(BAS × c-cent)`** telle que
rédigée au §4.6 — c'est la cellule qui **réfute `P-N`** tout en **confirmant sa prédiction de
centrage**, et le PI l'a **ratifiée avant mesure** (§14-1).

## 12. Questions pour `lab-math` — VERROUS LEVÉS

M1 à M9 traitées à l'Arbitrage. **Restent dues, avant le banc** :
- **Q-M1** : confirmation de la **masse cumulée à `p = 9`** (interpolée à 0.192) — **à recalculer**,
  pas à interpoler.
- **Q-M2** : réserve M5 — **≥ 8 paires par tige** pour la quasi-normalité : **confirmer par
  simulation** que l'IC bootstrap percentile ne gonfle pas `ind_L` sous ce plancher.
- **Q-M3** : validation de la reformulation de la clause « mêmes directions » (0-133).
- **Q-M4** : `P(classe)` des **10 objets** sous la nulle, par simulation, à comparer aux valeurs
  analytiques annoncées.

## 13. Questions pour le PI — **TRANCHÉES à la gate du 2026-08-23** (voir §14)

**P1** `ε*` canonique et `R` dérivé → **adoptés** · **P2** charge sur `Δ*` seul → **adoptée** ·
**P3+P12** budget et GPU conditionnel → **acceptés** (§14-4) · **P4** portée « pour cette `G` » →
**adoptée** · **P5** `Core` sans IC → **adoptée** · **P6** `X1-rect` note non franchie → **confirmée**
· **P7** candidate D30 → **proposée** (§14-5) · **P8** condition `auto` → **ADOPTÉE** (§14-3) ·
**P9** refus de `t−1` → **ratifié** · **P10** split-half principal si `min n_cell < 30` → **RATIFIÉ**
(§14-2) · **P11** purge des deux réserves d'`I2` avant `I3` → **gravée** · **P13** lecture de la
cellule `(BAS × c-cent)` → **RATIFIÉE telle quelle** (§14-1).

## 14. Amendements de la gate PI — 2026-08-23

1. **Cellule `(BAS × c-cent)` : lecture gravée ratifiée telle quelle, avant mesure.** Le PI confirme
   que cette cellule est la réalisation de son pari, qu'elle **réfute `P-N`** tout en **confirmant la
   prédiction de centrage de Neuro**, et que sa lecture du §4.6 lui convient — **de sorte que
   personne n'arbitrera après coup en faveur de l'un des deux paris**.
2. **Bascule split-half RATIFIÉE, automatique et pré-enregistrée** : si `min n_cell < 30`, le
   split-half **devient la mesure principale** et le LOO passe en contrôle, **sans nouvelle gate**.
   Motif : le biais résiduel du LOO (`+1/(n_cell−2) ≈ 0.05` de cosinus à `n_cell ≈ 20`) est **de la
   taille de la quantité décidée** — il fabriquerait `c-cent`. **La bascule étant pré-enregistrée,
   ce n'est pas un choix après lecture.**
3. **Condition `auto` ADOPTÉE** — cinquième niveau de la variable manipulée. Motif : coût marginal
   quasi nul ; **seul centrage calculable en ligne** ; **seul discriminant de `C-mod` avec une cause
   vérifiable dans le code des modèles** (`V-norm`, deux lignes de config).
4. **Budget accepté, re-forward autorisé.** ~10-14 h d'agent ; `< 30 min` CPU ; **0 GPU sous
   `V-cache` PASS**. Si les `.npz` sont absents ou sans hash vérifiable, le **re-forward est
   autorisé** (53,82 s, VRAM ≤ 4,688 Gio) — **cela contredit le « 0 GPU » du cadrage, et c'est
   assumé et écrit**.
5. **Candidate D30 proposée au PI** *(à graver par lui, hors labo)* : *« une borne géométrique se
   publie en deux versions — une **théorique pré-enregistrée** qui **prédit**, et une **réalisée par
   paire** qui **décide** ; c'est la réalisée qui porte la porte, la théorique qui porte
   l'engagement. »*

---

## Registre des engagements

| Engagement | Auteur | Signature | Statut | Motif |
| --- | --- | --- | --- | --- |
| **`P-N`** (niveau, `Λ(s) > 0` sur 4 strates × 3 modèles, `O_type ≳ 10/64`) | lab-neuro | 2026-08-23 | **SIGNÉE — DÉCISIONNELLE** | Antipode `C-strat` ⇒ la lecture forte d'`A3` tombe. |
| **`P-N-ord`** (ordre du **cosinus brut**) | lab-neuro | 2026-08-23 | **SIGNÉE — DESCRIPTIVE** | Antipode = ordre du **design** ⇒ sursis partiel pour l'attribution représentationnelle de X1, **à instruire ailleurs**. |
| **Ordinal du design `S3>S2>S1>S0`** | lab-director (brouillon) | — | **RETIRÉ, non reporté** | **Déjà démenti** par le brut de v4 (0-122). |
| **Prédiction de centrage** (`O_brut ≈ O_glob > O_type`, `O_plac ≈ O_brut`) | lab-neuro | 2026-08-23 | **SIGNÉE** | Antipode `c-mec`, formulation gravée mot pour mot. |
| **`auto` / `C-mod`** | lab-neuro | 2026-08-23 | **SIGNÉE**, sous `V-norm` | Cause **vérifiable dans le code des modèles**, pas une croyance. |
| **`Core` / `Core-G`** | lab-neuro | 2026-08-23 | **SIGNÉE** | `Core-G` élevé ⇒ **aucune lecture représentationnelle licite**. |
| **`C-anti`** | lab-neuro | 2026-08-23 | **GRAVÉE — consignée, non interprétée, NON MUETTE** | Trois descripteurs obligatoires ; **aucune cause nommée avant** ; **aucune suite de chantier**. |
| **`t−1`** | lab-neuro | 2026-08-23 | **REFUSÉ, NON REPORTÉ** ; ratifié par le PI | A4 + `V-t1` (`64/64` par arithmétique). |
| **Condition bloquante `f` / `σ±` / `(O, cos)`** | lab-neuro | 2026-08-23 | **INTÉGRÉE — porte `V-diag`** | Sans elle, le cycle **ne peut pas donner tort à son expert** (0-120). **Diagnostique, jamais correctif.** |
| **Phrase gravée (xvi)** | lab-neuro | 2026-08-23 | **PRIME sur celle du Directeur, verbatim** | Quatre contraintes chargées. |
| **Vocabulaire interdit (xviii)** | lab-neuro | 2026-08-23 | **GRAVÉE, verbatim** | Trois candidats bio écartés par propriété manquante nommée. |
| **`X1-rect`** | lab-neuro | 2026-08-23 | **GRAVÉ, NON FRANCHI** | **Même observé, ne licencie aucune modification de `engram/`.** |
| **`L(c)` version réalisée par paire** | lab-math | 2026-08-23 | **VERROU LEVÉ** | Zéro hypothèse ; la théorique **prédit**, la réalisée **décide**. |
| **`ε*` canonique + `R` dérivé** | lab-math | 2026-08-23 | **VERROU LEVÉ** | Recopiée à l'identique §4.5 / §7 (anti-0-82). |
| **`V-iid` par provenance ; MC supprimé** | lab-math | 2026-08-23 | **GRAVÉ** | Échec de `V-G` = **arrêt de provenance**. |
| **`≥ 8 paires` pour `Λ` et le corridor** | lab-math | 2026-08-23 | **GRAVÉ — `V-P8`** | `q95` par paire = **2 × le corridor**. |
| **Split-half principal si `min n_cell < 30`** | lab-math | 2026-08-23 | **RATIFIÉ par le PI** (§14-2) | Bascule **automatique et pré-enregistrée**. |
| **« Mêmes directions sur les 3 modèles »** | lab-math | 2026-08-23 | **REFORMULÉ (0-133)** | Arithmétiquement inexécutable ; appariement **par la norme**. |
| **Pari du PI** — `O_brut ≥ 0.5` (≥ 32/64) et **`O_type` sous `2n`** | **PI** | **2026-08-23, HORS PROTOCOLE, aucun poids décisionnel** | **CONSIGNÉ** | **Mutuellement exclusif avec `P-N` sur `O_type`** ; réalisation = cellule **`(BAS × c-cent)`** (0-129), **lecture ratifiée avant mesure**. |
| **Pari de `lab-director`** — `O_brut ∈ [12, 25]/64` ; `O_type ∈ [4, 10]/64` ; **`c-mec`** ; `Core ≠ ∅` avec **`Core-G > 0.8`** | lab-director | **2026-08-23, HORS PROTOCOLE** | **CONSIGNÉ** | Posé **avant** la mesure pour que son interprétation soit auditable. |
| **Pari du copilote** — `O_brut ∈ [25, 45]/64` ; `O_type` **chute fortement mais reste au-dessus de `2n`** ⇒ cellule **`HAUT × c-cent`** ; **`Core-G > 0.7`** | **copilote** | **2026-08-23, HORS PROTOCOLE, aucun poids décisionnel** | **CONSIGNÉ — MOTIF INVALIDÉ, PARI MAINTENU** | **Le motif invoqué était faux** : *« la constance du plancher 0.41-0.46 sur trois architectures sent le mode commun »* reposait sur la colonne **compression**, que le copilote n'avait pas vérifiée (0-136). **Le pari lui-même n'est pas modifié** — il a été posé, il est daté, il tient ou il tombe ; mais **son fondement est retiré**, et cela se lit avec lui. *La constance réelle est celle de la compression (0.408-0.466 sur trois architectures), qui reste un fait — mais elle ne dit pas ce que le motif lui faisait dire.* | Conséquences : **le pari du PI tombe**, **`P-N` survit**, **la prédiction de centrage de Neuro est confirmée**, **le `c-mec` du Directeur tombe**. Il **ne modifie aucun seuil, aucune classe, aucune porte**. |
| **D29 — contrainte de build** | PI | reconduite | **GRAVÉE** | **Tout correctif de plus de 10 lignes repasse le circuit complet.** |

---

## Ce qui reste indécidable, et le restera après ce run

*Écrit avant la mesure, pour qu'aucune de ces phrases ne soit tentée après.*

1. **L'effet aval.** Aucune injection, aucune NLL, aucun E1/E2/E3. `O` ne dit **rien** sur le rappel,
   sur le gate, sur la capacité, ni sur le 0/10 top-10.
2. **Le chemin d'écriture de `M`.** Le +57 % d'E2 de X1 est **acquis et intact** ; ce cycle porte sur
   la **lecture** de `φ`. **Seule son attribution représentationnelle est en jeu, et seulement par
   `P-N-ord`, qui est descriptive.**
3. **La généralité à une autre `G`.** Portée **« pour cette `G` », seed 0** (D9). `Core` est
   **G-spécifique**.
4. **La cause d'un `c-mec`.** Il dit que le placebo reproduit la chute ; il **ne dit pas** pourquoi
   la chute existe.
5. **La cause d'un `c-anti`.** Trois mécanismes candidats nommés par Neuro, **qu'il déclare ne pas
   savoir ordonner avant le run** ; **aucune cause ne sera nommée après** sans les trois descripteurs.
6. **`I3` et le profil par couche.** Ce cycle mesure **un locus**. Il **ne licencie pas `I3`**.
7. **`gate_keysim_mid` (Q-06).** Point de contact **nommé** au §4.9 ; **non instruit** ici.
8. **Le canal suffixe** (0-62, v4). Hors périmètre, successeur désigné, **matériau propre requis**.

---

## 15. Correction de provenance du 2026-08-26 — traçabilité complète

**Autorisation.** Les §4 et §6 étaient **gelés depuis le 2026-08-26**, et le protocole grave que
*« ni le PI, ni le copilote, ni les experts, ni un auditeur ne les modifient »*. **Le PI a levé le
gel pour cette correction seule**, sur le discriminant suivant, qu'il propose de graver en décision
d'architecture :

> **Une erreur trouvable SANS la mesure se corrige sans rompre le gel ; une erreur qui n'apparaît
> qu'à la lumière des résultats, jamais.** Le gel protège contre l'**ajustement post-hoc aux
> données** — il n'a jamais été écrit pour protéger une **erreur d'arithmétique ou de lecture**
> identifiable avant toute mesure. Le critère est **exécutable** : *une donnée du run est-elle
> nécessaire pour voir l'erreur ?* Ici **non** — elle a été trouvée en **9,1 s de CPU**, par les
> portes de provenance, **avant** toute mesure et sans une seule donnée de l'expérience.

**Ce qui est corrigé, avec ancienne valeur, nouvelle valeur, motif et auteur :**

| Section | Ancienne valeur | Nouvelle valeur | Motif | Relevé par |
| --- | --- | --- | --- | --- |
| **§3, ligne 1** | « cosinus de `φ(h)` : plancher **0.41-0.46**, étendue **0.01-0.03** » | **rectifiée et barrée** ; ligne neuve : `cos` mesuré **avec la `G` du projet**, plage **[0.1471, 0.3368]** | 0-136 : les chiffres nommaient la colonne **`compression`** et l'**amplitude de compression** ; 0-135 : `A3` n'a pas utilisé la `G` du projet | `lab-builder` (§6.I), confirmé par `lab-verifier` |
| **§4.2, prédiction** | `O ≥ 8 à 11 / 64`, **16 à 22 × le hasard**, sur `cos ∈ [0.41, 0.46]` | **`O ≥ 1 à 6 / 64`**, **1 à 6 × le corridor**, sur `cos ∈ [0.1471, 0.3368]` mesuré **ici** | 0-137 : dérivée du chiffre mal nommé ; l'ensemble conditionnant était **vide** | `lab-verifier` (recalcul), copilote (première alerte) |
| **§4.2, défaut 0-104** | « `C-sep` est **vide par arithmétique**, la classe n'est pas offerte » | **ne tient plus catégoriquement** : `BAS` exclue sur **11/12** cellules, **RE-OFFERTE sur `Qwen S0`** à la frontière exacte (`L = 2n`), et **fragilement** | idem | `lab-verifier` |
| **§4.7, `V-G`** | « reproduire `A3` en fp32 sur `hippocampus.phi`, bit-à-bit » | **v2** : instancier la `G` du projet, publier son hash et les `cos`, **publier la divergence avec `A3`** ; **aucune reproduction d'`A3` exigée** | 0-135 + 0-138 : la v1 était **inexécutable par construction** | `lab-builder`, confirmé par `lab-verifier` |
| **§4.9-2** | « plancher **0.41-0.46**, étendue **0.01-0.03** ⇒ **faible dynamique** » | « **0.147-0.337**, étendue **0.064-0.091** ⇒ **dynamique modérée** » ; « faible dynamique » **retiré** | 0-139 : les **deux** termes étaient faux | `lab-verifier` |
| **§2 (N7), §4.6 (cellule)** | seuils illustratifs `f ≈ 0.45`, `O_brut ≈ 4-6/64` | **forme conservée, valeurs retirées** | 0-139 : calibrées sur le chiffre faux | `lab-verifier` |
| **Registre, pari du copilote** | motif : « la constance du plancher **0.41-0.46** sur trois architectures » | **pari MAINTENU, motif INVALIDÉ et retiré** | 0-139 : motif fondé sur une colonne non vérifiée | copilote (auto-relevé) |

**Ce qui n'est PAS corrigé et reste gelé** : la structure des partitions, l'ordre d'évaluation, le
schéma 1×/2×, la ligne canonique de `ε*`, `R` dérivé, les portes autres que `V-G`, les engagements
signés de `lab-neuro` (`P-N`, `P-N-ord`, centrage, `auto`, `Core`), les critères d'abandon du §6, et
**les paris eux-mêmes**. *Aucune de ces sections ne dépendait du chiffre mal nommé.*

**Reste dû avant la reprise au banc** : **Q-M5** — `lab-math` confirme le re-paramétrage de `L(c)`
sur les `cos` de la `G` du projet (table ci-dessus), et dit si la ré-ouverture de `BAS` sur
`Qwen S0` est **robuste** ou **à l'intérieur de la marge** de 0-128.

## 16. Borne serrée `p_sym` — adoptée le 2026-08-26 sous D30 alinéa 2

**Ce que `lab-math` a trouvé (Q-M5).** La dérivation gelée passe par `min(m_φ, m_ψ) ≥ cos²`, obtenue
en majorant `m_ψ ≤ 1` — **cette marche jette un facteur `cos`**. Or les **deux** masses sont bornées
par `T(p)`, d'où l'identité **serrée**, de même statut et sans hypothèse supplémentaire :

> **`cos ≤ √(T_φ(p) · T_ψ(p)) ≤ T(p)`  ⇒  `p_sym = min{p : cum(p) ≥ cos}`** — et non `≥ cos²`.

À `cos ≈ 0.45` l'écart entre les deux bornes était modeste ; **à `cos ≈ 0.15` il vaut un facteur ~7**.
Trois valeurs dérivées par `lab-math` : **`Qwen S0 → 7`**, **`gpt2 S3 → 14`**, **`SmolLM2 S3 → 18`**.

**Décision du PI, 2026-08-26 — `p_sym` devient la borne DÉCISIONNELLE**, sous le **second alinéa de
D30** (critère de **direction**) :

> La borne gelée n'était **pas fausse** — elle était **correcte et lâche**. La remplacer n'est donc
> pas une correction d'erreur (D30 alinéa 1) mais une **amélioration**, normalement interdite après
> gel. Elle est licite ici parce qu'elle rend la prédiction **PLUS DURE à satisfaire** :
> **`O ≥ 7 à 18` au lieu de `O ≥ 1 à 5`**, soit **14 à 36 × la nulle** au lieu de 1 à 5 ×.
> *Un changement qui rend sa propre hypothèse plus difficile à confirmer n'est pas une bifurcation.*

**Conséquences, gravées :**

1. **`V-borne` porte désormais `p_sym`** : `|A∩B| ≥ p_sym(cos_pair)` sur **100 %** des paires, en
   version **réalisée par paire** (`T` calculée sur les `φ_i²` observés), la version théorique
   restant la **prédiction pré-enregistrée** que la réalisée doit encadrer (D26 inchangé).
2. **Le défaut 0-104 est RE-FERMÉ, cette fois avec la bonne `G`** : `p_sym ≥ 7` sur la cellule au
   `cos` le plus bas ⇒ **`BAS` est arithmétiquement exclue sur 12/12**, et **`C-sep` n'est plus
   offerte**. *La fermeture d'origine était vraie par accident — bonne conclusion, mauvaise borne,
   mauvaise `G`, mauvaise colonne. Elle est maintenant vraie pour ses raisons.*
3. **`p_pair` (borne lâche) reste publiée** en descriptif, avec l'écart aux deux versions — **double
   mesure D26**, et trace de ce que le gel portait.
4. **La trivialité relevée en Q-M5 disparaît** : `p_sym ≥ 7` n'est jamais l'énoncé « les supports se
   touchent ». La borne **redevient prédictive sur 12/12**.

### 16.1 `Q-M6` — table et douze `p_sym`, livrées le 2026-08-26 (`lab-math`, FAVORABLE)

**Provenance (D14-R)** : dérivée **à la main** par Newton sur `Q(z) = φ(z)·R(z)`, ratio de Mills en
fraction continue (profondeur 10-14), **ancrée sur six quantiles exacts** re-vérifiés (`Q` recalculée
à < 10⁻⁴ relatif), **aucune valeur interpolée**. Incertitude numérique **≤ ±1.5·10⁻⁴** absolu sur
`cum` ; les **sept recoupements** avec la table de `lab-verifier` coïncident **tous à la 4ᵉ décimale**.

> **CLAUSE D'IMPLÉMENTATION GRAVÉE** : **le banc grave la version `fp64` (`scipy.special.ndtri`) ;
> la table ci-dessous est le CONTRÔLE, pas la gravure** (tolérance `±1.5e−4`). Et la colonne
> **`Z = 563.9` est un HYBRIDE DÉCLARÉ** — profil d'ordre **théorique** × masse totale **réalisée** ;
> **la vraie table réalisée est PAR CLÉ**, calculée au banc.

| `j` | `t_j²` | `cum` (`Z = 569.6`) | `cum` (`Z = 563.9`) |
| --- | --- | --- | --- |
| 1 | 14.7604 | 0.025914 | 0.026176 |
| 2 | 13.4566 | 0.049538 | 0.050039 |
| 3 | 12.6971 | 0.071829 | 0.072555 |
| 4 | 12.1599 | 0.093178 | 0.094120 |
| 5 | 11.7441 | 0.113795 | 0.114946 |
| 6 | 11.4051 | 0.133819 | 0.135172 |
| 7 | 11.1189 | 0.153340 | 0.154890 |
| 8 | 10.8715 | 0.172426 | 0.174168 |
| 9 | 10.6535 | 0.191130 | 0.193061 |
| 10 | 10.4587 | 0.209491 | 0.211608 |
| 11 | 10.2828 | 0.227543 | 0.229843 |
| 12 | 10.1223 | 0.245314 | 0.247794 |
| 13 | 9.9749 | 0.262826 | 0.265483 |
| 14 | 9.8385 | 0.280099 | 0.282930 |
| 15 | 9.7116 | 0.297149 | 0.300152 |
| 16 | 9.5930 | 0.313990 | 0.317164 |
| 17 | 9.4818 | 0.330638 | 0.333979 |
| 18 | 9.3769 | 0.347099 | 0.350607 |
| 19 | 9.2779 | 0.363387 | 0.367060 |
| 20 | 9.1840 | 0.379511 | 0.383347 |

**Les douze `p_sym` = `min{p : cum(p) ≥ cos}`.** *Critère de robustesse : `p_sym` identique sous les
deux normalisations ⇒ robuste ; la marge citée est la distance relative à la frontière la plus proche.*

| cellule | `cos` | **`p_sym`** | `L = p/64` | marge | verdict |
| --- | --- | --- | --- | --- | --- |
| Qwen S0 | 0.147121 | **7** | 0.1094 | 4.1 % | **ROBUSTE** |
| Qwen S2 | 0.163589 | **8** | 0.1250 | 5.1 % | **ROBUSTE** |
| Qwen S1 | 0.188322 | **9** | 0.1406 | 1.5 % | robuste, **limite** |
| gpt2 S0 | 0.188646 | **9** | 0.1406 | 1.3 % | robuste, **limite** |
| gpt2 S2 | 0.206957 | **10** | 0.1563 | 1.2 % | robuste, **limite** |
| Qwen S3 | 0.218442 | **11** | 0.1719 | 4.0 % | **ROBUSTE** |
| **gpt2 S1** | 0.247890 | **12-13** | 0.188-0.203 | **0.04 %** | **FRONTIÈRE FRANCHE — indécidable en théorique, la RÉALISÉE tranche** |
| SmolLM2 S2 | 0.256561 | **13** | 0.2031 | 2.4 % | **ROBUSTE** |
| SmolLM2 S0 | 0.256770 | **13** | 0.2031 | 2.3 % | **ROBUSTE** |
| gpt2 S3 | 0.276588 | **14** | 0.2188 | 1.25 % | robuste, **limite** |
| **SmolLM2 S1** | 0.317424 | **16-17** | 0.250-0.266 | **0.08 %** | **FRONTIÈRE FRANCHE — la RÉALISÉE tranche** |
| SmolLM2 S3 | 0.336841 | **18** | 0.2813 | 1.9 % théorique, **0.9 % réalisée** | quasi-frontière 17/18 ; `p = 18` sous **les deux** tables, mais marge réalisée < 1 % ⇒ **la réalisée tranche** |

> ### **PRÉDICTION FINALE : `O ≥ 7 à 18 indices sur 64`** — soit **14 à 36 × la nulle** (`n = 0.5`).
>
> Huit cellules **robustes**, deux **frontières franches** (`gpt2 S1`, `SmolLM2 S1`, marges 0.04 % et
> 0.08 % — **sous le plancher numérique de ±0.07 %**, donc décidables par **aucune** table
> théorique), deux **limites** (1.2-1.5 %, stables sous les deux normalisations).
>
> **Propriété gravée, et elle est décisive pour la lecture** : **chaque frontière basculerait vers le
> `p` INFÉRIEUR**, donc dans le sens **CONSERVATEUR** pour l'hypothèse. ***Aucune ne peut rendre la
> prédiction plus facile qu'annoncé.*** *C'est ce qui autorise à écrire « `O ≥ 7 à 18 »` sans
> réserve : l'incertitude résiduelle ne joue que contre nous.*

### 16.2 `p_sym` est une IDENTITÉ — confirmé (`Q-M6` (c))

Par paire :
`|cos| = |Σ_{A∩B} φ_iψ_i| ≤ Σ_{A∩B}|φ_i||ψ_i| ≤ √(Σ_{A∩B}φ_i²)·√(Σ_{A∩B}ψ_i²) ≤ √(T_φ(|A∩B|)·T_ψ(|A∩B|))`
— **Cauchy-Schwarz**, puis *« la masse d'un sous-ensemble de cardinal `p` est majorée par celle des
`p` plus grandes coordonnées **réalisées** »*. **Deux pas exacts, zéro hypothèse** ; le membre droit
est **croissant en `p`**, donc `p_sym^pair` est bien défini.

> **Toute violation de `|A∩B| ≥ p_sym(cos_pair)` en version réalisée est un BUG DE MESURE, jamais un
> résultat.** `V-borne` peut donc la porter.

**Précision d'implémentation, gravée** : la version réalisée se calcule avec **`|cos_pair|`** et la
**MOYENNE GÉOMÉTRIQUE** des masses top-`p` des **deux clés de la paire** — **pas** avec la table
moyenne.

### 16.3 Statut à la livraison de `Q-M6` (2026-08-26)

**`Q-M6` est LIVRÉE et intégrée** (§16.1). La table de contrôle est écrite en
`experiments/results/recouvrement-supports/cum_table_QM6.json`, **avec ses contrôles de sanité
exécutés avant publication** : décroissance des `t_j²`, croissance et **concavité** de `cum`, et les
**sept ancres de `lab-verifier` retrouvées à `≤ 1.5·10⁻⁴`**. **Le banc DOIT la recalculer en `fp64`
(`scipy.special.ndtri`) et comparer à cette tolérance** — elle est le **contrôle**, jamais la gravure.

**`V-borne` est DÉBLOQUÉE.** Son membre (a) — l'identité par paire — était déjà **PASS** (0 violation
sur 40 paires synthétiques). Son membre (b) — *« la réalisée encadre la théorique »* — dispose
désormais de sa table.

*Chiffre du banc à conserver* : sur les 40 paires synthétiques, **`p_sym ∈ [7, 18]` contre
`p_pair ∈ [1, 6]`**, écart **min 6 / max 12 / moyen 8,5**, et **`p_sym ≥ p_pair` sur 100 %** — la
borne serrée domine la lâche partout, comme l'identité l'exige.

## Historique

- **2026-08-23** — Brouillon (mode cadrage). **Treize défauts acquittés (0-104 … 0-116), dont quatre
  critiques.**
- **2026-08-23** — **Avis `lab-math` : FAVORABLE, verrous levés** (M1 `L(c)` confirmée + correctif
  « version réalisée par paire » ; M2 nulle hypergéométrique et plancher de 8 paires ; M3 `V-iid` par
  provenance et **deux budgets morts** ; M4 LOO insuffisant ; M5 corridor absolu vérifié par calcul ;
  M6 partition et deux écritures ; M7 ligne canonique de `ε*` et `R` dérivé ; M8 puissance déclarée ;
  M9 `Core`).
- **2026-08-23** — **Avis `lab-neuro` : FAVORABLE sous une condition bloquante** (`f`, `σ±`,
  `(O, cos)` conjoint). Retire l'ordinal du design, refuse `t−1`, ajoute `auto`, `Core-G`, l'entrée
  (xviii), la phrase gravée (xvi), `C-anti` non muette, sa branche de tort, et la purge des deux
  réserves d'`I2`.
- **2026-08-23** — **Consolidation.** Condition bloquante **intégrée** ; **dix-huit défauts de plus
  (0-117 … 0-134), dont sept critiques** ; **quatre contradictions tranchées** (dont deux clauses
  d'expert **inexécutables**, relevées par le copilote) ; ordinal du design **retiré**, `t−1`
  **refusé**, `auto` **ajouté**, les **deux budgets morts** retirés et §9 re-chiffré.
  **Total du cycle : trente-et-un défauts (0-104 … 0-134), aucun octet mesuré, aucun GPU touché.**
- **2026-08-23** — **Gate PI** : cellule `(BAS × c-cent)` **ratifiée telle quelle avant mesure** ;
  bascule split-half **ratifiée** ; condition `auto` **adoptée** ; budget et **re-forward
  conditionnel acceptés**. Pari du copilote **consigné**. Candidate **D30** proposée.
- **2026-08-23** : proposé.
- **2026-08-26** : **PRÉ-ENREGISTRÉ par le PI.** Prédictions (§4) et critères d'abandon (§6)
  **gelés définitivement** — ni le PI, ni le copilote, ni les experts, ni un auditeur ne les
  modifient désormais ; toute lecture ultérieure qui les contredirait est un **résultat**, pas un
  motif d'amendement. Suite : `eval/support_overlap.py` puis **banc D14-S complet (`E = 0`, cas
  échouants obligatoires pour `V-borne`, `V-P8`, `V-core-S`, `V-ulp`, `V-seed`, `V-t1`)** ;
  **aucune mesure avant PASS intégral du banc**.
- **2026-08-26** — **BUILD & RUN : arrêt à la deuxième porte.** `V-cache` **PASS** (hashes gravés,
  première gravure ; 0 GPU, re-forward non déclenché). **`V-G` FAIL** : reproduction bit-à-bit par
  `hippocampus.phi` **fausse sur 12/12**, écarts `3.3e−03` à `2.6e−02` ; le chemin **réellement
  exécuté en v4** (`descriptif_A3`) reproduit **`0.000e+00` sur 12/12**. **§6.A appliqué : arrêt de
  provenance (D14-R)**, banc non écrit, aucune mesure, aucune constante dérivée. **9,1 s de CPU.**
  Second relevé par la clause **§6.I** : deux chiffres du §3 nomment la mauvaise colonne.
- **2026-08-26** — **VERIFY : `APPROVED`** (`tests_passed`, `protocol_followed`, `rerun_consistent`).
  L'arrêt est **fondé, correctement appliqué et structurellement matérialisé**. Les deux constats
  sont **reproduits par un troisième chemin** écrit par le vérifieur. **Isolement causal** : une
  seule des quatre différences (le **tirage de `G`**) porte l'échec. **Huit défauts de plus
  (0-135 … 0-142), dont trois critiques**, tous portant sur le **protocole** et sur le **journal de
  v4**, aucun sur ce run.
- **2026-08-26** — **REPRISE : banc `E = 0`, arrêt à `V-borne` (Q-M6 non encore rendue).**
  **`V-G` v2 PASS sur 3/3** (hashes de `G` publiés ; écart fp64−fp32 `≤ 3.8e−09` contre une tolérance
  de `1e−06` ; **`corrcoef(G projet, G d'A3)` = −0.0008 / +0.0009 / −0.0012** — deux tirages
  **indépendants**, le diagnostic 0-135 est confirmé numériquement). **`V-norm` PASS**, cité **par
  ligne de config** : gpt2 **LayerNorm** (`centre = True`), SmolLM2 et Qwen **RMSNorm**
  (`centre = False`) — **la base de la prédiction `auto`/`C-mod` de `lab-neuro` est vérifiée, pas
  crue**. **0-141 fermé.** Banc `dgov` : **`E = 0`**, 28 clauses, 66 cas, couverture 100 %, **2,4 s** ;
  `E(v4) = 0` et 228 tests restent verts. **Le banc a trouvé DEUX défauts dans le code du Builder
  avant toute mesure** (première passe `E = 5`) : **0-144** (corridor **64 × trop large**, détecté par
  le **comptage du cardinal** : 3 cellules au lieu de 12) et **0-145** (`V-seed` rejetant `seed = 0`).
  **`V-borne` rendue `EN-ATTENTE`** — le Builder **refuse de poser un seuil** en l'absence de `Q-M6` :
  *« ce serait 0-52, troisième occurrence du cycle »*. **Aucune mesure lancée**, étage de mesure écrit
  mais **structurellement fermé**. Tension **relevée et non tranchée** par lui : la portée de `V-t1`
  (⇒ **0-143**).
- **2026-08-26** — **`Q-M6` livrée** (table `j = 1..20`, douze `p_sym`, identité confirmée) et
  **0-143 tranché** : `V-t1` reçoit une **portée en trois membres**, les paires intra-tige **restent**
  à `t`. **`V-borne` débloquée.**
- **2026-08-26** — **`Q-M5` (`lab-math`, FAVORABLE)** : table `cum` **exacte** (aucune
  interpolation), **deux cellules corrigées** — `Qwen S3 → p = 2` (marge 3.6 %, robuste) et
  `SmolLM2 S3 → frontière 5/6` à **0.26 %**, sous la marge de 1 % (mode 0-128 rejoué à `p = 5/6`,
  **par l'interpolation de la session principale**). Règle gravée : ***aucune valeur de `cum` n'est
  jamais interpolée***. `0-104` **tranché fermement** : ré-ouverture de `BAS` sur `Qwen S0`
  **ROBUSTE** (16.4 % contre ~1 %). Et **borne serrée `p_sym` dérivée** — la marche `m ≤ 1` jetait un
  facteur `cos`.
- **2026-08-26** — **DÉCISION PI : `p_sym` adoptée comme borne décisionnelle**, sous le **second
  alinéa de D30** (critère de **DIRECTION** : un changement post-gel trouvable sans la mesure est
  licite s'il rend la prédiction **plus dure**, interdit s'il la rend plus facile). Prédiction :
  **`O ≥ 7 à 18`** au lieu de `1 à 5`. **`C-sep` re-fermée sur 12/12, avec la bonne `G`** — §16.
  **`Q-M6` dû avant le banc** : table `cum(j)` exacte `j = 1..20` et les douze `p_sym`.
- **2026-08-26** — **DÉCISION PI : correction de provenance autorisée** sur le discriminant *« une
  erreur trouvable sans la mesure se corrige ; une erreur qui n'apparaît qu'à la lumière des
  résultats, jamais »*. §3, §4.2, §4.7 (`V-G` v2), §4.9-2, §2, §4.6 et le Registre **rectifiés avec
  traçabilité complète (§15)**. **Le cycle reprend au banc**, sous réserve de **Q-M5**.
