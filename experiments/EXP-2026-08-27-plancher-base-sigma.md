# EXP — plancher de `top64(G·μ_global)` et clôture de `σ±` (Étape A, priorité 1)

Statut : PRE-ENREGISTRE

*Cadré par `lab-director` le 2026-08-27, sous avis **`lab-neuro` RÉSERVÉ** (7 corrections) et
**`lab-math` RÉSERVÉ** (9 corrections). **Vingt-cinq défauts fermés (0-157 … 0-181), dont huit
critiques, avant qu'une mesure existe.** 0 GPU, 0 CPU de calcul.*

> **Déclaration en tête, exigée, à recopier dans le rapport.**
> **Les deux énoncés ratifiés de ce cycle sont, après dérivation, quasi forcés, et leurs deux
> antipodes quasi inatteignables.** `P-Base ≥ 0.50` est prédite à **0.55–0.75 par le cosinus seul**
> (`Q-M9`) ; `C-Base-mort` exigerait `γ ≈ 0.5`, contredit par `‖μ‖/‖h‖ = 0.73–0.85`. La clause 2 de
> `P-σ` a un **support vide** et son antipode est **exclu par l'identité du barycentre**.
> **Ce cycle ne se court donc PAS pour décider ce que ses deux énoncés prétendent décider.** Il se
> court comme **calibrateur assumé (D27)** — il **produit le plancher que D20 exige pour D32** — et
> il porte **deux questions neuves**, `R_Base` et `Ψ`, qui sont les seules à avoir un contenu.
> **Cela est déclaré avant la mesure et ne sera pas renégocié après.**

---

## Arbitrage

### A. Tranchages

| # | Point | Décision | Motif |
| --- | --- | --- | --- |
| **A-1** | **`P1` — clause centroïde à support vide : (A) `SANS OBJET` de `lab-neuro` contre (B) extension aux paires inter-cadres du Directeur** | **(A) RETENUE. (B) ÉCARTÉE — le Directeur est le perdant, et l'argument est celui de `lab-neuro`.** **Ratifié PI le 2026-08-27.** | Sous (B), une paire inter-cadres oppose **deux états de la MÊME unité décisionnelle vus dans deux cadres de prompt** : c'est un contraste d'**invariance au cadre**, l'objet d'`I2` et d'`A4`. Le porter sous le nom `P-σ` clause 2 serait **le mode 0-122 exact, transposé de l'étage du chiffre à l'étage de l'énoncé** — faire signer à un expert une prédiction dont l'objet n'est pas le sien. L'argument du Directeur (« (B) est plus dur, donc licite sous D30 al. 2 ») **reste vrai et ne suffit pas** : **D30 alinéa 2 autorise à durcir une prédiction, jamais à lui substituer un autre objet.** **(B) n'est pas morte : elle s'ouvrira sous le nom neuf `P-cadre`, avec son antipode propre, signée à froid par qui la conçoit — elle n'hérite d'AUCUNE signature de `lab-neuro`. NON OUVERTE ici.** |
| **A-2** | **`Ξ_perm` — `lab-math` : « il a un objet » (effet de cadre, `β ≈ 10 %`) ; `lab-neuro` : « cet objet n'est pas le mien »** | **`Ξ_perm` RETIRÉ de ce cycle. `lab-math` perd sa correction (6).** **Ratifié PI le 2026-08-27.** | **Les deux ont raison, et cela ne suffit pas à garder la quantité.** Trois raisons indépendantes, chacune suffisante : **(i)** sous A-1, la clause qu'`Ξ_perm` devait porter **n'a pas de domaine de définition** — porter une clause `SANS OBJET` par une primaire, c'est **licencier une borne depuis un support vide**, interdit par **D23** ; **(ii)** l'objet réel d'`Ξ_perm` est le **cadre de capture**, pas les centroïdes de type — le publier sous le nom de la clause 2 rejoue **0-122** au même endroit qu'en A-1 ; **(iii) défaut 0-168** — le diagnostic de départage que `lab-math` propose lui-même (*décomposition de `cos_paire` intra-cadre / inter-cadre*) est **inexécutable dans l'univers de paires gelé** : **toutes** les paires sont intra-cadre (0-157), le bras inter-cadre est **vide**. **Un départage à un seul bras n'est pas un départage.** ⇒ `Ξ_perm` et le `β ≈ 10 %` partent **intacts** dans la fiche `P-cadre`, **non ouverte**. *Aucune des deux dérivations de `lab-math` n'est perdue ; seul leur rattachement à `P-σ` l'est.* |
| **A-3** | **`Q-M9` — le cycle vaut-il d'être couru ?** | **OUI, sous D27 : scission calibrateur / question.** Ni « tel quel », ni « réduit puis abandonné ». **Ratifié PI le 2026-08-27.** | **Position de `lab-neuro` retenue sur le périmètre** (*« il vaut par son plancher D32 et par la clôture de `σ±` ; à 10 minutes de CPU il est bon marché et il ne doit rien retarder »*) ; **position de `lab-math` retenue sur le contenu** (*« la seule quantité à contenu au-delà est le résidu `ρ_Base − ρ̂_Base(γ)` »*). Les deux sont **compatibles** et forment ce protocole. |

### A-3 détaillé — la valeur du cycle

**Le fait nouveau.** `Q-M9` établit, sur la **propriété pivot** démontrée par `lab-math` — *pour deux
vecteurs fixés, les couples `(z_a,i, z_b,i)` sont iid en `i`, gaussiens bivariés de corrélation
**exactement** `ρ = cos(x_a, x_b)`, sur l'aléa de `G`* — que `ρ̂_Base ≈ 0.55–0.75` **se dérive du seul
cosinus `γ = cos(h, μ_global)`**. Le seuil de succès ratifié vaut 0.50. **La marge du succès annoncé
(0.05 à 0.25) est intégralement fournie par `γ`.**

| Option | Instruite | Sort |
| --- | --- | --- |
| **(i) Courir tel quel, calibrateur assumé** | Position de `lab-neuro`. **Vraie sur le périmètre, insuffisante sur le contenu** : elle laisse le cycle sans **aucune** quantité dont l'issue ne soit pas prévisible avant mesure — c'est-à-dire **sans mode de succès négatif**, le mode D28 à l'étage de l'énoncé signé (défaut **0-169**). | **RETENUE pour le périmètre et le budget, ÉCARTÉE comme forme complète.** |
| **(ii) Re-spécifier autour du résidu `ρ_Base − ρ̂_Base(γ)`** | Position de `lab-math`. C'est **une différence** (D16), sa **nulle est dérivée et vaut 0 par construction du modèle pivot** (zéro hypothèse sur `h`), et son antipode est **atteignable**. **C'est la seule quantité du cycle dont l'issue n'est pas prédictible.** | **RETENUE comme question principale.** |
| **(iii) Réduire et passer à `V2-D(b)`** | **ÉCARTÉE.** Sauter la production du plancher laisserait **D32 (ii) innocentante** — ce que D20 proscrit. Le coût de la production est **< 10 min CPU**. *On ne renonce pas à une dette de 10 minutes.* | **ÉCARTÉE, motif écrit.** |

**Forme retenue — D27, scission des fonctions :**

- **CALIBRATEUR** (`ρ_Base`, `Δ_Base`, les deux énoncés ratifiés) : **il a le droit d'être dominé par
  `γ` — il ne décide rien.** Il fournit le **plancher** que D20 exige, la nulle exacte et la santé de
  l'instrument. Ses seuils ratifiés sont **évalués verbatim**, et son **poids probant est déclaré nul
  d'avance** sous `V-ident` et l'entrée (xxv).
- **QUESTION** (`R_Base`, `Ψ`) : définie **hors du domaine où `γ` règne** — `R_Base` est le résidu
  *après* retrait de ce que `γ` prédit, `Ψ` le résidu *après* retrait de ce que la loi de queue
  prédit. **Leur plancher n'est pas dominé : il est nul par dérivation.**

**Ce que le cycle ne peut plus rendre** : *« `P-Base` est confirmée »* comme un résultat. **Un plafond
atteint n'est pas un succès** — la jurisprudence (xiii) du plafond lexical de v4 s'applique mot pour
mot, à l'autre extrémité.

### B. Corrections de `lab-neuro` — 7 remarques, 7 intégrées

| # | Remarque | Sort | Motif |
| --- | --- | --- | --- |
| **N1** | Il **retire son mécanisme** : sa cible était les centroïdes de type au sens **(tige, domaine)**, `Ξ_perm` porte sur les centroïdes de **cadre de capture**. | **INTÉGRÉE** | *Mécanisme mort avant mesure, coût zéro.* **Conduite consignée au dossier méthode** : un expert qui retire sa propre cible avant mesure épargne un cycle — pendant exact de la conduite du Builder (« refuser de trancher, déclarer, demander l'arbitrage »), validée quatre fois. |
| **N1-bis** | Refus de (B) ; nom neuf `P-cadre` si elle s'ouvre. | **INTÉGRÉE** — voir A-1. | — |
| **N5** *(compte fermé par les données)* | `σ±` **uniforme en `\|A∩B\|`** ⇒ incompatible avec un porteur de rang dominant ; **non-concentration** (top-15 = 513/5083) ⇒ `Core` ne le porte pas ; *« l'effet vit dans le volume, pas dans le noyau »*. | **INTÉGRÉE** en acquis contraignant (§3) | Ferme d'avance deux explications qui auraient été proposées après coup. **Aucune géométrie de centroïdes ne peut le porter** — **confirme A-2 par un second chemin**. |
| **N5** *(exigence bloquante)* | Dépendance LOO **positive** ⇒ `N` effectif ≪ 5083 : le **« 12 à 15 σ » du journal du 2026-08-26 n'est pas publiable en l'état**. | **INTÉGRÉE, BLOQUANTE** — porte `V-grappe`, classe **`Σ-mort`** | **Défaut 0-170.** **Rectification du journal autorisée par le PI le 2026-08-27** sous D30 alinéa 1, traçabilité complète. Conséquence gravée : si l'écart ne survit pas, **`N-queue` est sans objet et `Q-M1` tombe pour une raison qui n'est pas la sienne**. |
| **Correction de `Q-M1`** | La sélection réalisée est un **RANG** (top-64 de son **propre** vecteur), **pas un seuil commun** ; seuil réalisé fluctuant (Gumbel, SD 15-20 %). | **INTÉGRÉE** — `Q-M1-bis` | **Défaut 0-171.** *« Sans cela, `Ψ` est un résidu à une loi qui n'est pas celle du dispositif. »* Nulle par **simulation exacte à sélection top-64 par vecteur**, **ou** erreur d'approximation publiée — **l'un des deux, jamais ni l'un ni l'autre**. |
| **N2** | Il **signe `Σ-epuise` d'avance** : *« la clause 1 de `P-σ` se réalise pour une raison qui ne lui appartient pas et ne compte pas à son crédit : retirée du registre comme **réalisée-sans-mérite** »*. | **INTÉGRÉE verbatim** | Lecture gravée avant mesure ⇒ **personne n'arbitrera après coup**. |
| **N4** | Il **signe `Δ-nul`**, avec deux bornes : (1) elle **ne licencie rien** ; (2) **D34 doit exiger un COUPLE, jamais un nombre**. | **INTÉGRÉE** ; **D34 GRAVÉE le 2026-08-27** dans ses termes | *« Une clause qui publie son propre contrôle apparié produit toujours un plancher »* (D26). **Meilleure réparation que celle du Directeur** : nommer la clé nulle est **déclaratif**, publier le couple est **exécutable**. |
| **N6** | Ordinal signé : **`Δ_Base(S0) ≥ Δ_Base(S3)`**, conjonction bornée par le **`min`** (0-63), **bootstrap joint**. Cellule modale signée d'avance : `Δ-nul` sur les deux ⇒ **l'ordre n'a pas de domaine ⇒ `SANS OBJET`, jamais `0`**. | **INTÉGRÉE** | Il signe **sa propre cellule d'évanouissement** — D13 tenu jusqu'au bout. |
| **N3 + façades** | Il rédige **(xxi)** et **(xxii)** lui-même, **nomme sa propre façade** pour la faire interdire, et grave que **`σ± < 0.5` n'est pas un fait de neurosciences**. | **INTÉGRÉES verbatim** ; les brouillons du Directeur sont **retirés** au profit des siens | *Cinquième façade tuée par la méthode « nommer la propriété manquante »* — après Hasselmo, 0-59, 0-55 et 0-121. **Et la première nommée par son propre auteur.** |

### C. Corrections de `lab-math` — 9 remarques, 8 intégrées, 1 écartée

| # | Remarque | Sort | Motif |
| --- | --- | --- | --- |
| **(1)** | `N-queue` gelée en **`σ̂±_pool`**, **jamais `σ̂±(cos̄)`**. | **INTÉGRÉE, gelée littéralement** | **Défaut 0-172.** *« Utiliser `σ̂±(−0.002)` global, c'est construire le facteur manquant DANS la primaire »* — le pooling pondéré **est** le facteur ~10 recherché. Mode 0-155 à l'étage de l'agrégation. |
| **(2)** | Écriture de `Δ_Base` et agrégation des `R` tirages **gravées littéralement** (C1). | **INTÉGRÉE** | **Défaut 0-173** — deux lectures licites, **troisième occurrence du mode 0-155**. |
| **(3)** | **`μ_global` LOO-par-paire en MESURE PRINCIPALE**, non-LOO en descriptif. | **INTÉGRÉE, inconditionnelle.** La bascule conditionnelle du Directeur est **écartée**. **Confirmé PI (P7).** | **Défaut 0-174.** Biais directionnel ≈ **+5·10⁻³** sur `ρ_Base(μ_global)` **seul**, donc **directement dans `Δ_Base`**, plausiblement **un demi-`ε_B`**. Mode **0-117** : *un biais de la taille de l'effet n'est pas une correction de second ordre.* Coût nul. |
| **(4)** | **`ρ̂_Base(γ_cell, t)` et `Δ̂_Base(γ, c)` calculées au banc et gravées AVANT mesure**, avec la **cellule modale**. | **INTÉGRÉE — pivot de la re-spécification** | Sans elles, `ρ_Base` élevé serait lu comme un résultat. **Avec elles, le résidu devient la question.** |
| **(5)** | `N-état` apparié en **(tige, domaine) aux DEUX membres** ; **`SANS OBJET` si support vide**. | **INTÉGRÉE** | **Défaut 0-175.** Sans appariement, `cos(h_j, paire)` varie de **0.15 à 0.34** selon la strate ⇒ **`Δ_Base` mesurerait la strate**. Nécessaire **et non suffisant** : la suffisance n'existe que **jointe à `Δ̂_Base(γ,c)` publiée d'avance**. |
| **(6)** | Clause centroïde **portée par `Ξ_perm` seul**. | **ÉCARTÉE — `lab-math` est le perdant.** | Voir **A-2** : la clause n'a **pas de domaine** ; porter une clause `SANS OBJET` par une primaire **licencie une borne depuis un support vide** (D23) ; et le départage proposé est **inexécutable** (0-168). |
| **(7)** | **`Δ_min` par cellule publié après pilote `R₀ = 20`, AVANT la mesure principale.** | **INTÉGRÉE** | `Δ_min ≈ 1.14·σ_tige`, `σ_tige` inconnue avant pilote. **Publié, jamais absorbé.** |
| **(8)** | **8 buckets dyadiques de 8 rangs**, planchers **≥ 200 trials ET ≥ 8 paires**, fusion **pré-déclarée**, **définition du rang gelée**. | **INTÉGRÉE** | **Défauts 0-176 et 0-171 (second membre).** 64 rangs ⇒ `se ≈ 0.056` : **un profil illisible est une porte absente.** **Rang gelé : `r = max(rang_A, rang_B)`** — l'appartenance à `A∩B` est **bornée par le rang le pire des deux** ; les trois autres lectures **énumérées et rejetées par écrit** (D31). |
| **(9)** | Corriger « 250× » en **~170×**. | **INTÉGRÉE** en provenance | Le ratio des écarts à 0.5 est **~170**, dont le conditionnement fournit **~14** : **il manque ~11**. |
| **C5** | La structure de cadre (`β ≈ 10 %`) explique **à la fois** `cos_type` et potentiellement `σ±` : **ne pas la compter deux fois**. | **INTÉGRÉE** en interdit **(xxix)** | **Défaut 0-181.** *Une seule cause candidate pour deux quantités ne produit pas deux confirmations indépendantes.* |
| **C6** | `G` unique (seed 0) ; `Q-M1`/`Q-M9` sont des **espérances sur `G`**, écart `O(1/√D)`. | **INTÉGRÉE** | Le terme `O(1/√D)` **entre dans `ε_R`** (`Q-M11`) — sans quoi le résidu serait comparé à une nulle trop étroite. |
| **C7** | Publier `n_vide` par cellule. | **INTÉGRÉE** — porte `V-vide` | — |
| **Pièges numériques** | `t = ndtri(1 − 1/256)` **au banc, jamais 2.66 en dur** ; fp64 partout ; `V-ulp` sur **toutes** les conditions y compris LOO (358 vs 360) ; **~92 % des rééchantillons de 10 grappes ont une grappe manquante** ⇒ cardinal effectif et méthode de quantile publiés (D24) ; **cosinus à ≥ 4 décimales**. | **TOUS INTÉGRÉS** | **Défauts 0-177 à 0-179.** La pente `(φ/Q)²` varie de **1.5 %** entre 2.66 et 2.6601 ; **−0.0019 contre −0.0023 pèsent 20 % l'un sur l'autre** dans `σ̂±`. |
| **`Q-M2`** | **`σ±` n'est PAS un effet du centrage** : l'artefact LOO pousse vers l'**accord** (`+0.0169`, soit `+0.075` en `σ±`). Anti-corrélation **intrinsèque ≈ −0.019**. | **INTÉGRÉE en acquis** (§3) | **Écarte par dérivation la branche « le cycle n'a pas d'objet ».** *Le Directeur perd une hypothèse, et c'est un gain : la question `Ψ` est vivante.* |
| **`Q-M4`** | Sous permutation de cadres, `P(cos ≤ −0.15) ≈ 1 − 3·10⁻⁷`. **Poids probant ≈ nul confirmé.** | **INTÉGRÉE** | Confirme 0-159 par calcul. Sans effet décisionnel (A-2), **conservée en provenance**. |
| **`Q-M3`** | Cardinal `N-cadre` = `720^(U−1)` ; dégénérescence seulement si `U = 1` ⇒ **sans objet**. | **CONSERVÉE en provenance** ; `N-cadre` **retirée avec `Ξ_perm`** | La vérification D31 reste due **pour `N-état`** (§11), où **le danger est réel et confirmé**. |

### D. Ce qui n'a pas été fait

Aucun énoncé ratifié n'a été **retiré, affaibli ni réinterprété**. Les seuils `≥ 0.50` et `< 0.25` sont
**évalués verbatim**. La clause 1 de `P-σ` est retirée **du registre des prédictions par son auteur** —
retrait qui **ne rend rien plus facile** (il supprime une clause **déjà satisfaite**, donc il durcit la
revendication de succès : D30 al. 2 respecté). La clause 2 est **`SANS OBJET`** parce que **son support
est vide**, ce que D23 impose : un constat de domaine, pas un affaiblissement.

---

## 0. Défauts acquittés

**0-1 … 0-156** : reconduits sans changement (patrimoine, v4, recouvrement-supports).

### 0-157 … 0-167 — relevés au cadrage

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-157** *(critique)* | **La clause centroïde de `P-σ` a un SUPPORT VIDE.** Les paires sont construites **à l'intérieur d'une cellule de capture** (`np.triu_indices(n_dec, 1)` sur `H[c]`, `eval/support_overlap.py:2246`, boucle `for c in cellules:` l. 2257) ⇒ **les deux membres de toute paire partagent leur `μ_type`**, sur **4/4 strates**. L'ensemble *« paires de cadres `t ≠ t'` représentées en S0 »* est **∅**. | Pris à la lettre, `cos(μ_t − μ_g, μ_t − μ_g) = +1` ⇒ **`P-σ` serait falsifiée par arithmétique de définition, pas par le monde.** Mode **0-137** aggravé : ici l'ensemble vide ne rend pas la clause *sans objet*, il la rend **fausse**. D23 : un support vide se publie **`SANS OBJET`, jamais `0`** — et **ne licencie aucune borne**. |
| **0-158** *(critique)* | **L'antipode `C-σ-mort` est quasi INATTEIGNABLE par identité.** `μ_global` est la moyenne **non pondérée** des 6 centroïdes de cadre (cellules équi-cardinales, `n_global = n_dec × 6`, l. 2241-2243) ⇒ **`Σ_t (μ_t − μ_global) = 0` exactement** ⇒ `cos` moyen par paire ≈ **−1/(T−1) = −0.20** à `T = 6`, contre un seuil `\|cos\| ≤ 2/√d` = **0.072 / 0.065 / 0.051**. | **Un protocole dont l'antipode ne peut pas mordre n'a pas de mode de succès négatif** — mode 0-81/D28 transposé de la classe d'équivalence à l'antipode. `C-σ-mort` ne pourrait être « réalisé » que par le **vide** (0-157), ce que D23 interdit de lire comme une réalisation. Correctif : **`V-T`**, accessibilité déclarée d'avance (D31). |
| **0-159** *(critique)* | **La seconde clause de `P-σ` est une IDENTITÉ ALGÉBRIQUE à slack.** `≤ −1/(T−1) + 0.05` est **vrai par le barycentre** dès que les normes des centroïdes sont proches ; la marge `+0.05` est le seul jeu. | **Mode D32 exact** : *« `O ≥ p_sym` est une identité au poids probant nul »*. La clause **re-publierait l'identité du barycentre sous un nouveau nom**. Confirmé par calcul (`Q-M4`) : `P(cos ≤ −0.15) ≈ 1 − 3·10⁻⁷`. |
| **0-160** *(critique)* | **`P-Base` est un TAUX** — interdit comme statistique décisionnelle (**D16**). | Un **taux de détection sans taux de fausse alarme** est exactement ce que D16 grave après V2-D(a) v3. Correctif §4.3 : le taux **reste publié** (plancher dû à D20, nommé par D32) et la **décision migre** sur `Δ_Base`, puis sur `R_Base`. |
| **0-161** *(critique)* | **`P-Base` ne nommait pas la condition de centrage.** Sous `type`, `O ≈ 0.5 à 1.5 indice` ⇒ **dénominateur nul sur une large fraction des paires**, ratio **indéfini**. | Sous-spécification de la **statistique**, famille **0-155/D31** : deux lectures licites, l'une **non définie**. Correctif : condition **`aucun` primaire**, `type` en **double mesure D26**, agrégation gelée en **ratio des sommes**, `n_vide` publié. |
| **0-162** *(majeur)* | **`P-Base` est quasi FORCÉE sous le régime d'égalité démontré.** Si tout état est ≈ `μ_global` en direction, `top64(G·h) ≈ top64(G·μ_global)` pour **tout** `h`. | Le taux élevé ne distingue **pas** `μ_global` d'un **état quelconque** — publier un plancher qui ne distingue rien, c'est produire un **suspect innocenté** (D20). Correctif : nulle **`N-état`** appariée, puis **`N-quad`** (0-169). |
| **0-163** *(majeur)* | **La clause 1 de `P-σ` est DÉJÀ MESURÉE** (`cos_type` négatif sur les S0, 3/3, journal 2026-08-26). | *« Une mesure déjà faite se cite, ne se refait pas. »* Correctif : **citée**, retirée du registre des prédictions **par son auteur**. |
| **0-164** *(majeur)* | **`C-Base-mort` ne disait pas sur combien de strates.** | **0-156 rejoué** : grille à trous. Correctif : lecture gelée **4/4 strates** d'au moins **2 modèles/3**. |
| **0-165** *(majeur)* | **Le piège de coupure `topk` n'est pas un artefact à côté du mécanisme : c'est LE MÊME OBJET.** Le `ρ` d'une loi de queue conditionnée à la coupure **EST** le `cos` des états centrés. | Sans forme quantitative, *« exclu par calcul »* est **indécidable** — `P-σ` pourrait être « confirmée » par la coupure **et** « expliquée » par la géométrie, **au choix du lecteur** (mode 0-56/0-70). Correctif : **`Q-M1`**, primaire **`Ψ`**, **profil signé par rang**. |
| **0-166** *(mineur)* | **Les deux références n'ont pas le même estimateur** : `μ_global` **non-LOO** (la paire mesurée est dedans), `μ_type` **LOO par paire**. | Sans publication, `Δ_Base` mélangerait un **biais de fuite** avec l'effet. Frère de **0-117**. Correctif : **`V-mu`**, majorant publié, double mesure D26 — puis **`V-loo`** (0-174). |
| **0-167** *(mineur)* | **Le mot « cellule » est surchargé** : cellule de **capture** (6), cellule **décisionnelle** (12), cellule de **partition**. | Famille **0-136** à l'étage du vocabulaire. *« Les 6 cellules S0 »* n'était **pas décidable**. Correctif : **glossaire gelé au §3.1**, trois cardinaux publiés par énumération. |

### 0-168 … 0-181 — relevés à la consolidation

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-168** *(critique)* | **Le diagnostic de départage de `lab-math` est inexécutable dans l'univers de paires gelé.** Il demande la *« décomposition de `cos_paire` intra-cadre / inter-cadre »* pour départager `Ξ` et `ρ_eff` ; **toutes les paires sont intra-cadre** (0-157), le bras inter-cadre est **vide**. | **Un départage à un seul bras n'est pas un départage** — il aurait rendu un stratum et un `SANS OBJET`, et l'unique valeur aurait été lue comme le **résultat** du départage. Famille **0-137** (ensemble conditionnant vide) **dans un instrument de départage**, c'est-à-dire à l'endroit dont la fonction est précisément d'empêcher qu'une valeur unique décide. **Troisième raison, indépendante, du retrait d'`Ξ_perm`.** |
| **0-169** *(critique)* | **Les DEUX énoncés ratifiés sont quasi forcés, avec antipodes quasi inatteignables.** `Q-M9` : `ρ̂_Base ≈ 0.55–0.75` par le cosinus seul contre un seuil de 0.50 ; `C-Base-mort` exigerait `γ ≈ 0.5`, contredit par `‖μ‖/‖h‖ = 0.73–0.85`. Côté `P-σ` : identité du barycentre et support vide. | **Mode D28 à l'étage de l'énoncé signé** : *un protocole ainsi réglé n'a pas de mode de succès négatif.* Et le réflexe naturel — courir quand même parce que c'est bon marché — **produit un chiffre spectaculaire et vide**, exactement le mode que D32 vient de graver. Correctif : **D27**, scission calibrateur/question, primaire neuve **`R_Base`**, poids probant du calibrateur **déclaré nul d'avance**. |
| **0-170** *(majeur)* | **Le « 12 à 15 σ » du journal du 2026-08-26 n'est pas publiable en l'état.** La dépendance LOO est **positive** ⇒ `N` effectif ≪ 5083, **jamais établi**. | **Un écart en σ dont le `N` effectif est inconnu n'est pas un écart en σ.** Famille **D14-R** + **D24**. Fatal ici parce que **tout le brin `σ±` en dépend** : sans anomalie établie, `Q-M1` tombe **pour une raison qui n'est pas la sienne**. Correctif : **`V-grappe`** bloquante, classe **`Σ-mort`** gravée d'avance, **rectification du journal autorisée par le PI le 2026-08-27** (D30 al. 1). |
| **0-171** *(majeur)* | **`N-queue` était spécifiée sur un SEUIL de magnitude commun ; la sélection réalisée est un RANG par vecteur**, le seuil réalisé fluctuant (Gumbel, SD 15-20 %). **Et la définition du rang d'une coordonnée partagée admettait quatre lectures licites.** | *« `Ψ` serait un résidu à une loi qui n'est pas celle du dispositif »* — une primaire dont la nulle décrit **un autre appareil** que celui qui a produit les données. Second membre = **mode 0-155** : quatre lectures, aucune gelée. Correctif : `Q-M1-bis` ; **`r = max(rang_A, rang_B)` gelé**, les trois autres **énumérées et rejetées par écrit**. |
| **0-172** *(majeur)* | **`σ̂±(cos̄)` au lieu de `σ̂±_pool` construirait le facteur manquant DANS la primaire.** | Le facteur ~10 recherché (`ρ_eff ≈ −0.023` contre `−0.002`) **est plausiblement l'écart entre ces deux agrégations**. Comparer une quantité **poolée** à une prédiction **non poolée**, c'est **fabriquer le résidu qu'on prétend mesurer**. Correctif : `σ̂±_pool` gelée littéralement, **prédiction testable pré-enregistrée** : *la moyenne de `cos_paire` pondérée par `\|A∩B\|` ≈ −0.02*. |
| **0-173** *(majeur)* | **`Δ_Base` avait deux écritures licites** (ratio des sommes / apparié par paire) et **l'agrégation des `R` tirages n'était pas gravée**. | **Troisième occurrence du mode 0-155 dans le projet.** Deux écritures de la primaire décisionnelle ⇒ deux partitions possiblement **opposées** sur les mêmes données. Correctif : gravure littérale (§4.1). |
| **0-174** *(majeur)* | **`μ_global` non-LOO induit un biais directionnel ≈ +5·10⁻³ sur `ρ_Base(μ_global)` SEUL**, donc **directement dans `Δ_Base`** — plausiblement **un demi-`ε_B`**. La bascule conditionnelle était insuffisante. | **Mode 0-117 exact** : un biais **de la taille de la quantité décidée** n'est pas une correction de second ordre — **il fabrique la classe**. Et il est **asymétrique** : il n'affecte **qu'un** des deux termes de la différence. Correctif : **LOO-par-paire en principal, inconditionnellement**. |
| **0-175** *(majeur)* | **`N-état` non apparié en `(tige, domaine)` mesurerait la STRATE, pas le privilège** (`cos(h_j, paire)` varie de 0.15 à 0.34) ; **et l'appariement est nécessaire mais NON SUFFISANT.** | Sans appariement, `Δ_Base` porterait le facteur que les strates encodent — **le confondant que tout le cycle précédent a passé son temps à séparer**. Et la moitié « non suffisante » est la plus dangereuse : un appariement **paraît** clore la question. Correctif : appariement aux **deux** membres, **cardinal du vivier publié par strate**, **`Δ̂_Base(γ,c)` publiée d'avance**, **cellule D23** en S3. |
| **0-176** *(mineur)* | **64 buckets de rang ⇒ `se ≈ 0.056` : profil illisible.** | **Une porte de contrôle inexploitable est une porte absente** — et c'était la porte chargée d'exclure l'artefact de coupure. Correctif : **8 buckets dyadiques**, planchers, fusion pré-déclarée. |
| **0-177** *(mineur)* | **`t = 2.66` en dur** au lieu de `ndtri(1 − 1/256)` : la pente `(φ/Q)²` varie de **1.5 %**. | Famille **0-132 / D21**. 1.5 % sur la pente d'une primaire dont l'écart cherché vaut ~11 unités sur ~170 est **de l'ordre du dixième de l'effet**. |
| **0-178** *(mineur)* | **Le bootstrap de 10 grappes a ~92 % de rééchantillons à grappe manquante** ; cardinal effectif et méthode de quantile non publiés. | **D24** : une nulle qui perd sa variabilité **a l'apparence d'un plancher sans en être un**. Troisième occurrence du motif. |
| **0-179** *(mineur)* | **Cosinus publiés à moins de 4 décimales** : `−0.0019` et `−0.0023` **pèsent 20 % l'un sur l'autre** dans `σ̂±`. | Famille 0-132. **La précision d'affichage devient une variable manipulée** quand la primaire est un résidu. |
| **0-180** *(mineur)* | **`N-hyp` : un seul `S` partagé entre toutes les paires** ⇒ termes corrélés, `DEFF_S > 1` ⇒ **la formule binomiale est fausse**. | Nulle d'échelle **mal bornée**. Sans effet décisionnel (poids probant nul par D32), mais **une nulle fausse publiée reste une nulle fausse, et elle se recopie** (mode 0-139). Correctif : `q₉₅` sur `R` tirages de `S`, `DEFF_S` publié. |
| **0-181** *(mineur)* | **Une seule cause candidate (`β ≈ 10 %`) pour deux quantités**, comptée comme deux confirmations indépendantes. | Double comptage d'évidence — **le mode le plus discret de l'auto-confirmation**. Correctif : interdit **(xxix)**. |

**Total du cycle : vingt-cinq défauts (0-157 … 0-181), dont huit critiques. Aucune mesure conduite,
aucun GPU touché.** *Cinquième reconduction du motif : **les deux défauts critiques de la
consolidation ont été trouvés par dérivation, sans une donnée** — l'un par `lab-math` en quadrature,
l'autre par `lab-neuro` en relisant les lignes du code lui-même.*

---

## 1. Question

À la couche d'accroche, avec la `G` du projet (seed 0), le recouvrement de `A ∩ B` avec
`top64(G·μ_global)` **excède-t-il ce que les seuls cosinus prédisent** sous le modèle gaussien
bivarié exact — et l'écart de `σ±` à 0.5, **une fois établi sous bootstrap par grappe**, est-il
**entièrement compté** par la corrélation par paire et par la sélection top-64 ?

## 2. Hypothèses, antipodes, vocabulaire interdit

### 2.1 Calibrateur — `H-Base` *(ratifiée, évaluée verbatim, poids probant déclaré NUL d'avance)*

> `|A ∩ B ∩ top64(G·μ_global)| / |A ∩ B| ≥ 0.50` sur **les 4 strates** et **3/3 modèles**.

**Antipode `C-Base-mort` (ratifié)** : `< 0.25` sur **≥ 2 modèles/3** ⇒ la clause (B) de l'interdit
tombe, `top64(G·μ_global)` ne capture que la queue la plus stable, **`k²/D` n'est pas remplaçable
comme nulle opératoire**, et *le recouvrement brut contient une structure que ni le cosinus ni l'état
moyen ne prédisent* — **cellule qui justifierait un chantier, à instruire avec son propre
pré-enregistrement.**
**Lecture gelée (0-164)** : réalisé ⇔ point `< 0.25` sur **4/4 strates** d'au moins **2 modèles/3**.
**Atteignabilité déclarée d'avance (D31)** : `C-Base-mort` exigerait `γ ≈ 0.5` ; les normes mesurées
donnent `γ ≈ 0.75–0.89` ⇒ **classe déclarée QUASI INATTEIGNABLE avant mesure**, par `V-quad`.
*Déclarée d'avance, jamais découverte après.*
**Cellule d'indécision (ratifiée)** : part dans `[0.25, 0.50)` ⇒ *« indécidable ICI »*, résolution
publiée ; **jamais** « effet partiel ».

### 2.2 Calibrateur apparié — `Δ_Base` *(D16)*

> **`Δ_Base = [Σ_p n_p(μ_global^{LOO}) − Σ_p n̄_p(x_null)] / Σ_p |A∩B|_p`**, `n̄_p` = **moyenne sur les
> `R` tirages, PAR PAIRE** — écriture **gravée littéralement** (0-173).

**Antipode `Δ-nul`, signé `lab-neuro`** : `IC(Δ_Base) ⊂ [−2ε_B, +2ε_B]` ⇒ ***« `μ_global` n'est pas
privilégié : un état quelconque, apparié, fait aussi bien »***, mots exacts, sans habillage.
**Deux bornes signées** : (1) **`Δ-nul` ne licencie RIEN** — ni *« `μ_global` est sans effet »*
(D28 : *« effet borné par 2 × la résolution »*), ni lecture représentationnelle (**fausse par
provenance**), ni X1, ni D11 ; (2) la réparation de D32 n'est **pas** de l'affaiblir mais d'exiger
qu'elle **publie un couple, jamais un nombre** ⇒ **D34, gravée le 2026-08-27**.
**Second antipode `Δ-anti`** : `IC_sup(Δ_Base) < −ε_B` ⇒ **consigné, court-circuitant, NON INTERPRÉTÉ,
non muet** (trois descripteurs, §4.5).
**Cellule modale gravée d'avance (`lab-math` (4))** : ***« `ρ_Base ≈ ρ̂_Base(γ)` et `Δ_Base` dans le
couloir : `μ_global` n'est privilégié qu'à hauteur de sa proximité cosinus — plancher produit, contenu
= le chiffre lui-même. »***
**Ordinal signé `lab-neuro` (N6)** : **`Δ_Base(S0) ≥ Δ_Base(S3)`**, conjonction bornée par le **`min`**
(0-63), **bootstrap joint** sur le même rééchantillon de grappes. **Antipode** : ordre inverse au-delà
de la résolution jointe ⇒ **son motif est faux**. **Indécision** : `|écart| ≤ 2 ×` la résolution ⇒
classe d'équivalence. **Cellule modale signée d'avance** : `Δ-nul` sur les deux ⇒ **l'ordre n'a pas de
domaine ⇒ `SANS OBJET`, jamais `0`**.

### 2.3 Question #1 — `H-R` *(primaire neuve, signée `lab-director`)*

> **`R_Base = ρ_Base − ρ̂_Base(·)`** — résidu au niveau que **les seuls cosinus** prédisent sous le
> modèle gaussien bivarié **exact** (propriété pivot : `(z_a,i, z_b,i)` iid en `i`, bivariés de
> corrélation **exactement** `ρ = cos(x_a, x_b)`, sur l'aléa de `G`).

**`H-R`** : *à ce locus, sur ce matériau, le recouvrement de l'intersection avec `top64(G·μ_global)`
**excède** ce que les cosinus seuls prédisent* — `IC_inf(R_Base) > ε_R` sur ≥ 1 cellule.
**Antipode `R-nul`, énuméré (D13)** : `IC(R_Base) ⊂ [−2ε_R, +2ε_R]` ⇒ ***« le recouvrement avec la
projection de la moyenne empirique des 360 états est entièrement compté par les cosinus ; il n'y a
rien de plus à ce locus »*** — lecture TOST : **« résidu borné par 2 × la résolution »**, jamais
« aucune structure ».
**Second antipode `R-moins`** : `IC_sup(R_Base) < −ε_R` ⇒ **contredirait le modèle pivot lui-même** ⇒
**d'abord un suspect de bug**, consigné, trois descripteurs obligatoires, **aucune cause nommée**.
**C'est la seule quantité du cycle dont l'issue n'est pas prédictible avant mesure.**

### 2.4 Question #2 — `H-Ψ` *(primaire neuve, signée `lab-math`, sous correction de rang de `lab-neuro`)*

> **`Ψ = σ± − σ̂±_pool`**, `σ̂±_pool = Σ_p |A∩B|_p·σ̂±(cos_p, t) / Σ_p |A∩B|_p` — **gelée
> littéralement, jamais `σ̂±(cos̄)`** (0-172).
> Identité de Plackett : `σ̂±(ρ, t) = 1/2 + (ρ/2)(φ(t)/Q(t))² + O(ρ²)` ; à `t = ndtri(1 − 1/256) =
> 2.6601`, `(φ/Q)² = 8.81`, **pente 4.41**. **`t` calculé au banc, jamais en dur** (0-177).

**Préalable BLOQUANT (`lab-neuro`, N5)** : `σ±` **re-exprimé sous bootstrap par grappe** (unité
décisionnelle) **avant toute phrase le qualifiant**. Si l'écart à 0.5 **ne survit pas**, classe
**`Σ-mort`** : *il n'y a pas d'anomalie à expliquer*, **`N-queue` est sans objet** et **`Q-M1` tombe
pour une raison qui n'est pas la sienne** — à écrire dans ces mots.
**Prédiction testable pré-enregistrée (`lab-math`)** : la moyenne de `cos_paire` **pondérée par
`|A∩B|`** ≈ **−0.02**.
**Cellule D13 exigée par `lab-math`, gravée** : `Ψ` significatif = ***« modèle bivarié + agrégation
gelée insuffisants »***, **PAS** « structure nouvelle ».
**Signature `lab-neuro` (N2), verbatim** : si **`Σ-epuise`**, *« la clause 1 de `P-σ` se réalise pour
une raison qui ne lui appartient pas et **ne compte pas à son crédit** : elle est retirée du registre
des paris comme **réalisée-sans-mérite** »*.

### 2.5 Ce qui est déclaré `SANS OBJET` avant mesure (D23)

| Objet | Statut | Motif |
| --- | --- | --- |
| **`P-σ` clause 2** | **`SANS OBJET`**, `T` publié par `V-T` | Support vide : paires **intra-cadre par construction** (0-157). **Ne licencie aucune borne, pas même une équivalence TOST.** |
| **`C-σ-mort`** | **déclaré INATTEIGNABLE d'avance** (D31), par `V-T` | Identité du barycentre (`Q-M4`) contre un seuil `\|cos\| ≤ 2/√d ≈ 0.05–0.07`. **Le vide de 0-157 ne le réalise PAS.** |
| **`P-σ` clause 1** | **retirée du registre des prédictions par son auteur**, **citée** en acquis (§3) | Déjà mesurée. *Une mesure déjà faite se cite, ne se refait pas.* |
| **`Ξ_perm` / `N-cadre` / `P-cadre`** | **NON OUVERTS** | A-2. Fiche `P-cadre` déposée, **sans signataire**, à concevoir à froid avec son antipode propre. |
| **E1, E2, `E3 ≤ +0.05 nats/token`, « 0 write », VRAM, D7** | **`SANS OBJET`, nommés, jamais omis** | Aucune lecture injectée, aucune NLL modifiée, `M` jamais instanciée, 0 GPU. |

### 2.6 Vocabulaire interdit

**Reconduits sans exception** : **(i)-(viii)** d'I2 ; **(ix)-(xiii)** de v4, dont le **plafond
lexical** ; **(xiv)** rien sur le chemin d'**ÉCRITURE** depuis une mesure de **LECTURE** — *le +57 %
d'E2 de X1 reste acquis et intact* ; **(xv)** *« sémantique »* interdit, maximum licite
*« appartenance au sous-vivier déclaré »* ; **(xvi)** phrase gravée de `lab-neuro` ; **(xvii)** aucun
énoncé d'effet aval ; **(xviii)** le centrage n'a **aucune désignation biologique** ; **(xix)** *« le
noyau commun est le code de fond du gyrus denté »* ; **(xx)** *« le centrage répare `φ` »* et toute
formulation où une classe licencie `I3` ; **« séparation de patterns »** interdit ici (0-59) ; **bin
dur (A-5)** sous toute forme, y compris adverbiale ; **support vide ⇒ `SANS OBJET`, jamais `0`** ;
**indécidable ⇒ cause nommée**.

**Créés par ce cycle — (xxi) à (xxiv), rédigés par `lab-neuro`, verbatim :**

- **(xxi)** — *toute désignation biologique de `top64(G·μ_global)` est interdite : « code de fond »,
  « cellules préférentiellement recrutées », « noyau d'allocation », « ensemble partagé ».
  Formulation licite : **« indices retenus par `topk` appliqué à la projection de la moyenne empirique
  des 360 états, pour cette `G` (seed 0) »**. **Propriété manquante nommée** : dans le gyrus denté, le
  recrutement préférentiel est **fabriqué par l'expérience** et **s'épuise** (Yiu et al. 2014 ; Cai et
  al. 2016) ; ici `G` et `μ_global` n'ont vu **aucun** matériau, sans épuisement ni décroissance.*
- **(xxii)** — *extension de (xviii) à la **composante de rang ~1** : elle se nomme **« direction
  dominante des états à ce locus »**, jamais « mode commun », « activité de fond », « piédestal
  tonique », « drive tonique ».*
- **(xxiii)** *(façade nommée par son propre auteur)* — *« les résidus d'un ensemble fermé se
  repoussent » se lit comme une inhibition latérale et n'en a **aucune** propriété : interdite.*
- **(xxiv)** — *`σ± < 0.5` **n'est pas un fait de neurosciences** : les taux de décharge sont **non
  négatifs**, le signe d'une coordonnée projetée n'a **pas de référent**. **Aucun compte neuro ne doit
  en être propriétaire.***

**Créés par ce cycle — (xxv) à (xxix), `lab-director` :**

- **(xxv) — l'identité algébrique.** Toute quantité dont la valeur est **forcée par une identité**
  (barycentre, Cauchy-Schwarz, appartenance) **ne se lit jamais comme un résultat**. Formulation
  obligatoire : ***« valeur imposée par l'identité \<nommée\> ; poids probant nul »***.
- **(xxvi) — le plancher.** Un plancher **produit** ne s'écrit jamais *« le canal est innocenté »*.
  Licite : ***« toute mesure future de structure de support à ce locus doit dépasser \<valeur\> ; en
  deçà, elle re-publie l'état moyen sous un autre nom »***.
- **(xxvii) — la valeur prédite par le cosinus.** *« `ρ_Base` est élevé »* ne se lit **jamais** comme
  une propriété du support. Formulation obligatoire : ***« à hauteur de la proximité cosinus `γ` ;
  valeur prédite `ρ̂_Base(γ)` = ·, résidu `R_Base` = · »***. **Un `ρ_Base` publié sans son `ρ̂_Base`
  apparié rend le rapport non écrivable.**
- **(xxviii) — `σ±`.** *« La géométrie explique `σ±` »* interdite tant que `Ψ` n'est pas attribué ;
  *« expliqué partiellement »* interdit ; et **avant `V-grappe`, aucune phrase qualifiant l'ampleur de
  l'écart** (ni « massif », ni un chiffre en σ) n'est publiable.
- **(xxix) — le double comptage.** Une **seule** cause candidate expliquant **deux** quantités
  (`cos_type` et `σ±` par la structure de cadre) **ne produit pas deux confirmations indépendantes**.

**Phrase de périmètre, reconduite verbatim (`lab-neuro`)** :
> *« Cette Étape A vaut par son plancher D32 et par la clôture de `σ±`, pas par ce qu'elle pourrait
> apprendre sur le mécanisme ; à 10 minutes de CPU elle est bon marché, et **elle ne doit rien
> retarder**. Si `Q-M1` n'est pas livrée, la partition `Σ` est retirée, `σ±` reste `NON EXPLIQUÉ` avec
> sa cause nommée — **et `V2-D(b)` s'ouvre quand même**. »*

Ni la loi 2 ni **D11** ne sont touchées : aucune lecture n'est injectée, `M` n'est jamais instanciée,
aucune NLL n'est modifiée — **la quantité sur laquelle D11 se prononce n'existe pas dans ce run**.

---

## 3. Provenance (D14-R)

### 3.1 Glossaire gelé *(0-167)*

| Nom | Définition | Cardinal | Source |
| --- | --- | --- | --- |
| **cellule de capture** (`c`) | cadre de requête `(type, variante)` | **6** | `eval/materiel_v4.py:872` |
| **unité décisionnelle** (`i`) | un des 60 états par cellule, **les mêmes 60 dans les 6 cellules** | **60** (`U = 60`) | `support_overlap.py:2230-2232` ; `materiel_v4.py:126` |
| **cellule décisionnelle** | `(modèle, strate)` | **12** | `cellules_decisionnelles()`, `support_overlap.py:656` |
| **`T`** | nombre de **cadres** servant de `type` à `μ_type` | **à publier par énumération** (`V-T`) ; présumé 6 | à vérifier, jamais à croire |

> ***« Les 6 cellules S0 » se lit : la strate S0 dans chacune des 6 cellules de capture, par
> modèle.*** Lecture **gelée** ; toute autre est un défaut de protocole.

### 3.2 Estimateurs *(0-166, 0-174 — la ligne que le cycle précédent a manquée deux fois)*

| Symbole | Estimateur exact | LOO ? | Statut |
| --- | --- | --- | --- |
| **`μ_global^{LOO}`** | `(S − z_a − z_b)/358`, `S = Σ` des 360 états | **OUI, par paire** | **MESURE PRINCIPALE**, inconditionnelle (0-174, confirmé PI) |
| `μ_global` | `Σ_c Σ_i h_{c,i}/360` (l. 2241-2243) | NON | **descriptif**, écart publié (D26) |
| `μ_type` | `(Σ_{i∈c} z_i − z_a − z_b)/(n_dec − 2)`, `n_cell = 60` (l. 2270-2271) | **OUI**, par paire | condition `type` de la double mesure |
| `top64(G·x)` | `support_topk(G @ x, 64)` (l. 2332) | — | un ensemble d'indices par modèle et par référence |

**Biais publié, jamais absorbé** : perturbation directionnelle du non-LOO ≈ `(2/360)(‖h‖/‖μ‖) ≈ 0.007`
de cosinus ; sensibilité `dp/dγ ≈ 0.67` ⇒ **≈ +5·10⁻³ sur `ρ_Base(μ_global)` seul**, **plausiblement
un demi-`ε_B`** (`Q-M8`). **C'est pourquoi le LOO est principal et inconditionnel.**

### 3.3 Acquis, avec leur source

| Fait | Chiffre | Source, date | Étiquette |
| --- | --- | --- | --- |
| Propriété **pivot** | `(z_a,i, z_b,i)` **iid en `i`, gaussiens bivariés de corrélation exactement `ρ = cos(x_a, x_b)`**, sur l'aléa de `G` | `lab-math`, 2026-08-27, **démontrée** | **re-dérivée** — fonde `Q-M1`, `Q-M6`, `Q-M9` **sans hypothèse sur `h`** |
| `Q-M9` | `γ ≈ 0.75–0.89` ⇒ **`ρ̂_Base ≈ 0.55–0.75`** ; `x_null` même cellule : `cos ≈ 0.56–0.80` ⇒ **`Δ̂_Base ≈ 0.05–0.15`** | `lab-math`, 2026-08-27, quadrature 1-D | **re-dérivé** — fonde **0-169** et la scission |
| `Q-M1` | Plackett ⇒ pente **4.41** à `t = 2.6601` ; `σ̂±(−0.002) = 0.4912` ⇒ **le conditionnement fournit ~14 des ~170 requis, il manque ~11** ; il faudrait **`ρ_eff ≈ −0.023`, soit 10 ×** | `lab-math`, 2026-08-27, **démontrée** | **re-dérivée** — fonde `Ψ` et **0-172** |
| `Q-M2` | cos attendu sous LOO-par-paire = **`+1/(n−1) = +0.0169`, POSITIF** ; non-LOO et LOO-simple : `−0.0169`. **Mesuré −0.0019/−0.0023 : aucun des trois.** ⇒ **`σ±` n'est PAS un effet du centrage** ; **anti-corrélation intrinsèque ≈ −0.019** | `lab-math`, 2026-08-27, **démontrée** | **re-dérivée** — **écarte par dérivation la branche « le cycle n'a pas d'objet »** |
| `Q-M4` | Sous permutation de cadres, `cos` moyen concentre à **`−0.20` ± ~0.01**, `P(≤ −0.15) ≈ 1 − 3·10⁻⁷` | `lab-math`, 2026-08-27 | **re-dérivé** — confirme 0-159 |
| `σ±` sur les S0 sous `type` | **0.392 / 0.403 / 0.410** ; **uniforme en `\|A∩B\|`** ; **non concentrée** (top-15 = 513/5083) ; dépendance LOO **positive** | journal 2026-08-26 | vérifié — **l'ampleur en σ est RECTIFIÉE le 2026-08-27** (0-170) |
| `cos_type` sur ces cellules | **−0.0023 / −0.0021 / −0.0019** (**à republier à ≥ 4 décimales**, 0-179) | journal 2026-08-26 | vérifié — **clause 1 de `P-σ`, citée, non refaite** |
| Lecture `lab-neuro` | *uniformité* **incompatible** avec un porteur de rang dominant ; *non-concentration* **exclut** `Core` ⇒ *« l'effet vit dans le volume, pas dans le noyau »* | `lab-neuro`, 2026-08-27 | **acquis contraignant** — **confirme A-2 par un second chemin** |
| `cos` de `φ(h)` avec la `G` du projet | plage **[0.1471, 0.3368]** | `V-G` v2, `experiments/results/recouvrement-supports/V-G.json`, 2026-08-26 | mesuré, troisième chemin |
| Régime d'égalité | `cos/f ∈ [0.991, 0.994]` ⇒ égalité de Cauchy-Schwarz ⇒ `σ± = 1.0` **exact** sur brut | journal 2026-08-26 | vérifié — fonde 0-162 |
| `f` | enrichie de **1.066 à 1.102** ⇒ **`N7` fermée** | journal 2026-08-26 | vérifié |
| Excès `O − p_sym` | `aucun` **+2.437 ± 0.337** ; `type` **+0.509 ± 0.378** ; `plac` **+2.904 ± 0.715** | journal 2026-08-26 | vérifié |
| Classes `N` | S3 `HAUT`, S2 `BAS`, S1 `ind_L`, S0 `BAS`, 3/3 ⇒ **`C-strat`**, `H_sup` **REJETÉE** | journal 2026-08-26 | vérifié |
| `Core` / `Core-G` | **2 / 3 / 0** ; **1/1, 1/1, `SANS OBJET`** | journal 2026-08-26 | vérifié |
| Motif de `P-Base` | **les 12 indices les plus partagés sont tous dans `top64(G·μ_global)`, 3/3** | D32, 2026-08-26 | vérifié — **motif, jamais prédiction** |
| `‖μ_global‖/‖h‖` | **0.73–0.85** ; direction `1` : 0.010–0.023 | journal 2026-08-26 | vérifié — **entrée de `Q-M9`** |
| Cardinaux | S3 **360** · S2 **540** · S1 **2160** · S0 **7560** (unité de `P`) ; par cellule de capture **60 / 90 / 360 / 1260** ; `n_cell = 60`, `k = 64`, `D = 8192`, `d = 768/960/1536` | journal 2026-08-26 + code | vérifié |
| Structure des strates | **S0 = 100 % inter-tige** ; **S3/S2 = 100 % intra-tige** | 0-155 / D31 | vérifié — **entrée obligatoire du contrôle D31, §11** |
| Hashes des états v4 | **première gravure**, `V-cache` PASS, 0 GPU | 2026-08-26 | **reconduits à l'identique** |

---

## 4. Prédictions, nulles, portes, partitions

### 4.1 Quantités

| Symbole | Définition | Unité |
| --- | --- | --- |
| `ρ_Base(x)` | `Σ_p |A∩B∩top64(G·x)|_p / Σ_p |A∩B|_p` — **ratio des sommes, gelé** | numérateur et dénominateur **entiers publiés** |
| `ρ̂_Base(·)` | prédiction de quadrature sous le modèle pivot, **forme fonctionnelle gelée au banc** (`Q-M10`) | fraction, **≥ 4 décimales** |
| **`R_Base`** | `ρ_Base(μ_global^{LOO}) − ρ̂_Base(·)` | **PRIMAIRE — question #1** |
| **`Δ_Base`** | `[Σ_p n_p(μ) − Σ_p n̄_p(null)] / Σ_p |A∩B|_p`, `n̄_p` = moyenne des `R` tirages **par paire** | **PRIMAIRE — calibrateur apparié (D16)** |
| `Δ̂_Base(γ, c)` | prédiction de quadrature, **publiée d'avance** | fraction |
| `σ±(b)` | accord de signe par **bucket dyadique `b`** de 8 rangs, **`r = max(rang_A, rang_B)` gelé** | fraction, nulle 0.5 |
| `σ̂±_pool` | `Σ_p |A∩B|_p·σ̂±(cos_p, t)/Σ_p |A∩B|_p` — **gelée littéralement** | fraction |
| **`Ψ`** | `σ± − σ̂±_pool` | **PRIMAIRE — question #2** |
| `n_vide` | paires à `|A∩B| = 0` | compte entier, par cellule |
| `T`, `U`, cardinal du vivier `x_null` | comptés **par énumération** | comptes entiers |

### 4.2 Nulles — une par maillon, toutes bloquantes (D17)

| Maillon | Nulle | Type | Statut probant |
| --- | --- | --- | --- |
| **Échelle de `ρ_Base`** | `N-hyp` : `\|A∩B∩S\| ~ Hypergéom(8192, m, 64)`, `E = 1/128` ; **`q₉₅` sur `R` tirages de `S`, `DEFF_S` publié — jamais la formule binomiale** (0-180) | exacte, corrigée | **POIDS PROBANT NUL, déclaré** (c'est `k²/D`, D32) — **échelle seule** |
| **Niveau prédit par les cosinus** | **`N-quad`** : `ρ̂_Base(·)` par quadrature sous le modèle pivot ⇒ **nulle de `R_Base` = 0 par construction** | **dérivée, zéro hypothèse sur `h`** | **maillon décisionnel #1** |
| **Privilège de `μ_global`** | **`N-état`** : `x = h_j` hors paire, **apparié `(tige, domaine)` aux DEUX membres**, même cellule de capture, `R` tirages ; **`SANS OBJET` si le vivier est vide (S3)** | rééchantillonnage, statistique gelée §11 | **clé nulle appariée (D16)** — **nécessaire, non suffisante** sans `Δ̂_Base` |
| **Magnitude / anisotropie de `G`** | **`N-plac`** : `x = c·r`, `‖c·r‖ = ‖μ_global‖`, `R` directions | par construction | contrôle du décalage mécanique de magnitudes (0-106) |
| **`σ±` — existence de l'anomalie** | **`N-grappe`** : bootstrap par **grappe = unité décisionnelle**, cardinal effectif et méthode de quantile publiés (D24) | rééchantillonnage | **BLOQUANTE — préalable à tout `Ψ`** (0-170) |
| **`σ±` — compte par la sélection** | **`N-queue`** : `σ̂±_pool`, **sélection top-64 PAR VECTEUR** (rang, non seuil) — **simulation exacte, ou erreur d'approximation publiée** (`Q-M1-bis`, 0-171) | dérivée / simulée | **maillon décisionnel #2** |
| **Provenance de `G`** | `torch.randn(8192, d, generator=seed(0))`, lignes iid | par construction | échec de `V-G` ⇒ **arrêt de provenance** |

### 4.3 Le taux — licite, et à quel titre

1. **Comme quantité PUBLIÉE, le taux est OBLIGATOIRE** : D32 (ii) le nomme, D20 exige **un plancher à
   produire**. `ρ_Base` **est** le plancher.
2. **Comme statistique DÉCISIONNELLE, il est INTERDIT** (D16). La décision porte sur `Δ_Base`
   (calibrateur apparié) et sur **`R_Base`** (question).
3. **`Q-M9` durcit le point 2** : `ρ_Base` est **prédictible avant mesure à partir de `γ` seul**. Le
   publier sans son `ρ̂_Base` apparié est **interdit par (xxvii)** et **rend le rapport non écrivable**.
   **C'est exactement ce que D34 grave, le 2026-08-27 : un couple, jamais un nombre.**

### 4.4 `ε_B`, `ε_R`, `ε_Ψ` — lignes canoniques uniques, recopiées à l'identique au §7

> **`ε_B` = q₀.₉₅ de `|Δ_Base|` sur `B = 10⁴` rééchantillons bootstrap de tiges (`K_eff = 10`) sous
> permutation appariée, par paire, des étiquettes `{μ_global^{LOO}, x_null}`, l'agrégation des `R`
> tirages étant celle du §4.1 ; une valeur par modèle et par strate, calculée et gelée AVANT lecture
> des `Δ_Base` observés.**

> **`ε_R` = q₀.₉₅ de `|R_Base|` sur `B = 10⁴` rééchantillons bootstrap de tiges (`K_eff = 10`),
> INCLUANT le terme `O(1/√D)` de l'espérance sur `G` (C6, `Q-M11`) ; une valeur par modèle et par
> strate, gelée avant lecture.**

> **`ε_Ψ` = q₀.₉₅ de `|Ψ|` sous `N-queue` propagée par bootstrap de GRAPPES (unité décisionnelle,
> `B = 10⁴`), une valeur par modèle et par bucket, gelée avant lecture.**

**`R` est DÉRIVÉ** : `R = ⌈10·σ̂_ref²/σ̂_Δ²⌉`, **`σ̂_ref` gelée nommément** (`lab-math`), pilote
`R₀ = 20`, **plafond `R ≤ 300` publié**, dépassement **remonté au PI, jamais absorbé** (0-52).
**Puissance déclarée (`Q-M5`)** : `Δ_min ≈ (1.96 + 1.64)·σ_tige/√10 ≈ 1.14·σ_tige` ⇒ **`Δ_min` publié
PAR CELLULE après `R₀ = 20`, AVANT la mesure principale**.
**Région masquée, publiée** : `Φ((2ε_B − Δ̂)/se − 1.96)` sous l'alternative **`Δ̂_Base(γ, c)`** —
*l'alternative chiffrée qui manquait*. **Détection étouffée assumée, jamais silencieuse** (0-131).

### 4.5 Partitions — exhaustives, exclusives, ordre gravé (D18)

**Principe transversal (D28)** : marge de significativité **1×**, couloir d'équivalence **2×**,
**couloirs ABSOLUS**.

**Partition `B` — CALIBRATEUR, poids probant déclaré nul** (par modèle × strate, condition `aucun`,
`P ≥ 8`) :

| Ordre | Classe | Condition | Verdict gravé |
| --- | --- | --- | --- |
| 0 | `B-vide` | `Σ_p |A∩B|_p = 0` | **`SANS OBJET`** (D23), `n_vide` publié, **ne licencie aucune borne** |
| 1 | `B-haut` | `IC_inf(ρ_Base) ≥ 0.50` | `H-Base` soutenue **sur cette cellule** ; **plancher D32 (ii) produit = `IC_inf`** ; **lecture obligatoire (xxvii)** |
| 2 | `B-mort` | point `ρ_Base < 0.25` *(seuil ratifié, non durci)* | `C-Base-mort` — **classe déclarée QUASI INATTEIGNABLE d'avance** par `V-quad` |
| 3 | `B-ind` | complémentation, **en dernier** | *« indécidable ICI »*, résolution publiée ; **jamais** « effet partiel » |

**Partition `Δ` — CALIBRATEUR APPARIÉ** (par modèle × strate) :

| Ordre | Classe | Condition | Verdict gravé |
| --- | --- | --- | --- |
| 0 | `Δ-sansobjet` | vivier `x_null` apparié **vide** (attendu possible en S3) | **`SANS OBJET`**, cardinal publié, **jamais repli silencieux** |
| 1 | `Δ-anti` | `IC_sup(Δ_Base) < −ε_B` | **consignée, court-circuitante, NON INTERPRÉTÉE, non muette** : oblige (i) `ρ_Base(état)` par décile de `cos(h_j, μ_global)`, (ii) `\|top64(G·μ) ∩ top64(G·h_j)\|`, (iii) `‖μ_global‖/‖h‖`. **Aucune cause nommée avant ces trois chiffres.** |
| 2 | **`Δ-nul`** | `IC(Δ_Base) ⊂ [−2ε_B, +2ε_B]` | ***« `μ_global` n'est pas privilégié : un état quelconque, apparié, fait aussi bien »*** — mots exacts. **Ne licencie RIEN.** TOST : *« privilège borné par 2 × la résolution »*, **jamais** « pas de privilège ». |
| 3 | `Δ-priv` | `IC_inf(Δ_Base) > ε_B` | *« à ce locus, sur ce matériau, pour cette `G` »* — **ne licencie aucune ligne de `engram/`** |
| 4 | `Δ-ind` | complémentation, **en dernier** | sous la résolution ; publier `σ_tige`, `K_eff`, `Δ_min` |

**Partition `R` — QUESTION #1** (par modèle × strate) :

| Ordre | Classe | Condition | Verdict gravé |
| --- | --- | --- | --- |
| 0 | `R-vide` | `B-vide` ou `ρ̂_Base` non définie | **`SANS OBJET`** |
| 1 | `R-moins` | `IC_sup(R_Base) < −ε_R` | **contredit le modèle pivot** ⇒ **d'abord suspect de BUG** : consigné, trois descripteurs (`f`, `n_vide`, écart fp32/fp64), **aucune cause nommée** |
| 2 | **`R-nul`** | `IC(R_Base) ⊂ [−2ε_R, +2ε_R]` | ***« le recouvrement avec la projection de la moyenne empirique des 360 états est entièrement compté par les cosinus ; il n'y a rien de plus à ce locus »*** — TOST : **« résidu borné par 2 × la résolution »**, jamais « aucune structure » |
| 3 | `R-plus` | `IC_inf(R_Base) > ε_R` | résidu positif — **DESCRIPTIF**, aucune cause nommée, chantier **à instruire ailleurs avec son propre pré-enregistrement** |
| 4 | `R-ind` | complémentation, **en dernier** | sous la résolution, **cause nommée** (puissance / support / précision) |

**Partition `Σ` — QUESTION #2** (par modèle × bucket ; **conditionnée à `Q-M1-bis` livrée**) :

| Ordre | Classe | Condition | Verdict gravé |
| --- | --- | --- | --- |
| 0 | **`Σ-mort`** | l'écart de `σ±` à 0.5 **ne survit pas** à `N-grappe` | ***« il n'y a pas d'anomalie à expliquer »*** ; **`N-queue` sans objet**, **`Q-M1` tombe pour une raison qui n'est pas la sienne** |
| 1 | `Σ-vide` | bucket sous plancher (**< 200 trials OU < 8 paires**) | **`SANS OBJET`**, fusion dyadique **pré-déclarée** appliquée d'abord |
| 2 | **`Σ-epuise`** | `IC(Ψ) ⊂ [−2ε_Ψ, +2ε_Ψ]` sur **tous** les buckets **et** profil conforme à la loi | ***« l'écart de `σ±` à 0.5 est entièrement compté par la corrélation par paire et par la sélection top-64 »*** ⇒ **`σ±` cesse d'être `NON EXPLIQUÉ`**, **sans aucune géométrie** ⇒ clause 1 de `P-σ` **réalisée-sans-mérite** (N2, verbatim) |
| 3 | `Σ-residu` | `IC_inf(|Ψ|) > ε_Ψ` sur ≥ 1 bucket | ***« modèle bivarié + agrégation gelée insuffisants »*** — **PAS** « structure nouvelle » ; `σ±` **reste `NON EXPLIQUÉ`** |
| 4 | `Σ-ind` | complémentation, **en dernier** | sous la résolution, **cause nommée** |

**Partition `K` — réduite** : **`K-vide` déclarée d'avance** par `V-T`, `T` publié, **`C-σ-mort`
déclaré inatteignable**. Aucune autre classe. **Aucun `Ξ`.**

**Espace de verdict : `B` × `Δ` × `R` × `Σ` = 4 × 5 × 5 × 5 = 500 cellules formelles ; les cellules
décisionnelles atteignables sont recomptées PAR ÉNUMÉRATION au banc, jamais par affirmation.**

**Cellules dont la lecture est gravée d'avance :**

| Cellule | Lecture gravée |
| --- | --- |
| **`B-haut × Δ-nul × R-nul`** *(issue modale, signée `lab-math` (4) et `lab-neuro` (N4))* | ***« `ρ_Base ≈ ρ̂_Base(γ)` et `Δ_Base` dans le couloir : `μ_global` n'est privilégié qu'à hauteur de sa proximité cosinus — plancher produit, contenu = le chiffre lui-même. »*** ⇒ **D32 (ii) lue en couple (D34)**. **Aucune lecture représentationnelle, dans les deux sens.** |
| **`B-haut × R-plus`** | Le support capture **au-delà** de ce que les cosinus prédisent — **descriptif**, **aucune cause nommée**, chantier avec son propre pré-enregistrement. **Le seul résultat positif que ce cycle puisse rendre.** |
| **`B-mort × ·`** | `C-Base-mort` — **mais la classe est déclarée quasi inatteignable d'avance** : sa réalisation impose d'abord un **audit d'instrument** (`V-quad`, `V-loo`, `V-appar`) avant toute lecture. |
| **`Δ-nul` sur S0 ET S3** | **L'ordinal N6 n'a pas de domaine ⇒ `SANS OBJET`, jamais `0`** — signée d'avance par `lab-neuro`. |
| **`Σ-mort`** | *Il n'y a pas d'anomalie.* Le brin `σ±` s'éteint **sans que `Q-M1` soit en cause** — et **`V2-D(b)` s'ouvre quand même**. |
| **`Σ-epuise`** | La clause 1 de `P-σ` se réalise **pour une raison qui ne lui appartient pas et ne compte pas à son crédit** : **réalisée-sans-mérite**. |
| **`R-ind × Σ-ind`** | *« indécidable ICI »* — **cause nommée** avant toute suite. *« Augmenter la résolution » n'est pas une suite par défaut.* |

### 4.6 Portes — toutes exécutables, toutes bloquantes

| Porte | Contenu | Coût |
| --- | --- | --- |
| **`V-cache`** | Hashes des `raw/etats-*.npz` **identiques à la gravure du 2026-08-26**, republiés. Divergence ⇒ **ARRÊT** (changement de matériau ; **pas de re-forward** — le périmètre grave 0 GPU ; **confirmé PI, P8**) | s |
| **`V-G` v2** | `G` instanciée, **hash publié par modèle** ; `cos` fp32 **et** fp64, écart publié, tolérance `1e−6`. Échec ⇒ **arrêt de provenance** | s |
| **`V-mu`** | Formules citées **par ligne de code** ; `μ_global` **recalculé par un second chemin**, écart publié ; **majorant de fuite publié**. Chiffre **recopié** au lieu de relu ⇒ **arrêt** | min |
| **`V-loo`** *(NEUVE — 0-174)* | `μ_global^{LOO}` est **la mesure principale** ; non-LOO **descriptif** ; **les deux publiés**, écart comparé à `ε_B`. Inversion ⇒ **arrêt** | banc |
| **`V-quad`** *(NEUVE — le pivot)* | `ρ̂_Base(γ_cell, t)` et `Δ̂_Base(γ, c)` **calculées et publiées PAR CELLULE, AVANT toute mesure**, forme **gelée** (`Q-M10`) ; **et déclaration d'atteignabilité de `B-mort`** (D31). Non-exécutée ⇒ **`ρ_Base` retiré de l'interprétation** | banc |
| **`V-T`** | `T`, cardinal des paires de cadres `t ≠ t'` en S0, trois cardinaux du glossaire — **par énumération** ; **déclaration d'avance de l'inatteignabilité de `C-σ-mort`**. Antipode inatteignable non déclaré ⇒ **run invalide** | banc |
| **`V-vide`** | `n_vide` publié **par cellule et par condition** ; dénominateur nul ⇒ **`SANS OBJET`, jamais `0`** | banc |
| **`V-appar`** *(NEUVE — 0-175)* | `x_null` apparié en **`(tige, domaine)` aux DEUX membres** ; **cardinal du vivier publié PAR STRATE** ; vivier vide ⇒ **`Δ-sansobjet`**, jamais repli | banc |
| **`V-ecriture`** *(NEUVE — 0-173)* | Écriture littérale de `Δ_Base` et agrégation des `R` **vérifiées contre le texte gravé**, avec un cas où les deux lectures **divergent** | banc |
| **`V-grappe`** *(NEUVE, BLOQUANTE — 0-170, 0-178)* | `σ±` re-exprimé sous **bootstrap par grappe** ; **cardinal effectif** et **méthode de quantile** publiés (D24). **Aucune phrase qualifiant l'ampleur de `σ±` avant PASS** ((xxviii)) | banc |
| **`V-queue`** *(NEUVE — 0-171, 0-172, 0-177)* | `σ̂±_pool` **gelée littéralement** ; **`t = ndtri(1 − 1/256)` calculé au banc, jamais en dur** ; **sélection top-64 PAR VECTEUR** | banc |
| **`V-rang-def`** *(NEUVE — 0-171, D31)* | **`r = max(rang_A, rang_B)` gelé** ; les trois autres lectures **énumérées et rejetées par écrit** ; **8 buckets dyadiques**, planchers **≥ 200 trials ET ≥ 8 paires**, **fusion pré-déclarée** | banc |
| **`V-ident`** *(NEUVE — (xxv))* | Toute quantité **forcée par une identité** déclarée comme telle **avant mesure**, avec sa dérivation ; le banc l'exhibe à sa valeur forcée **sur des données aléatoires**. Ici : `ρ_Base` sous régime `γ`, `O ≥ p_sym`, `cos` des centroïdes. Non-déclaration ⇒ **quantité retirée de l'interprétation** | banc |
| **`V-S`** *(NEUVE — 0-180)* | `q₉₅` de `N-hyp` sur `R` tirages de `S`, **`DEFF_S` publié** ; formule binomiale **interdite** | banc |
| **`V-decimales`** *(NEUVE — 0-179)* | Cosinus, `ρ_Base`, `R_Base`, `Ψ` publiés à **≥ 4 décimales** ; `O` en **entiers exacts** | banc |
| **`V-P8`** | Toute quantité agrégée sur **< 8 paires** exclue, cardinal publié | banc |
| **`V-ulp`** | Marge à la coupure `k = 64` publiée pour **toutes** les conditions, **y compris `μ_global^{LOO}`** (358 vs 360 change des rangs de queue) ; **fp64 = chemin nominal, déclaré partout** (D21) | banc |
| **`V-seed`** | Seed, générateur, règle de tirage **gelés et publiés** ; **test de PRÉSENCE, jamais de vérité** (0-145) ; cardinal **par modèle** (0-133) | banc |
| **`V-diag`** | `ρ_Base`, `ρ̂_Base`, `R_Base`, `Δ_Base`, `Δ̂_Base`, `n_vide`, `f`, `σ±`, `σ̂±_pool` publiés **par strate, par condition, par modèle**. Sans eux, **le rapport ne peut pas être écrit** | banc |
| **`V-perimetre`** | §4.7, quatre éléments obligatoires | rédaction |

### 4.7 Hors-périmètre déclaré — `V-perimetre`

1. **Mécanisme.** Aucun effet aval : `M` jamais instanciée, aucune lecture injectée, aucune NLL
   modifiée. Quantités **géométriques**, cortex gelé.
2. **Limite nommée — le point de contact `keysim`.** `read_gate=keysim` se calcule sur
   `cos(φ(h), clés)` ; plancher mesuré **0.147-0.337**, étendue inter-strates **0.064-0.091** ⇒
   **variable à offset marqué et dynamique modérée**. **Ce n'est pas un verdict sur le gate**
   ((xvii)) : c'est une **désignation de chantier**.
3. **Successeur désigné.** **Q-06**, calibration de `gate_keysim_mid` par modèle. **Non ouvert,
   résultat non anticipé.**
4. **Phrase de périmètre de `lab-neuro`, reconduite verbatim** (§2.6).

---

## 5. Contrôles et baselines

1. **Configuration courante** : `EngramConfig()` par défaut — **citée** ; **ni `M` ni backprop** (D8),
   `G` gelée (D9).
2. **`M` reset / D7** : **`SANS OBJET`**, nommé. Contrôle homologue : **`G` gelée, seed 0, identique
   aux trois références** — seule l'entrée `x` change.
3. **Contrôle de l'explication triviale — TROIS, pas un** (D26 : la difficulté migre) :
   **`N-plac`** (magnitude, 0-106) ; **`N-état` apparié aux deux membres** (privilège) — *seul
   contrôle qui mord sous le régime rang-1* ; **`N-quad`** (niveau prédit par les cosinus) — *le seul
   qui mord sous `Q-M9`*. **Les deux premiers étaient nécessaires et non suffisants ; le troisième est
   la réponse à la migration.**
4. **Double mesure D26, aux quatre endroits où la difficulté peut migrer** : `aucun` / `type` ;
   `μ_global^{LOO}` / non-LOO ; fp32 / fp64 ; **`ρ_Base` / `ρ̂_Base`**.
5. **Flag à off, même code** : les trois références passent par **le même chemin** — différence
   **uniquement dans `x`**.
6. **Ordre des conditions** : mêmes paires, même ordre, **même cache `Z`** ⇒ **aucun effet d'ordre**.
7. **Contrôle de l'artefact de coupure `topk`** : `σ±` **en 8 buckets dyadiques**,
   `r = max(rang_A, rang_B)` gelé, **prédiction signée** sous `N-queue` — la loi prédit que **l'écart
   à 0.5 CROÎT avec la magnitude**. **Profil plat ⇒ la loi est fausse ; profil conforme ⇒ `σ±` épuisé
   sans géométrie.** *Ce n'est pas « exclure un artefact » : le `ρ` de la loi EST le `cos_paire`
   (0-165), et la seule chose qui les départage est le profil.*
8. **Unité d'échange** : **tiges** (`K_eff = 10`) pour `Δ_Base` et `R_Base` ; **unités décisionnelles**
   (grappes) pour `σ±` ; IC 95 % bootstrap, `B = 10⁴`, **cardinal effectif et méthode de quantile
   publiés** ; conjonctions par **`min`** et **bootstrap joint** sur le **même** rééchantillon (0-63 —
   **tout produit de p-valeurs par modèle rend le run invalide**).
9. **Portée** : **« pour cette `G` », seed 0** (D9, C6). Les prédictions `Q-M1`/`Q-M9` sont des
   **espérances sur `G`**, écart `O(1/√D)` **inclus dans `ε_R`**.

---

## 6. Critères d'abandon — portes exécutables (D22)

**Ce qui INVALIDE le run** : **A.** Banc D14-S à `E ≠ 0`. · **B.** `V-cache` : hash ≠ gravure du
2026-08-26. · **C.** `V-G` échec. · **D.** `V-mu` / `V-loo` échec, ou chiffre **cité de mémoire**. ·
**E.** `V-quad` non exécutée **avant** mesure. · **F.** `V-T` non exécutée, ou antipode déclaré
atteignable sans énumération (D31). · **G.** `V-ecriture` : écriture de `Δ_Base` non conforme au texte
gravé. · **H.** NaN **propagé** au lieu d'arrêté ; `ρ_Base ∉ [0,1]` ; valeurs sous la précision de
`V-decimales`. · **I.** `V-diag` incomplète ⇒ **rapport non écrivable**. · **J.** `V-P8` violée :
cellule exclue ; **> 2 strates** sur un modèle ⇒ **modèle exclu**. · **K.** Toute phrase qualifiant
l'ampleur de `σ±` publiée **avant** PASS de `V-grappe` ((xxviii)).

**Ce qui TUE `H-R`** : **`R-nul`** ⇒ *« entièrement compté par les cosinus »*. **C'est un résultat,
pas un échec.**
**Ce qui TUE `H-Ψ`** : **`Σ-residu`** ⇒ *« modèle bivarié + agrégation gelée insuffisants »*, **jamais**
« structure nouvelle ».
**Ce qui ÉTEINT le brin `σ±` sans le tuer** : **`Σ-mort`** ⇒ *il n'y a pas d'anomalie à expliquer*.
**Ce qui TUE `H-Base`** : `C-Base-mort` (**déclarée quasi inatteignable d'avance** ⇒ audit
d'instrument obligatoire avant lecture).

**Ce qui NE tue rien et se consigne** : `Δ-anti`, `R-moins` (trois descripteurs chacun, **aucune cause
nommée**) ; `K-vide`, `Δ-sansobjet`, `B-vide`, `Σ-vide` (**`SANS OBJET`**) ;
`R-ind`/`Δ-ind`/`Σ-ind`/`B-ind` (*« indécidable ICI »*, **cause nommée**).

**`SANS OBJET` ici, nommés et jamais omis** : « 0 write », **« E3 ≤ +0.05 nats/token »** (aucune
lecture injectée, **la lecture du cortex ne change pas**), E1, E2, VRAM, D7.

---

## 7. Variables fixées

- **Modèles / couches** : GPT-2 (`--layer 6`), `HuggingFaceTB/SmolLM2-360M` (`--layer 16`),
  `Qwen/Qwen2.5-1.5B` (`--layer 14`). `d` = **768 / 960 / 1536** ; directions **non transportables**
  (0-133) ; **les modèles ne sont pas des réplicats** (0-63).
- **`G`** : gelée, `seed = 0`, `dg_dim = 8192`, `dg_topk = 64` — **portée « pour cette `G` »**.
- **Matériau** : états `h` du **matériau qualifié v4**, capturés à **`t`** (`t−1` refusé), **mêmes
  hashes `V-cache`**, `U = 60`, 6 cellules de capture.
- **Strates** : **360 / 540 / 2160 / 7560** (unité de `P`) ; **jamais poolées**.
- **Inférence** : `K_eff = 10` **tiges** (`Δ_Base`, `R_Base`) ; **grappes = unités décisionnelles**
  (`σ±`) ; `B = 10⁴` ; **cardinal effectif et méthode de quantile publiés** (D24).
- **`ε_B`, `ε_R`, `ε_Ψ`** : **lignes canoniques uniques, recopiées à l'identique du §4.4** (anti-0-82).
- **`R`** : dérivé, `σ̂_ref` **gelée nommément**, pilote `R₀ = 20`, **plafond 300 publié**.
- **`t`** : **`ndtri(1 − 1/256)`, calculé au banc en fp64** (0-177).
- **Rang** : **`r = max(rang_A, rang_B)`**, gelé ; **8 buckets dyadiques**, planchers **≥ 200 trials ET
  ≥ 8 paires**, fusion **pré-déclarée**.
- **Couloirs** : **absolus**, `2 × ε` (D28).
- **Précision** : **fp64 chemin nominal, déclaré partout** (D21) ; fp32 en contrôle ; **≥ 4 décimales**.
- **Rien d'autre ne bouge** : aucun hyperparamètre modifié, **aucun fichier de `engram/` touché**,
  **0 GPU**.

## 8. Variable manipulée

**Une seule : le vecteur de référence `x` dont on prend `top64(G·x)`**, à **trois niveaux** :

| Niveau | `x` | Statut |
| --- | --- | --- |
| **`mu`** | **`μ_global^{LOO}`** (principal) ; `μ_global` (descriptif) | condition d'intérêt — la quantité que D32 (ii) nomme |
| **`etat`** | `h_j` hors paire, **apparié `(tige, domaine)` aux deux membres**, `R` tirages | **clé nulle appariée (D16)** |
| **`plac`** | `c·r`, `‖c·r‖ = ‖μ_global‖`, `R` directions | contrôle du décalage mécanique de magnitudes |

**`Ψ` n'a pas de variable manipulée** : c'est une **stratification** d'une quantité déjà mesurée,
comparée à une **loi dérivée**. **`R_Base` non plus** : c'est un **résidu à une prédiction calculée
avant mesure**. *Cela se dit ; c'est le statut honnête du cycle.*

---

## 9. Budget

| Poste | Coût | GPU |
| --- | --- | --- |
| Portes de provenance (`V-cache`, `V-G`, `V-mu`, `V-loo`) | < 30 s CPU | 0 |
| `V-quad` (quadratures par cellule) | **< 10 s CPU** | 0 |
| Cache `Z` fp64 | 51,9 Mo/modèle, ~156 Mo RAM | 0 |
| `ρ_Base` × 3 références × 4 strates × 2 conditions × 3 modèles, en LOO-par-paire | < 3 min CPU | 0 |
| `N-queue` par simulation exacte (sélection top-64 par vecteur) | **négligeable** | 0 |
| Bootstraps `ε_B`, `ε_R`, `ε_Ψ`, `N-grappe` (`B = 10⁴`) | < 2 min CPU | 0 |
| `σ±` en 8 buckets | < 1 min CPU | 0 |
| Banc D14-S étendu (cellules énumérées, cas échouants, cardinaux comptés) | ~4-6 min CPU | 0 |
| **Total mesure** | **< 10 min CPU, ~160 Mo RAM, 0 VRAM** — **conforme au périmètre gravé** | **0** |
| **Temps agent** | ~8-12 h (banc, portes, partitions, `Q-M1-bis`, `Q-M10`) | — |

**`V-cache` échec ⇒ ARRÊT, pas de re-forward.**

---

## 10. Livrables attendus

- **Aucun flag `EngramConfig`, aucune modification de `engram/`.** Aucune classe de ce cycle ne
  licencie une ligne de code : elles licencient **un énoncé, un locus, un matériau, cette `G`**.
- **Script** : **options ajoutées à `eval/support_overlap.py`** (`--base-ref {mu,etat,plac}`,
  `--mu-loo`, `--sigma-rang`, `--quad`) — **motif D29** : les portes, le cache `Z`, `p_sym` et les
  fractions exactes y sont **déjà écrits et déjà relus** ; les réécrire, c'est **19 lignes →
  9 défauts**. SPDX AGPL-3.0-or-later. **Tout correctif > 10 lignes repasse le circuit COMPLET** (D29).
- **Banc D14-S obligatoire AVANT toute mesure, `E = 0`** : un cas **passant** et un cas **échouant
  MORDANT** par clause ; **cardinaux comptés par ÉNUMÉRATION** ; cas échouants explicitement dus pour
  `V-T` (support vide), `V-vide`, `V-appar` (vivier vide en S3), `V-quad` (non exécutée), `V-ecriture`
  (les deux lectures divergent), `V-grappe` (grappe manquante), `V-rang-def` (bucket à 199 trials),
  `V-ident` (quantité forcée non déclarée), `V-S` (`DEFF_S > 1`), `V-seed` (`seed = 0` accepté),
  `V-loo` (inversion).
- **Livrables dus AVANT le banc** : **`Q-M10`** (forme fonctionnelle de `ρ̂_Base`) — **BLOQUANTE** ; et
  **`Q-M1-bis`** (rang, non seuil) — **non bloquante par décision PI** : si elle manque, **la partition
  `Σ` est retirée**, `σ±` reste `NON EXPLIQUÉ` avec sa cause nommée, `R_Base` survit seule, **et
  `V2-D(b)` s'ouvre quand même**.
- **Tests CPU** : `.venv\Scripts\python -m pytest tests/ -q` — les 228 restent verts.
- **Entrée de journal**, format maison.
- **Ligne du tableau `docs/EXTENSIONS.md` §4** : **déclaration explicite qu'aucune ligne n'est due** —
  ni E1, ni E2, ni E3.

---

## 11. Cardinal des permutations effectives, et satisfiabilité du banc (D31 + D14-S)

**Deux nulles de ce protocole sont de rééchantillonnage** : **`N-état`** et **`N-grappe`**.
**`N-cadre` est retirée avec `Ξ_perm`** (A-2) ; sa dérivation (`Q-M3` : cardinal `720^(U−1)`,
dégénérescence seulement si `U = 1`, donc **sans objet** à `U = 60`) est **conservée en provenance**
pour la fiche `P-cadre`.

### 11.1 Statistiques complètes gelées

| Élément | `N-état` | `N-grappe` |
| --- | --- | --- |
| **Unité d'échangeabilité** | **la paire** `(a, b)` | **l'unité décisionnelle** `i` (grappe) |
| **Schéma** | `x_null = h_j`, `j ∉ {a,b}`, **même cellule de capture**, **apparié `(tige, domaine)` aux DEUX membres**, `R` tirages | rééchantillonnage **avec remise des grappes**, `B = 10⁴` |
| **Règle d'agrégation** | **`n̄_p` = moyenne des `R` tirages PAR PAIRE**, puis **ratio des sommes** — **jamais** moyenne des ratios, **jamais** ratio après pooling des `R` | `σ±` **poolé pondéré par `\|A∩B\|`**, cohérent avec `σ̂±_pool` |
| **Traitement des atomes** | vivier apparié **vide** ⇒ **`SANS OBJET`**, cardinal publié, **jamais repli** | rééchantillon à **grappe manquante** (~92 %) ⇒ **compté, publié**, méthode de quantile **déclarée** |
| **Règle de signe** | **aucune** — la statistique est une **différence de deux taux**, pas un produit de signes ⇒ **le mode 0-155 ne se rejoue pas** (sous réserve de la gravure littérale) | aucune |

### 11.2 Vérification D31 explicite

> **C'est ce qui a tué la primaire d'hier** (0-155 : la stratification **ÉTAIT** la relation
> d'identité de l'unité d'échangeabilité).

| Nulle | Unité d'échangeabilité | Stratification | Coïncidence ? |
| --- | --- | --- | --- |
| **`N-état`** | la paire `(a,b)` | **strate** | **DANGER CONFIRMÉ par `lab-math`.** `cos(h_j, paire)` varie de **0.15 à 0.34** selon la strate ⇒ sans appariement, `Δ_Base` **mesure la strate**. **L'appariement est NÉCESSAIRE ; il n'est SUFFISANT que joint à `Δ̂_Base(γ,c)` publiée d'avance.** Les deux sont exigés. **Cellule D23 prévue** : en **S3**, le vivier **peut être VIDE** ⇒ `Δ-sansobjet`. |
| **`N-grappe`** | unité décisionnelle `i` | **bucket de rang** | **NON** — le bucket est une propriété de la **coordonnée**, l'unité une propriété de l'**état**. **À vérifier par énumération au banc, jamais par cette affirmation.** |
| **Bootstrap `R_Base`** | tige (`K_eff = 10`) | strate | **PARTIELLE et connue** (S3/S2 intra-tige). **Aucune règle de signe n'intervient** ⇒ mode 0-155 non rejoué. **À confirmer par `lab-math` (`Q-M14`).** |

### 11.3 Ce que le banc DOIT publier avant le gel

1. **Le cardinal effectif par cellule**, compté **par exécution** : vivier de `x_null` **par strate**,
   rééchantillons de grappes **distincts**, trials **par bucket**.
2. **Les classes déclarées inatteignables d'avance** (D31) : **`C-σ-mort`** (`V-T`), **`B-mort`**
   (`V-quad`) — *déclarées avant, jamais découvertes après*.
3. **`T` et le cardinal des paires de cadres `t ≠ t'`** — **présumé 0**, à **confirmer par énumération**.
4. **`ρ̂_Base` et `Δ̂_Base` par cellule** (`V-quad`), **avant toute mesure**.
5. **Un cas passant et un cas échouant MORDANT par clause**, `E = 0`, **couverture 100 %**. *Rappel
   0-144 : le comptage a détecté un facteur 64 qu'aucune relecture n'aurait vu, en 2 s.*

---

## 12. Décisions du PI — 2026-08-27

| # | Question | Décision |
| --- | --- | --- |
| **P1** | Forme du cycle | **Scission D27** — calibrateur (`ρ_Base`, `Δ_Base`, poids probant nul déclaré) + questions neuves (`R_Base`, `Ψ`). |
| **P2** | Clause 2 de `P-σ` : (A) contre (B) | **(A) `SANS OBJET` ratifiée.** `P-cadre` **NON OUVERTE**, sans signataire. |
| **P3** | `Ξ_perm` | **Retrait confirmé.** `β ≈ 10 %` versé intact à la fiche `P-cadre`. |
| **P4** | `D34` | **GRAVÉE le 2026-08-27** dans les termes de `lab-neuro` ; D32 (ii) se lit désormais **en couple**. |
| **P5** | Rectification du journal du 2026-08-26 | **AUTORISÉE**, D30 alinéa 1, traçabilité complète. Seule **l'ampleur en σ** est retirée ; les valeurs, l'uniformité, la non-concentration et le corrélat **restent acquis**. |
| **P6** | Blocage du banc | **`Q-M10` bloquante ; `Q-M1-bis` non bloquante** — si elle manque, `Σ` est retirée et **`V2-D(b)` s'ouvre quand même**. |
| **P7** | `μ_global` LOO-par-paire | **Principal, inconditionnel.** |
| **P8** | `V-cache` divergent | **ARRÊT sans re-forward.** |

---

## 13. Questions restantes aux experts

### 13.1 `lab-math`

- **`Q-M10` (BLOQUANTE)** — **forme fonctionnelle exacte de `ρ̂_Base`** à geler au banc : quels
  cosinus entrent (`γ_a`, `γ_b`, `ρ_ab`), et la **dérivation** de la quadrature. *Sans elle, `R_Base`
  n'a pas de nulle.*
- **`Q-M1-bis` (non bloquante, décision PI)** — nulle avec **sélection top-64 PAR VECTEUR** :
  **simulation exacte** ou **erreur d'approximation seuil-fixe/rang publiée**. **L'un des deux, jamais
  ni l'un ni l'autre.**
- **`Q-M11`** — **`ε_R`** : dérivation, **avec le terme `O(1/√D)`** (C6).
- **`Q-M12`** — **`DEFF_S`** pour `N-hyp`, et le `q₉₅` sur `R` tirages.
- **`Q-M13`** — **`N` effectif de `σ±` sous grappes** (0-170) : résolution réelle, et **`Σ-mort`
  est-elle une issue plausible** ?
- **`Q-M14`** — confirmer que **`Δ_Base` et `R_Base` ne rejouent pas 0-155** sous l'écriture littérale
  gravée, **et** que le bootstrap de tiges y est licite malgré la coïncidence partielle S3/S2.

### 13.2 `lab-neuro`

- **`Q-N1`** — signature de la **cellule modale gravée** `B-haut × Δ-nul × R-nul`, **avant mesure**.
- **`Q-N2`** — sous `Q-M9`, `Δ̂_Base ≈ 0.05–0.15` **partout** : **maintient-il l'ordinal N6**, sachant
  que sa propre cellule modale (`SANS OBJET`) est la plus probable ?
- **`Q-N3`** — **`R_Base` est une primaire neuve qu'il n'a pas signée.** A-t-il un **ordinal ou un
  signe** à poser dessus, avec son antipode (D13) — ou la laisse-t-il **sans engagement neuro**, ce qui
  est licite et se consigne ?
- **`Q-N4`** — sous **`Σ-mort`**, confirme-t-il que ce résultat **ne rouvre aucune de ses hypothèses
  antérieures** et se lit *« il n'y avait rien à expliquer »*, sans habillage ?
- **`Q-N5`** — la fiche **`P-cadre`** : **confirme-t-il qu'il n'en est pas l'auteur** ?

---

## Registre des engagements

| Engagement | Auteur | Statut | Antipode / cellule modale |
| --- | --- | --- | --- |
| **`P-Base` / `C-Base-mort`** | `lab-neuro`, ratifié PI | **MAINTENUS, évalués verbatim — POIDS PROBANT DÉCLARÉ NUL** | `C-Base-mort` **déclaré quasi inatteignable d'avance** |
| **`P-σ` clause 1** | `lab-neuro` | **RETIRÉE DU REGISTRE PAR SON AUTEUR** ; citée | sous `Σ-epuise` : **réalisée-sans-mérite** |
| **`P-σ` clause 2 / `C-σ-mort`** | `lab-neuro` | **`SANS OBJET`** / **INATTEIGNABLE déclaré d'avance** | — |
| **`Δ-nul`** | `lab-neuro` (N4) | **SIGNÉE**, deux bornes | ne licencie **rien** ; **D34** |
| **Ordinal `Δ_Base(S0) ≥ Δ_Base(S3)`** | `lab-neuro` (N6) | **SIGNÉE**, `min`, bootstrap joint | cellule modale **`SANS OBJET`**, signée d'avance |
| **`Σ-epuise`** | `lab-neuro` (N2) | **SIGNÉE d'avance, verbatim** | — |
| **(xxi)-(xxiv)** | `lab-neuro` | **GRAVÉES verbatim** | quatre façades nommées, **dont la sienne** |
| **`Ψ` / `σ̂±_pool` / pente 4.41** | `lab-math` | **SIGNÉE**, sous correction de rang de `lab-neuro` | `Σ-residu` = *« modèle + agrégation insuffisants »* |
| **`ρ̂_Base(γ)` / `Δ̂_Base(γ,c)`** | `lab-math` (`Q-M9`) | **GRAVÉES au banc AVANT mesure** | fondent 0-169 et la scission D27 |
| **`R_Base`** | **`lab-director`** | **SIGNÉE — primaire neuve** | `R-nul` (TOST), `R-moins` (suspect de bug) |
| **`Ξ_perm` / `P-cadre`** | — | **NON OUVERTE, SANS SIGNATAIRE** | à concevoir à froid |
| **Pari de `lab-director`**, hors protocole, aucun poids décisionnel | `lab-director`, 2026-08-27 | **CONSIGNÉ** | `ρ_Base ∈ [0.55, 0.72]` · **`R-nul`** · **`Δ-nul`** · **`Σ-epuise`** · `σ±` **survit** à `V-grappe`. *Posé avant mesure pour que l'interprétation soit auditable — et trois cycles consécutifs viennent d'être perdus.* |
| **D29** | PI | **RECONDUITE** | tout correctif > 10 lignes ⇒ circuit **complet** |

---

## Ce qui reste indécidable, et le restera après ce run

*Écrit avant la mesure, pour qu'aucune de ces phrases ne soit tentée après.*

1. **L'effet aval.** Aucune injection, aucune NLL. Rien sur E1, E2, E3, le gate, la capacité, ni le
   0/10 top-10.
2. **Le chemin d'écriture de `M`.** Le **+57 % d'E2 de X1 est acquis et intact** ; ce cycle porte sur
   la **lecture** de `φ` et **ne le touche pas** ((xiv), 0-75).
3. **La généralité à une autre `G`.** Portée **« pour cette `G` », seed 0**.
4. **La cause d'un `R-plus`.** Il dirait qu'un résidu existe ; **il ne dirait pas pourquoi**.
5. **L'objet de `Ξ` / `β ≈ 10 %`.** Réel selon `lab-math`, **hors cible** selon `lab-neuro` :
   **`P-cadre`, non ouverte**.
6. **`I3` et le profil par couche.** Ce cycle mesure **un locus** ; `I3` reste conditionné à **`P11`**.
7. **`gate_keysim_mid` (Q-06).** Point de contact **nommé**, **non instruit**.
8. **Le canal suffixe** (0-62). Hors périmètre, **matériau propre requis**.

---

## Historique

- **2026-08-27** — **Brouillon (mode cadrage).** Onze défauts (0-157 … 0-167), dont six critiques ;
  `P-Base` reformulée en différence ; trois arbitrages ouverts.
- **2026-08-27** — **Avis `lab-neuro` : RÉSERVÉ.** **Retire son propre mécanisme (N1)**, refuse
  l'option (B) sous le mode 0-122, **ferme son compte par ses propres données (N5)**, **suspend le
  « 12 à 15 σ »**, corrige la spécification de `Q-M1` (rang, non seuil), signe `Σ-epuise` et `Δ-nul`
  d'avance, pose l'ordinal N6 **avec sa cellule d'évanouissement**, rédige (xxi)-(xxiv) **dont sa
  propre façade**, et fixe le périmètre : *« elle ne doit rien retarder »*.
- **2026-08-27** — **Avis `lab-math` : RÉSERVÉ.** Démontre la **propriété pivot**, `Q-M1`, `Q-M2`
  (*« `σ±` n'est PAS un effet du centrage »*), `Q-M4`, et surtout **`Q-M9`** — qui établit que **les
  deux énoncés ratifiés sont quasi forcés**. Neuf corrections.
- **2026-08-27** — **Consolidation.** Quatorze défauts de plus (0-168 … 0-181). **Trois tranchages** :
  (A) l'emporte sur (B) — *le Directeur perd* ; **`Ξ_perm` retiré** — *`lab-math` perd sa correction
  (6)* ; **scission D27** retenue. **16 corrections sur 16 traitées : 14 intégrées, 1 écartée avec
  motif, 1 renforcée au-delà de ce que son auteur demandait.**
- **2026-08-27** — **Gate PI : huit décisions rendues** (§12). **D34 gravée**, **rectification 0-170
  autorisée et portée**. `Statut : PROPOSE`.
- **2026-08-28** — **PRÉ-ENREGISTRÉ PAR LE PI.** À partir d'ici, le **§4 (Prédictions, nulles,
  portes, partitions)** et le **§6 (Critères d'abandon)** sont **GELÉS — plus jamais modifiés, par
  personne**. Suite : `Q-M10` (bloquante) à `lab-math`, puis banc D14-S à `E = 0`, puis mesure.
  **Aucune mesure avant PASS intégral du banc.**
