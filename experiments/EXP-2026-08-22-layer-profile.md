# EXP — I2 : profilage par couche du cortex (`layer_profile`)

**Statut : PROPOSE**

Origine : `docs/EXTENSIONS.md` §2 (I2) · cadrage PI du 2026-08-22.
Dépendances d'ordonnancement : **après** la clôture du cycle méthode D14-ext (banc `eval/gate_bench.py` à 100 %), **avant** toute mesure GPU de V2-D(a) v3.
Instrument, pas un mécanisme : aucune injection, aucune écriture, `M = 0` partout, `engram/` non modifié.

---

## 0. Écarts entre le cadrage et l'état réel — levés explicitement

| # | Écart | Levée |
| --- | --- | --- |
| **O-1** | Le cadrage demande l'insertion « **avant** la rédaction de v3 ». **v3 est déjà rédigé** (`EXP-2026-08-22-knn-borne-logits-v3.md`, `PROPOSE`, commit `10627d2`). | La **valeur prédictive est préservée** : aucun GPU n'a tourné, la gate du banc est fermée, aucun résultat E1 de v3 n'existe. I2 peut donc être consigné **avant que v3 ne mesure quoi que ce soit**. **Ce qui est perdu et n'est pas masqué** : I2 **ne peut pas revendiquer avoir motivé** le bras L6 de v3 — ce bras vient de P6. I2 en **prédit** l'issue ; il n'en est pas la cause. Cette phrase est à recopier au journal. |
| **O-2** | Le cadrage prescrit « les gabarits `VARIED_PAIRS` prévus pour v3 ». **`VARIED_PAIRS` n'est plus le jeu de v3** (décision 14 du cycle : remplacé par `pool.fact_pairs(30)` + `POOL_PARAPHRASES`). | L'exigence de fond — *cohérence des instruments* — pointe vers le jeu **réellement** utilisé par v3. Et il est **strictement meilleur ici** : `VARIED_PAIRS` offre 10 gabarits distincts avec **un seul indice exact chacun**, il n'a pas les 3 paraphrases par fait que le score contrastif exige ; `POOL_PARAPHRASES` a exactement cette forme (30 faits × 3 paraphrases, règles déterministes gelées). **Corpus (a) = jeu d'unités v3.** |
| **O-3** | **Confondant introduit par les règles gelées de `POOL_PARAPHRASES`** : para3 se termine par `"It's "` **pour les 30 unités**. C'est une vertu pour la primaire de v3 (prior local constant entre unités) et un **confondant direct** pour le score contrastif — à la position de fin d'indice, la similarité **inter-faits** est gonflée sur para3, ce qui **déprime le ratio** et peut **déplacer l'argmax**. | **Pré-enregistré** : le score est ventilé par **couple de types** de paraphrase, et le score-titre est la version **équilibrée par type** (§4.2). Le confondant est mesuré, pas supposé : porte **V-suffixe** (§4.3). |
| **O-4** | La référence secondaire est annoncée « Lad et al., arXiv:2404.xxxx — vérifier le numéro exact avant gravure ». | **Vérifié le 2026-08-22 : le numéro est faux.** La référence est **Lad, Gurnee & Tegmark, arXiv:2406.19384**. Corrigé dans `EXTENSIONS.md` et ici. |
| **O-5** | Le critère d'échec dit « si P-B ou P-C échoue (**les deux** quantités désignent des couches éloignées) ». **Ambigu** : l'échec exige-t-il que les deux soient loin, ou une seule suffit-elle ? | **Désambiguïsé et pré-enregistré en trois cellules** (§4.4). C'est le genre exact d'ambiguïté qui a invalidé trois runs de V2-D(a) : elle se tranche **avant** mesure. La cellule « une près, une loin » n'est pas un échec — c'est **le résultat le plus informatif du lot**, et le cadrage le dit lui-même. |
| **O-6** | Si l'instrument désigne pour GPT-2 une couche-clé **différente de 6**, faut-il déplacer le bras L6 de v3 ? | **Hypothèse posée, à confirmer ou infirmer par le PI** : **non, les deux bras de v3 restent F et L6 inchangés.** Raison : ajouter un troisième bras ré-ouvrirait la multiplicité que la décision 12 du cycle vient de fermer (deux bras portent une multiplicité gérée ; quatre non). La désignation de I2 est alors consignée comme **prédiction en attente**, et v3 la teste au cycle suivant. **Si le PI préfère l'inverse, c'est une modification de v3 avant pré-enregistrement — légitime, mais c'est sa décision, pas la mienne.** |

---

## 1. Question

Où, dans la profondeur d'un cortex gelé, vit (a) l'invariance à la paraphrase lexicale et (b) la vallée de compression — et ces deux couches coïncident-elles avec la couche d'injection validée empiriquement par D3 ?

## 2. Hypothèse

**H_I2** — les deux quantités sont **non plates en profondeur** et désignent chacune une couche **médiane**, et au moins l'une des deux tombe dans la fenêtre de la couche D3 validée du modèle.

**H₀** — les deux courbes sont plates, ou leurs extrema tombent aux bords (couche 0 ou couche L−1), auquel cas l'instrument ne porte aucune information de profondeur et le balayage empirique de D3 reste la seule méthode.

**Distinction gravée, maintenue dans tout le protocole** : **couche-clé** (lecture / reconnaissance — le rôle qui subsiste dans M_out, qui injecte aux logits) et **couche-injection** (écriture dans le flux résiduel, mécanisme v1) sont **deux rôles distincts**. **Leur coïncidence est une SORTIE de l'instrument, jamais une hypothèse.** Toute phrase du rapport qui traite les deux comme un seul « bon endroit » est interdite d'écriture.

**Interdictions de vocabulaire (avant mesure)** :

- **(i)** L'argmax du score contrastif s'écrit **« la couche où l'invariance à la paraphrase lexicale est la plus contrastée »**, jamais « la couche qui comprend le sens ».
- **(ii)** Le minimum d'entropie matricielle s'écrit **« vallée de compression »**, jamais « goulot d'information » ni « couche optimale ».
- **(iii)** Une coïncidence des deux extrema **ne démontre pas** que lire et écrire se font au même endroit : elle est **compatible** avec cette lecture et avec plusieurs autres. La propriété non mesurée s'appelle **tolérance à l'injection** — I2 ne l'approche pas, il ne touche jamais au flux résiduel.

**Conformité** : D8 (aucun gradient) · D11 (sans objet : aucun canal, aucune injection, rien à évaluer par position) · D12 (aucune association ΔNLL ici) · D13 (antipodes, §4.4) · D14 (chaque seuil porte sa dérivation) · **D14-S** (toutes les portes passent au banc) · D14-R (aucun chiffre de P6 dans une porte).

## 3. Ce que le projet sait déjà — statut de provenance

| Fait | Chiffre | Source | Statut D14-R |
| --- | --- | --- | --- |
| Couche d'injection validée | GPT-2 **6/12** ; SmolLM2 **16/32** ; Qwen **14/28** *(non validée — posée par la règle n/2)* | D3, journal v1 / v1.2 | **orientation et cible de comparaison** ; les valeurs 6 et 16 sont **validées par balayage**, celle de Qwen **ne l'est pas** |
| Profondeurs des trois modèles | **L = 12 / 32 / 28** | **re-mesuré le 2026-08-22** (`AutoConfig`, CPU) | re-mesuré ; **à re-lire du `config` au run**, jamais recopié |
| P6 : clé couche 6 h = 1.00 vs état final h = 0.33 | `N_eff = 3` | run v2 **INVALIDÉ** | **motive le design, n'entre dans aucune porte ni prédiction** |
| Aplatissement, lecture sans composante directionnelle | cos ≈ 0 ; ΔH +0.141 | X7 | orientation seule |

**Dérivations portées en entier :**

- **Les trois couches D3 valent exactement `L/2`** : 6 = 12/2, 16 = 32/2, 14 = 28/2. La « règle du milieu » est donc littéralement `⌊L/2⌋` sur les trois modèles.
- **Fenêtre de tolérance, dérivée et non posée** : le cadrage fixe pour GPT-2 la zone 5-7, soit **±1 sur L = 12**, c'est-à-dire une fraction de profondeur **1/12**. Transposée : `w(L) = max(1, ⌊L/12⌋)` ⇒ **GPT-2 : ±1** (fenêtre [5, 7]) ; **SmolLM2 : ±2** (fenêtre [14, 18]) ; **Qwen : ±2** (fenêtre [12, 16]). Le **plancher** (⌊·⌋ plutôt que ⌈·⌉) est le choix **conservateur** : une fenêtre plus étroite rend la prédiction **plus facile à falsifier**, ce qui est la bonne direction pour un instrument dont on veut tester la valeur prédictive. Une seule règle, trois fenêtres — pas trois constantes à la main.
- **Score contrastif** : pour un fait *f* et deux paraphrases *p ≠ q*, `s_intra(f) = moyenne_{p≠q} cos(h_ℓ(f,p), h_ℓ(f,q))` ; pour deux faits *f ≠ g*, `s_inter = moyenne cos(h_ℓ(f,p), h_ℓ(g,q))`. **Score = `s_intra / s_inter`**. La forme **ratio** annule le décalage de mode commun dû à l'anisotropie ; le cosinus brut ne l'annulerait pas. **Antipode arithmétique déclaré** : si `s_inter ≤ 0` à une couche, le ratio change de signe et devient ininterprétable ⇒ cette couche est **NON ÉVALUABLE**, comptée et publiée, jamais silencieusement omise.
- **Entropie matricielle**, Eq. 1 de Skean et al., cas α → 1 : sur la matrice des représentations du prompt à la couche ℓ, Gram normalisé, valeurs propres normalisées à somme 1, `H = −Σ λ_i log λ_i`. **La convention exacte de normalisation doit être re-dérivée depuis la source par le Builder avant implémentation — elle n'est pas recopiée de mémoire ici** (D14-R). Si la source est inaccessible, le protocole s'arrête sur ce point plutôt que de deviner.

## 4. Prédictions chiffrées

### 4.1 Corpus

- **(a) faits déclaratifs** : le **jeu d'unités de v3** — `pool.fact_pairs(30)` + table d'unités v3 (secret 5 substitué) + `POOL_PARAPHRASES` (3 indices par unité). État capturé **à la position de fin d'indice**.
- **(b) texte neutre** : `NEUTRAL_TEXT` de `eval/collateral.py` (343 positions) — **contrôle** : la vallée se déplace avec la nature de l'entrée (Skean et al., Fig. 3). Sans ce bras, une vallée médiane est indiscernable d'un artefact d'architecture.

**Indépendance instrument / juge** : le corpus (a) est construit à partir des gabarits v3 **avant** qu'un seul résultat E1 de v3 n'existe. **Vérifiable** : `experiments/results/knn-borne-logits-v3/` doit être **absent** au lancement — porte **V-vierge**.

### 4.2 Quantités

Par modèle (**GPT-2, SmolLM2-360M, Qwen2.5-1.5B — les trois d'un coup, le coût le permet**), par couche `ℓ ∈ [0, L)` :

| Quantité | Définition | Candidate |
| --- | --- | --- |
| `contrast(ℓ)` | `s_intra / s_inter`, **équilibré par couple de types** de paraphrase (moyenne des 3 couples {1,2}, {1,3}, {2,3} pondérés également, et non moyenne brute sur les paires) — traitement du confondant O-3 | **couche-clé** |
| `H(ℓ)` sur (a) | entropie matricielle α → 1 | **couche-injection** |
| `H(ℓ)` sur (b) | idem sur `NEUTRAL_TEXT` | contrôle de déplacement |

**Ventilations obligatoires** (descriptives, hors clause) : `contrast(ℓ)` par **couple de types** (3 courbes) ; `s_intra` et `s_inter` **séparément** (une hausse du ratio par effondrement de `s_inter` n'est pas la même chose qu'une hausse de `s_intra`) ; 30 valeurs par unité à la couche argmax.

Sorties : CSV + tracé par modèle ; argmax / argmin consignés avec leur valeur ; `experiments/results/layer-profile/`.

### 4.3 Portes

| Porte | Clause | Dérivation |
| --- | --- | --- |
| **V-vierge** | `experiments/results/knn-borne-logits-v3/` **absent** au lancement | indépendance instrument / juge (§4.1). Présent ⇒ **arrêt** : l'instrument aurait pu être réglé sur le résultat qu'il doit prédire |
| **V-hooks** | les hooks sont posés sur **les L blocs** et **retirés** en fin de run ; `engram/` **non modifié** (diff vide) ; `M` jamais instanciée | garde-fou du cadrage. Un hook qui survit contaminerait tout run ultérieur du même processus |
| **V-1pass** | le nombre de forwards par (modèle, corpus) est **1**, compté et rapporté | c'est la propriété qui rend l'instrument XS : profiler L couches coûte **un** passage. > 1 ⇒ l'implémentation boucle sur les couches, à corriger |
| **V-L** | `L` **re-lu du `config` du modèle** au run, comparé à {12, 32, 28} | D14-R : re-mesuré, jamais recopié. Écart ⇒ **arrêt** (mauvais checkpoint) |
| **V-suffixe** *(confondant O-3)* | fraction des paires **inter-faits** partageant leur **dernier token BPE**, par type de paraphrase, **mesurée et publiée avant** toute lecture des courbes | para3 partage `"It's "` sur les 30 unités ⇒ attendu **≈ 1.0 pour para3**, ≈ 0 pour para1/para2. Si l'attendu n'est **pas** observé, les règles gelées de `POOL_PARAPHRASES` ne sont pas celles qui ont tourné ⇒ **arrêt** |
| **V-signe** | compte des couches où `s_inter ≤ 0` | ratio ininterprétable ⇒ couche **NON ÉVALUABLE**, publiée comme telle. Si > L/2 couches sont NON ÉVALUABLES ⇒ `INCONCLUSIF — ratio dégénéré` |
| **V-plat** | amplitude relative `(max − min)/median` de chaque courbe | **< 0.05 ⇒ courbe déclarée PLATE**, son argmax n'est **pas** interprété (bruit). Dérivation : en dessous de 5 % l'ordre des couches n'est pas séparable de la variabilité inter-unités attendue ; le seuil est **pré-enregistré, pas ajusté après vue** |
| **V-bord** | argmax / argmin en couche **0** ou **L−1** | ⇒ `H₀` non rejetée pour cette quantité : un extremum au bord est le comportement attendu d'une courbe monotone, pas une vallée |

### 4.4 Prédictions décisionnelles et leurs antipodes (D13)

| # | Prédiction | Si vraie | Si fausse | Antipode (D13) |
| --- | --- | --- | --- | --- |
| **P-A** | GPT-2 : `contrast(ℓ)` **maximal en zone médiane** et **strictement supérieur** à `contrast(L−1)` | argmax ∈ [3, 8] **et** `contrast(argmax) > contrast(11)` | sinon | `contrast(11)` **maximal** ⇒ l'état final est la meilleure clé ⇒ **P6 est falsifiée avec un N réel**, et le bras L6 de v3 perd sa motivation. **Résultat majeur, à consigner comme tel** |
| **P-B** | GPT-2 : `argmax(contrast)` **et** `argmin(H)` tombent **tous deux** dans [5, 7] | les deux dans la fenêtre | voir la table à trois cellules ci-dessous | — |
| **P-C** | SmolLM2 : les deux tombent dans **[14, 18]** *(validation croisée : 16 est validée indépendamment par D3 v1.2)* | les deux dans la fenêtre | idem | — |
| **P-D** *(la seule qui engage l'avenir)* | Qwen2.5-1.5B : les couches désignées sont **consignées au journal AVANT tout balayage E1 sur Qwen** | consignation faite | — | le balayage {7, 14, 21} de **Q-08 devient le JUGE** de l'instrument, **jamais la méthode de sélection**. Si le balayage tourne avant la consignation, **P-D est morte** et ne peut pas être ressuscitée |

**Table à trois cellules — désambiguïsation de O-5, pré-enregistrée** (par modèle, pour P-B et P-C) :

| Cellule | Observation | Lecture pré-enregistrée |
| --- | --- | --- |
| **C1 — les deux dans la fenêtre** | `argmax(contrast)` **et** `argmin(H)` dans la fenêtre D3 | **P-B / P-C tenue.** L'instrument prédit le point d'injection. Décision candidate au PI : D3 gagne un prédicteur mécanique |
| **C2 — une dans la fenêtre, une hors** | exactement une des deux | **NI succès NI échec — c'est le résultat le plus informatif du lot.** À consigner : *« qualité représentationnelle ≠ tolérance à l'injection »*, distinction que Skean et al. ne font pas. **Nommer laquelle** : si `argmin(H)` est dans la fenêtre et `argmax(contrast)` hors, la couche-injection est prédictible et la couche-clé ne l'est pas (et réciproquement). v3 retombe sur le balayage classique pour le rôle non prédit, **et pour lui seul** |
| **C3 — les deux hors** | ni l'une ni l'autre | **P-B / P-C échouée.** L'instrument ne prédit pas le point d'injection. Consigner *« qualité représentationnelle ≠ tolérance à l'injection »* comme **résultat en soi**, et v3 retombe intégralement sur le balayage classique. **Ce n'est pas un échec de cycle** : c'est une réponse |

**Interdit** : lire les courbes puis choisir laquelle des deux quantités « comptait ». Les deux sont déclarées ici, à égalité, avec leur fenêtre.

### 4.5 Bras descriptifs

- **Contrôle (b)** : `argmin(H)` sur `NEUTRAL_TEXT` **vs** sur les faits. Prédiction signée : **les deux diffèrent** (la vallée bouge avec la nature de l'entrée, Skean Fig. 3). **Antipode** : identiques à la couche près ⇒ la vallée est un **artefact d'architecture**, indépendant de l'entrée — ce qui **affaiblit** `H(ℓ)` comme candidate couche-injection, et doit être écrit.
- **Profil des quatre stades** (Lad et al. 2406.19384) : les courbes sont confrontées **qualitativement** aux quatre stades (détokenisation / ingénierie de traits / ensemblage / affûtage résiduel). **Descriptif, aucune clause** — trois modèles ne testent pas une théorie des stades.
- **Cohérence `L/2`** : les trois D3 valent exactement `⌊L/2⌋`. Rapporter l'écart `argmax(contrast) − ⌊L/2⌋` et `argmin(H) − ⌊L/2⌋` **en fraction de L**, ce qui rend les trois modèles comparables.

## 5. Contrôles

1. **`M = 0`, jamais instanciée** ; aucune écriture, aucun `stream()`, aucun `force_write`.
2. **`engram/` non modifié** — `git diff --stat engram/` vide en fin de run (porte V-hooks).
3. **Hooks posés et retirés** dans le même bloc `try/finally`.
4. **Un seul forward par (modèle, corpus)** — porte V-1pass.
5. **`L` re-lu du config** — porte V-L.
6. **Corpus (a) figé et hashé** (SHA-256 de la table d'unités + `POOL_PARAPHRASES` + `OWNER_OBJ`) avant capture ; **même hash que celui scellé pour v3**.
7. **Aucun résultat v3 sur disque** au lancement — porte V-vierge.
8. **Le confondant de suffixe est mesuré, pas supposé** — porte V-suffixe.
9. **fp32 pour le Gram et les cosinus**, même si le cortex tourne en fp16 (cohérent avec la règle projet « M reste en fp32 ») ; les valeurs propres en fp64.
10. **Ordre des modèles fixé** (gpt2, SmolLM2, Qwen) et VRAM libérée entre les trois — 6 Go.

## 6. Critères d'abandon

**Ce qui tue H_I2** : les deux courbes **PLATES** (V-plat) sur les trois modèles, ou extrema **aux bords** (V-bord). Verdict `REJETE` : l'instrument ne porte pas d'information de profondeur.

**Ce qui invalide le run** (`INCONCLUSIF`, cause nommée) : V-vierge échouée (un résultat v3 existe déjà) ; `engram/` modifié ; hooks non retirés ; plus d'un forward par corpus ; `L` différent du config ; V-suffixe non conforme à l'attendu (les règles gelées ne sont pas celles qui ont tourné) ; > L/2 couches NON ÉVALUABLES (ratio dégénéré) ; NaN / inf ; repli CPU silencieux ; convention de normalisation de l'entropie **non re-dérivée depuis la source** (D14-R — deviner est un motif d'arrêt, pas une approximation acceptable).

**Ce qui n'est PAS un critère d'abandon** : la cellule **C2** (une quantité près, une loin) — c'est un résultat ; la cellule **C3** — c'est aussi un résultat, et une réponse à D3 ; un désaccord entre modèles (Qwen est **instruct + RLHF**, un profil différent est **attendu**, c'est même la raison d'être de P-D).

**Interdit (D14)** : amender une clause après lecture de la première courbe. Les fenêtres, la table à trois cellules et les seuils de V-plat sont fixés ici.

## 7. Variables fixées

`seed = 0` ; `M = 0` ; aucun hyperparamètre d'`EngramConfig` autre que `model` et `device` n'intervient — **aucun nouveau champ n'est demandé**.
Modèles : `gpt2` (L = 12) ; `HuggingFaceTB/SmolLM2-360M` (L = 32) ; `Qwen/Qwen2.5-1.5B` (L = 28) — **`L` re-lu du config au run**.
Fenêtres : `w(L) = max(1, ⌊L/12⌋)` autour de `⌊L/2⌋` ⇒ [5, 7] / [14, 18] / [12, 16].
Corpus (a) : jeu d'unités v3 (`fact_pairs(30)` + table v3 + `POOL_PARAPHRASES`). Corpus (b) : `NEUTRAL_TEXT`.
Numérique : cosinus et Gram en **fp32**, valeurs propres en **fp64**.

## 8. Variable manipulée

**Une seule : la couche** `ℓ ∈ [0, L)`. Le modèle et le corpus sont des **facteurs de réplication**, pas des variables manipulées : chaque (modèle, corpus) porte son propre profil, aucune comparaison inter-modèles n'est décisionnelle.

## 9. Budget

**XS.** Un forward pass instrumenté par (modèle, corpus) : 6 forwards au total, hooks sur toutes les couches simultanément. Ordre de grandeur : **< 2 min GPU** pour les trois modèles, VRAM dominée par Qwen2.5-1.5B (**< 4 Go sur 6**, chargement fp16). Analyse hors-ligne CPU, gratuite.
**Règle pré-enregistrée** : au-delà de **10 min** GPU, c'est une **anomalie à signaler** — probablement un forward par couche (V-1pass), pas une dérive à absorber.

## 10. Livrables attendus

- **`eval/layer_profile.py`** (nouveau, CPU + GPU, SPDX `AGPL-3.0-or-later`) — hooks sur toutes les couches, deux quantités, trois modèles, deux corpus. **`engram/` non modifié** ; réutilise `engram.cortex._find_blocks` en lecture seule.
- **Banc** : les portes de ce protocole (V-vierge, V-hooks, V-1pass, V-L, V-suffixe, V-signe, V-plat, V-bord) et la table à trois cellules **passent au banc de satisfiabilité `eval/gate_bench.py`** — chacune exhibée **passante ET échouante** sur données synthétiques. **Aucune exemption** : cet instrument arrive **après** le cycle méthode, pas à côté de lui.
- **Sorties** : `experiments/results/layer-profile/` — un CSV par (modèle, corpus), les tracés, `argmax`/`argmin` avec leurs valeurs, les ventilations par couple de types, `s_intra` et `s_inter` séparés, le compte de couches NON ÉVALUABLES, et le hash du corpus (a).
- **Tests CPU** : (i) V-1pass détecte une implémentation qui boucle sur les couches ; (ii) V-hooks détecte un hook non retiré ; (iii) V-plat déclare PLATE une courbe synthétique à 3 % d'amplitude et non plate à 8 % ; (iv) V-signe compte correctement les couches à `s_inter ≤ 0` ; (v) la fenêtre `w(L)` rend bien 1 / 2 / 2 pour L = 12 / 32 / 28 ; (vi) le score contrastif équilibré par type diffère du score brut sur un jeu où un type partage son dernier token ; (vii) V-suffixe détecte para3 à ≈ 1.0 ; (viii) la table à trois cellules classe correctement C1 / C2 / C3.
- **Entrée de journal** : datée, protocole recopié, **la phrase O-1 recopiée mot pour mot** (*I2 prédit le bras L6 de v3, il ne l'a pas motivé*), les trois interdictions de vocabulaire §2, la cellule atteinte (C1 / C2 / C3) **nommée**, les couches désignées pour Qwen **consignées avant tout balayage** (P-D), les deux références avec leurs numéros **vérifiés** (2412.09563 et **2406.19384**), et la décision candidate au PI.
- **`docs/EXTENSIONS.md` §4** : une ligne au tableau des ablations/instruments (proposée par le labo, appliquée par le PI).

## 11. Questions pour le PI

1. **O-6 — bras de v3. — TRANCHÉE PAR LE PI (2026-08-22) : NON.** Les bras **F et L6 de v3 restent inchangés**, quelle que soit la couche que I2 désignera. Motif du PI : *« on ne tire pas de plans sur la comète avant d'avoir eu des résultats de I2 »*. L'hypothèse O-6 est donc **confirmée** et cesse d'être une hypothèse. **Conséquence opératoire** : la désignation de I2 est consignée au journal comme **prédiction en attente**, et c'est un **cycle ultérieur** qui la teste — jamais v3, dont le design est clos sur ce point. La multiplicité fermée par la décision 12 du cycle méthode reste fermée.
2. **Ordonnancement.** *Sans objet — déjà fixé par le cadrage* : I2 s'insère après la clôture du banc D14-ext et avant le GPU de v3. La question était une redite de ma part ; l'ordre du cadrage fait foi.
3. **Avis d'experts.** Ce protocole n'a **pas** été soumis aux tours Math / Neuro — le cadrage ne le demandait pas. Vu que les deux tours ont trouvé des clauses fausses dans v3 **à chaque passage**, je recommande **un tour** avant pré-enregistrement. **Ouverte.** Rien ne presse : le banc est bloquant et aucune mesure ne peut partir avant qu'il soit vert.

## Historique

- 2026-08-22 : cadrage PI (instrument I2, motivation à trois sources, prédictions P-A à P-D)
- 2026-08-22 : références **vérifiées** — Skean et al. arXiv:2412.09563 confirmée ; **Lad et al. corrigée en arXiv:2406.19384** (le cadrage annonçait 2404.xxxx) ; `L` = 12 / 32 / 28 **re-mesuré** ; les trois couches D3 valent exactement `⌊L/2⌋`
- 2026-08-22 : six écarts cadrage / état réel levés explicitement (§0) ; fenêtre de tolérance **dérivée** de la zone GPT-2 au lieu d'être posée trois fois ; ambiguïté du critère d'échec **désambiguïsée** en trois cellules avant mesure
- 2026-08-22 : **PI — question 1 (O-6) tranchée : NON**, les bras F et L6 de v3 sont
  clos ; la désignation de I2 sera une prédiction en attente, testée par un cycle
  ultérieur. Question 2 sans objet (l'ordre du cadrage fait foi). Question 3 ouverte.
- 2026-08-22 : **PROPOSE**. Étape suivante : décision du PI sur les trois questions §11, puis (si tour d'experts) avis Math / Neuro, puis banc, puis pré-enregistrement.
