# EXP — v4-matériel : construire un matériau qualifié pour un instrument représentationnel

Statut : PROPOSE

*Protocole consolidé par `lab-director` le 2026-08-23, sous les avis Math (RÉSERVÉ, 6 points) et
Neuro (RÉSERVÉ, 5 points), les positions PI A-1..A-7, et l'instruction du point D exigée par le PI.
Amendé à la gate du 2026-08-23 (§14).*

---

## D. Instruction du point D — préalable à tout banc

Le PI a exigé que les trois questions (a)(b)(c) soient instruites **avant adoption**. Elles le sont
ici, avec leur verdict. Le reste du protocole est écrit **sous** ces verdicts.

### D.0 Le schéma de construction sur lequel porte l'instruction

L'instruction n'est pas possible sur « 4 × 5 × 3 » en abstrait : la réponse dépend de **comment**
les 3 membres d'une famille se ressemblent. Le schéma constructif est donc fixé, et c'est lui qui
est instruit.

- **Entité = `<Tige> <Suffixe>`**, la tige et le suffixe faisant **exactement 1 token** dans chacun
  des trois tokenizers (avec espace de tête). ⇒ `L_e = 2` tokens partout, **par construction et non
  par recherche**.
- **Famille = tige partagée** ; 3 membres = 3 suffixes distincts. ⇒ le partage BPE est
  **intra-famille**, et il est **exactement d'un token**, mesurable, pas approximatif.
- **Suffixes tous distincts** sur l'ensemble du matériau (72 suffixes pour 24 familles) ⇒ deux
  unités de familles différentes partagent **zéro** token d'entité, sauf si elles partagent la tige.
- **Pont (`bridge`)** : une tige peut être portée par **deux familles situées dans deux domaines
  différents**. C'est le mécanisme, à coût nul, qui remplit la cellule manquante de Neuro
  (« domaines différents **avec** token partagé »).

Ce schéma rend le plan de strates **2 × 2 exact** (token partagé × domaine partagé), plus fort que
le « gradient » à quatre strates, et répond à la demande de Neuro de renoncer au mot.

| Strate | Token d'entité partagé | Domaine | Réalisation | # paires (72 unités) |
| --- | --- | --- | --- | --- |
| **S3** | oui (tige) | même | intra-famille | 24 × 3 = **72** |
| **S2** | oui (tige) | différent | pont | 10 × 9 = **90** |
| **S1** | non | même | intra-domaine inter-famille | 4 × 135 = **540** |
| **S0** | non | différent | inter-domaine | **1 854** |

*Vérification de somme (copilote, par exécution) : 72 + 90 + 540 + 1854 = 2556 = C(72,2). ✓*

### D.1 Question (a) — les 3 paires internes sont-elles échangeables au sens de la nulle `1/37` ?

**Verdict : NON. Le partage intra-famille casse l'échangeabilité — et il la casse aussi dans le
design à 2 membres, où le défaut était simplement invisible.**

Dérivation. L'exactitude de `1/37` démontrée par Math repose sur le fait que, **conditionnellement à
la requête**, les 37 candidats sont **identiquement distribués sous H0**. Les candidats ne diffèrent
que par le contenu du slot entité ; ils sont donc identiquement distribués **ssi leur relation
lexicale à l'entité de la requête est la même pour tous**. Dès qu'un concurrent partage la tige de
la requête, le pool contient **deux relations lexicales** (partage 1/2 token, partage 0/2 token),
donc **deux distributions**. La loi du rang n'est plus uniforme : `P(rang = 1)` de la cible n'est ni
`1/37`, ni borné par `1/37` dans une direction utile — le co-familial bat la cible par recouvrement
de surface **sans qu'aucune identité ne soit en jeu**.

C'est le mode de mort de la **dominance de type** d'I2 (journal 2026-08-23, facette 2), réincarné en
**dominance de surface**. Le reconnaître avant qu'un octet n'existe est la contrepartie exacte de
D24-b.

**Correctif — clause `C5`, structurelle, pas statistique** :

> Le pool de 36 concurrents d'une requête **exclut toute unité partageant au moins un token BPE avec
> l'entité de la requête** (co-famille **et** pont). Le pool est tiré **une fois**, seedé, **gelé**,
> et **jamais rééchantillonné** (défaut Q1 de Math). Aucun rang n'est jamais recalculé à l'intérieur
> d'un échantillon bootstrap.

Sous `C5`, les 37 candidats d'une requête sont **tous à recouvrement d'entité nul** :
l'échangeabilité est restaurée **exactement**, et `1/37` redevient la loi vraie du rang, par requête.

**Et c'est ici que se joue l'arbitrage de budget, sur un argument arithmétique, pas de préférence :**

| Design | Unités | Exclus du pool (soi + co-famille + pont) | Éligibles | Pool requis | Marge |
| --- | --- | --- | --- | --- | --- |
| 40 (`K = 20`, 2 membres) | 40 | 1 + 1 + 0 | 38 | 36 | **2** |
| 40 + ponts | 40 | 1 + 1 + 2 | 36 | 36 | **0 — insatisfiable au premier item mort** |
| **60 (`K = 20`, 3 membres)** | 60 | 1 + 2 + 3 | **54** | 36 | **18** |
| 60 + réserve | 72 | 1 + 2 + 3 | **66** | 36 | **30** |

Le design à 40 **ne peut pas porter la cellule manquante de Neuro** (S2) : y ajouter les ponts vide
la marge du pool à zéro. Le design à 60 la porte avec un facteur 1.5 de marge. **Argument
structurel, pas argument de prudence.**

**Ce que (a) coûte en plus** : les paires S3 et S2 **ne peuvent plus apparaître dans le test
décisionnel** (elles sont exclues du pool). Elles ne sont pas perdues : elles portent la
**prédiction ordinale de Neuro**, avec **leur propre nulle** (permutation des étiquettes de strate à
l'intérieur de la famille de requête) et **leur propre cardinal D24-b**. Deux maillons, deux nulles
(D17) — pas une demi-mesure. *(Ce coût est amendé au §14 : le PI a exigé une seconde primaire à pool
surface-apparié.)*

### D.2 Question (b) — les quatre conditions de Math survivent-elles au passage à 3 membres ?

**Verdict : NON, trois des quatre changent. Elles sont re-dérivées, pas supposées stables.** Le
résultat reste une nulle exacte, mais sous **cinq** conditions, pas quatre.

| Cond. Math | Statut à 3 membres | Re-dérivation |
| --- | --- | --- |
| **(a)** moule byte-identique hors slot entité | **CHANGE — devient cellulaire** | L'axe de variante exigé par Neuro (Q2) crée **2 moules par type**. La byte-identité ne vaut plus globalement mais **par cellule `(type, variante)`**. ⇒ `C1` : *l'échangeabilité est exacte à l'intérieur d'une cellule ; toute nulle poolée à travers les variantes est proscrite ; le contraste de variante est **entre cellules**, apparié, jamais mélangé.* Sans cette re-dérivation, la nulle serait fausse d'un facteur inconnu. |
| **(b)** longueur BPE de l'entité constante | **CHANGE — de contrainte de filtrage à contrainte de construction** | À 2 membres on pouvait espérer trouver des entités de même longueur ; à 3 membres avec **partage de token imposé**, la recherche libre échoue (la morphologie qui crée le partage change la longueur). ⇒ `C2` : entité = **tige (1 token) + suffixe (1 token)** ⇒ `L_e = 2` **par construction**, sur chacun des trois tokenizers. **Précision qui débloque tout** : la constante est **par tokenizer**, pas commune aux trois — la nulle est par modèle. `L_e^gpt2 = L_e^smol = L_e^qwen = 2` est exigé ; rien n'exige qu'ils soient égaux entre eux (ils le sont ici par chance de construction : on le vérifie, on ne s'en sert pas). |
| **(c)** point de capture défini par le moule | **STABLE EN FORME, DURCI EN EXIGENCE** | `C3` : indice de capture = `len(préfixe) + L_e − 1`, **entier constant par cellule**. Exige que les **préfixes des deux variantes soient égalisés en nombre de tokens** (le clivage se paie avant l'entité, pas après) — condition de Neuro Q5 (position linéaire identique, offset constant) **et** de D24-b (contraste réalisé **avant** la capture). Ces deux exigences, venues de deux experts différents, **sont la même contrainte** et se vérifient par une seule porte. |
| **(d)** règle d'égalités gravée | **CHANGE — sa portée s'étend** | `C4` : mid-rank + bris seedé, gravé avant mesure ; **plus** un moniteur de **quasi-égalités** : sous `C5` les co-familiaux sortent du pool, mais l'écart de cosinus entre concurrents devient **plus serré** (pool homogène). Couplé à Q5 de Math : `δ̂ = max|Δcos|` entre bf16 et fp32 sur `m = 60` états, et **borne déterministe = fraction des paires décisionnelles à écart < 2δ̂**. |
| — | **NOUVELLE** | `C5` : règle d'exclusion du pool (§D.1). Elle **n'existait pas** dans la démonstration de Math parce que le matériau à 2 membres sans pont ne la rendait pas nécessaire. **C'est précisément l'acquis qu'il ne fallait pas recopier.** |

**Conclusion de (b) : la nulle `1/37` est exacte sous `C1..C5`, pas sous les quatre conditions
d'origine.** Le repli n'est pas déclenché : la re-dérivation **aboutit**, elle ne bute pas.

### D.3 Question (c) — coût de génération réel des 60 (+12) unités, toutes contraintes cumulées

Chiffré **avant** adoption. Le schéma tige+suffixe transforme la génération d'un **problème de
recherche** (rendement inconnu, mode d'échec par épuisement) en un **problème de produit cartésien**
(rendement calculable).

**Besoin brut** : 14 tiges (10 portées par 2 familles de domaines différents = les 10 ponts ; 4
portées une fois) × 24 familles ; 72 suffixes distincts (+12 pour la réserve = 84) ; 40 noms communs
S0 hors domaines (Neuro Q4) ; ~20 pseudo-mots (nulle de nouveauté, séparée) ; 6 gabarits (3 types ×
2 variantes).

| Poste | Volume | Nature | Coût |
| --- | --- | --- | --- |
| Candidats bruts (tiges, suffixes, noms S0, pseudo-mots) | **90** / ~250 / ~120 / ~20 | rédaction | ~1 h 20 |
| `eval/pool_v4.py` (génération + table des garanties D25) | ~400 lignes | code | ~2-3 h |
| Qualification BPE sur 3 tokenizers | ~480 chaînes × 3 | CPU | **< 10 s** (tokenizers en cache) |
| Gabarits + égalisation du préfixe des variantes (`C3`) | 6 | rédaction itérée sous script | ~1-2 h |
| Bandes de fréquence (serpentin sur rang de fusion, **borne par énumération**) | 24 familles | script | ~0.5 h, < 1 s CPU |
| Banc D14-S étendu (≈ 30 clauses × 2 cas + 4 cas de bande + contre-exemple `fact_pairs`) | ~70 cas | tests CPU | ~3-4 h, ~10 s |
| **Total avant tout GPU** | | | **~9-12 h d'agent, < 30 s de CPU** |
| Première mesure I2 sur le matériau (3 modèles) | ~870 séquences/modèle | GPU | **~3-5 min**, VRAM **en réservé** |

**Rendements attendus, avec leur mode d'échec** :

- tiges 1-token sur 3 tokenizers, capitalisées, avec espace de tête : rendement estimé **~30-40 %**
  ⇒ ~30 retenues sur **90 candidats** (vivier porté de 45 à 90 par décision PI du 2026-08-23,
  §14-3), besoin **14** ⇒ **marge ×2**.
- suffixes 1-token sur 3 tokenizers (noms communs) : rendement estimé **~50-60 %** ⇒ ~140 sur 250,
  besoin **84**. Marge ×1.7.
- noms S0 sans token partagé deux à deux **ni avec aucune entité** : contrainte forte mais vérifiable
  en O(n²) trivial ; rendement ~50 % ⇒ ~60 sur 120, besoin **40**.

**Le chiffre décisif pour l'arbitrage 40 / 60** : les postes fixes (script, gabarits, banc, table
D25) représentent **~85 %** du coût. Le passage de 40 à 60 unités coûte **~12 suffixes de plus**,
soit **~15 min**. La réserve de 4 familles coûte **~12 suffixes + 4 tiges**, soit **~20 min**. **La
résolution supplémentaire et la marge sont, ici, quasi gratuites toutes les deux** — la ligne
« acheter de la marge, pas de la résolution » n'a plus d'arbitrage à faire : le schéma constructif
achète les deux.

### D.4 Verdict d'instruction

| Question | Verdict | Conséquence |
| --- | --- | --- |
| (a) échangeabilité | **Échouée telle quelle, réparée par `C5`** — et **seul le design ≥ 60 la rend satisfiable avec la cellule S2** | adoption de 60 |
| (b) quatre conditions | **Trois changent**, re-dérivées en `C1..C5` ; la nulle reste exacte | adoption sous `C1..C5` |
| (c) coût | **~9-12 h agent, < 30 s CPU, 0 GPU** ; surcoût 60 vs 40 ≈ **15 min** | adoption de 60 **+ réserve** |

**Le repli pré-enregistré (48 / `K = 16`) n'est PAS déclenché.** Son **amendement** est **adopté de
façon inconditionnelle** : **4 familles surnuméraires (12 unités) générées et qualifiées au même
banc, substituables uniquement avant le premier token du run (D14)**. `K` reste **20** en toute
circonstance ; la réserve **substitue**, elle n'ajoute pas.

---

## Arbitrage

### A. Positions du PI (A-1 … A-7) — toutes reconduites, aucune contradiction remontée

| Position | Traitement | Où |
| --- | --- | --- |
| **A-1** scission `N-a`/`N-b`/`N-ind` | **Reconduite intégralement**, avec les trois exigences (3 classes au banc dont `N-ind`, verdict gravé « indécidable ICI », interdit binomial en clause de **portée**) | §4.4, §6.C, §10 |
| **A-2** S-4 lecture faible | **Reconduite** — et **satisfaite par construction** : `L_e = 2` constant + moule byte-identique hors slot ⇒ longueur **constante intra-type à travers les unités**, **libre entre types**. Le corollaire (« la statistique s'apparie par type, pas le matériau ne s'uniformise ») est gravé en `C1` | §4.1-C1/C2 |
| **A-3** un type capitalisé obligatoire | **Reconduite** — le type 3 place l'entité en **tête de phrase** ; la porte BPE `V-casse` mord sur du réel | §4.1-C6, §10 |
| **A-4** D24-b | **Reconduite et opérationnalisée** : capture = `len(préfixe) + L_e − 1` ; **tout** contraste (variante, type, identité) est **avant** ce point ; **tous** les cardinaux sont **tronqués à la capture** | §4.1-C3, §5, §10 |
| **A-5** bin dur descriptif | **Reconduite** — ligne gravée mot pour mot, extension du vocabulaire interdit du §2 | §4.5, §6.E |
| **A-6** D25 exécutable | **Reconduite** — table des garanties **dans la source** `eval/pool_v4.py`, prérequis **déclarés par l'instrument**, vérification **mécanique**, `fact_pairs` en **contre-exemple échouant obligatoire** au banc | §10 |
| **A-7** annulation de la prédiction signée de Neuro | **Reconduite** — section traçable dédiée, motif consigné, re-signature exigée avant gel | Registre des engagements |

### B. Avis Math — 6 points

| # | Traitement | Raison |
| --- | --- | --- |
| **Q1** cluster = famille, `K = 20`, marge survit au DEFF à ρ = 1 | **Intégrée** | Reprise telle quelle. **Précision ajoutée** : le clustering se fait par **la famille de la REQUÊTE**, ce qui donne `K = 20` **dans toutes les strates**, y compris S2 où le comptage par paire de familles n'aurait donné que 10. |
| **Q1 défaut** — pool de concurrents non couvert par le bootstrap | **Intégrée, durcie** | Devient `C5` + clause gravée : *pool fixe, seedé, gelé, jamais rééchantillonné ; rangs jamais recalculés dans l'échantillon* (cela changerait `s`, donc le hasard `1/37` — interdit). Cas de banc passant **et** échouant. |
| **Q2** alpha nul pour gatekeeping ; défaut du seuil zéro | **Intégrée** | `ΔR1_ident > 0` en point estimé est retiré de toute décision ; partition en 3 classes sur **IC cluster à 95 %**, alpha propre 0.05. |
| **Q3** nulle `1/37` exacte sous 4 conditions ; interdit d'agrégat binomial | **Intégrée, et les conditions RE-DÉRIVÉES** (§D.2) : elles deviennent `C1..C5`. L'interdit binomial devient **clause de portée** (§6.C) | La demande explicite du PI était de ne pas les supposer stables. |
| **Q4** bin dur indécidable (~23 familles requises) | **Intégrée** | Déclaré **descriptif avant mesure**, vocabulaire interdit étendu. |
| **Q5** bf16 : porte au niveau des états, `m = 60`, `δ̂`, borne déterministe | **Intégrée** | Devient la porte `V-dtype` (§4.5) avec seuil exécutable : fraction des paires décisionnelles à écart `< 2δ̂` **> 1 %** ⇒ `INCONCLUSIF-précision`, repli fp32 déclaré **avant** le run. |
| **Q6** la fréquence se **construit** (serpentin sur rang de fusion, borne par énumération) ; défaut du couloir en non-rejet | **Intégrée** | Aucun test d'homogénéité nulle part. Serpentin sur le rang de fusion **gpt2**, puis **borne exacte par énumération** du déséquilibre résiduel sur les **trois** tokenizers, publiée comme **nombre**, jamais comme valeur-p. |

### C. Avis Neuro — 5 points

| # | Traitement | Raison |
| --- | --- | --- |
| **Q1** 2,5 familles/domaine impossible ; puissance inversée ; cellule croisée manquante | **Intégrée — c'est le déclencheur du redesign** | 4 × 5 × 3 = 60 ; le facteur de surface devient **intra-famille**, donc orthogonal au domaine par construction. La cellule manquante est réalisée par les **10 ponts** (S2, 90 paires, `K = 20` par famille de requête). |
| **Q2** variante possiblement NO-OP ; second axe (ordre / clivage) | **Intégrée** | Le no-op est traité par D24-b/`C3` (contraste **avant** capture, préfixes égalisés). Le second axe devient la **variante** : deux moules par type, entité à position absolue **inchangée**. `R1_ident` a bien **deux barreaux**. |
| **Q3** croiser la fréquence, pas l'ajuster ; ordre S-4 puis fréquence | **Intégrée** | Cascade de génération **gravée dans cet ordre** : (1) `L_e = 2` sur 3 tokenizers → (2) structure tige/suffixe et ponts → (3) moules et égalisation de préfixe → (4) **bandes de fréquence dans le sous-ensemble survivant** → (5) type capitalisé → (6) banc. Convention de rang pour entités multi-tokens : **rang de fusion de la TIGE** (le token porteur de famille), gravée. |
| **Q4** pseudo-mots = pire choix ; 40 noms communs hors domaines | **Intégrée** | La nulle de cadre utilise **40 noms communs**, sans token partagé deux à deux ni avec aucune entité ⇒ paires internes **≡ S0 par construction**. Pseudo-mots **conservés séparément**, étiquetés **« nulle de nouveauté »**, jamais mélangés à la nulle de cadre. |
| **Q5** position linéaire identique, offset constant | **Intégrée** | Fusionnée avec `C3` : c'est la même porte que D24-b. Argument retenu tel quel (« on referait I2 sur du matériau neuf »). |
| **« gradient » → « quatre strates stipulées »** | **Intégrée, renforcée** | Le plan devient un **2 × 2 exact** (token × domaine). L'ordre `S3 > S2 > S1 > S0` devient une **prédiction ordinale pré-enregistrée falsifiable**, avec **son antipode nommé** (`S1 > S2` : le domaine domine le token). |
| Note d'instrument : ne pas interdire `cos(topk(G·h))` | **Intégrée** (coût nul) | Les états bruts sont conservés ; `G` gelée, seedée par `cfg.seed`, D8 intacte. Aucune décision n'en dépend. |

### D. Écarté

| Élément | Raison |
| --- | --- |
| Repli 48 / `K = 16` comme **design** | (a) et (b) n'échouent pas — ils se réparent (`C5`, `C1..C5`). Et à 48 avec ponts, la marge de pool tombe à ~8 : le repli est **plus fragile** que ce qu'il devait protéger. |
| **Son amendement** (4 familles de réserve) | **NON écarté — adopté inconditionnellement**, hors condition de déclenchement (coût ~20 min, marge dans le matériau). |
| Bin dur en décision | Math Q4 : indécidable arithmétiquement à ce `N`. |
| Test d'homogénéité de fréquence | Math Q6 : un couloir en **non-rejet** est un blanc-seing (famille N11 / D20). |
| Pseudo-mots dans la nulle de cadre | Neuro Q4 : régime OOV, non apparié. Conservés **à part**. |
| Uniformisation de la longueur **entre types** | Position A-2 du PI (lecture faible). |
| Toute réutilisation de `pool.fact_pairs` | D25 + décision PI du 2026-08-23. `pool.py` reste **gelé et intouché** ; le matériau v4 vit dans un fichier neuf. |

---

## Registre des engagements — annulation consignée (position A-7)

| Engagement | Auteur | Date de signature | Statut | Motif de l'annulation |
| --- | --- | --- | --- | --- |
| Prédiction signée de Neuro sur le contraste de variante (formulée sous le design **2 membres / variante par adjoint postposé**) | lab-neuro | tour matériau, 2026-08-23 | **ANNULÉE — non reportée** | **Le design en supprime le fondement, pas la mesure.** La prédiction portait sur un contraste de variante réalisé **par un adjoint circonstanciel postposé** ; D24-b (gravée le même jour) établit que ce contraste est **bit-identique au point de capture**, donc **inexistant pour l'état mesuré**. La quantité prédite n'a pas de référent dans le design consolidé. Elle n'est **ni falsifiée ni confirmée** : elle est **sans objet**. |
| Prédiction ordinale de strates `S3 > S2 > S1 > S0` | lab-neuro | ce tour | **EN ATTENTE DE SIGNATURE** | Nouvelle, sur le plan 2 × 2 ; portée par §4.6. Antipode nommé : `S1 > S2`. |
| Re-signature sur le design final | lab-neuro | — | **DUE avant gel** | Le PI exige : Neuro **re-signe ou déclare s'abstenir** sur le design consolidé, avant gel. Une abstention déclarée est un statut valide ; un silence ne l'est pas. |

---

## 0. Défauts acquittés

**0-1 … 0-33 : reconduits sans changement**, numérotation inchangée, aucun ré-ouvert (0-1..0-15 du
§0 d'I2 `layer_profile` ; 0-16..0-33 des tours v3 et du brouillon v4-matériel). Ce tour en ajoute
douze (0-34..0-45, le dernier venant de la gate PI).

| # | Défaut | Pourquoi il était fatal |
| --- | --- | --- |
| **0-34** | **Variante NO-OP** : la variante intra-type proposée était un **adjoint postposé** ⇒ les deux états d'une même unité auraient été **bit-identiques au point de capture** ⇒ `R1_ident = 1` **par arithmétique, pas par représentation**. | **Fatal à la primaire.** D24 ne le voyait pas (les séquences complètes étaient bien distinctes). Trouvé avant qu'un octet de matériau n'existe. Gravé en **D24-b**. |
| **0-35** | **Non-intégrité du plan** : 10 familles / 4 domaines = **2,5 familles par domaine**. Arithmétiquement impossible ⇒ le facteur de surface devenait **corrélé au domaine**, contre l'esprit de S-5. | **Fatal au design** : un plan non entier n'est pas un plan sous-optimal, c'est un plan **inexistant**. |
| **0-36** | **Puissance inversée** : ~600 paires sur `S0` (la strate sans intérêt) contre **10 vs 10** sur le contraste porteur `S3 − S2`. | La puissance était entièrement dépensée là où aucune décision ne se prend. |
| **0-37** | **Croisement incomplet** : aucune cellule « domaines différents **avec** token partagé ». | Le plan ne pouvait pas séparer l'effet du **token** de celui du **domaine** — un confondant de plan, pas de mesure. |
| **0-38** | **Seuil zéro sur quantité bruitée** : `ΔR1_ident > 0` en **point estimé** est, à `K = 20`, **un tirage au sort une fois sur deux**. | **Fatal à la frontière de bande** : une frontière qui décide à pile ou face n'est pas une frontière. |
| **0-39** | **Agrégat binomial sur 240 indicateurs de rang** : l'échangeabilité `1/37` vaut **par requête**, jamais sur l'agrégat. | C'est **exactement** l'erreur du Copilote sur `R1 = 0` d'I2 (`P = 1.4e-3` annoncé sur une nulle fausse). **Deuxième occurrence ⇒ clause de portée** (A-1c). |
| **0-40** | **Bin dur indécidable** : 2AFC exige **≥ 15/20** familles à α = 0.05 ; puissance à p = 0.75 ≈ **0.60** ; il faudrait **~23 familles**. | Une clause qui **découvrirait** son indécidabilité après mesure (leçon 0-13). Déclarée **descriptive avant mesure**. |
| **0-41** | **Couloir de fréquence défini comme NON-REJET** d'un test d'homogénéité. | **Acceptation sous-puissante = blanc-seing** — la version lexicale de N11 (D20). Remplacé par une **construction** + une **borne par énumération**. |
| **0-42** | **Pseudo-mots comme nulle de cadre** : régime OOV, maximalement nouveaux ⇒ non appariés à quoi que ce soit. | Une nulle non appariée ne borne rien (famille D24). Reclassés en **« nulle de nouveauté »**, séparée. |
| **0-43** | **Position linéaire de l'entité laissée libre** entre types/variantes. | La distance à la capture est un **intervalle de rétention** : la laisser varier **garantirait** que le type reste dominant — « on referait I2 sur du matériau neuf ». |
| **0-44** | **Bootstrap de familles ne couvrant pas le PARTAGE DES CONCURRENTS** ; et, plus profondément, **le pool de concurrents n'était pas échangeable** dès qu'une famille y met deux membres. | **Fatal à la nulle exacte.** Réparé par `C5` (exclusion) + pool **gelé, jamais rééchantillonné**. **C'est un acquis de régime qui aurait été recopié** : la démonstration de Math était exacte **sous le matériau à 2 membres sans pont**. |
| **0-45** | **`C5` restaure l'exactitude en supprimant la difficulté** : sous `C5`, les 36 concurrents ne partagent **aucun** token d'entité avec la requête, alors que la cible partage l'entité **en entier** ⇒ le rang de la cible peut être bon **par lecture lexicale de l'état**, sans identité représentationnelle. Le §5 consolidé ne comportait **aucune baseline lexicale à 0 forward** sur `ΔR1_inv`. | **Fatal à l'interprétation de la bande `M`.** C'est le mur mesuré par I2 (max AUC **sous** le plancher lexical : 0.4306 / 0.5224 / 0.5442 contre 0.6160 / 0.6078 / 0.6166) — et `C5` rend ce plancher **plus fort, pas plus faible**. Réparé au §14-2 : baseline lexicale obligatoire + seconde primaire à pool surface-apparié. Relevé par le copilote à la gate, avant tout banc. |

---

## 1. Question

Peut-on construire, et **certifier au banc avant tout GPU**, un matériau dont les propriétés
vérifiées satisfont simultanément S-1..S-8, `C1..C5` et D24-b — au point que la mesure d'identité
intra-unité intra-type y soit **décidable** (verdict `M`, `N-a` ou `N-b`) plutôt qu'indécidable
(`N-ind`) ?

## 2. Hypothèse

`H_mat` : **un matériau à moules (entité = tige + suffixe, 1 token chacun sur les trois tokenizers),
4 domaines × 5 familles × 3 membres, 3 types dont un capitalisé, 2 variantes à préfixe égalisé, 10
ponts inter-domaines, satisfait les 8 specs et les 5 conditions par construction vérifiable ; et
l'instrument I2, appliqué à ce matériau, rend un verdict décisionnel.**

**Antipode explicite (D13)** : au moins une porte de génération est **insatisfiable** (le banc le
montre, à coût CPU, avant tout GPU), **ou** l'instrument rend `N-ind` — auquel cas la conclusion
gravée est *« indécidable ICI, matériel ou `K` insuffisant »*, la suite est **« augmenter la
résolution »**, et **jamais** « conclure prudemment ».

**Interdiction de vocabulaire (§2, étendue)** : les formulations interdites d'I2 (i)-(viii) sont
reconduites ; **s'y ajoute le bin dur** (position A-5) : aucune phrase du rapport ne peut le citer
comme évidence, ni sous forme adverbiale (« tendance », « suggère », « va dans le sens de »).

## 3. Ce que le projet sait déjà — provenance (D14-R)

| Fait | Chiffre | Source, date | Étiquette |
| --- | --- | --- | --- |
| Le cosinus sur ces états mesure **la forme de la paraphrase**, pas l'identité du fait | ppv même TYPE **1.000 / 0.996 / 1.000** (hasard 0.331) ; ppv même FAIT **0.000 / 0.008 / 0.000** (hasard 0.008) | I2, journal 2026-08-23 | re-mesuré depuis `experiments/results/layer-profile/` — **motive S-1, ne prédit rien** (run INCONCLUSIF) |
| `R1_36 = 0` sur toutes les couches × 3 modèles **n'est pas une anomalie** | rang médian de la cible **19/37** (moyennes 20.1 / 18.6 / 18.5) | I2, 2026-08-23 | idem |
| Les max AUC sont **sous le plancher lexical à 0 forward** | 0.4306 / 0.5224 / 0.5442 contre 0.6160 / 0.6078 / 0.6166 | I2, 2026-08-23 | idem — **motive la baseline obligatoire du §14-2** |
| Appariement de longueur sur `fact_pairs` : **impossible**, pas difficile | violé pour **16/16** valeurs de `k` | Neuro, vérifié par exécution, 2026-08-23 | vérifié |
| Période 20 sur `(entity, verb)` ⇒ 20 couples distincts à N = 30, **10 collisions** | — | banc v3, 2026-08-22 | vérifié par exécution |
| Coût GPU de l'instrument | **53.82 s** pour 12 forwards / 240 séquences ; VRAM Qwen **4.688 Gio réservés** (fp16) | I2, 2026-08-23 | re-mesuré |
| Deux fois de suite, la porte qui trouve le défaut est celle **qui n'a besoin d'aucune donnée** | `V-slot`, `V-ident` | cycle D14-ext, 2026-08-22 | vérifié |

## 4. Prédictions de construction — chacune avec dérivation et antipode

*Toutes décidables **sans GPU**, au banc. Une prédiction fausse ici est un **résultat**, obtenu pour
< 30 s de CPU.*

### 4.1 Conditions de construction `C1..C6`

| # | Prédiction | Dérivation | Antipode (résultat de signe opposé, et sa conséquence) |
| --- | --- | --- | --- |
| **C1** | Pour chaque cellule `(type, variante)`, les 72 séquences sont **byte-identiques hors du slot entité** ; cardinal de séquences **distinctes tronquées à la capture** = **72**. | Moule unique par cellule, entité seule variable, `L_e` constante ⇒ toute différence vit dans le slot. | Cardinal < 72 ⇒ **deux entités collision** ⇒ **arrêt de génération**, pas un avertissement. |
| **C2** | `L_e = 2` tokens pour **72/72** entités, **sur les trois tokenizers**. | Tige 1 token + suffixe 1 token, vérifiés à la qualification. | Une seule entité à `L_e ≠ 2` sur un tokenizer ⇒ **rejet de l'entité**, remplacement dans le vivier, re-qualification complète (pas de rustine locale). |
| **C3** | L'indice de capture est un **entier constant par cellule** ; les **deux variantes d'un type ont le même nombre de tokens de préfixe** sur les trois tokenizers ; le contraste de variante est **entièrement dans le préfixe**. | D24-b + Neuro Q5 : capture `= len(préfixe) + L_e − 1`. | Préfixes de longueurs différentes ⇒ la position absolue de l'entité varie ⇒ **0-43 réintroduit** ⇒ **arrêt**. |
| **C4** | Aucune paire de séquences décisionnelles n'est **byte-identique tronquée à la capture** ; règle d'égalités (**mid-rank**, bris seedé par `cfg.seed`) gravée **avant** mesure. | D24-b + Math Q3(d). | Une égalité byte ⇒ contraste no-op (0-34) ⇒ **arrêt**. |
| **C5** | Pour **72/72** requêtes, le nombre d'unités **à recouvrement de token d'entité nul** est **≥ 60** ; le pool de 36 est tiré une fois, seedé, **gelé**. | 72 − 1 (soi) − 2 (co-famille) − 3 (pont) = **66 ≥ 36 + 24 de marge**. | Éligibles < 36 pour une seule requête ⇒ **le design est insatisfiable** ⇒ retour au PI, **pas d'ajustement du pool**. |
| **C6** | **Exactement un** des trois types place l'entité en **position capitalisée** ; pour ce type, la porte `V-casse` **mord** : la minusculisation de la tige change le token pour **≥ 90 %** des tiges sur les trois tokenizers. | Position A-3 ; leçon `Her`→`her` (2332 → 607, aucun partage) de v3. | < 90 % ⇒ la clause serait **vraie par vacuité** ⇒ **troisième occurrence du mode 0-6/0-8** ⇒ **arrêt** (le PI a explicitement refusé la vacuité déclarée). |

### 4.2 Plan de strates (2 × 2 exact) et cardinaux D24-b

| Strate | Définition | # paires | `K` (famille de requête) | Cardinal distinct **tronqué à la capture** |
| --- | --- | --- | --- | --- |
| S3 | token partagé, même domaine | 72 | 20 | à publier, attendu 72 |
| S2 | token partagé, domaine différent | 90 | 20 | à publier, attendu 90 |
| S1 | pas de token partagé, même domaine | 540 | 20 | à publier |
| S0 | pas de token partagé, domaine différent | 1 854 | 20 | à publier |
| Nulle de cadre | 40 noms communs hors domaines, sans token partagé deux à deux ni avec aucune entité | 40 / cellule | 20 | **40** par cellule — porte D24 |
| Nulle de nouveauté | ~20 pseudo-mots, **séparée, jamais mélangée** | 20 / cellule | — | **20** par cellule |

### 4.3 Primaires décisionnelles

**Primaire 1 — pool `C5` (nulle exacte)** : `ΔR1_inv = R1(variante correcte de la même unité) −
R1(clé nulle appariée)`, rang parmi **37** candidats du pool `C5`, statistique **de cluster**
(famille de requête), `K = 20`, **bootstrap de familles côté requêtes uniquement, pool jamais
rééchantillonné, rangs jamais recalculés**.

**Primaire 2 — pool surface-apparié (amendement de gate, §14-2)** : même quantité, pool construit
pour **contenir** les concurrents à recouvrement de tige (S3 et S2), avec **sa propre nulle** (non
`1/37` — à dériver par Math) et **son propre `K`**. C'est elle qui répond à *« l'adressage
apporte-t-il quelque chose ? »*.

**Baseline lexicale à 0 forward (amendement de gate, §14-2)** : `ΔR1_inv` calculé **sans aucun
forward**, sur les seuls identifiants de tokens de l'entité. **Elle est un plancher, pas un
contrôle** : aucune bande supérieure à `N` ne peut être prononcée si l'IC de la primaire ne domine
pas ce plancher. Porte `V-lex`.

`R1_ident` (identité brute, legs d'I2) : **plafond descriptif**, hors décision.

### 4.4 Partition exhaustive des bandes (D18, A-1)

Seuils : `T = 0.25` (couloir), `τ = 0.10` (plancher de résolution). IC = bootstrap de familles à
**95 %**, alpha propre **0.05** (Math Q2 : coût en alpha nul pour le gatekeeping).

| Bande | Condition (exhaustive et exclusive) | Verdict gravé | Suite gravée |
| --- | --- | --- | --- |
| **M** | `IC_inf(ΔR1_inv) > T` **et** `V-lex` PASS | identité **adressable** sous cet instrument | ouvrir v4 sur le matériau |
| **N-b** | non-`M` **et** [`IC_inf(ΔR1_inv) > 0` **ou** ordinal de strate significatif] | identité présente mais **sous-dominante à la surface** | manipuler la surface, pas la résolution |
| **N-a** | non-`M`, non-`N-b`, **et** `IC ⊂ [−τ, +τ]` | **pas d'identité** décelable, à résolution suffisante | réorienter : l'identité n'est pas dans cette géométrie |
| **N-ind** | tout le reste (IC plus large que `[−τ, τ]` sans franchir `0` ni `T`) | **« indécidable ICI, matériel ou `K` insuffisant »** | **« augmenter la résolution »** — *jamais un demi-`N-a`* |

**Non-vacuité** : chaque bande a **un cas synthétique** au banc (§10), y compris `N-ind` (D18 : la
totalité de la fonction est un item de couverture).

### 4.5 Portes

| Porte | Contenu | Passe / échoue |
| --- | --- | --- |
| `V-C1..V-C6` | les six conditions ci-dessus | un cas synthétique passant **et** un échouant chacune |
| `V-pool` | pool gelé, hash avant/après, **aucun rang recalculé dans un échantillon bootstrap** | cas échouant = pool rééchantillonné |
| `V-lex` | **baseline lexicale à 0 forward** calculée et publiée ; toute bande supérieure à `N` exige que l'IC de la primaire **domine** ce plancher | plancher non dominé ⇒ **la bande `M` est interdite**, verdict rabattu sur la partition `N` |
| `V-dtype` | bf16 épinglé (D21) ; `m = 60` états fp32 ; `δ̂ = max\|Δcos\|` ; **fraction des paires décisionnelles à écart `< 2δ̂` ≤ 1 %** | > 1 % ⇒ `INCONCLUSIF-précision`, repli **fp32 déclaré avant le run** |
| `V-freq` | serpentin sur rang de fusion gpt2 ; **déséquilibre résiduel par énumération exacte** sur les 3 tokenizers ≤ **0.15** (échelle [0,1]) | **aucun test d'homogénéité** nulle part (0-41) |
| `V-bindur` | **le bin dur est marqué `DESCRIPTIF` dans le rapport, et son champ décisionnel est absent du schéma de sortie** | présence d'un champ décisionnel ⇒ échec du pipeline |
| `V-fact-pairs` | **contre-exemple obligatoire** : `fact_pairs` soumis à la table de garanties D25 doit **ÉCHOUER** sur `C1`, `C2` et S-1 | s'il passe, c'est la **table** qui est fausse |

### 4.6 Prédiction ordinale (Neuro, signature due)

`cos̄(S3) > cos̄(S2) > cos̄(S1) > cos̄(S0)` — « le token domine le domaine ».
**Antipode nommé** : `cos̄(S1) > cos̄(S2)` — le domaine domine le token ; conséquence gravée : la
géométrie est **sémantique et non lexicale**, et `C5` devient **insuffisante** (il faudrait exclure
aussi le domaine du pool) ⇒ le matériau retourne au banc, la mesure ne s'interprète pas.

### 4.7 Prédictions d'instrument reconduites

`P-A`, `P-B`, `P-C`, `P-D` **restent pré-enregistrées telles quelles** (décision PI 2026-08-23) et
sont jugées **sur ce matériau**. Elles ne s'amendent pas.

## 5. Contrôles et baselines

1. **Configuration courante** : `EngramConfig()` par défaut (layer=6, λ=2.0, cap=0.5, η=0.2,
   decay=1e-3, thr=4.0, dg=8192/64, read_gate=keysim, seed=0) — **citée comme référence de projet** ;
   ce protocole **n'instancie ni ne lit `M`** (aucune injection, D8 intacte).
2. **M reset / D7** : sans objet ici (aucune écriture) ; le contrôle homologue est le **pool gelé** —
   même matériau, mêmes états, seule la clé change.
3. **Nulles par maillon (D17)** : matériau → `V-C1..C6` ; clé → clé nulle appariée (`ΔR1_inv`, D16) ;
   sélection → pool `C5` gelé ; **lecture de surface → baseline lexicale à 0 forward (`V-lex`)** ;
   surface → nulle de cadre (40 noms communs) ; nouveauté → nulle de pseudo-mots, **séparée**.
4. **Contrôle qui tue l'explication triviale** : `fact_pairs` passé au **même banc** doit **échouer**
   — sinon le banc ne discrimine rien (D25 mécanique).
5. **Ordre des conditions** : la cascade de génération est **gravée** (S-4 puis fréquence, Neuro Q3)
   et le banc vérifie que l'exécution suit l'ordre gravé.

## 6. Critères d'abandon — **portes exécutables** (D22)

Chaque clause **bloque l'écriture du rapport**. Un drapeau calculé puis ignoré est **lui-même** un
motif d'invalidation.

**A. Invalidation de génération** — toute porte `V-C1..V-C6` en échec ⇒ **aucun run**, retour au
vivier ou au PI.

**B. Clause NaN (D23), portée gravée** : « NaN / inf » invalide le run s'il apparaît dans **(i)** les
états capturés, **(ii)** toute quantité intermédiaire d'une réduction (AUC, IC, bootstrap),
**(iii)** toute quantité publiée. Les réductions sont **NaN-strictes** : elles propagent ou
s'arrêtent.

**C. Clause de PORTÉE de l'interdit binomial (position A-1c, même famille que B)** :

> **Aucune** statistique décisionnelle, **aucun** IC, **aucune** valeur-p, **aucune** annotation de
> figure ne peut être obtenue par **agrégation binomiale** sur les indicateurs de rang individuels.
> L'échangeabilité `1/37` vaut **par requête** et **seulement** pour la loi du rang d'une requête
> unique, sous `C1..C5`. Portée : quantités **publiées**, quantités **intermédiaires**, et
> **figures**. Toute inférence agrégée passe par le **bootstrap de familles** (`K = 20`, pool gelé).
> **Violation détectée ⇒ run invalide**, porte exécutable dans le pipeline.

**D. Ce qui tue `H_mat`** : `C5` insatisfiable (éligibles < 36 pour une requête) ; ou `C6` vacuée
(< 90 %) ; ou verdict `N-ind` avec `V-dtype` PASS et `K = 20` (⇒ le matériau ne suffit pas, la suite
est **augmenter la résolution**).

**E. Bin dur** : *« le bin dur ne peut entrer en décision qu'à ≥ 23 familles ; à ce `N` il est
descriptif, et aucune formulation du rapport ne le cite comme évidence »*. Porte `V-bindur`.

**F. Baseline lexicale** : si `V-lex` échoue (plancher non dominé), **la bande `M` est interdite** et
le rapport ne peut employer aucune formulation d'adressabilité. Porte exécutable.

**G. Invalidation classique** : 0 write (sans objet ici), NaN, E3 > +0.05 nats/token — **E3 est sans
objet dans ce run** (aucune lecture, aucune injection) et **redevient bloquant** au premier run qui
injecte.

## 7. Variables fixées

`seed = 0` · modèles : `gpt2` (L=12), `HuggingFaceTB/SmolLM2-360M` (L=32, layer de référence 16),
`Qwen/Qwen2.5-1.5B` (L=28, layer de référence 14) — `L` re-lu du config · **dtype du forward
épinglé : bf16** (D21), contrôle fp32 sur `m = 60` états · cosinus/Gram fp32, valeurs propres fp64 ·
**aucune injection, `M` jamais instanciée**, `engram/` non modifié · tokenizers : les trois, tous les
tests BPE se font **simultanément** sur les trois · `T = 0.25`, `τ = 0.10`, `K = 20`, pool = 36 ·
batch ≤ 240 séquences, **VRAM rapportée en réservé**.

Jeux de données : **`eval/pool_v4.py` (nouveau)** — 60 unités décisionnelles + 12 de réserve, 40 noms
communs S0, ~20 pseudo-mots, 6 gabarits. `data/…` et `SECRETS` : **non utilisés**. `pool.fact_pairs`
: **utilisé uniquement comme contre-exemple échouant** du banc.

## 8. Variable manipulée

**Une seule : le matériau** (`pool.fact_pairs` → `pool_v4`). Instrument, seeds, modèles, couches,
métriques, seuils : identiques à I2.

## 9. Budget

- **Génération + banc** : ~9-12 h d'agent, **< 30 s de CPU**, **0 GPU**, 0 VRAM (tokenizers seuls).
  *C'est le poste qui décide de tout ; il ne consomme pas de GPU.*
- **Première mesure I2 sur le matériau** : ~870 séquences/modèle × 3 modèles ⇒ **~3-5 min GPU**
  (extrapolé de 53.82 s pour 240 séquences × 3 modèles, facteur ~3.6). **VRAM : à rapporter en
  réservé, pas à confirmer** — l'estimation de VRAM du projet a été fausse trois fois (journal
  2026-08-23).
- **Runs** : 1 run de génération, 1 run de banc, 1 run de mesure. **Aucun balayage.**

## 10. Livrables attendus

1. **`eval/pool_v4.py` (fichier neuf, SPDX AGPL-3.0-or-later)** — génération + **table des garanties
   D25 en en-tête de la source** : une ligne par propriété (`C1..C6`, indépendance longueur/unité,
   paires intra-unité intra-type, diversité de type, survie à l'effacement de casse, absence de
   période sur tout slot et tout couple de slots, cardinaux des nulles **tronqués à la capture**,
   éligibles de pool), chacune avec **son résultat et sa porte**. `eval/pool.py` **n'est pas touché**
   (gel de `fact_pairs`).
2. **Déclaration de prérequis par l'instrument** + **vérification mécanique** matériau × instrument ;
   incompatibilité = **arrêt**, pas avertissement. **Liste blanche manuelle proscrite.**
3. **Flag `EngramConfig`** : `dataset = "fact_pairs"` par défaut (**= comportement actuel**),
   `"pool_v4"` en option. Aucun défaut de config existant n'est modifié.
4. **Extension du banc `eval/gate_bench.py`** — pour **chaque** clause : **un cas passant et un cas
   échouant**. Plus :
   - **`fact_pairs` en contre-exemple échouant OBLIGATOIRE** (doit échouer `C1`, `C2`, S-1) ;
   - **un cas synthétique par bande** : `M`, `N-a`, `N-b`, **`N-ind`** ;
   - un cas échouant pour `V-pool` (pool rééchantillonné), pour `V-bindur` (champ décisionnel
     présent) et pour **`V-lex`** (plancher lexical non dominé) ;
   - **tous les cardinaux D24-b publiés tronqués au point de capture**, jamais sur séquences
     complètes.
5. **Tests CPU** : `.venv\Scripts\python -m pytest tests/ -q`, sans téléchargement HF au-delà des
   tokenizers en cache.
6. **Entrée de journal** au format maison, incluant le **Registre des engagements**.
7. **Conservation des états bruts** de façon à ne pas interdire `cos(topk(G·h))` (note d'instrument
   de Neuro, coût nul, `G` gelée, D8/D9 intactes).

## 11. Questions pour lab-neuro

1. **Re-signature (exigée par le PI avant gel)** : signes-tu la prédiction ordinale
   `S3 > S2 > S1 > S0` **sur le plan 2 × 2** (token × domaine), avec son antipode `S1 > S2` et sa
   conséquence gravée ? Ou déclares-tu t'abstenir ? *Un silence n'est pas un statut valide.*
2. Le **second axe de variante** est réalisé par **clivage à préfixe égalisé** (`C3`). Est-ce un
   contraste **de constituants** suffisant à tes yeux, ou exiges-tu un axe qui déplace l'entité dans
   la **structure** (au prix de `C3`) ?
3. Les **10 ponts** (tige partagée à travers deux domaines) créent des entités de type « même nom de
   lieu dans deux domaines ». Est-ce écologiquement acceptable, ou introduis-tu une **ambiguïté
   référentielle** qui contaminerait S2 ?
4. Les **40 noms communs S0** doivent être hors des quatre domaines **et** sans token partagé avec
   aucune entité. Confirmes-tu que « hors domaine » se vérifie par **appartenance lexicale
   déclarée**, et non par une mesure de similarité (qui serait circulaire) ?
5. *(gate)* La **primaire 2 à pool surface-apparié** réintroduit S3/S2 dans un test décisionnel. Que
   prédis-tu, biologiquement, de l'écart entre primaire 1 et primaire 2 — et ce qu'un écart nul
   signifierait ?

## 12. Questions pour lab-math

1. **`C5` restaure-t-elle l'exactitude de `1/37` à ta satisfaction** — ou l'exclusion des unités à
   token partagé introduit-elle un **biais de sélection** du pool que ta démonstration ne couvre pas
   (le pool n'est plus un tirage uniforme parmi les unités) ?
2. **Clustering par famille de REQUÊTE** dans **toutes** les strates : est-ce légitime pour S2/S3, où
   une paire appartient à **deux** familles ? Faut-il un bootstrap **à deux voies**, et si oui à quel
   coût en `K` effectif ?
3. **`T = 0.25` / `τ = 0.10`** : dérive-les ou corrige-les. Précisément : à `K = 20` et statistique de
   cluster bornée dans [0,1], quelle **demi-largeur d'IC** attends-tu (i) sous `ΔR1_inv ≈ 0` et (ii)
   sous `ΔR1_inv ≈ 0.5` ? La bande `N-a` est-elle **atteignable** (non vide) à ces seuils, ou
   reproduit-elle le 0-13 ?
4. **`V-freq`** : le seuil de déséquilibre résiduel **0.15 par énumération** est-il la bonne échelle,
   et quelle est la borne **exacte** atteignable par serpentin sur 24 familles / 4 domaines / 3
   tokenizers non alignés ?
5. **`V-dtype`** : le seuil **1 %** de paires décisionnelles à écart `< 2δ̂` — dérivable, ou faut-il
   le poser au niveau de la **probabilité d'inversion de rang** plutôt qu'à celui des paires ?
6. **`N` de la prédiction ordinale** : à 72 / 90 / 540 / 1854 paires et `K = 20`, quelles transitions
   du 2 × 2 sont **décisionnelles** et lesquelles doivent être déclarées **descriptives avant
   mesure**, sur le modèle du bin dur ?
7. *(gate, BLOQUANT)* **Primaire 2** : dérive la **nulle exacte** du pool surface-apparié — elle
   n'est pas `1/37`, puisque les concurrents à recouvrement de tige y sont admis **par construction**.
   Donne sa loi, ses conditions d'exactitude, son `K` effectif, et son cardinal D24-b. Si aucune
   nulle exacte n'existe, dis-le : la primaire 2 devient alors **descriptive avant mesure** plutôt
   que décisionnelle, et cela se grave **maintenant**.
8. *(gate, BLOQUANT)* **`V-lex`** : formalise le plancher lexical à 0 forward pour `ΔR1_inv`, et
   énonce le critère de **domination** (l'IC de la primaire doit dominer quoi, exactement : le point
   estimé du plancher, sa borne supérieure d'IC, ou une quantité appariée ?).

## 13. Questions pour le PI — **TRANCHÉES à la gate du 2026-08-23** (voir §14)

1. Adoption du schéma tige+suffixe, **60 + 12 réserve, `K = 20`** → **ADOPTÉ**.
2. Vivier de tiges 45 → 90 → **ADOPTÉ**.
3. Coût de `C5` / seconde primaire → **ADOPTÉ, avec baseline lexicale obligatoire**.
4. Aucun GPU avant PASS intégral du banc, `fact_pairs` en échec compris → **confirmé** (doctrine
   D14/D22, jamais rouverte).

## 13-bis. Principe de méthode — **D26 : la difficulté migre**

*Position du PI, 2026-08-23, versée au chapitre méthode et gravée en `docs/ARCHITECTURE.md` §3.*

> **`C5` illustre que la difficulté ne se supprime pas, elle migre.** La nulle inexacte — dominance
> de surface dans le pool — a été réparée **en purifiant le pool** ; et la difficulté a réapparu
> **instantanément dans le contrôle** : l'accès lexical à la cible. Chaque purification du matériau
> déplace le confondant **vers l'endroit que le protocole ne regarde pas encore**. La seule réponse
> stable est celle que la **double primaire** incarne : **mesurer des deux côtés de la purification,
> jamais d'un seul.**
>
> C'est la **cinquième fois** que le projet redécouvre cette structure — signal/bruit de X7, les deux
> nulles de v3, `N-a`/`N-b`, les deux pools, le plancher lexical. À la cinquième occurrence, ce n'est
> plus une astuce locale : **c'est la signature méthodologique du labo**, au même rang que l'arrêt
> dur et le banc de satisfiabilité.

**Portée opératoire dans ce protocole** : la primaire 1 (pool `C5`, purifié) et la primaire 2 (pool
surface-apparié, non purifié) ne sont pas une redondance de prudence — elles sont **les deux côtés
d'une même purification**, et le rapport ne peut prononcer aucune bande supérieure à `N` sur la seule
primaire 1. `V-lex` est le troisième point de mesure du même principe : il regarde ce que la
purification a rendu **plus facile**.

## 14. Amendements de la gate PI — 2026-08-23

1. **Design adopté : 60 unités décisionnelles + 12 de réserve, `K = 20`.** Le repli 48/`K = 16`
   n'est pas déclenché et son amendement (réserve substituable **avant le premier token**, jamais
   additive) est adopté inconditionnellement.
2. **Seconde primaire à pool surface-apparié, ET baseline lexicale à 0 forward — les deux, pas l'un
   ou l'autre.** Motif gravé (défaut 0-45) : *`C5` restaure l'exactitude de la nulle en supprimant la
   difficulté de la tâche* — un pool sans aucun recouvrement de tige, face à une cible qui partage
   l'entité en entier, est résoluble **par lecture lexicale de l'état**. I2 a mesuré ce mur (max AUC
   sous le plancher lexical) ; `C5` le rend **plus fort**. Conséquences : porte `V-lex` (§4.5),
   critère d'abandon F (§6), questions bloquantes 7 et 8 à Math (§12), question 5 à Neuro (§11).
   **La primaire 2 ne part au banc qu'une fois sa nulle dérivée par Math** ; si aucune nulle exacte
   n'existe, elle est **descriptive avant mesure**, et cela se grave avant génération.
3. **Vivier de tiges porté de 45 à 90 candidats** (+20 min, marge ×2 au lieu de ×1.1). Motif : `C2`
   interdit la rustine locale — un rendement inférieur à l'attendu impose une **re-qualification
   complète**, pas un remplacement ponctuel.
4. **Ordre d'exécution confirmé** : instruction Math (Q7, Q8) et re-signature Neuro → banc complet
   (`E = 0`, `fact_pairs` en échec compris) → **gel du matériau** → une seule mesure I2 sur les trois
   modèles. **Aucun GPU avant PASS intégral.**

---

## Historique

- **2026-08-23** — Brouillon v4-matériel (mode cadrage), sous les 4 positions PI « avant
  consolidation » et les specs S-1..S-8 du journal.
- **2026-08-23** — Avis **Math RÉSERVÉ** (6 points, 5 bloquants) et **Neuro RÉSERVÉ** (5 points, 4
  bloquants).
- **2026-08-23** — **Positions PI post-avis** (5 points), **D24-b** et **D25** gravées ; consigne :
  *rien au banc avant instruction du point D*.
- **2026-08-23** — **CONSOLIDATION** par `lab-director`. Instruction du point D : (a)
  **échangeabilité cassée**, réparée par `C5` ; (b) **trois des quatre conditions re-dérivées**,
  `C1..C5` ; (c) **coût chiffré**, surcoût 60 vs 40 ≈ 15 min. Plan de strates **2 × 2 exact**.
  Onze défauts acquittés (0-34 à 0-44). Annulation de la prédiction signée de Neuro **consignée avec
  motif**.
- **2026-08-23** — **Gate PI** : design **60 + 12** adopté ; vivier de tiges porté à **90** ; défaut
  **0-45** relevé par le copilote (`C5` supprime la difficulté qu'elle prétend assainir) ⇒ **seconde
  primaire surface-appariée + baseline lexicale `V-lex`** adoptées, avec deux questions **bloquantes**
  à Math. `Statut : PROPOSE` — **non pré-enregistrable** tant que Math n'a pas répondu à Q7/Q8 et que
  Neuro n'a pas re-signé.
- **2026-08-23** : proposé.
