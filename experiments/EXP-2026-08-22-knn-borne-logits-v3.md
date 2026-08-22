# EXP — V2-D(a) v3 : borne de l'étage des logits, design refondé, sous gate de satisfiabilité

**Statut : TERMINE — INCONCLUSIF**

Origine : `docs/EXTENSIONS.md` §2 (V2-D, candidat a) · « Suite » du journal 2026-08-22 · décision PI « cycle méthode d'abord », arbitrages délégués au copilote (2026-08-22).
Antécédents : `experiments/EXP-2026-08-21-knn-borne-logits.md` (INVALIDE) · `experiments/EXP-2026-08-21-knn-borne-logits-v2.md` (INVALIDE + ERRATUM).
Avis, **deux tours** : tour 1 — Math (RÉSERVÉ, 3 amendements, 5 défauts) et Neuro (FAVORABLE, 2 amendements, 3 façades) ; tour 2 sur le consolidé — Math (FAVORABLE sous 2 corrections bloquantes + 2 scellements) et Neuro (RÉSERVÉ → FAVORABLE sous 2 corrections de fichier). **Les deux tours ont trouvé des clauses fausses.**

---

## Arbitrage

### A. Les erreurs du Directeur (acquittées)

| # | Erreur | Trouvée par | Traitement |
| --- | --- | --- | --- |
| **E-D3** | `1/λ* = 20.50413` et `w₁(c=1) = 0.27972` — **faux au dernier digit** (20.504167 ; 0.27971) | Math, tour 1 | **Corrigés** (§3). Un chiffre faux au dernier digit dans un document qui grave D14-R est une faute de forme grave : elle est consignée. |
| **E-D4** | « 4 ULP / 8 ULP dérivés par comptage d'opérations » — **ce n'est pas une borne d'erreur valide** | Math Q1, tour 1 | **Justification remplacée par l'argument structurel de Math** (§4.3). Les constantes survivent, leur fondement change : c'était une dérivation de façade. |
| **E-D5** | « h est l'image exacte de P1 par `q = r²(3−2r)` ⇒ les deux primaires ne peuvent pas se contredire » — suppose l'**indépendance conditionnelle des 3 paraphrases**, que le §7 prédit fausse | Neuro (façade iii), tour 1 | **h rétrogradée en descriptive** (décision 6). *Façade de dérivation, même genre que celles qui ont tué les runs 1 et 2.* |
| **E-D6** | **`P5f-borne` écrite sur le seul `v_nn` est FAUSSE** : avec k = 8 voisins, la bascule peut être portée par la valeur d'un voisin 2..8 ⇒ la borne sous-compte les positions à risque et **peut déclarer « bug » une implémentation correcte** | Math Q4, tour 2 | **Corrigée** : borne sur `min_{v ∈ V_k}`, et `≤` au lieu de `<` (§4.4). Bloquant levé. |
| **E-D7** | **`τ_promu` = « le plus grand τ admissible » est FAUX** : `E3(τ)` **n'est pas monotone** — le soulagement T2 donne `δ(p) < 0` à certaines positions, donc l'ensemble admissible peut être **non connexe** | Math Q1, tour 2 | **Corrigée** : règle du **préfixe admissible connexe** (§4.5). *C'est ma propre correction de l'amendement de Math qui était fausse : la passe (i) avait le bon diagnostic (le plus petit τ est trivial) et la mauvaise dérivation.* |

### B. Corrections du tour 2 — Math

| # | Point | Statut | Traitement |
| --- | --- | --- | --- |
| **Q1** | `E3(τ)` non monotone (T2) ⇒ `τ_promu = max{τ : E3(τ') < 0.05 ∀ τ' ≤ τ}` | **BLOQUANT, intégré** | E-D7. Sous monotonie la règle coïncide avec le sup ; sans elle, elle reste unique et **interdit de sauter une violation**. Si le 1ᵉʳ décile viole déjà, `τ_promu` n'existe pas ⇒ G descriptif, pas de promotion |
| **Q4** | `P5f-borne` sur `v_nn` est fausse ; forme exacte sur `V_k`, inégalité **large** | **BLOQUANT, intégré** | E-D6. Dérivation reprise en entier au §4.4 |
| **Q2** | risque I = **1.528e-5** (ma valeur 1.53e-5 juste ; l'« ≈ 2e-5 » de Math tour 1 était faux) | intégré | table des 6 premiers termes en rationnels exacts, §3 |
| **Q3** | grise à p = 0.3 = **0.7641**, **irréductible** : a = 12 est **forcé** (a = 11 ⇒ risque I 8.91e-5 ; a = 13 ⇒ puissance 0.8192), seul levier b, et monter b convertit de l'INCONCLUSIF en **échec ferme erroné** | intégré | la contrainte implicite est explicitée : `P(échec ferme \| p = 0.3) ≤ 10 %` ⇒ b = 5 maximal |
| **Q3, défaut d'affichage** | puissance exacte à p = 0.5 = **0.89976**, strictement < 0.90 | intégré | écrit **0.8998** partout ; « ≥ 0.90 » proscrit |
| **Q6** | **k(30) = 20** confirmé en entiers ; table n = 11..30 ; frontières **n = 16, 23, 30** | intégré | table complète en §3 ; les trois frontières deviennent des cas obligatoires du test CPU (xx) |
| **Q7** | B = 20 surdimensionné pour la porte, mais **médiane ambiguë à B pair** ⇒ **B = 21** | intégré | +1 permutation, une ambiguïté de moins. Transport « ≤ 3/30 » plus exigeant que « ≤ 1/10 » (breakeven p ≈ 0.125 vs 0.165) — **conservateur**, noté au journal |
| **Q5** | tolérance P5f au descriptif : rien de falsifiable n'est perdu | confirmé | seule perte : une alarme de dérive de régime, conservée en descriptif |
| **Q8** | (c) en pct-espace est **exactement équivalente** à la comparaison des médianes de cos brut (la médiane commute avec toute transformation monotone) ⇒ anisotropie absorbée **intégralement** ; seul résidu = censurage aux bords, **conservateur** | intégré | ajouts descriptifs : vérifier `médiane(pct_inter) > 0`, rapporter les médianes de cos brut à côté |

### C. Corrections du tour 2 — Neuro

| # | Point | Statut | Traitement |
| --- | --- | --- | --- |
| **1** | **Collision secret/entité** : `SECRETS_80[5] = "lighthouse"` et `ENTITIES[16] = "lighthouse"` — l'unité 5 a pour secret un mot présent dans les 4 indices de l'unité 16 | **BLOQUANT — VÉRIFIÉ VRAI, corrigé** | **Vérifié par exécution** (`pool.py` l.20 et l.26). Correction par **règle déterministe** : le secret de l'unité 5 devient le **premier `SECRETS_80[j]`, j ≥ 30, qui passe V-tok** = `SECRETS_80[30]` = **`walrus`**. **`SECRETS_80` n'est pas modifié** (il gèle X9) : la substitution vit dans la table d'unités de v3 |
| **2** | **Porte V-tok** à ajouter ; suspicion de collision `catapult`/`cathedral` sur « cat » | **porte intégrée ; la suspicion est VÉRIFIÉE FAUSSE** | Pré-vérification CPU exécutée : les **30 premiers tokens BPE sont deux à deux distincts, 0 collision** ; la seule collision avec le pool est `lighthouse` (= point 1). Neuro avait raison d'exiger la vérification et a eu raison de ne pas l'affirmer. La porte est conservée : `POOL_PARAPHRASES` introduira des mots neufs |
| **4** | **Aliasing verbe/entité** : `verb = i mod 5` et `entity = i mod 20` ⇒ `i mod 5 = (i mod 20) mod 5`. Le verbe est une **fonction déterministe** de l'entité — plan **aliasé**, pas croisé | **VÉRIFIÉ VRAI, intégré** | Vérifié par exécution sur les 30. Conséquence : **aucune prédiction signée sur le verbe** ; ventilation publiée en descriptif, **avec l'aliasing en légende**. Façade (iv) de Neuro : « produit combinatoire ⇒ facteurs croisés » est faux ici |
| **3** | N11 : F `n ≤ 6/30`, `h_F ≤ 0.35` ✔ (taux, se transpose) ; L6 **`n ∈ [8, 22]/30`** (l'intervalle n'était pas un taux : part épistémique + part d'échantillonnage, seule la seconde rétrécit) ; **clause d'audit si `n_L6 ≥ 25/30`** | intégré | c'est la prédiction de Neuro, à lui de la signer. L'audit à ≥ 25/30 est un garde-fou de fuite, pas un critère de succès |
| **5** | **V-para étendue** : recouvrement vérifié contre **les 30 faits**, pas seulement celui de l'unité (fuite croisée) ; publier la **matrice 30 × 4** des recouvrements BPE | intégré | gratuit, et le partage OWNERS/ENTITIES rend la fuite croisée plausible |
| **6** | nulle inter-unités **ventilée** « 0 attribut partagé » vs « 1 attribut partagé » ; **refuser** de restreindre les paires | intégré, **descriptif hors clause** | restreindre sélectionnerait une nulle plus facile — *un biais conservateur qui protège la conclusion se garde ; on ne l'échange pas contre un biais anti-conservateur qui l'embellit* |
| **7** | **FAÇADE (v)** : interdiction de « complétion de patterns » / CA3 pour décrire un succès L6 — propriété manquante = le **régime attracteur** | intégré | gravée en §2 (iv) |
| **8** | règles de génération de `POOL_PARAPHRASES` (para1/para2/para3 + table `OWNER_OBJ`) | intégré en entier | §7. *L'arbitraire doit être global (un choix appliqué 30 fois), non local (30 choix)* |
| **9** | formulation du renversement d'asymétrie | intégrée verbatim | §2, portée |
| **3'** | cadre attributif unique : **le construit survit**, le jeu n'est pas refusé | acté | analogue = **récupération sous indice dégradé** (partial-cue recall, Nakazawa et al. 2002), pas la généralisation de schéma (Tse et al. 2007). Refuser le jeu confondrait l'effet de cadre avec la comparaison primaire F vs L6 et ferait tomber la puissance à 0.62 |

### D. Écarté

Escalade en deux étapes (Math A2) · Holm sur deux bras (Math Q2) · seuil ΔP6 « 8/10 sur les 10 » (Neuro A-1) · h co-primaire (E-D5) · `P1(λ*) ≥ 1` comme clause de non-vacuité (Math Q7) · `τ_promu` = plus petit τ **et** = plus grand τ (les deux faux, cf. E-D7) · `P5f-borne` sur `v_nn` (E-D6) · tolérance P5f comme **porte** · N9 (sans objet) · **prédiction signée sur le verbe** (aliasé, inanalysable) · restriction des paires inter-unités (sélectionnerait une nulle plus facile) · profil de profondeur 4 taps (refusé pour ce cycle) · toute clause portant sur `d²_min` dans V0 · `VARIED_PAIRS` comme jeu d'unités · **modification de `SECRETS_80`** (il gèle X9).

---

## 0. Pourquoi cette étape maintenant — arbitrage (A) vs (B)

**(B) : v3 rédigé intégralement, banc de satisfiabilité en pré-condition bloquante du pré-enregistrement.**

1. Le critère de sortie du cycle méthode est **auto-référentiel** (« toute porte **de v3** a un contre-exemple exécutable ») : un banc ne peut pas précéder les portes qu'il teste, sans quoi le Builder écrit le protocole — le travail dont trois échecs ont montré qu'il ne se délègue pas.
2. (B) n'omet aucune étape, il les ordonne : v3 est écrit **sans mesure**, le banc s'exerce sur v3, la gate est franchie ensuite. **Le GPU est interdit tant que la gate n'est pas verte.**
3. Amender une clause **après le banc et avant toute donnée réelle** est du pré-enregistrement ; amender après données est ce qui a tué le run 2. La frontière est le premier token de donnée réelle.
4. (B) rend le cycle méthode **mesurable** : la métrique est **E** (§4.1).
5. Le coût GPU est de l'ordre du quart d'heure ; le vrai budget est en heures de rédaction et de revue.

**Validation empirique de (B), à jour** : avant qu'aucun GPU ne tourne, ce cycle a trouvé **sept** défauts — E-D3/E-D4/E-D5 (tour 1), `τ_promu` version Math (passe (i) du §4bis), E-D6/E-D7 (tour 2, dont un dans **ma propre correction**), et deux défauts de **données** (collision `lighthouse`, aliasing verbe/entité) que seule une exécution a pu trancher. Cette dernière ligne est le vrai argument : **deux des sept défauts n'étaient vérifiables que par exécution**, ce qui est exactement ce que le banc automatise.

## 1. Question

Un datastore brut de paires (état caché → token suivant), mélangé à la distribution de sortie du cortex gelé, hisse-t-il le premier token BPE du secret en top-10 sous **indice paraphrasé**, à λ conforme au budget E3 ≤ +0.05 nats/token — et l'étage de clé qui y parvient, s'il existe, est-il l'**état final pré-`lm_head`** ou l'**état de la couche d'injection (6)** ?

## 2. Hypothèse

**H** — il existe au moins un des deux étages de clé pour lequel le mélange aux logits, pris au supremum de sa famille de températures et à λ* = 0.048770575499286, place le premier token BPE du secret en top-10 sur ≥ 2 des 3 indices paraphrasés, pour **≥ 12 des 30 unités** ; et ce gain disparaît sous permutation des valeurs.

**H₀** — même avec la réponse littéralement présente dans le datastore et l'adressage au supremum de la famille, aucun des deux étages ne hisse quoi que ce soit au-delà du match d'état quasi identique.

**H_méthode** — les trois passes du §4bis suffisent désormais : le banc ne trouve **aucune** clause défectueuse échappée (**E = 0**).

### Interdictions de vocabulaire — GRAVÉES AVANT MESURE

- **(i)** Les 30 unités sont distinctes **lexicalement**, pas **structurellement** : un unique cadre attributif `<GN possessif> <entité> <verbe attributif> {secret}`, une seule relation sémantique, une seule position de la cible. Écrire **« invariance à la paraphrase lexicale à l'intérieur d'un cadre attributif unique »**, **jamais** « généralisation ». La propriété **non** mesurée s'appelle **transfert inter-constructionnel**.
- **(ii)** Deux taps ne font pas un profil de profondeur. Seule phrase autorisée : **« la couche 6 bat l'état final »**. **Jamais** « l'invariance vit dans les couches intermédiaires ».
- **(iii)** (D15) La borne `ΔNLL_t ≤ −log(1−λ)` appartient à la **convexité du mélange**, **pas** à l'étage des logits. Le candidat (b) M_out est un **biais additif non convexe et non borné**. **L'innocuité constatée sur (a) ne se transporte pas à (b).**
- **(iv)** *(Neuro, façade v)* Un succès du bras L6 ne s'écrit **jamais** « complétion de patterns » ni « CA3 ». Il n'y a ici ni récurrence, ni itération, ni point fixe : une clé, un lookup, un mélange. La propriété manquante est le **régime attracteur** (bassin d'attraction, convergence en plusieurs pas) — elle reste du ressort de X2. Seul mot autorisé : **« lookup sous clé dégradée »**.

**Note de design gravée** *(Neuro, façade iv — vérifiée par exécution)* : dans `pool.fact_pairs`, `entity = i mod 20` et `verb = i mod 5`, or `i mod 5 = (i mod 20) mod 5`. **Le verbe est une fonction déterministe de l'entité** : le plan est **aliasé**, pas croisé. Aucun écart entre niveaux de verbe n'est attribuable au verbe. Toute ventilation par verbe est **descriptive et porte cette phrase en légende**.

**Portée, à écrire au journal quoi qu'il arrive** : ce run borne ce qu'un mélange à l'étage des logits peut faire avec des clés égales à des états du cortex gelé, à ces deux étages, dans un cadre attributif unique. L'analogue biologique est la **récupération sous indice dégradé** (partial-cue recall ; Nakazawa et al. 2002), **pas** la généralisation de schéma (Tse et al. 2007). La borne ne se retourne pas : un échec ne dit pas « la localisation est le problème » ; un succès ne démontre aucun mécanisme (le datastore contient la réponse). Le mélange à `c → 0` est un **WTA imposé, non émergent**.

**Renversement d'asymétrie** *(formulation de Neuro, à recopier au journal)* :

> « Le canal d'état (`λ·M·φ(h)` dans le résiduel) a un gain borné par la géométrie de la couche et un dommage non borné, puisqu'une direction fausse déplace le logit d'un montant arbitraire ; le mélange convexe aux logits `(1−λ)·p_cortex + λ·p_mem` renverse exactement cette asymétrie — dommage borné par `−log(1−λ)` quelle que soit la mémoire, gain non borné jusqu'à la masse que le cortex refusait au bon token. »
>
> *Clause de non-transport* : ce renversement est une propriété de la **convexité du mélange dans l'espace des probabilités**, pas de l'étage des logits ; il disparaît dès qu'un canal écrit additivement sur les logits — (b) M_out est non borné dans les deux sens et ne bénéficie d'aucune des deux moitiés de la phrase.
>
> Ligne biologique : le résiduel est une **entrée synaptique dans un état encore en train d'être calculé** ; le mélange convexe est un **vote à la lecture**, où la contribution d'une population est bornée par son poids — le régime de la microstimulation de MT (Salzman, Britten & Newsome 1990), déjà la référence de D11.

**Conformité** : D7 · D8 · D9 · D11 · D12 · D13 · D14 · D14-S · D14-R · D15.

## 3. Ce que le projet sait déjà — avec statut de provenance

**Règle structurelle (D14-R)** : *aucun chiffre historique n'entre dans une porte de v3.*

| Fait | Chiffre | Source | Statut D14-R |
| --- | --- | --- | --- |
| Pas de composante directionnelle du canal d'état | cos ≈ −0.01 vs 0.136 ; ΔH +0.141 | X7 | orientation seule |
| Verrou top-10 | 0/10 depuis X0 | X0→X8 | orientation seule |
| Référence courante (défauts X8) | E1 +1.353 ± 1.58 / 0/10 / E3 −0.014 | X8 | **re-mesuré** (porte V-base) |
| Généralisation paraphrase (Δlogp, jamais rang) | 0.68 [0.56, 0.99] ; 0.38 sous gate | E1b / COR-02 | orientation seule |
| d²_min run 1 sous indice exact | 0.00449220464 (après ERRATUM) | ERRATUM 2026-08-22 | **retiré de toute porte** |
| Le canal récupère à question fixée | P1-exact 10/10, rang 2 | run 2 (**INVALIDE**) | interdit de citation hors mention « run invalidé » ; re-mesuré |
| Observation P6 : h 1.0000 (L6) vs 0.3333 (F), **N_eff = 3** | id. | run 2 (**INVALIDE**) | **motive le design** ; **n'est ni prédiction ni seuil** |
| Distracteur 30 k détruit tout | id. | run 2 (**INVALIDE**) | orientation ; P7 re-mesurée |
| Coût GPU d'une passe complète à 10 unités | 105 s | run 2 (**INVALIDE**) | budget indicatif ; re-mesuré (§9) |
| Anti-pseudo-réplication : unité ≠ clé partagée | — | Q-01b §7(i) ; C4 run 2 | **règle**, porte **V-indep** |
| **Collision `lighthouse` secret/entité ; aliasing verbe/entité** | — | **vérifié par exécution le 2026-08-22** sur `eval/pool.py` | **corrigé** (unité 5 → `walrus`) et **gravé** (§2, note de design) |

**Acquis analytiques — re-dérivés intégralement :**

- `ΔNLL_t = −log[(1−λ) + λ·p_kNN(y_t)/p_LM(y_t)] ≤ −log(1−λ)`, **égalité ssi `p_kNN(y_t) = 0`**.
- **λ* = 1 − e^(−0.05)** ; e^(−0.05) = 0.951229424500714 ⇒ **λ* = 0.048770575499286**.
- **1/λ* = 20.504167** *(corrigé, E-D3)*.
- **λ*/(1−λ*) = e^{0.05} − 1 = 0.0512711** (identité exacte).
- top-10 à `p_kNN(cible) = 1`, `p_LM(cible) ≈ 0` : `λ₁₀ = p₁₀/(1+p₁₀)` ; **F₁₀ = 20.504167·p₁₀/(1+p₁₀)** ; budget mordant **ssi `p₁₀ < 0.0512711`**.
- Température, k = 8 : `w₁ = 1/(1 + 7·e^(−1/c))` = **0.800182 / 0.279709 / 0.166231** pour c = 0.3 / 1 / 3 *(corrigé, E-D3)*.
- **ULP fp64 au bord** : `−log1p(−λ*) = 0.05 ∈ [2⁻⁵, 2⁻⁴)` ⇒ ULP = **2⁻⁵⁷ = 6.938893903907228e-18** ; 4·ULP = **2.7755575615628914e-17** ; 8·ULP = 2⁻⁵⁴ = **5.551115123125783e-17**.
- **Chaînage paraphrase → unité** : `P(≥ 2 succès sur 3 | r) = r²(3 − 2r)`.

**Plan à N = 30, seuils 12/30 et 5/30 — dérivations exactes (Math, tour 2) :**

- **Puissance à p = 0.5 : `1 − 107 636 402/2³⁰` = 0.89976** ⇒ **écrire 0.8998**, jamais « ≥ 0.90 ».
- **Risque I à p = 0.1 : Σ_{k≥12} C(30,k)·9^(30−k)/10³⁰**, termes : 1.29822e-5 + 1.99726e-6 + 2.69471e-7 + 3.19374e-8 + 3.3268e-9 + 3.044e-10 + (queue < 2.5e-11) = **1.528e-5**. *(L'« ≈ 2e-5 » du tour 1 était faux.)*
- **FWER deux bras (Bonferroni, majorant même sous corrélation positive) ≤ 3.06e-5.**
- **`a = 12` est FORCÉ** : `a = 11` ⇒ risque I 8.91e-5 > 3e-5 ; `a = 13` ⇒ puissance `1 − 194 129 627/2³⁰` = 0.8192 < 0.90. Le seul levier est `b`.
- **Zone grise** : `P(6 ≤ X ≤ 11 | Bin(30, ½))` = `(107 636 402 − 174 437)/2³⁰` = **0.1001** (contre 0.366 à N = 10) ; `P(6 ≤ X ≤ 11 | Bin(30, 0.3))` = **0.7641** (contre 0.7004 à N = 10). **Irréductible** sous la contrainte explicitée `P(échec ferme | p = 0.3) ≤ 10 %` : monter `b` réduit la grise mais convertit de l'INCONCLUSIF en **échec ferme erroné** (`P(X ≤ b | Bin(30, 0.3))` = 0.0766 / 0.1595 / 0.2814 pour b = 5 / 6 / 7) ⇒ **b = 5 maximal**. *La vraie amélioration de N = 30 est la grise à p = 0.5 (0.366 → 0.1001) : la grise se concentre sur les p réellement ambigus.*
- **Image des seuils P1 sur h** : `r²(3−2r) = 0.400` ⇒ r ≈ 0.433 ; `= 0.1667` ⇒ r ≈ 0.259. À N = 10 : 0.50 / 0.20. **Un seuil dérivé qui bouge avec N est l'image d'un plan d'expérience, pas d'une hypothèse.**

**Table `k(n) = min{k : Σ_{j≥k} C(n,j)/2ⁿ ≤ 0.10}`** — ancres à la main {6→6, 7→6, 8→7, 9→7, 10→8} ; **vérifiée en entiers exacts par Math jusqu'à 30** :

| n | 11 | 12 | 13 | 14 | 15 | **16** | 17 | 18 | 19 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k(n) | 9 | 9 | 10 | 10 | 11 | **12** | 12 | 13 | 13 | 14 |

| n | 21 | 22 | **23** | 24 | 25 | 26 | 27 | 28 | 29 | **30** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k(n) | 14 | 15 | **16** | 16 | 17 | 17 | 18 | 18 | 19 | **20** |

`k(30) = 20` : `Σ_{j≥19} C(30,j) = 107 636 402 / 2³⁰ = 0.100244 > 0.10` (rejeté) ; `Σ_{j≥20} = 53 009 102 / 2³⁰ = 0.049369 ≤ 0.10`. **Frontières à la 3ᵉ décimale — cas obligatoires du test CPU : n = 16 (0.10506), n = 23 (0.10502), n = 30 (0.100244).** La règle est recalculée **en entiers** par le script, jamais lue dans cette table ; la table est la **référence** du test.

## 4. Prédictions chiffrées

### 4.1 Volet méthode (mesurable sur le banc, avant tout GPU)

| Métrique | Si H_méthode vraie | Si H_méthode fausse | Seuil + dérivation |
| --- | --- | --- | --- |
| **E** = clauses de v3 déclarées **insatisfiables**, **vacuées** (par insatisfaction ou par satisfaction) ou **à variance nulle par construction** par le banc, **et non listées par les trois passes du §4bis ni par les deux tours d'avis** | **E = 0** | **E ≥ 1** | Base honnête : run 1 (1 passe) → 2 échappées ; run 2 (2 passes) → 2 échappées ; ce cycle → **7 défauts trouvés avant banc**, dont **2 par les seuls experts après les trois passes** (E-D6, E-D7) et **2 non décidables sans exécution** (collision `lighthouse`, aliasing verbe/entité). E ≥ 1 ⇒ H_méthode **rejetée**, banc obligatoire à vie ; protocole corrigé, banc rejoué en entier. |
| **E ≥ 3** | — | — | **Réduction de portée obligatoire** (§6). |
| Couverture | **100 %** des clauses ont un contre-exemple **passant** et un **échouant** exécutables | < 100 % | Gate de pré-enregistrement, binaire, sans marge. |

**Note d'honnêteté pré-enregistrée** : le tour 2 d'avis a trouvé **deux clauses fausses** (E-D6, E-D7) que les trois passes du §4bis avaient laissées passer. `H_méthode` telle qu'énoncée (« les trois passes suffisent ») est **déjà mise à mal** avant le banc ; `E` reste défini **sur le banc seul**, mais le journal devra porter la ligne : *les passes du Directeur n'ont jamais, à ce jour, suffi sans un tour d'expert derrière.*

### 4.2 Volet scientifique — cellule et bras

Unité = **le gabarit**, **N = 30**. **Deux bras co-égaux, décidés séparément, jamais poolés** : **F** (`knn_key_layer="final"`) et **L6** (`knn_key_layer="inject"`, layer 6). Cellule décisionnelle : **`sup_c`** sur `c ∈ {un-hot, 0.03, 0.1, 0.3, 1, 3}`, à λ*, k = 8, `T_q = c·med_{j≥2}(d²_j − d²_min)` par requête ; le c-du-sup est loggé. Multiplicité de température **absorbée par P3, évalué au même `sup_c`, dans le même bras**. Cible = **premier token BPE** de `" <secret>"`. Ex-æquo : un-hot ≡ uniforme sur l'argmin-set, compte loggé. Dispersion rapportée : `n/30` exact par bras avec **IC 95 % de Clopper-Pearson** (Math).

### 4.3 Portes d'intégrité et d'environnement

| Porte | Clause | Dérivation du seuil |
| --- | --- | --- |
| **V-base** | E1 / top-10 / E3 aux défauts `EngramConfig()` reproduits ; **produit et scelle `borne_marge`** | comparaison à la valeur **re-mesurée** au centième ; écart ⇒ **arrêt** |
| **V-cap** | \|lm_head(h) − logits\|_max ≤ **1e-5** | logits d'amplitude ~30 en fp32 : 30·2⁻²³ = 3.6e-6 ⇒ 1e-5 |
| **V-drift** | contrôle croisé bit-à-bit de la passe V0 contre `experiments/results/knn-borne-logits-v2/raw/` | **vérification d'environnement seulement** ; un écart est une anomalie à signaler, **jamais** une autorisation de réutiliser les bruts |
| **V-bord** | `bord` est **le flottant produit par l'expression `−log1p(−λ)` et par ce code seul** ; égalité **bit-à-bit** avec le `bord` utilisé par V1b-1/V1b-2 | si `bord` provient de `−log(1−λ)`, l'écart atteint ~2 ULP et **mange la moitié de la marge de V1b-2**. Sans cette porte, V1b n'a pas de fondement |
| **V0** | R1 = 1 sous indice **exact**, 1 unité par bras. **Aucune clause sur d²_min** | échec ⇒ store/indexation cassés ⇒ **invalide** |
| **V-indep** | (a) 30 × 4 clés de requête **deux à deux distinctes** (bit-à-bit) ; (b) sd inter-unités de d²_min **> 0** ; (c) sd inter-unités de p₁₀ **> 0** ; (d) **max des cosinus inter-unités, rapporté** (descriptif) | opérationnalise C4. Le pool partage OWNERS (mod 16), ENTITIES (mod 20), VERBS (mod 5) : **distinctes ≠ décorrélées**, d'où (d). Échec de (a)/(b)/(c) ⇒ **arrêt** |
| **V-tok** *(nouvelle — Neuro, tour 2)* | (a) les **30 premiers tokens BPE des secrets** sont deux à deux **distincts** ; (b) **aucun** ne coïncide avec le 1ᵉʳ token BPE d'un mot de `OWNERS ∪ ENTITIES ∪ VERBS` ; (c) **aucun secret n'est un mot du pool** | la primaire est « 1ᵉʳ token BPE en top-10 » : une collision rend un succès attribuable au datastore d'une **autre** unité. **Pré-vérifiée le 2026-08-22** : (a) 30/30 distincts, 0 collision ; (b)+(c) une seule violation, `lighthouse` — **corrigée** (unité 5 → `walrus`). La porte reste **bloquante** : `POOL_PARAPHRASES` introduit des mots neufs |
| **V-para** *(étendue — Neuro, tour 2)* | pour **chaque** unité : (a) aucun de ses 3 indices paraphrasés n'est sous-chaîne de son fait ; (b) Jaccard BPE(indice, fait \ secret) **strictement inférieur** à celui de son indice exact ; (c) **fuite croisée** : le recouvrement de chaque paraphrase de l'unité *i* est vérifié **contre les 30 faits**, pas seulement contre le fait de *i* | sinon une « paraphrase » est un indice exact déguisé, ou l'indice d'une **autre** unité. Seuil **relatif par unité**, sans constante arbitraire. **Livrable : matrice 30 × 4 des recouvrements BPE publiée.** Jaccard conservé comme covariable **C1** |
| **V-hash** | SHA-256 de `POOL_PARAPHRASES`, de `OWNER_OBJ` et de la table d'unités identiques avant et après le run ; SHA-256 de `data/rfc9293.txt` re-vérifié | les paraphrases sont des données, elles se gèlent comme `SECRETS` |
| **V-tie** | compte d'ex-æquo de d²_min rapporté ; > 0 ⇒ un-hot appliqué uniforme sur l'argmin-set | — |
| **V1a** | sur `p_kNN(y_t) = 0` : `max_t \|ΔNLL_t + log1p(−λ)\| ≤ 1e-6` | identité algébrique ; erreur attendue ~1e-15 ⇒ 9 ordres de marge |
| **V1b-1** *(volet faible)* | sur `p_kNN(y_t) > 0` : `ΔNLL_t ≤ bord + 4·ULP`, **100 %** | **Argument structurel** : `δ̂ = log1p(t)`, `t ≥ 0` ⇒ `δ̂ ≥ 0` **exactement en machine** ; l'arrondi au plus près ne franchit jamais un représentable ⇒ `fl(bord − δ̂) ≤ bord` **exactement**. La clause tient **à marge 0** ; les 4 ULP sont du mou délibéré. **Satisfiable en tout régime, y compris `r ≲ 1e-16`** |
| **V1b-2** *(volet strict, restreint)* | sur `{δ̂ ≥ 8·ULP}` : `ΔNLL_t < bord` **strictement**, 100 %. Sur le complément : `\|ΔNLL_t − bord\| ≤ 8·ULP`, **compte rapporté, non nul attendu** | `δ̂ ≥ 2⁻⁵⁴` ⇒ `fl(bord − δ̂) ≤ bord − 8·ULP < bord`. **C'est la porte que le Builder a dû substituer après données au run 2 : elle est ici pré-enregistrée sous sa forme correcte** |
| **V1c** | E3 mesuré vs recomposé hors-ligne ≤ **1e-6 nats** | test d'intégrité **régime-libre** |
| **V-var** | `var(D_t)` sur `p_kNN = 0` = **0 à 1e-12** | identité : D_t constant. Remplace toute porte de corrélation (NaN) |
| **V2** *(faisabilité, rapportée AVANT P1)* | `n_faisable` = unités avec `p₁₀ < 0.0512711` sur ≥ 2/3 paraphrases. **≤ 15/30 ⇒ `INCONCLUSIF — budget arithmétique`** | transposition du seuil v2 (≤ 5/10 = 50 %) ; le canal n'a pas eu sa chance |
| **V-λ0** | `knn_lambda = 0.0` ⇒ logits **bit-exacts** vs baseline | rejoué en fin de run |

### 4.4 Prédictions décisionnelles

| # | Métrique | Si H vraie | Si H fausse | Seuil + dérivation | Antipode (D13) |
| --- | --- | --- | --- | --- | --- |
| **P1** *(SEULE DÉCISIONNELLE, ITT, par bras)* | n = unités avec ≥ 2/3 paraphrases en top-10, à λ*, `sup_c` | **n ≥ 12/30** | **n ≤ 5/30** | puissance **0.8998** à p = 0.5 ; risque I **1.528e-5** à p = 0.1 ; **FWER ≤ 3.06e-5** ; grise [6, 11] = 0.1001 à p = 0.5, 0.7641 à p = 0.3. **`a = 12` forcé** (§3). Test des signes exact, IC Clopper-Pearson. **Pas d'escalade** | n ≥ 12 mais **R1v > 1 sur la majorité des succès** ⇒ **ininterprétable** |
| **P1-cellule-dégénérée** | — | — | — | majorité des échecs avec `R1v > 1` **et** sup **au bord de grille** ⇒ `INCONCLUSIF — cellule dégénérée` | — |
| **P1-exact** *(ancrage, par bras)* | idem, indice exact | 30/30 | < 30/30 ⇒ **suspect** | l'entrée correcte est à d² minimal ⇒ top-10 ssi `p₁₀ < 0.0512711` ; échec à p₁₀ conforme = bug | — |
| **ΔP6** *(arbitrage des deux bras)* | test des signes **conditionnel aux paires discordantes**, **unilatéral** α = 0.10, direction pré-déclarée par N2′ (L6 > F) | **succès L6 ≥ k(n_disc)** | < k(n_disc) | table §3, **recalculée en entiers** par le script. **n_disc < 5 ⇒ NON ÉVALUABLE** ; n_disc = 5 ⇒ unanimité | L6 ≤ F sur les discordantes ⇒ **N2′ falsifiée** |
| **ΔP6-sec** | Wilcoxon apparié sur la **médiane par unité de `log₂ R1(para)`** | L6 inférieure d'**≥ 1 bit** | — | résolution continue là où h n'a que 4 valeurs ; **secondaire, ne renverse jamais ΔP6** | — |
| **P3** *(BLOQUANT, hors-ligne, même `sup_c`, même bras)* | valeurs permutées, **21 permutations** | médiane ≤ 3/30 | ≥ 12/30 | identité exacte sur `valeur(argmin) ≠ cible`. **B = 21 et non 20** : à B pair la médiane est la moyenne des 10ᵉ/11ᵉ stats d'ordre, donc demi-entière — ambiguïté supprimée pour +1 permutation (Math Q7). SE de la médiane ≈ 0.44 unité sous `Bin(30, 0.03)` ⇒ franchissement sous la nulle ~1e-14 ; détection à ~93 % contre une fuite à p = 0.15. Transport « ≤ 3/30 » **plus exigeant** que « ≤ 1/10 » (breakeven p ≈ 0.125 vs 0.165) — **conservateur, noté** | identité violée ⇒ `INCONCLUSIF` |
| **P4** *(spécificité)* | store unité A, indice unité B | ≤ 3/30 | ≥ 12/30 | > 12/30 ⇒ sélectivité nulle | — |
| **P5** *(porte d'implémentation, PAS un test)* | E3 sur `NEUTRAL_TEXT` à λ* | — | — | **`E3(λ*) ≤ 0.05` est un THÉORÈME** : dépassement = **bug**. Prédiction **[0.046, 0.050)** ; ordinal : E3(fait-seul) > E3(distracteur) | E3(λ*) > 0.05 ⇒ **bug, jamais un résultat** |
| **P5f-borne** *(porte d'implémentation, zéro paramètre libre — CORRIGÉE, E-D6)* | taux global de bascule d'argmax | — | — | **≤ `borne_marge`**, avec **`borne_marge` = #{p : min_{v ∈ V_k(p)} (p_max(p) − p_LM(v)) ≤ 0.0512711} / T**, `V_k(p)` = les valeurs des **8 voisins** à la position p, mesurée dans la passe V-base **sans kNN** (la retrieval ne dépend pas du mélange), hashée et scellée avant la passe A. **Inégalité LARGE** (`≤`, pas `<`) : aux ex-æquo l'argmax bascule par convention d'indice à marge exactement égale à la borne. Dépassement ⇒ **bug** | — |

**Dérivation de `P5f-borne`** (Math Q4, à recopier au journal) : soit `a = argmax p_LM`, `p_mix = (1−λ*)p_LM + λ*p_kNN`. Si `argmax p_mix = w ≠ a`, alors `p_mix(w) ≥ p_mix(a)`, d'où `(1−λ*)(p_max − p_LM(w)) ≤ λ*(p_kNN(w) − p_kNN(a)) ≤ λ*·p_kNN(w) ≤ λ*`, donc `p_max − p_LM(w) ≤ λ*/(1−λ*) = 0.0512711`, **avec `w ∈ V_k`** (un token hors `V_k` a `p_kNN = 0` et ne peut pas basculer). **La version écrite sur le seul `v_nn` sous-comptait les positions à risque et pouvait déclarer « bug » une implémentation correcte.**

**E3 ≤ +0.05** est porté par **P5** (théorème), **P7** (store distracteur) et **G(τ)**.

### 4.5 Bras descriptifs (aucun ne décide ; tous rapportés in extenso)

| # | Contenu | Clause |
| --- | --- | --- |
| **h** *(rétrogradée)* | taux de récupération par paraphrase, **30 valeurs in extenso**, médiane + IC bootstrap **« indicatif, N = 30 »**, ventilation par **type de paraphrase (C3)**, sd inter-unités (N13) | **h n'entre dans aucune porte et dans aucun critère d'ouverture.** Si h = 0 partout : borne binomiale exacte unilatérale de Clopper-Pearson sur les 90 couples groupés par unité |
| **Ventilation par verbe** *(Neuro, tour 2)* | 30 valeurs de P1 par niveau de verbe (6 par niveau, équilibré) | **descriptif, aucune prédiction signée** : le verbe est **aliasé sur l'entité** (§2). La légende porte cette phrase, obligatoirement |
| **F₁₀** | taux de succès ; `log F₁₀` conditionnelle aux succès ; nb d'échecs | bimodale et **constante entre bras par construction** ⇒ **jamais comparée entre bras, jamais une médiane seule** |
| **P5c-id** | partition {y_t ∈ valeurs recevant de la masse} vs complément : `f_relief`, distribution de D_t | `f_relief(un-hot)` **strictement <** `f_relief(c fini)`. **Aucune bande absolue.** Antipode : ≥ ⇒ normalisation de p_kNN fausse |
| **P5f-cond** | **taux conditionnel de bascule sachant marge fine**, ventilé par décile de NLL_base ; **identité du token gagnant sur chaque bascule confiante** | **semi-analytique** : re-description de la distribution des marges du cortex gelé, **ne crédite aucune théorie**. IC 95 % par **bootstrap par blocs (L = 25)**, secours `1.96·σ_rel·√(1+2Σ₁⁵ρ̂_k)`, **les deux rapportés** — aucune porte n'en dépend. **Antipode** : bascule **confiante** dont le gagnant est **une valeur du store** = **intrusion mnésique** (informatif) ; token arbitraire = **bug** (invalide) |
| **P7** | h et P5 sous store **30 k + le fait** | dégradation ≤ 30 % si sélectif ; `corr(dégradation, densité locale) > corr(dégradation, taille du store)` |
| **P8** | rang vs d²_min, poolé (~120 points/bras) | graduelle, monotone, **sans genou** |
| **G** *(CORRIGÉE, E-D7)* | `α = 1[d²_min ≤ τ]` : courbes `E3(τ)`, `P1(τ)` sur les **déciles empiriques dédupliqués** de d²_min mesurés dans le run | **Non-vacuité** : τ évalué seulement si `#{α=1} ≥ 1` **et** `#{α=0} ≥ 1`, comptes publiés. **`τ_promu` = max{τ ∈ grille : `E3(τ') < 0.05` pour TOUT τ' ≤ τ de la grille}** — le plus grand élément du **préfixe admissible connexe**. **Dérivation** : `E3(τ) = (1/T)Σ_p 1[d²_min(p) ≤ τ]·δ(p)` est une somme cumulée le long des positions triées par d²_min ; elle est croissante **ssi tous les `δ(p) ≥ 0`**, ce que le mécanisme **T2 falsifie** (aux positions dont la cible est une valeur du store, `δ(p) < 0`). L'ensemble admissible peut donc être **non connexe** ; promouvoir un τ au-delà d'une zone violée accepterait un gate dont un sous-gate strictement inclus casse le budget. Si le 1ᵉʳ décile viole déjà, **`τ_promu` n'existe pas ⇒ pas de promotion**. **Déclencheur « G → organe »** : `P1(τ_promu) ≥ max(1, P1(λ*) + 1)` **et** `E3(τ_promu) < 0.05`, à λ = 0.25 |
| **Multi-clé** | `pct(q, k)` = fraction des clés du distracteur 30 k avec `cos ≥ cos(q, k)` (résolution 3.33e-5) ; **`m(q, k) = cos(q, k) − max_j cos(q, d_j)`** | **ARMÉ ssi, sur ≥ 18/30 unités** : (a) R1 > 1 sur ≥ 2/3 paraphrases ; (b) médiane de `pct(q_para, k_correcte) ≥ 1e-3` alors que `pct(q_exact, k_correcte) ≤ 3.33e-5` ; (c) médiane `pct` **intra-unité** **<** médiane **inter-unités**. **(c) fausse ⇒ NE PAS ARMER.** **Saturation** : deux `pct` à 0 ⇒ (c) se décide sur `m`. **Contrôle** : nulle **stratifiée par bucket de position** à côté de la globale ; divergence qualitative sur (b) ou (c) ⇒ **NON ÉVALUABLE** |
| **Multi-clé, descriptifs ajoutés** *(tour 2)* | (i) nulle inter-unités **ventilée « 0 attribut partagé » vs « 1 attribut partagé »** (Neuro) ; (ii) `médiane(pct_inter) > 0` vérifiée ; (iii) **médianes de cos brut** rapportées à côté des `pct` (Math) | **descriptifs, hors clause.** (i) : si la clé porte l'identité d'unité, on attend `pct_intra < pct_(1 partagé) < pct_(0 partagé)` — monotonie ordinale gratuite ; si `pct_(1 partagé) ≈ pct_intra`, la clé encode le **gabarit de surface**, pas l'unité — diagnostic que (c) seule ne rendrait pas. (iii) : (c) en pct-espace est **exactement équivalente** à la comparaison des médianes de cos brut (la médiane commute avec toute transformation monotone), donc l'anisotropie est absorbée **en mode commun** ; seul résidu = **censurage aux bords**, de sens **conservateur** |
| **Additivité, E1c** | `M` au point courant + kNN ; `λ_renv` | lus **après** verdict. **P2 sans fourchette** : seule clause `F ≤ 10.252` (`p_P − p_M ≤ 1 ⇒ λ_renv ≤ 0.5 ⇒ F ≤ 0.5/λ*`). Étiquette obligatoire : *« mesure du prior de GPT-2, pas du canal »* |

### 4.6 Prédictions signées de Neuro, pré-mesure

- **N2′** — L6 > F sur les paires **discordantes** : au plus une paire favorise F ; médiane par unité de `log₂ R1(para)` **inférieure d'≥ 1 bit** en L6.
- **N10** — médianes par type : `h(para1) ≥ h(para2) ≥ h(para3)`, écart le plus large en bras F.
- **N11** *(révisée au tour 2 — la transposition proportionnelle était fausse pour L6)* — bras **F : `n ≤ 6/30`, `h_F ≤ 0.35`** (bornes sur un **taux**, se transposent) ; bras **L6 : `n ∈ [8, 22]/30`** — l'intervalle du plan à 10 n'était pas une prédiction de taux mais un **aveu d'ignorance** (`N_eff = 3`) **plus** la marge d'échantillonnage : la part épistémique ne rétrécit pas avec N, la part d'échantillonnage si (sd ≈ 0.09 en taux à N = 30) ⇒ cœur épistémique ≈ [0.35, 0.70], bande observable ≈ [0.27, 0.73]. Deux corrections vont vers le bas : 24 des 30 unités partagent un attribut de surface (plus d'interférence qu'au jeu ad hoc), et l'homogénéité du gabarit resserre la dispersion **sans déplacer la moyenne**. **Clause d'audit signée : `n_L6 ≥ 25/30` n'est PAS un succès** mais déclenche un audit de fuite (V-para croisée, V-tok) — à λ*, hisser 5/6 des unités sous indice paraphrasé est plus vraisemblablement un artefact de recouvrement lexical qu'un effet mémoire.
- **N12** — bras F : (c) vraie sur ≥ 18/30 ; bras L6 : (c) plus faible ou fausse. **(c) vraie en F et fausse en L6 = le meilleur argument mécaniste pour L6, indépendant de h.**
- **N13** — sd inter-unités de h **strictement > 0**, étendue couvrant ≥ 2 des 4 niveaux {0, ⅓, ⅔, 1} — sinon **pseudo-réplication d'issue**, que V-indep n'attraperait pas.
- **N14** — bascules aux positions confiantes **< 0.5 %** ; quand elles existent, le token gagnant est **une valeur du store** (intrusion), pas un token arbitraire (bug).

## 4bis. Auto-vérification D14 — trois passes, ré-exécutées sur le consolidé

### Passe (i) — cohérence logique

| # | Phénomène prédit par ce protocole | Clause exposée | Verdict |
| --- | --- | --- | --- |
| 1 | T2 (soulagement sur les tokens-valeurs) | V1a conditionnée sur `p_kNN = 0` ; T2 vit sur le complément | non contredite |
| 2 | T2 baisse E3 | λ* est le pire cas | non contredite |
| 3 | T-perm | P3 restreint à `valeur(argmin) ≠ cible`, 21 permutations | non contredite |
| 4 | T-exact | V0 sans clause d²_min ; G sur déciles mesurés + non-vacuité | corrigée (défaut C3/G du run 2) |
| 5 | T-const | V-var teste la variance, pas la corrélation | non contredite |
| 6 | T-sup | P3 au même `sup_c`, même bras | non contredite |
| 7 | T-bras (2 bras) | décisions séparées ; FWER ≤ 3.06e-5 ; P3/P4 rejoués dans le bras gagnant **comme contrôle de validité, PAS comme correction de multiplicité** | non contredite |
| 8 | T-item | ITT sur les 30 ; 30 valeurs publiées ; V-para par unité **et croisée** ; C1/C2/C3 | non contredite |
| 9 | T-cadre (unique cadre attributif) | interdiction §2(i) ; « transfert inter-constructionnel » nommé comme non mesuré | traité |
| 10 | P5 ne peut pas échouer à λ* | porte d'**implémentation** | non contredite |
| 11 | P5f impliquée par la baseline | scindée : `P5f-borne` / `P5f-cond` | corrigée |
| 12 | h et P1 sous corrélation des types de paraphrase | h rétrogradée ⇒ plus qu'une primaire | corrigée (E-D5) |
| 13 | `τ_promu` = plus petit τ | corrigé en « plus grand » à cette passe… | **puis re-corrigé : « plus grand » était FAUX aussi (E-D7)** |
| 14 | **T2 casse la monotonie de `E3(τ)`** | **la passe (i) avait déjà T2 en ligne 1 et ne l'a pas confrontée à G.** Corrigé par Math tour 2 : règle du préfixe admissible connexe | **DÉFAUT ÉCHAPPÉ AUX TROIS PASSES** |
| 15 | **La masse kNN est répartie sur k = 8 valeurs** | `P5f-borne` écrite sur `v_nn` seul | **DÉFAUT ÉCHAPPÉ AUX TROIS PASSES** (E-D6, Math tour 2) |
| 16 | **Le pool réutilise ses attributs (mod 16/20/5)** | conséquence non tirée sur les **secrets** (`lighthouse`) ni sur l'**aliasing** verbe/entité | **DÉFAUT ÉCHAPPÉ, non décidable sans exécution** (Neuro tour 2 + vérification) |
| 17 | E3 > 0.05 aux λ élevés | prédit analytiquement, **non** critère d'abandon | non contredite |
| 18 | V2 | impossibilité arithmétique ⇒ INCONCLUSIF, pas REJETE | non contredite |

**Lecture de cette passe** : les lignes 14-16 sont l'aveu du cycle. Les trois passes vérifient qu'une clause n'est pas contredite par un phénomène **déjà listé** ; elles ne vérifient pas que la liste est complète, ni qu'une clause est vraie **arithmétiquement** (14, 15), ni qu'elle est vraie **des données** (16). D'où le banc.

### Passe (ii) — satisfiabilité machine (D14-S)

| Clause | Risque examiné | Résultat |
| --- | --- | --- |
| **V1b-1** | insatisfiabilité fp64 (mort du run 2) | **levée par preuve** : `δ̂ ≥ 0` en machine + monotonie de l'arrondi. **Conditionnée à V-bord** |
| **V1b-2** | strictitude non résoluble | **levée** sur `{δ̂ ≥ 2⁻⁵⁴}` ; le complément est **compté, pas jugé** |
| **V-bord** | — | condition d'existence des deux précédentes |
| **`P5f-borne` corrigée** | `borne_marge` sur `V_k` calculable **sans kNN** ? | **oui** : la retrieval (requête = état caché du LM) ne dépend pas du mélange ⇒ `V_k(p)` est connu à la passe V-base. `borne_marge = 0` ⇒ la clause exige **0 bascule** : satisfiable, pas de NaN (l'inégalité est absolue, pas un rapport) |
| **`τ_promu` corrigée** | `E3(τ)` non monotone ⇒ ensemble non connexe | **traité par construction** : le préfixe connexe est toujours bien défini et unique sur grille finie dédupliquée. Aucun τ admissible ⇒ **G NON ÉVALUABLE** ; le banc doit exhiber une grille **non connexe** et vérifier que le τ post-violation **n'est pas** promu |
| **Multi-clé (c)** | vacuité par satisfaction résolue ; **censurage aux bords** | (c) est une comparaison d'ordre équivalente au cos brut ⇒ anisotropie absorbée. **Résidu conservateur déclaré** : deux médianes au-delà du max des 30 000 clés ⇒ les deux `pct` à 0 ⇒ bascule sur `m`. Le banc doit exhiber le cas saturé |
| **Marge continue `m`** | `m` négative partout (anisotropie) | sans conséquence : comparaison d'ordre, invariante par translation commune |
| **P3 à B pair** | médiane demi-entière | **résolu par B = 21** |
| **Table k(N_eff)** | table tronquée ; frontières à la 3ᵉ décimale | règle **exacte en entiers** + test CPU sur la table de référence, **n = 16, 23, 30 obligatoires** |
| **V-tok / V-para croisée** | non décidables sans tokenizer | **pré-exécutées** le 2026-08-22 sur les secrets ; à rejouer sur `POOL_PARAPHRASES` une fois gelé |
| **`h` avec Clopper-Pearson** | h = 0 partout ⇒ bootstrap dégénéré | repli pré-déclaré ; h n'est plus décisionnelle |

**Statistiques dont l'écart-type inter-unités pourrait être nul par construction — liste AVANT mesure :**

| Statistique | Risque | Traitement pré-enregistré |
| --- | --- | --- |
| d²_min, R1, R1v, p₁₀ entre unités | **cause C4 du run 2** | **V-indep (a)(b)(c)**, échec ⇒ arrêt ; **(d)** rapporte la corrélation résiduelle |
| **h entre unités** | zéro possible **sans être structurel** ; V-indep ne l'attraperait pas (elle teste les clés, pas les issues) | **N13** est ce contrôle ; sd = 0 ⇒ **pseudo-réplication d'issue**, h retirée du rapport |
| F₁₀ entre bras | **constante par construction** | diagnostic seul |
| ΔNLL sur `p_kNN = 0` | nulle **par identité** | c'est une **porte** (V-var), pas une statistique |
| P3 à permutation unique | nulle par construction | **21 permutations** |
| V-drift, V-bord, V-λ0, V-tok | zéro exact attendu | portes, pas des statistiques |
| ΔP6 si tous les bras échouent à 0 | tous ex-æquo | ex-æquo **exclus** ; **n_disc < 5 ⇒ NON ÉVALUABLE** |
| Nulle stratifiée par position | buckets vides | bucket < 1 000 clés fusionné ; < 3 buckets ⇒ **NON ÉVALUABLE** |
| **Ventilation par verbe** | 6 unités par niveau, mais **verbe aliasé sur l'entité** | **descriptif sans prédiction**, légende obligatoire |

### Passe (iii) — provenance de chaque chiffre cité (D14-R)

| Chiffre | Emploi | Statut | Commande / dérivation |
| --- | --- | --- | --- |
| λ* = 0.048770575499286 | cellule | **re-dérivé** | `1 − exp(−0.05)`, §3 |
| 0.0512711 | V2, P5f-borne, bascule | **re-dérivé** | `e^0.05 − 1`, §3 |
| 20.504167 | F₁₀ | **re-dérivé, corrigé** | `1/λ*` (E-D3) |
| 0.800182 / 0.279709 / 0.166231 | grille de température | **re-dérivé, corrigé** | `1/(1+7e^(−1/c))` (E-D3) |
| 2⁻⁵⁷ / 4·ULP / 8·ULP | V1b | **re-dérivé** | exposant binaire de 0.05 |
| **0.8998 / 1.528e-5 / 3.06e-5 / 0.1001 / 0.7641** | P1 | **re-dérivé en entiers exacts (Math, tour 2)** | queues de Bin(30, ½) et Bin(30, 0.1) ; §3 |
| **a = 12 forcé (8.91e-5 ; 0.8192)** | justification du seuil | **re-dérivé** | §3 |
| **table k(n), n = 6..30** | ΔP6 | **re-dérivée en entiers** | §3 ; recalculée par le script, jamais lue dans la table |
| 3.33e-5 / 5e-6 | planchers de rang | **re-dérivé** | 1/30000 ; 1/200000 |
| 10.252 | P2 (descriptif) | **re-dérivé** | `0.5/λ*` |
| **collision `lighthouse` ; aliasing verbe/entité ; 30 tokens BPE distincts** | V-tok, note de design | **re-mesuré par exécution** | tokenizer GPT-2 sur `eval/pool.py`, 2026-08-22 |
| E1 / top-10 / E3 de référence | **porte V-base** | **re-mesuré** | `.venv\Scripts\python eval\fact_injection.py` puis `eval\collateral.py`, défauts `EngramConfig()` |
| `borne_marge` | **porte P5f-borne** | **re-mesuré dans le run** | passe V-base sur `NEUTRAL_TEXT`, `min sur V_k`, **hashée et scellée avant la passe A** |
| grille τ de G | bras G | **re-mesuré dans le run** | déciles dédupliqués de d²_min, passe A |
| ECDF de la nulle (globale, stratifiée, ventilée 0/1) | multi-clé | **re-mesuré dans le run** | passe E |
| d²_min = 0.00449220464 (run 1) | **aucun** | **interdit de porte** | `eval\knn_ceiling.py --phase analysis` sur `results\knn-borne-logits\raw\gpu_raw.npz` |
| h 1.0000 / 0.3333, R1 5.03 → 1.27, 105 s, 4.257 % | **aucun** | motivation de design et budget indicatif, étiquetés « run invalidé » | — |

**Zéro chiffre du journal n'entre dans une porte de v3.**

## Banc de satisfiabilité — livrable préalable au pré-enregistrement (D14-S)

Livrable : `eval/gate_bench.py` (CPU, sans HF sauf tokenizer, sans GPU, SPDX AGPL-3.0-or-later). Pour **chaque** clause, deux jeux synthétiques exécutés, rapport `PASS/FAIL` conforme à l'attendu. **Critère de sortie : 100 %.**

**Usage méta borné** : les portes d'**intégrité seules** (V-cap, V-bord, V1a, V1b-1, V1b-2, V1c, V-var, V-drift) peuvent être rejouées sur `experiments/results/knn-borne-logits{,-v2}/raw/`. Le banc **n'a pas le droit d'émettre une statistique décisionnelle** (P1, ΔP6, P3, P4, h, multi-clé, G) sur les bruts archivés : ceux du run 2 portent des valeurs déjà publiées, les faire traverser les portes de v3 reviendrait à calibrer v3 sur son propre résultat.

| Clause | Contre-exemple **passant** | Contre-exemple **échouant** |
| --- | --- | --- |
| V-base | métriques égales à la référence au centième | une métrique décalée de 0.02 |
| V-cap | logits recalculés à l'identique | écart injecté 1e-4 |
| V-drift | deux tableaux identiques | un bit modifié |
| **V-bord** | `bord` par `−log1p(−λ)` des deux côtés ⇒ égalité bit-à-bit | `bord` par `−log(1−λ)` d'un côté ⇒ **FAIL**, écart ~2 ULP exhibé |
| V0 | clé exacte à d² minimal ⇒ R1 = 1 | entrée parasite plus proche ⇒ R1 = 2 |
| **V-indep (a)(b)(c)** | 120 clés distinctes, sd > 0 | **les 30 unités partagent la même clé** (réplique de C4) ⇒ FAIL |
| **V-indep (d)** | cos max inter-unités < 1 | deux clés bit-identiques ⇒ FAIL par (a) |
| **V-tok** | 30 premiers tokens BPE distincts, aucun dans le pool | **le jeu réel AVANT correction** (`lighthouse` en unité 5) ⇒ **FAIL, unité 5 nommée** ; **et** deux secrets à 1ᵉʳ token identique ⇒ FAIL |
| **V-para (a)(b)(c)** | Jaccard paraphrase < Jaccard exact sur les 30, et croisé | paraphrase = préfixe littéral de son fait ⇒ FAIL ; **et** paraphrase de l'unité *i* recouvrant le fait de *j ≠ i* plus que le sien ⇒ **FAIL croisé, paire (i, j) nommée** |
| V-hash | jeu inchangé | un caractère modifié |
| V-tie | store sans ex-æquo (compte 0) | deux entrées de d² identiques ⇒ compte 2, un-hot uniforme |
| V1a | `p_kNN = 0` ⇒ écart ~1e-16 | ΔNLL perturbé de 1e-5 |
| **V1b-1** | `r = 1e-30` (décrément sous l'ULP) ⇒ **doit PASSER** (cellule qui a tué le run 2) | ΔNLL forcé à `bord + 1e-12` ⇒ FAIL |
| **V1b-2** | `r = 1e-3` ⇒ strictement < bord | `r = 1e-3` avec ΔNLL forcé = bord ⇒ FAIL ; **et** `r = 1e-30` rangé dans le complément avec compte > 0 (attendu, **non-FAIL**) |
| V1c | E3 recomposé identique | décalage 1e-4 |
| V-var | D_t constant ⇒ var = 0 | un D_t perturbé ⇒ var > 1e-12 |
| V2 | p₁₀ < 0.0512711 sur 20 unités ⇒ n_faisable = 20 | p₁₀ tous à 0.2 ⇒ n_faisable = 0 ⇒ `INCONCLUSIF budget` |
| V-λ0 | logits bit-identiques | un bit modifié |
| P1 | n = 14 ≥ 12 | n = 4 ; n = 5 ⇒ H fausse ; **n = 8 ⇒ zone grise, INCONCLUSIF exhibé** |
| P1-antipode | succès avec R1v = 1 | succès avec R1v = 3 majoritaire ⇒ **ininterprétable** |
| P1-cellule-dégénérée | sup à l'intérieur de la grille | sup au bord + R1v > 1 majoritaire ⇒ `INCONCLUSIF — cellule dégénérée` |
| **ΔP6 (table k)** | n_disc = 10, 9 succès ⇒ k(10) = 8 ⇒ ARMÉ | n_disc = 10, 6 succès ⇒ non ; **n_disc = 4 ⇒ NON ÉVALUABLE** ; **n = 16, 23, 30 : k(n) recalculé en entiers et comparé à la table §3 — les trois frontières à 0.1050 / 0.1050 / 0.1002 sont obligatoires** |
| ΔP6-sec | Wilcoxon avec écart ≥ 1 bit | écart nul ⇒ vérifier que le secondaire **ne peut pas** changer le verdict |
| **P3** | 21 permutations, médiane 1/30 | médiane 14/30 ⇒ `INCONCLUSIF` ; **et B pair ⇒ le banc doit REFUSER** (médiane ambiguë) ; **et** une permutation unique ⇒ refus (variance nulle) |
| P4 | croisé 1/30 | croisé 20/30 |
| P5 | E3 = 0.047 | E3 = 0.06 ⇒ **bug**, pas un résultat |
| **P5f-borne** | taux 0.6 × `borne_marge` ; **et** une bascule portée par la valeur du **voisin 5** ⇒ **doit PASSER** (cellule de E-D6 : la version `v_nn` échouerait ici) | taux 1.4 × `borne_marge` ⇒ **bug** ; **et** `borne_marge = 0` avec taux 0 ⇒ PASS (pas de NaN) ; **et** ex-æquo d'argmax à marge exactement égale à 0.0512711 ⇒ **doit PASSER** (inégalité large) |
| P5f-cond + token gagnant | bascule confiante, gagnant = **valeur du store** ⇒ **intrusion** | bascule confiante, gagnant **absent du store** ⇒ **bug**, run invalide |
| P5c-id | `f(un-hot) = 1 % < f(c=1) = 4 %` | `f(un-hot) = 5 % ≥ f(c=1) = 3 %` ⇒ FAIL |
| P7 | dégradation 20 % | dégradation 100 % |
| P8 | série monotone sans genou | série à genou |
| **G : non-vacuité + `τ_promu` connexe** | `#{α=1} = 18`, `#{α=0} = 12` ; E3 admissible sur les déciles 1..7 ⇒ `τ_promu` = 7ᵉ ; `P1(λ*) = 3`, `P1(τ_promu) = 6` ⇒ promotion armée | **grille NON CONNEXE** : E3 admissible aux déciles 1-3, violé au 4ᵉ, ré-admissible aux 5-9 ⇒ **`τ_promu` = 3ᵉ, PAS le 9ᵉ** (cellule de E-D7) ; **`P1(λ*) = 0`, `P1(τ_promu) = 0` ⇒ NON ÉVALUABLE** ; **`#{α=0} = 0` ⇒ τ non évalué** ; **1ᵉʳ décile déjà violé ⇒ G NON ÉVALUABLE, pas de promotion** |
| **Multi-clé (a)(b)(c) + m** | (a) vraie ; (b) 5e-3 vs 0 ; (c) intra 1e-4 < inter 5e-3 ⇒ **ARMÉ** | (c) fausse ⇒ **NON ARMÉ** ; **(c) avec les deux `pct` à 0 ⇒ décision sur `m`, PAS de NON ÉVALUABLE** ; **nulle stratifiée de verdict qualitatif opposé ⇒ NON ÉVALUABLE** |
| **Ventilation 0/1 attribut partagé** | trois médianes ordonnées `intra < 1 partagé < 0 partagé` | `1 partagé ≈ intra` ⇒ rapporté comme **clé de gabarit de surface** (descriptif, pas un FAIL) |

Sortie : `experiments/results/gate-bench/report.json` — une ligne par clause (`pass_case`, `fail_case`, `expected`, `observed`, `ok`) + le compte **E**.

## 5. Contrôles et baselines

1. **Configuration courante** — défauts `EngramConfig()` (porte V-base), avant tout ajout.
2. **D7** — KV vidé avant chaque rappel ; datastore jamais alimenté pendant `logprob_continuation` (test CPU bloquant).
3. **M reset sur le même indice** — baseline `M = 0` **et** datastore vide ; c'est aussi la condition principale.
4. **Flag à off, même code** — `knn_lambda = 0.0` ⇒ logits bit-exacts (V-λ0), rejoué en fin de run.
5. **Contrôle qui élimine l'explication triviale** — **P3** (valeurs permutées, 21 tirages, même `sup_c`, même bras) et **P4** (store unité A / indice unité B). **Rejoués dans le bras gagnant** — contrôle de validité, **pas** une correction de multiplicité.
6. **Contrôles de design** — **V-indep**, **V-tok**, **V-para** par unité **et croisée**, **N13**.
7. **Traces égalisées** — une entrée de store par token du fait, même compte que `force_write=True` ; **tailles de store par unité publiées**.
8. **Clés jamais projetées par G/DG** (test CPU, D9).
9. **Ordre des conditions** fixé par la table d'unités et `POOL_PARAPHRASES` scellés, identique dans les deux bras.
10. **Bras F et L6 capturés dans la même passe forward** ⇒ aucun effet d'ordre entre bras, par construction.
11. **Nulle empirique triple** — globale, **stratifiée par bucket de position**, **ventilée 0/1 attribut partagé** ; divergence qualitative globale/stratifiée ⇒ NON ÉVALUABLE.
12. **Covariables pré-déclarées, toutes calculables avant la passe A** — **C1** Jaccard BPE(indice, fait \ secret) ; **C2** p₁₀ et NLL_base à la position de requête (cortex seul) ; **C3** type de paraphrase (facteur croisé à 3 niveaux, **vrai facteur intra-unité**, contrairement au verbe). **C3 n'entre dans aucune porte.** Plus : `corr(rang_mix, logp_base)` par unité.
13. **Re-collecte intégrale à neuf** ; les bruts archivés ne sont lus que par V-drift et l'usage méta borné du banc.

## 6. Critères d'abandon

**Ce qui tue H** : dans **les deux** bras, `P1 ≤ 5/30`, avec `n_faisable ≥ 18/30`. Verdict `REJETE`, formulé avec la portée §2 : *« ni l'état final ni l'état de la couche 6 du cortex gelé ne sont des clés invariantes à la paraphrase lexicale, dans un cadre attributif unique, à l'échelle du top-10 »* — **et non** « l'espace de sortie n'est pas le bon point d'attaque ».

**Table d'attribution (à remplir AVANT tout verdict, par bras)** — P1 échoue avec `R1 > 1`, `pct(para) ≫ pct(exact)`, `R1v > 1`, R3 faible ⇒ **Branche A, échec côté CLÉ**, V2-D non réfuté ; P1 échoue avec `R1 = R1v = 1`, R3 élevée ⇒ **Branche B, échec côté INJECTION** — *le seul résultat qui parle contre le candidat (b)*.

**Ce qui tue H_méthode** : `E ≥ 1` au banc.

**Ce qui impose une réduction de portée** : `E ≥ 3` ⇒ v3 est ramené au noyau {V-base, V-cap, V-drift, V-bord, V0, V-indep, V-tok, V-para, V-hash, V1a, V1b-1, V1b-2, V1c, V-var, V2, V-λ0, P1, P1-exact, ΔP6, P3, P4, P5, P5f-borne} ; **h, ventilation par verbe, F₁₀, P5c-id, P5f-cond, P7, P8, G, multi-clé, additivité, E1c reportés**.

**Ce qui invalide le run** (`INCONCLUSIF`, cause nommée) : V-base non reproduite ; V-cap > 1e-5 ; V-bord non bit-identique ; V0 R1 ≠ 1 ; V-indep (a)/(b)/(c) échouée ; **V-tok échouée** ; V-para (a)/(b)/(c) échouée ; V-hash échouée ; V1a > 1e-6 ; **V1b-1 violée une seule fois** ; V1b-2 violée sur son sous-ensemble ; V1c > 1e-6 ; V-var > 1e-12 ; P3 ≥ 12/30 ou identité violée ; majorité des échecs de P1 avec `R1v > 1` et sup au bord ⇒ `INCONCLUSIF — cellule dégénérée` ; bascule confiante dont le token gagnant est absent du store ; datastore vide / 0 écriture ; NaN ou inf ; fp16 dans la chaîne de distance ; DG appliqué aux clés ; `knn_lambda = 0` non bit-exact ; repli CPU silencieux ; **E3(λ*) > 0.05** ; **taux de bascule > `borne_marge`**.

**INCONCLUSIF sans être invalide** : `n_faisable ≤ 15/30` ⇒ `budget arithmétique` ; `n ∈ [6, 11]` ⇒ `zone grise` (0.1001 sous p = 0.5 ; 0.7641 sous p = 0.3, **irréductible**, §3) ; `n_disc < 5` ⇒ ΔP6 `non évaluable` ; divergence globale/stratifiée ⇒ multi-clé `non évaluable` ; aucun τ admissible ⇒ G `non évaluable`.

**Ce qui n'est PAS un critère d'abandon** : E3 > 0.05 aux λ élevés (prédit) ; F ou F₁₀ grands ; `corr(D, H) ≈ 0` (identité) ; compte non nul dans le complément de V1b-2 (attendu) ; h quelconque (descriptive) ; `n_L6 ≥ 25/30` (⇒ **audit de fuite**, pas un succès, N11).

**Interdit (D14 / D14-S)** : amender une clause après lecture du **premier token de donnée réelle**. Une porte échouée **arrête le run** et se journalise comme résultat (porte + dérivation + chiffre), jamais comme échec de cycle. Amender une clause **après le banc et avant le run** est au contraire obligatoire si le banc l'exige — et **le banc est alors rejoué en entier**.

## 7. Variables fixées

`seed = 0` ; `model = gpt2` ; `layer_index = 6` ; `lam = 2.0` ; `cap = 0.5` ; `eta = 0.2` ; `decay = 1e-3` ; `threshold = 4.0` ; `dg = 8192/64` ; `read_gate = "keysim"` ; `prune = 512/0.10` ; `knn_k = 8` ; `knn_temp_c` balayé sur la grille §4.2 ; `knn_gate_tau = 0.0` hors bras G ; `M = 0` en condition principale.

### Table d'unités (N = 30) — **additive, `eval/pool.py` non modifié dans ses constantes**

Faits : `pool.fact_pairs(30)` inchangé. Secrets : `SECRETS_80[i]` pour `i ≠ 5`, et **`secret(5) := SECRETS_80[30] = "walrus"`**.
**Règle de la substitution** (déterministe, à graver) : *le premier `SECRETS_80[j]`, `j ≥ 30`, qui passe V-tok*. Motif : `SECRETS_80[5] = "lighthouse"` **est** `ENTITIES[16]`, présent dans les quatre indices de l'unité 16 (vérifié par exécution le 2026-08-22). **`SECRETS_80` n'est pas modifié** : il gèle la courbe de capacité X9.

### `POOL_PARAPHRASES` — règles de génération (Neuro, tour 2), gelées et hashées avant toute mesure

Principe : *l'arbitraire doit être global (un choix appliqué 30 fois), jamais local (30 choix)* — sinon C3 mesure le talent du rédacteur.

- **para1 — substitution du verbe attributif** : `verb_index → (verb_index + 1) mod 5` sur `VERBS` gelée. Un seul nombre, et 1 est le moins arbitraire des nombres. **Sécurité prouvée** : deux unités partagent OWNER ssi `i ≡ j (mod 16)` et ENTITY ssi `i ≡ j (mod 20)`, donc les deux ssi `i ≡ j (mod 80)` — impossible pour n ≤ 30 ⇒ **le couple (owner, entity) est unique** ⇒ aucune para1 ne peut coïncider avec l'indice exact d'une autre unité. En revanche para1 réutilise un verbe présent verbatim dans 5 autres faits : **c'est la condition la plus dure pour la clause (c), et c'est voulu** — le régime de clés corrélées est celui que la séparation de patterns est censée traiter.
- **para2 — cadre + marqueur temporel** : préfixe unique gelé, appliqué identiquement —
  `"Years later, everyone still remembered that " + owner_lc + " " + entity + " " + verb`
  avec `owner_lc` = minusculisation mécanique du premier caractère (aucun nom propre dans `OWNERS`, la règle est totale). **Verbe d'origine conservé** : un seul facteur manipulé par type, sinon C3 ne veut rien dire.
- **para3 — changement de cadre syntaxique et de registre** : une seule transformation gelée —
  `"So what's the name of the " + entity + " that belongs to " + OWNER_OBJ[owner] + "? It's "`
  Elle change simultanément (a) le cadre (possessif prénominal → génitif postnominal + relative), (b) le type de phrase (déclarative attributive → interrogative + réponse), (c) le registre (contractions *what's / It's*). Seul intrant par unité : **`OWNER_OBJ`**, table gelée et hashée de **16 entrées** mappant chaque OWNER vers sa forme objet (« The captain's » → « the captain » ; « Her » → « her » ; « His » → « him » ; « Our » → « us » ; « Their » → « them » ; « My uncle's » → « my uncle » ; …). 16 décisions, toutes forcées par la grammaire anglaise.
  **Vertu méthodologique** : les 30 unités partagent le **même token immédiatement avant la cible** (« It's »). Le prior local du cortex à la position de mesure est donc **constant sur les 30 unités et identique entre les bras F et L6** — ce qui purge la comparaison primaire d'un effet n-gramme différentiel. C'est l'argument qui fait préférer cette forme à toute autre.

**Autres données** : `NEUTRAL_TEXT` de `eval/collateral.py` (343 positions) ; distracteur = **30 000** premiers tokens de `data/rfc9293.txt` (SHA-256 re-vérifié) ; `E1C_FACT`/`E1C_QUESTION` de `eval/read_gate.py` (descriptif). **`VARIED_PAIRS` n'est pas utilisé.**

**Statistique** : unité = le gabarit (N = 30) ; test des signes exact ; **une seule primaire par bras (P1)** ; IC 95 % **Clopper-Pearson** sur `n/30` ; bootstrap 10 000 tirages **« indicatif, N = 30 »** ; 30 valeurs in extenso pour h et F₁₀ ; permutations au plancher écrites « p < 10⁻³ » ; par position, ρ̂ jusqu'au lag 5, **L = 25 si ρ̂₁ ≤ 0** ; **P3 à B = 21**.

**Numérique** : distances **fp32** ; `p_LM` en log-probs fp32, **exponentiation fp64** ; rang par **comptage strict** ; softmax kNN après soustraction de d²_min ; mélange en **log-espace** ; **`bord := fl(−log1p(−λ))`, même expression et même code partout** (V-bord) ; `M` fp32 ; **table `k(n)` recalculée en entiers Python (pas de flottant)**.

## 8. Variable manipulée

**Une seule : l'étage de la clé** (`knn_key_layer` ∈ {`"final"`, `"inject"`}), à λ = λ* fixé. Le balayage λ × c est **hors-ligne et gratuit** (recomposition analytique) et ne constitue pas une seconde variable manipulée : la cellule décisionnelle est fixée d'avance à (λ*, `sup_c` sur grille déclarée). Bras descriptifs lus **après** verdict.

## 9. Budget

**Étape 1 — banc (bloquante)** : CPU seul. `.venv\Scripts\python eval\gate_bench.py` — ~33 clauses × 2 cas, **< 60 s d'exécution**.

**Étape 2 — v3 (après gate verte)** :

| Passe | Contenu | Durée estimée |
| --- | --- | --- |
| V-base | E1 + E3 aux défauts ; calcul et scellement de `borne_marge` (sur `V_k`) | ~3 min |
| V-cap / V-bord / V0 / V-drift / V-tok / V-para | smoke, capture, égalité de `bord`, contrôle croisé, intégrité des données | ~1 min |
| A | store fait-seul, M = 0 : **30 unités × 4 indices** ; états **final ET couche 6 dans la même passe** | ~3 min |
| D | E3 par position sur `NEUTRAL_TEXT` | ~1 min |
| E | store distracteur 30 k + P7 + ECDF globale, stratifiée **et ventilée 0/1** | ~5 min |
| descriptifs | additivité (M chargée) + E1c | ~2 min |
| — | re-run `knn_lambda = 0` bit-exact | ~1 min |
| **Total GPU** | **~15 min**, **pas d'escalade** | |

**Base et honnêteté de l'estimation** : la seule mesure disponible est **105 s pour la phase GPU complète à 10 unités** (run 2, invalide — à re-mesurer). **Règle pré-enregistrée** : si le GPU dépasse **30 min**, c'est une **anomalie à signaler**, pas une dérive à absorber. *(Le v2 budgétait 27-29 min pour 105 s réelles — facteur ~15. Le goulot n'est pas le calcul, c'est l'écriture des portes.)* **VRAM < 0.8 Go sur 6.**

Hors-ligne (CPU, gratuit) : grille λ × c, P1, P1-exact, ΔP6, h, F₁₀, P3, P4, P5c-id, P5f-cond, P8, frontière, courbe G, R1v, `pct`, `m`.

## 10. Livrables attendus

- **Décisions gravées par le PI AVANT le banc** : **D12, D13, D14, D14-S, D14-R, D15** — faites le 2026-08-22, `docs/ARCHITECTURE.md` §3.
- **Flags `EngramConfig`** : aucun nouveau champ. `knn_lambda = 0.0` (inerte), `knn_k`, `knn_temp_c`, `knn_key_layer`, `knn_gate_tau` existent déjà (`engram/config.py` l.65-74). **Aucune modification de `engram/` n'est demandée.**
- **Nouveau script** : `eval/gate_bench.py` (CPU) → `experiments/results/gate-bench/report.json` + compte **E**.
- **Script d'éval amendé** (pas réécrit) : `eval/knn_ceiling.py`.
- **Données, purement additives à `eval/pool.py`** : `POOL_PARAPHRASES`, `OWNER_OBJ`, et la table d'unités de v3 (substitution du secret 5). **`OWNERS`, `ENTITIES`, `VERBS`, `SECRETS_80`, `fact_pairs` intouchés** ; `eval/fact_injection.py` **non modifié**.
- **Tests CPU** (sans HF sauf tokenizer) : les 12 du v2 §10, plus (xiii) V-indep détecte des clés dupliquées sur 30 ; (xiv) V-para détecte un indice sous-chaîne **et** une fuite croisée, paire nommée ; (xv) V-bord détecte `−log(1−λ)` vs `−log1p(−λ)` ; (xvi) V1b-1 passe à `r = 1e-30` ; (xvii) V1b-2 range `r = 1e-30` dans le complément ; (xviii) G rend NON ÉVALUABLE si `P1(λ*) = 0` **et** si aucun τ n'est admissible ; **(xix) `τ_promu` est le plus grand τ du PRÉFIXE CONNEXE — testé sur une grille non connexe** ; **(xx) table `k(n)` exacte contre référence pour n = 5..30, avec n = 16, 23, 30 obligatoires** ; (xxi) (c) bascule sur `m` quand les `pct` saturent ; (xxii) repli Clopper-Pearson quand h = 0 partout ; (xxiii) `borne_marge = 0` ne produit pas de NaN ; **(xxiv) V-tok échoue sur le jeu réel AVANT correction (`lighthouse`) et passe après** ; **(xxv) `P5f-borne` passe sur une bascule portée par la valeur du voisin 5** ; **(xxvi) P3 refuse un B pair**.
- **Entrée de journal** : datée, protocole recopié, **compte E du banc**, table d'attribution `R1/R1v/R3` **par bras**, décomposition à trois facteurs, covariables C1/C2/C3 + tailles de store, `P5c-id`, `P5f-borne`/`P5f-cond`, ΔP6, G, multi-clé (3 nulles), bruts hashés, **les quatre interdictions de vocabulaire §2 recopiées mot pour mot**, **la note de design sur l'aliasing verbe/entité**, **le renversement d'asymétrie**, **contraintes (α)/(β) du futur (b)**, et la ligne d'honnêteté du §4.1.
- **`docs/EXTENSIONS.md` §4** : une ligne dans le tableau des ablations/instruments (proposée par le labo, appliquée par le PI).
- **Corrections documentaires du run 1** : appliquées le 2026-08-22 (erratum `d²_min`, façades « PLAFOND » et « familiarité »).

## 11. Questions pour lab-neuro — **RÉPONDUES (tour 2)**

Les six questions ont reçu réponse ; l'intégration est au §C de l'arbitrage. Verdict Neuro : **RÉSERVÉ → FAVORABLE** sous les points 1 et 2, qui sont *« des corrections de fichier, pas de design »* — **tous deux traités** (vérifiés par exécution, corrigés, gravés).

## 12. Questions pour lab-math — **RÉPONDUES (tour 2)**

Les huit questions ont reçu réponse ; l'intégration est au §B de l'arbitrage. Verdict Math : **FAVORABLE** sous les corrections 1 (`P5f-borne`) et 2 (`τ_promu`), toutes deux **bloquantes et intégrées**, et les scellements 3 (0.8998) et 4 (B = 21), intégrés.

## 13. Décisions déléguées au copilote (2026-08-22) — gate de pré-enregistrement

**Règle de départage appliquée à tous les arbitrages** : *à qualité de preuve égale, choisir la version qui compte le moins de clauses. Trois cycles sont morts de la complexité du système de portes, pas de la difficulté de la science.*

| # | Décision | Justification |
| --- | --- | --- |
| 1 | **30 unités d'emblée par bras, escalade SUPPRIMÉE** | +puissance (0.623 → 0.8998 à p = 0.5 ; grise 0.366 → 0.1001) et **−une mécanique procédurale**, pour un coût GPU de quelques minutes. |
| 2 | **`bord` := `fl(−log1p(−λ))`, même expression, même code** ; constantes 4/8 ULP conservées, **justification remplacée par l'argument structurel** | une constante juste avec une dérivation fausse est une façade. Promue en **porte V-bord**. |
| 3 | **Math A3 adopté, tolérance P5f confinée au descriptif** | aucune porte ne dépend d'un paramètre libre ; `P5f-borne` en fournit une à zéro paramètre. |
| 4 | **ΔP6 = signes conditionnel aux discordances, unilatéral, règle `k(n)` exacte** | 8/10 sur l'ensemble complet est une machine à faux négatifs. Règle exacte parce que la table de Math s'arrêtait à n = 10 alors que n_disc peut monter à 30 — **et `k(30)` s'est révélé être un cas frontière à la 3ᵉ décimale**. |
| 5 | **Marge continue `m(q,k)` ; (c) décidée sur `m` quand les `pct` saturent** | *un NON ÉVALUABLE déclenché par la résolution de l'instrument et non par la donnée est exactement l'issue qui a rendu le NON ARMÉ du run 2 non probant.* |
| 6 | **h RÉTROGRADÉE en descriptive ; P1 seule décisionnelle par bras** | deux primaires dont la non-contradiction repose sur une hypothèse que le protocole prédit fausse sont un risque net. **Critère d'ouverture de (b) : `P1 ≥ 12/30` dans au moins un bras nommé, sans clause « ou h ».** |
| 7 | **Chiffres corrigés** : `1/λ* = 20.504167`, `w₁(c=1) = 0.27971`, **`risque I = 1.528e-5`**, **`puissance = 0.8998`** | D14-R. Les deux derniers viennent du tour 2 : « ≈ 2e-5 » (Math tour 1) et « ≥ 0.90 » étaient faux. |
| 8 | **Quatre interdictions de vocabulaire gravées** (§2 i-iv) | les façades de vocabulaire survivent aux verdicts ; celle de Hasselmo a tenu trois entrées de journal. La (iv) est préventive : elle interdit « complétion de patterns » **avant** que L6 gagne. |
| 9 | **Covariables C1/C2/C3 ; le type de paraphrase n'entre dans aucune porte** ; **aucune prédiction signée sur le verbe** | C3 est un vrai facteur intra-unité ; le verbe est **aliasé sur l'entité** (vérifié) et donc inanalysable. |
| 10 | **Identité du token gagnant loggée sur les bascules confiantes** | sans elle l'antipode de P5f est ambigu : intrusion (informatif) vs bug (invalidant). |
| 11 | **Trois nulles : globale, stratifiée par position, ventilée 0/1 attribut partagé** | interdit de choisir la nulle après coup ; la ventilation diagnostique une clé de **gabarit de surface**, ce que (c) seule ne rendrait pas. |
| 12 | **Profil de profondeur (couches 3 et 9) REFUSÉ pour ce cycle** | deux bras portent déjà une multiplicité gérée ; quatre en porteraient une qui ne l'est pas. |
| 13 | **Conditions (α) et (β) de Neuro consignées comme contraintes du futur (b)** | (α) *le piège DG revient transposé* : (a) mesure des clés **non projetées**, (b) lira à travers `φ = G/top-k`. (β) (b) porte son propre E3 par position. |
| 14 | **`pool.fact_pairs(30)` comme jeu d'unités, `SECRETS_80` NON modifié, substitution du secret 5 par règle déterministe** | le jeu porte la propriété voulue à N = 30 ; modifier `SECRETS_80` casserait la reproductibilité de X9. La substitution est **additive et dérivée d'une règle**, pas d'un choix. |
| 15 | **Gate de pré-enregistrement** : banc **bloquant**, GPU **interdit** tant qu'il n'est pas à 100 % ; `E ≥ 3` autorise le banc à **amputer** v3 | c'est l'objet du cycle méthode. |
| **16** | **Corrections bloquantes du tour 2 intégrées sans négociation** : `P5f-borne` sur `V_k` avec inégalité large (E-D6) ; `τ_promu` = préfixe admissible **connexe** (E-D7) ; **B = 21** ; **porte V-tok** ; **V-para croisée** | deux clauses étaient **fausses telles qu'écrites**, dont une était **ma propre correction d'un amendement d'expert**. Aucune des deux n'ajoute de paramètre libre : la règle de simplicité n'a pas eu à arbitrer. |
| **17** | **Les deux affirmations de fichier de Neuro ont été VÉRIFIÉES par exécution avant intégration** — la première vraie (`lighthouse`), la seconde fausse (`catapult`/`cathedral` ne collisionnent pas) | D14-R s'applique aussi aux **avis d'experts** : un chiffre ou un fait cité dans un avis n'entre pas dans le protocole sans être re-mesuré. La porte V-tok est conservée malgré la suspicion infirmée, parce que `POOL_PARAPHRASES` introduira des mots neufs. |

## 15. Amendement post-banc (2026-08-22) — obligatoire, banc à rejouer en entier

Le banc D14-S a rendu **E = 2**. `H_méthode` est **REJETÉE** : les trois passes du §4bis ne
suffisent pas, le banc devient obligatoire à vie. `E < 3` ⇒ **pas** de réduction de portée.
D14 impose d'amender **après le banc et avant le premier token de donnée réelle**, puis de
**rejouer le banc en entier**. Aucun GPU n'a tourné : la frontière n'est pas franchie.

### A. Le défaut de fond — deux fuites que le banc a confondues

Distinction apportée par Neuro, et c'est le vrai résultat du cycle :

- **Fuite structurelle** — une règle de génération **lit la ligne d'une autre unité**.
  C'est para1 : `verb_index + 1 mod 5` va chercher le verbe de l'unité *i+1*, alors que
  `VERBS` recycle tous les 5 indices. Ce n'est pas un défaut lexical, c'est un défaut de
  **définition de l'unité** : para1(i) contient verbatim du matériel qui **identifie** six
  autres unités, donc **la vérité-terrain est détruite** — on ne peut plus dire de quoi un
  succès sous para1 est un succès. **Interdit sans condition, détectable sans mesure.**
- **Fuite de cadre** — un type partage du matériel avec les 29 autres paraphrases du même
  type, matériel qui **n'identifie aucune unité**. C'est para3, et c'est le **prix
  inévitable d'un vrai recadrage**.

Le banc les a mesurées avec le même Jaccard brut (70 + 80 violations). **C'est l'instrument
qui doit se scinder, pas la donnée qui doit s'affaiblir.**

**Erreur du §7 acquittée** : la phrase « la réutilisation d'un verbe présent verbatim dans 5
autres faits est la condition la plus dure pour la clause (c), et c'est voulu — le régime de
clés corrélées est celui que la séparation de patterns est censée traiter » est **fausse**.
Un régime de clés corrélées se teste avec des **clés distinctes qui se ressemblent**, pas en
logeant dans l'indice de *i* le matériel identifiant de *i+1*. Le second cas ne mesure pas
l'interférence, il **détruit la vérité-terrain**. Le régime corrélé appartient à un **bras
déclaré et séparé** (store distracteur, X9, X6), où il est une variable manipulée — jamais à
la définition d'une paraphrase, dont la seule fonction est de **préserver l'identité de
l'unité**.

**Règles gravées** : *l'arbitraire doit être global — **et jamais indexé par l'unité**.* Et :
*on n'affaiblit jamais le stimulus pour satisfaire une porte ; on corrige la porte, ou on
abandonne la mesure.*

### B. Amendements

| # | Clause | Amendement |
| --- | --- | --- |
| **A-1** | **§7, para1** | La rotation `+1 mod 5` est **supprimée**. Remplacée par un **verbe attributif unique, global aux 30 unités, hors `VERBS`** : **`"bears the codename"`**. Owner, entity et absence de cadre ajouté inchangés — le facteur manipulé reste la seule substitution du verbe. *« The captain's ship bears the codename walrus. »* **Propriété prouvée, pas espérée** : un token qui n'appartient à aucun fait entre au dénominateur de **toutes** les comparaisons `J(para1(i), fait_j)` et à **aucun** numérateur — il dilue uniformément et **ne peut jamais** faire monter *j* au-dessus de *i*. **Preuve d'existence dans le jeu** : para2 satisfait déjà cette contrainte (préfixe gelé absent des faits) et rend **0 violation** — elle n'est pas passée par chance. Repli déterministe si collision : `"translates as"`, puis `"carries the tag"`, puis `"denotes"` — premier candidat conforme, aucun choix à la main. |
| **A-2** | **§7, para3** | **INCHANGÉE.** L'exemption (« `V-para (c)` ne s'applique pas aux types à cadre lourd ») est **refusée** : c'est exactement le mode d'échec que **D14(c)** interdit — la porte mélangerait l'**identité** (l'indice désigne-t-il son unité ?) et le **recadrage** (le phénomène mesuré), et l'exemption soustrairait la partie gênante du mélange au lieu de le défaire. Un cadre plus léger est **refusé** : ce serait affaiblir le stimulus pour satisfaire l'instrument, et para3 est le seul type qui change simultanément cadre, type de phrase et registre — le chiffre-titre du PoC est un **ratio de généralisation**, le mesurer sur trois quasi-variantes le viderait de sens. Passer à 2 paraphrases est **refusé** : cela casse le chaînage 2/3 de P1 **et** ne laisserait que deux transformations du même type (substitution locale sous cadre déclaratif), supprimant le gradient de difficulté qui rend la primaire interprétable. |
| **A-3** | **`V-para (c)` → `V-para (c′)`** | Le Jaccard porte désormais sur le **contenu**, pas sur les tokens bruts. **Définition indépendante de la position** : `F_t := ⋂_{i=1..30} tokens_BPE(para_t(i))`, `C_t(i) := tokens_BPE(para_t(i)) \ F_t` ; symétriquement `F_fait := ⋂_i tokens_BPE(fait_i)`, `C(fait_i) := tokens_BPE(fait_i) \ F_fait`. **Clause** : pour tout *i*, tout *t*, `J(C_t(i), C(fait_i)) > J(C_t(i), C(fait_j))` pour tout `j ≠ i`, **strictement — une égalité compte comme violation** (si l'indice n'est pas strictement plus proche de son fait, il ne désigne pas son unité). **Pourquoi PAS « plus long préfixe/suffixe commun »** : le cadre de para3 est **entrelacé** (`So what's the name of the` + entity + `that belongs to` + owner_obj + `? It's`) — un préfixe/suffixe laisserait `" that belongs to "` dans le contenu, **recréant la fuite**. `F_t` attrape le cadre **où qu'il soit**, se calcule **sans regarder aucun résultat** (données gelées seules) et capture gratuitement le nouveau verbe global de A-1. **Gain D14-S** : le Jaccard est un rapport de petits entiers ⇒ comparaison en **arithmétique entière exacte** par produit croisé (`a·d > c·b` sur `int`), **jamais en flottant** — l'analyse ULP disparaît au lieu d'être à faire. |
| **A-4** | **`V-slot` (NOUVELLE, structurelle)** | Pour tout type *t* et toute unité *i* : l'ensemble des **valeurs de ligne** des tables indexées par l'unité (`OWNERS`, `ENTITIES`, `VERBS`, `SECRETS_80`, `OWNER_OBJ`) apparaissant **verbatim** dans `para_t(i)` est **inclus dans les slots de l'unité i**. Toute occurrence d'un slot d'une unité `j ≠ i` ⇒ **arrêt**. **Raison d'être** : `V-para (c′)` **neutralise** la fuite structurelle (elle la met hors contenu) mais **ne la détecte pas** — une règle future qui lirait la ligne d'une autre unité pourrait passer alors que la vérité-terrain serait détruite. Une fuite de **règle** se prend par une porte de **règle**. `V-slot` échoue sur l'ancienne para1 **de façon déterministe et sans aucune mesure** : c'est la porte qui aurait tué le défaut avant que 150 triples ne soient comptés. Elle est aussi la seule qui reste correcte si le tokenizer ou le modèle change. |
| **A-5** | **`OWNER_OBJ`** | Porte ajoutée : `OWNER_OBJ` **injective**, et ses 16 images **deux à deux distinctes en BPE**. Motif : la table écrase de l'information (« Her » → « her », « His » → « him », « Our » → « us ») ; deux owners tombant sur la même forme objet rendraient deux unités **indiscernables dans para3**. **Vérifié par exécution le 2026-08-22 : 16/16 distinctes en chaînes ET en BPE.** La porte est conservée parce qu'elle doit tenir si la table bouge. |
| **A-6** | **`V-bord`** | Deux corrections. (i) **Évaluée sur toute la grille λ**, pas au seul λ* : à λ* calculé les deux expressions sont **bit-identiques** (0 ULP) ⇒ la porte y est **vacuée par satisfaction** ; elle est discriminante ailleurs (λ = 0.02 → 5 ULP ; 0.05 → 6 ULP ; 0.10 → 2 ULP). (ii) **λ\* est défini comme l'EXPRESSION `1 − exp(−0.05)`, jamais comme un décimal recopié.** Défaut trouvé hors banc : le littéral `0.048770575499286` du protocole est à **2 ULP** de `1 − exp(−0.05)` = `0.048770575499285984` — c'est la classe de bug que `V-bord` garde, **un étage au-dessus**. Le décimal reste dans le document comme **documentation seule**, étiqueté comme tel (D14-R). |
| **A-7** | **Descriptifs ajoutés** | Le **Jaccard brut** (avec cadre) reste **publié**, par type, avec son compte de violations — on ne cache rien : 80 violations sur para3 se lisent *« mesure du poids du cadre »*, 0 sur para1/para2 *« absence de fuite d'identité »*. **Deux quantités, deux noms, une seule bloquante.** Plus **`V-partage`** en descriptif (plus long préfixe **et** plus long suffixe communs en BPE, par type, sur les 30) : elle mesure une **troisième** propriété — position et volume du matériel commun — que ni `V-para (c′)` ni le Jaccard brut ne rapportent. C'est elle qui voit le préfixe de ~8 tokens de para2, que le Jaccard croisé laisse passer à 0 violation. |
| **A-8** | **N10 re-signée** | L'ancienne N10 (`h(para1) ≥ h(para2) ≥ h(para3)`) était calibrée sur une para1 qui gardait un verbe partagé avec son propre fait. La nouvelle para1 **perd tout recouvrement de verbe** avec son fait. **Neuro re-signe : `h(para2) ≥ h(para1) ≥ h(para3)`.** Motif mécanique : para2 est le seul type conservant le verbe d'origine, donc le recouvrement lexical maximal avec le fait. **Motif empirique convergent, vérifié dans `docs/JOURNAL.md`** : para2 +1.006 ± 0.918, para1 +0.345 ± 0.347, para3 +0.168 ± 0.505 — *mais sur le jeu `QUESTIONS` de v1, qui n'est **PAS** `POOL_PARAPHRASES`* : c'est une **analogie orientante**, pas une dérivation ; c'est le motif mécanique qui porte la prédiction. **Re-signature datée AVANT le GPU**, comme l'exige le pré-enregistrement. |

### C. Cascade D14(b) — obligatoire, la re-dérivation fait partie de l'amendement

L'amendement change les **données gelées** ⇒ **nouveau SHA-256**, et **chaque** porte antérieure
se re-dérive sous les nouvelles données :

- **`V-tok`** : re-jouée — les tokens du nouveau verbe global contre les 30 premiers tokens BPE des secrets.
- **`V-hash`** : nouveau hash de `POOL_PARAPHRASES`, `OWNER_OBJ` et de la table d'unités.
- **`V-para (a)/(b)`** : re-jouées sur les nouveaux indices.
- **`V-indep`** : re-jouée — les clés de requête changent avec les indices.
- **Budget de la passe A** : la longueur des indices change ⇒ le compte de tokens du store change.
- **`N10`** : re-signée (A-8).
- **I2 hérite de l'amendement** : son corpus (a) **est** le jeu d'unités v3 ⇒ le hash scellé pour I2
  doit être le hash **amendé**. Deux des douze points de l'avis Neuro sur I2 (score-titre sans para3 ;
  ventilation même-verbe / verbe-différent de `s_inter`) deviennent **partiellement sans objet**, la
  fuite par bloc disparaissant. À traiter au traitement d'I2, **pas ici**.

### D. Prédiction signée avant re-run du banc

**Neuro signe : 0 violation sur les trois types sous `V-para (c′)`.** Vérification de plausibilité,
à **re-mesurer et non à croire** : après retrait de `F_t`, `C_para3(i) = {entity_i, owner_obj_i}`
contre `C(fait_i) ⊇ {owner_i, entity_i, verb_i}` — l'intersection avec son propre fait contient
owner **et** entity, celle avec tout autre fait au plus l'un des deux, et l'union est plus grande.
*(Preuve du partage : deux unités partagent OWNER ssi `i ≡ j (mod 16)`, ENTITY ssi `i ≡ j (mod 20)`,
les deux ssi `i ≡ j (mod 80)` — impossible pour n ≤ 30. Elle ne vaut que si A-5 tient.)*
**Si des violations subsistent, ce n'est plus un défaut de règle mais un recyclage de table, et il
faut le nommer.**

### E. Vérifications d'exécution faites AVANT d'écrire cet amendement (D14-R)

| Affirmation | Source | Verdict |
| --- | --- | --- |
| 150 violations, 23/30 unités, para1 = 70 / para2 = 0 / para3 = 80 | banc | **reproduit indépendamment**, chiffres identiques |
| `OWNER_OBJ` injective et distincte en BPE | Neuro (résidu 1) | **vérifié : 16/16 en chaînes et en BPE** |
| Journal v1 : para2 +1.006 / para1 +0.345 / para3 +0.168 | Neuro (A-8) | **les trois trouvés dans `docs/JOURNAL.md`**, valeurs exactes |
| `"bears the codename"` ne contient aucune **valeur de ligne** des 5 tables | Neuro (A-1) | **conforme.** *Note de méthode : un premier test à recouvrement de **mots** l'avait signalé pour l'article « the », partagé avec des `OWNERS` — ce test était **plus strict que la règle**, qui porte sur les **valeurs de ligne verbatim**. Et le point est doublement sans objet : le verbe étant identique sur les 30, **tous** ses tokens sont dans `F_para1`, donc hors contenu par construction.* |
| `Qwen/Qwen2.5-1.5B` est **base**, pas instruct | Neuro (avis I2) | **vérifié sur la carte de modèle** : « Training Stage: Pretraining », « We do not recommend using base language models for conversations ». *Le test local par présence d'un `chat_template` disait « instruct » — il est **trompeur**, Qwen livre le template avec les tokenizers base.* |

### F. Ce que l'amendement ne fait pas

Il ne touche **ni** aux prédictions décisionnelles (P1, ΔP6, P3, P4), **ni** aux seuils
(12/30, 5/30, `k(n)`, λ*), **ni** aux bras. Les §4.4 et §6 sont **inchangés**. Seules changent
les **données** et les **portes d'intégrité** — c'est-à-dire exactement ce que D14 autorise à
amender entre le banc et le run, et rien d'autre.


## 16. Amendement du jeu d'unités (2026-08-22) — cause résiduelle du banc

Second passage du banc : **E = 1**, `V-para (c′)`, **10 violations** (para1 = 0, para2 = 5,
para3 = 5), **toutes des égalités exactes**, jamais des renversements, sur les paires
`(1,21) (2,22) (3,23) (21,1) (22,2)`. La refonte de para1 (A-1) a rendu **0 violation** :
la fuite structurelle est réparée. Ce qui reste n'est **ni une règle ni une porte** — c'est
le **jeu d'unités**, donc la décision 14 de ce cycle.

### A. La cause, dérivée et vérifiée par exécution

Dans `pool.fact_pairs`, `entity = i mod 20` et `verb = i mod 5`. **5 divise 20**, donc le
couple (entity, verb) a une **période de 20** : à N = 30 il n'existe que **20 couples
distincts**, et **10 collisions sont inévitables**. Pour ces paires, l'owner est le seul
discriminateur — et il est **effacé en BPE** dans deux des trois types, para2 minusculisant
le premier caractère et para3 passant en `OWNER_OBJ` :

| unité | owner | BPE dans le fait | BPE dans l'indice | partage |
| --- | --- | --- | --- | --- |
| 1 | `Her` | `[2332]` | `[607]` | **aucun** |
| 2 | `His` | `[2399]` | `[465]` | **aucun** |
| 3 | `Grandma's` | `[338, 2611, 5675]` | `[338, 49890]` | `[338]` |

⇒ **`pool.fact_pairs(30)` ne peut pas porter 30 unités identifiables.**

**Ma preuve du §7 était vraie et insuffisante** : « deux unités partagent les deux slots ssi
`i ≡ j (mod 80)`, impossible pour n ≤ 30 » exclut le partage **owner+entity**, pas le partage
**entity+verb sous owner effacé**.

### B. Une correction incomplète, écartée avant d'être écrite

Ma première proposition — *30 triplets à couples (entity, verb) deux à deux distincts* — est
**insuffisante**, et il faut le dire avant de la remplacer. Elle répare para1 et para2, qui
contiennent le verbe. Mais **para3 ne contient aucun verbe** : `So what's the name of the
{entity} that belongs to {OWNER_OBJ}? It's` a pour contenu `{entity, owner_obj}`. Si l'owner
s'efface, il ne reste que l'entité — et il n'y a que **20 entités pour 30 unités**. La
collision reviendrait par la même porte, sur para3 seule.

### C. La condition de design, dérivée

Pour que **tout** indice désigne strictement son unité, sous **tout** type, il suffit que
chaque type conserve **au moins deux slots discriminants dont le couple est unique**. Comme
para3 n'a que `{entity, owner_obj}`, la condition mordante porte sur l'**owner** :

> **C-1** — `(owner, entity)` deux à deux distincts sur les 30 unités.
> *(Déjà acquis : partage des deux ssi `i ≡ j mod 80`, impossible pour n ≤ 30.)*
> **C-2** — `(entity, verb)` deux à deux distincts sur les 30 unités.
> *(Répare para1/para2 ; **non satisfait** par `fact_pairs(30)`, période 20.)*
> **C-3** — pour **chaque** owner retenu, sa forme para2 (minusculisée) **et** sa forme
> `OWNER_OBJ` partagent **au moins un token BPE** avec sa forme du fait.
> *(Répare para3 ; **non satisfait** par les owners pronominaux `Her`, `His`, `Our`, `Their`,
> dont la forme objet est un mot entièrement différent.)*

Sous C-1 ∧ C-3, le contenu de para3 rencontre son propre fait sur **deux** slots (entity **et**
un token d'owner) et tout autre fait sur **au plus un** — l'inégalité devient **stricte par
construction**. Sous C-2, para1 et para2 ont la même propriété via le verbe. **La discrimination
ne dépend alors plus de la casse, du tokenizer ni du modèle.**

### D. Règle de sélection — déterministe, pré-déclarée

Les 30 unités sont les **30 premiers triplets `(owner, entity, verb)`, dans l'ordre
d'énumération `for e in range(len(ENTITIES)): for o in range(len(OWNERS)): for v in
range(len(VERBS))`, qui satisfont C-1, C-2 et C-3** — C-3 étant évaluée **par exécution du
tokenizer**, pas par jugement. Premier conforme, aucun choix à la main. Si moins de 30 triplets
conformes existent, **arrêt** : ce serait un fait sur le pool, à consigner, pas à contourner.

**Additif** : `OWNERS`, `ENTITIES`, `VERBS`, `SECRETS_80`, `fact_pairs` **intouchés** — X9
reste reproductible. La table d'unités de v3 est une **nouvelle fonction** de `eval/pool.py`.
La substitution du secret de l'unité 5 (A-1 du §15) est **recalculée** sur la nouvelle table,
par la même règle (*premier `SECRETS_80[j]`, j ≥ 30, qui passe V-tok*), et peut donc changer
d'unité ou disparaître si la collision `lighthouse` ne se présente plus.

### E. Nouvelle porte `V-ident` (structurelle, sans mesure)

> Les 30 unités vérifient **C-1**, **C-2** et **C-3**. Toute violation ⇒ **arrêt**.

Comme `V-slot`, elle est **décidable sans aucune mesure** et elle aurait tué la décision 14
avant le premier passage du banc. Contre-exemples obligatoires : `fact_pairs(30)` lui-même
doit **échouer sur C-2** (période 20) et **sur C-3** (owners pronominaux) ; le nouveau jeu doit
**passer les trois**.

### F. Pourquoi ce n'est pas un contournement

**Règle d'arbitrage, journalisée le 2026-08-22** :

> *Tricher, c'est changer le **critère** pour que le design existant passe. Corriger, c'est
> changer le **design** pour qu'il satisfasse un **critère inchangé**.*

`V-para (c′)` n'est **pas** amendée — ni son seuil, ni sa définition, ni sa strictitude, ni
l'égalité comptée comme violation. Aucune donnée n'a été mesurée : ce qui change est un **plan
d'expérience**, pas un résultat.

**Options écartées, avec leur coût mesuré :**

| Option | Coût | Verdict |
| --- | --- | --- |
| **N = 20** (collisions impossibles par construction) | puissance **0.5881**, grise 0.4117 à p = 0.5 — *pire que le plan à 10 unités que ce cycle a servi à abandonner* (0.623). *(N = 24 : 0.7294 ; N = 30 : 0.8998.)* | honnête et **inutile** |
| **Porte insensible à la casse** | — | **la seule vraie triche.** **Le modèle lit du BPE lui aussi** : si `" Her"` et `" her"` sont deux tokens sans rapport, l'ambiguïté entre *i* et *i+20* est **réelle du point de vue du cortex**, pas un artefact de l'instrument. Une porte aveugle à la casse déclarerait distinguable ce que le modèle ne distingue peut-être pas |
| **Retoucher para2/para3** | — | affaiblir le stimulus pour satisfaire une porte — refusé par la règle gravée au §15 ; et inopérant, les owners pronominaux de para3 étant contraints par la grammaire |

### G. Cascade D14(b) — obligatoire

Le jeu d'unités change ⇒ **nouveau SHA-256**, et re-dérivation de : `V-tok` (et la substitution
du secret 5), `V-hash`, `V-para (a)/(b)/(c′)`, `V-slot`, `V-indep`, `V-suffixe`, budget de la
passe A. **N10** est **inchangée** (`h(para2) ≥ h(para1) ≥ h(para3)`) : son motif mécanique —
para2 est le seul type conservant le verbe d'origine — ne dépend pas du jeu d'unités.
**I2 hérite** : son corpus (a) **est** le jeu d'unités v3.

### H. Prédiction avant re-run — la mienne, cette fois, et signée

**0 violation sur les trois types sous `V-para (c′)`**, par la dérivation de C. Contrairement à
la prédiction précédente, celle-ci ne repose pas sur une inspection des surfaces mais sur une
**condition de design vérifiée par une porte** (`V-ident`) avant la mesure. **Si des violations
subsistent, la porte ne sera PAS amendée** — ce sera un fait sur le pool, à consigner comme tel.


### I. Résultat du re-run, et ce que la gate ne voit pas — À DÉCLARER AVANT PRÉ-ENREGISTREMENT

**Banc rejoué en entier : 39 clauses, 115 cas, couverture 100 %, `E = 0`.** La prédiction du
§16 H est **tenue** : `V-para (c′)` rend **0 violation sur les trois types** (para1 0, para2 0,
para3 0 ; le passage précédent donnait 0 / 5 / 5), **porte inchangée**. `V-ident` **échoue**
sur `pool.fact_pairs(30)` — C-2 sur les 10 paires exactement prévues `(0,20) … (9,29)`, C-3 sur
les owners non conformes — et **passe** sur le nouveau jeu. Nouveau SHA-256 :
`7380b285f89c06ca…`. 92 tests, exit 0. Sélection : **450 triplets examinés** pour 30 conformes
(rejets C-1 = 60, C-2 = 252, C-3 = 108).

**Correction de mon §16 C, par l'exécution** : j'y écrivais que C-3 excluait « les owners
pronominaux `Her`, `His`, `Our`, `Their ` ». Le tokenizer en donne **cinq** : `Grandma's` est
aussi non conforme — elle partage `[338]` (`'s`) côté para2 mais **rien** côté `OWNER_OBJ`
(`"Grandma's"` → `[338, 2611, 23581]` vs `" grandma"` → `[49890]`). Onze owners sur seize sont
conformes. *C'est précisément pourquoi C-3 est évaluée par exécution et non par jugement : mon
jugement était faux d'une entrée.*

**Deux coûts que le banc ne peut pas voir, parce que ce ne sont pas des défauts de clause mais
des propriétés du plan** — tous deux **mesurés**, et à recopier au journal :

| Propriété | `fact_pairs(30)` | Nouveau jeu | Conséquence |
| --- | --- | --- | --- |
| **Diversité lexicale** | 16 owners, 20 entités | **5 owners, 6 entités** (6 et 5 occurrences chacun) | Les unités se ressemblent **beaucoup plus** entre elles. `V-indep (d)` rapportera un cosinus inter-unités maximal nettement supérieur. Effet sur la clause multi-clé (c) : **conservateur** (elle devient plus dure à satisfaire) — donc acceptable, mais **à publier**, jamais à découvrir après. |
| **Aliasing du verbe** | `verbe = f(entité)` (vérifié) | **`verbe = f(owner)`** (vérifié) | **L'aliasing n'est pas supprimé, il est DÉPLACÉ.** C'est la même classe de défaut que la façade (iv) de Neuro, sur un autre slot. |

**Conséquences opératoires, pré-enregistrées :**

1. **Aucune prédiction signée sur le verbe NI sur l'owner.** Les deux sont aliasés l'un sur
   l'autre : tout écart entre niveaux de verbe est indiscernable d'un effet d'owner, et
   réciproquement. La ventilation reste **descriptive** et sa légende obligatoire devient
   **« verbe et owner sont aliasés (`verbe = f(owner)`) ; aucun écart n'est attribuable à l'un
   plutôt qu'à l'autre »** — elle remplace la mention « aliasé sur l'entité » du §2, qui
   décrivait `fact_pairs` et n'est plus vraie du jeu retenu.
2. **`C3` (type de paraphrase) reste le seul vrai facteur intra-unité**, donc le seul
   analysable — inchangé.
3. **La diversité réduite est déclarée comme limite**, pas comme neutralité : ce run mesure
   l'invariance à la paraphrase lexicale dans un cadre attributif unique **et sur un
   vocabulaire de 5 owners et 6 entités**. L'interdiction de vocabulaire (i) du §2 s'applique
   *a fortiori*.

**Ce que ces deux coûts ne changent pas** : ni P1, ni ΔP6, ni P3, ni P4, ni aucun seuil. Ils
n'entrent dans **aucune** porte. Ils sont déclarés ici parce qu'un pré-enregistrement doit
porter ce que le plan a perdu, pas seulement ce qu'il a gagné.

**Fait de méthode à consigner** : pour la deuxième fois consécutive, la porte qui trouve le
défaut est celle qui **n'a besoin d'aucune donnée** — `V-slot` a tué l'ancienne para1, `V-ident`
a tué `fact_pairs(30)`. Les deux sont décidables par lecture des règles et des tables. C'est
l'enseignement le plus réutilisable du cycle méthode.


## Historique

- 2026-08-22 : **TERMINE — verdict INCONCLUSIF** (confondant d'instrument : `knn_k ≥ |store|`,
  pas d'étape d'adressage). Arrêt formel conforme par P3[L6], médiane 4/30.

- 2026-08-22 : **PRE-ENREGISTRE par le PI** (gate franchie apres banc E = 0).
  A partir d'ici, les sections Predictions chiffrees (4.1-4.6) et Criteres d'abandon (6)
  ne sont plus jamais modifiees, par personne.

- 2026-08-22 : brouillon du Directeur (v3, option B)
- 2026-08-22 : **tour 1** — avis Math (RÉSERVÉ, 3 amendements, 5 défauts) et Neuro (FAVORABLE, 2 amendements, 3 façades) ; chacun trouve une erreur de dérivation du Directeur (E-D3, E-D4, E-D5)
- 2026-08-22 : arbitrages délégués au copilote ; §4bis ré-exécuté en trois passes — un défaut trouvé dans un amendement d'expert (`τ_promu`, correction elle-même erronée)
- 2026-08-22 : D12 / D13 / D14 / D14-S / D14-R / D15 gravées dans `docs/ARCHITECTURE.md` §3 ; correctifs documentaires du run 1 appliqués
- 2026-08-22 : **tour 2** sur le consolidé — Math **FAVORABLE** sous 2 corrections bloquantes (E-D6 `P5f-borne`, E-D7 `τ_promu`) + 2 scellements ; Neuro **RÉSERVÉ → FAVORABLE** sous 2 corrections de fichier (collision `lighthouse`, porte V-tok) + façade (iv) aliasing et façade (v) complétion de patterns
- 2026-08-22 : **vérification par exécution** des deux affirmations de fichier — collision `lighthouse` **confirmée** et corrigée par règle déterministe ; suspicion `catapult`/`cathedral` **infirmée** (30 premiers tokens BPE distincts) ; aliasing verbe/entité **confirmé**
- 2026-08-22 : **banc rejoué après §15 — E = 1** (`V-para (c′)`, 10 égalités exactes sur
  les paires `(i, i+20)`). para1 refondue rend **0 violation** : la fuite structurelle est
  réparée. Cause résiduelle = **le jeu d'unités** (décision 14), pas une règle ni une porte.
- 2026-08-22 : **banc rejoué après §16 — `E = 0`, GATE FRANCHIE** (39 clauses, 115 cas,
  couverture 100 %). `V-para (c′)` rend **0 violation sur les trois types**, porte
  inchangée. `V-ident` échoue sur `fact_pairs(30)` (C-2 sur les 10 paires prévues) et
  passe sur le nouveau jeu. Deux coûts déclarés au §16 I : diversité lexicale réduite
  (5 owners / 6 entités) et **aliasing du verbe déplacé** de l'entité vers l'owner.
- 2026-08-22 : **amendement §16** — jeu d'unités refondu sous trois conditions de design
  dérivées (C-1/C-2/C-3), porte structurelle `V-ident`, `V-para (c′)` **inchangée**.
  Ma première correction (couples (entity, verb) distincts) était **insuffisante** :
  para3 ne contient aucun verbe. Banc à rejouer en entier. Aucun GPU n'a tourné.
- 2026-08-22 : **banc D14-S exécuté — E = 2, `H_méthode` REJETÉE** (34 clauses, 101 cas,
  couverture 100 %) : `V-para (c)` insatisfiable sur les données gelées (150 violations,
  23/30 unités) et `V-bord` vacuée par satisfaction à λ*. Défaut annexe trouvé hors banc :
  le littéral λ* du document est à 2 ULP de `1 − exp(−0.05)`.
- 2026-08-22 : **amendement post-banc (§15)** — para1 refondue (verbe global hors tables),
  para3 inchangée, `V-para (c′)` sur contenu défini par intersection des 30, porte
  structurelle `V-slot` ajoutée, `V-bord` sur toute la grille λ, N10 re-signée. Cascade
  D14(b) faite. **Banc à rejouer EN ENTIER.** Aucun GPU n'a tourné.
- 2026-08-22 : **protocole consolidé, corrections des deux tours intégrées — PROPOSE**. Étape suivante : livraison du banc `eval/gate_bench.py` par le Builder, puis gate de pré-enregistrement (**E = 0 exigé**), puis GPU.
