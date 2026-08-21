# EXP — V2-D(a) v2 : kNN-LM nu, borne de l'étage des logits

Statut : TERMINE — INVALIDE

> **ERRATUM (2026-08-22, après run)** — deux défauts de ce protocole, consignés sans
> retoucher les sections scellées : (1) **l'acquis de régime §3 « d²_min = 0.0 bit-à-bit
> 10/10 » (run 1) est FAUX** — recalcul depuis les bruts du run 1 : **0.00449220464**,
> argmin 2, pour les 10 secrets ; la justification §4bis-4 de la courbe G (« règle quantile
> dégénérée car d²_min = 0.0 ») reposait dessus, et la parenthèse de la porte V0
> (« R1 = 1 (d²_min = 0.0) ») est fausse — la porte elle-même (R1 = 1) reste valide.
> (2) **La porte V1b (« ΔNLL_t < −log(1−λ), 100 %, strict ») est insatisfiable en fp64**
> dès que le décrément réel est inférieur à un ULP du bord (r ≲ 1e-16) : elle est
> logiquement vraie et machinalement fausse. Ces deux défauts fondent la décision
> candidate **D14-ext** (satisfiabilité machine des portes ; provenance des chiffres cités).

**Origine** : `docs/EXTENSIONS.md` §2 (V2-D, candidat a) ; « Suite » du journal 2026-08-21
(V2-D(a) invalidé). Décisions candidates D12/D13/D14 appliquées ; D15 candidate créée ici.
**Antécédent** : `experiments/EXP-2026-08-21-knn-borne-logits.md` (v1, TERMINE — INVALIDE).
**Avis** : Math (RÉSERVÉ, 3 amendements) et Neuro (FAVORABLE, 2 amendements) — **chacun a
trouvé une erreur de dérivation dans le brouillon du Directeur**.

---

## 0. Pourquoi cette expérience maintenant

1. **La feuille de route la désigne nommément** : `docs/EXTENSIONS.md` V2-D — « LE chantier v2
   prioritaire », « ordre de la course : (a) kNN-LM nu **d'abord** ». Trois documents
   indépendants convergent (fiche V2-D, « Suite » de Q-01b, « Suite » du run invalidé).
2. **La question décisionnelle n'a jamais été mesurée** : le run 1 s'est arrêté à ses portes.
   Or **P1 seule** ouvre ou ferme le candidat (b) M_out — tant qu'elle n'est pas mesurée,
   (b) et (c) sont bloqués.
3. **Le coût marginal est au plancher** : script, tests, champs de config et capture
   pré-`lm_head` sont livrés et validés. Il reste ~28 min de GPU et une ré-écriture de portes.

**Contre-argument examiné et écarté** : la file d'audit propose Q-03 (E3 keysim avec σ) dans
l'ordre « fiabiliser d'abord ». Écartée pour ce tour — elle change une *formulation* sur un
mécanisme déjà RETENU et ne débloque aucune décision. Aucune question de la file ne lève un
verrou de route ; V2-D(a) si.

---

## Arbitrage

### A. Les deux erreurs du Directeur (acquittées)

| # | Erreur | Traitement |
| --- | --- | --- |
| **E-D1** (Neuro F1) | « La limite un-hot est la borne supérieure exacte de la famille de températures » — **faux**. `p_kNN(cible ; c) → 1` quand c → 0 **seulement si valeur(argmin) = cible** ; sinon → **0**, et un-hot est le **minimum**. Le store ayant une entrée par token du fait, plusieurs entrées peuvent porter la cible : c'est le régime attendu sous paraphrase. La cellule aurait produit un **REJETE non attribuable** — le défaut exact qu'elle prétendait supprimer. | Cellule décisionnelle = **`sup_c`** sur la grille déclarée ; **R1v** loggé ; partition {R1v = 1}/{R1v > 1} en table d'attribution ; clause `INCONCLUSIF — cellule dégénérée` |
| **E-D2** (Math D-6) | Prédiction P2 « F ∈ [5, 15] » : incohérente avec sa propre fourchette (p_P ∈ [0.20, 0.42] ⇒ F ∈ [3.42, 6.06]) et **partiellement impossible** (p_P − p_M ≤ 1 ⇒ λ_renv ≤ 0.5 ⇒ **F ≤ 10.252 toujours**). | P2 **sans fourchette pré-enregistrée** ; seule subsiste la borne dure F ≤ 10.252 ; ancre de régime F ≈ 6.77 (run 1) |

Les deux sont du même genre que celle qui a tué le run 1 : **un seuil écrit sans que sa
dérivation soit poussée jusqu'au bout.** D14 fonctionne.

### B. Les quatre arbitrages du Directeur

**B-1 — Amendement 1 de Neuro : intégré, en forme renforcée.** La cellule décisionnelle est
**`sup_c` sur toute la grille, uniformément** — le véritable supremum de la famille. Trois
raisons : (i) une statistique unique évite un verdict à deux régimes ; (ii) `sup_c` est une
fonction déclarée d'une grille pré-enregistrée, pas un choix post-hoc (le c atteignant le sup
est loggé) ; (iii) la multiplicité sur 6 valeurs de c est **absorbée par le contrôle P3, évalué
au même `sup_c`** (le contrôle porte la même inflation que le test). La partition
{R1v = 1}/{R1v > 1} est conservée intégralement comme table d'attribution et clause
d'invalidation. Coût GPU : nul.

**B-2 — Conflit P2 : tranché par un troisième terme.** Je ne départage pas les deux fourchettes,
**je supprime la fourchette**. Dérivation : au un-hot avec récupération correcte, λ_renv ne
dépend que de `p_LM` (Neuro N4) — c'est une mesure du prior de GPT-2, pas du canal ; et p_P est
**directement lisible dans le log**. Une « prédiction » sur une constante du modèle gelé,
immédiatement consultable, n'est pas falsifiable au sens de H : c'est une consultation. Que deux
experts compétents dérivent [3.4, 6.1] et [6, 12] à partir de *devinettes* sur p_P démontre que
l'intervalle ne prédit rien de l'hypothèse.

**B-3 — F₁₀ dégradée de co-primaire à diagnostic ; le co-primaire devient h.** F₁₀ vaut
20.504·p₁₀/(1+p₁₀) sur les succès et **+∞** sur les échecs (médiane = vote majoritaire déguisé),
et **conditionnellement à un succès elle est constante entre bras** (p₁₀ est une propriété du
cortex à la position — mêmes prompts, stores différents). Remplacée par le **taux de récupération
par paraphrase h**, dont les seuils se dérivent exactement de ceux de P1 par la règle de chaînage
de Math `q = r²(3−2r)` : n ≥ 5/10 ⟺ q ≥ 0.5 ⟺ **r ≥ 0.50** ; n ≤ 1/10 ⟺ q ≈ 0.104 ⟺
**r ≤ 0.20**.

**B-4 — Escalade : seuils re-dérivés, autorisation obtenue AVANT lancement.** 15/30 est mauvais
(P(n ≥ 15 | p = 0.5) = 0.572). Seuils re-dérivés : **12/30 et 5/30**. L'avertissement procédural
de Neuro est intégré : l'escalade est **autorisée par le PI avant lancement** (§13.4) — aucune
décision en cours de cycle, c'est ce qui a tué le run 1.

### C. Avis Math — intégration

| # | Remarque | Statut |
| --- | --- | --- |
| D-1..D-5 vérifiées (identité ; λ\* = 0.0487706 ; λ₁₀/F₁₀ ; w₁ = 1/(1+7e^(−1/c)) ; ρ = 0.928-0.938) | **Actées**, reprises en §3 |
| **D-6** P2 fausse, borne 10.252 | **Intégrée** (B-2) |
| **⇔ → ⇐** : « E3 ≤ 0.05 ⟺ λ ≤ λ\* » n'est vrai que dans le sens **⇐** | **Corrigée partout** : le théorème donne une condition suffisante universelle ; la réciproque est empiriquement fausse (le soulagement fait passer E3 sous 0.05 pour λ > λ\*) |
| **Q1** λ\* conservateur : oui — ρ ne se transporte pas au distracteur 30 k ; λ_emp utiliserait une donnée d'un run invalidé **dans une porte** (proscrit par D14) ; enjeu 7 % | **Intégrée** |
| **Q3** P(grise \| p=0.3) = 0.700 **irréductible** ; escalade **12/30 et 5/30** | **Intégrée** (B-4) |
| **Q4** F₁₀ en log ; IC bootstrap n=10 couverture réelle 88-92 % ⇒ « indicatif, N=10 » ; publier les 10 valeurs | **Intégrée**, appliquée à h **et** F₁₀ |
| **Q5** pas de confondant mais une **factorisation exacte** ; **ITT primaire** ; conditionnelle = diagnostic mécaniste ; publier les facteurs et la table de covariables | **Intégrée avec réserve** : sous `sup_c` la factorisation devient à **trois** facteurs ; l'identité à deux facteurs reste exacte dans la sous-cellule un-hot, rapportée comme telle |
| **Q6** acter que D11 ne se **mesure** pas sur ce canal ; porte sur **var(D_t) = 0 à 1e-12**, pas sur la corrélation (NaN) | **Intégrée** — ma porte de corrélation était encore un test déguisé |
| **Q2** recomposition exacte sous 4 conditions ; **ex-æquo** = un-hot ≡ uniforme sur l'argmin-set, compte loggé ; **P3 : la condition exacte est `valeur(argmin) ≠ cible`**, plus large que la mienne | **Intégrée** ; ma partition P3 conservée et **écrite comme sous-ensemble conservateur valide** |
| Prédictions : E3(λ\*) = 0.0464-0.0469 ; n_faisable ≥ 7/10 ; F₁₀ ∈ [0.1, 0.6] < 1 | **Intégrées** — « le verdict se jouera sur la récupération, pas sur le budget » |

### D. Avis Neuro — intégration

| # | Remarque | Statut |
| --- | --- | --- |
| **F1** un-hot n'est pas le maximum | **Intégrée** (E-D1 / B-1) |
| **Amendement 1** R1v, partition, clause d'invalidation | **Intégré, renforcé** |
| **Amendement 2 (a)** T2 testé par **identité** (partition exacte), pas par proxy « déciles de H » | **Intégré** — argument décisif : *tester par corrélation là où on dispose d'une identité, c'est le glissement exact qui a produit la porte V1 du run 1* |
| **Amendement 2 (b)** D11 pour un canal de sortie = **taux de bascule d'argmax**, ventilé par décile de NLL_base | **Intégré avec réserve** : le critère est **dérivable** (bascule ⟺ p_max − p_LM(v_nn) < λ/(1−λ)) ⇒ « concentrées aux marges fines » est quasi analytique ; seuls le **taux** et l'**antipode** sont informatifs |
| **Q1** R2 loggée **dans les deux conditions** et **normalisée** (anisotropie : Ethayarajh 2019, Timkey & van Schijndel 2021 — un cosinus brut n'a pas de seuil) | **Intégrée** : R2 devient **z = (cos − μ_null)/σ_null**, nulle empirique du distracteur 30 k |
| **Q1 (i)** un-hot est **structurellement aveugle à la force**, donc incapable d'innocuité sélective ; **G est le seul endroit du run où une pondération par la force intervient** | **Intégrée** : G promu de bras de design à **bras descriptif obligatoire**, avec déclencheur de promotion écrit |
| **Q1 (ii)** un-hot = WTA **imposé**, non émergent : un mauvais gagnant est amplifié à λ plein | **Intégrée** en §2 (portée) |
| **Q2** version mono-métrique **retirée** : le rang est saturé/bimodal, F₁₀ **constante par construction** entre bras. **E1-exact < E1b(fait-seul) < E1b(distracteur) se mesure sur h ; E1c sur F** | **Intégrée** — renforce B-3 |
| **N2** (principale) l'état final est le **pire étage** pour l'invariance à la paraphrase ; h(exact) = 1.00, **h(fait-seul) ∈ [0.20, 0.50]**, chute distracteur ≥ 0.10 ; **n ∈ [1,3], mode 2 ⇒ zone grise probable** ; **P6 devient la prédiction la plus informative** | **Intégrée** : P6 promue de descriptif à **prédiction ordinale signée** (couche 6 strictement meilleure sous paraphrase sur ≥ 6/10) |
| **N1, N3, N5, N6, N7** | **Intégrées**. N5 reprise avec sa clause d'auto-neutralisation (leçon Q-01b). **Micro-correction du Directeur** : la borne haute 0.0500 de N3 est le supremum théorique, atteint seulement si f_relief = 0 — que Neuro exclut lui-même ⇒ bande **[0.046, 0.050)**, stricte |
| **Q4 / F2** l'innocuité de (a) appartient à la **convexité du mélange**, pas à l'étage ; (b) M_out est un **biais additif non convexe et non borné** (peut envoyer p(y_t) → 0) | **INTERDICTION GRAVÉE avant mesure** (§2), **D15 candidate** (§13.6) |
| **Q4 (1)** **renversement d'asymétrie** : canal d'état = gain borné / dommage non borné (D12) ; mélange convexe = **dommage borné / gain non borné** | **Intégrée** — candidat au « résultat conceptuel du run » |
| **F3** « profil inverse » = **artefact de normalisation**, pas un profil ; un canal à dommage absolu constant est **NON SÉLECTIF** | **Intégrée, ma formulation retirée** (§4bis-6) |
| **Q5** déclencheur multi-clé, trois clauses en unités z, **(c) fausse ⇒ NE PAS ARMER** | **Intégré mot pour mot** (§6) |
| Note factuelle : correctifs de façade V2-D **non appliqués** ; le v2 ne doit pas ré-hériter du vocabulaire de la fiche | **Intégrée** : ce protocole n'emploie ni « plafond » ni « familiarité » — **« borne de l'étage des logits »** |

### E. Écarté

Cellule un-hot **seule** (fausse, E-D1) · fourchettes P2 des **deux** experts (B-2) · F₁₀
co-primaire (bimodale, constante entre bras) · seuil d'escalade 15/30 · porte `var > seuil` puis
corrélation (NaN) · P5c par déciles comme **test** de T2 (proxy d'une identité disponible) ·
« un canal aux logits est moins adversarial » (F2) · « profil inverse » (F3) · vocabulaire
« plafond » / « familiarité ».

---

## 1. Question

Un datastore brut de paires (état final pré-`lm_head` → token suivant), mélangé à la distribution
de sortie du cortex gelé, hisse-t-il le **rang** du premier token du secret en top-10 sous
**indice paraphrasé**, à un λ conforme au budget E3 ≤ +0.05 nats/token — et, à défaut, l'échec
est-il imputable à la **clé** ou à l'**injection** ?

## 2. Hypothèse

**H** — l'état final du cortex gelé est une clé **suffisamment invariante à la paraphrase** pour
que le mélange aux logits, pris **au supremum de sa famille de températures** et à
λ\* = 0.0487706, place le premier token BPE du secret en top-10 sur au moins 2 des 3 paraphrases,
pour au moins 5 secrets sur 10 ; et ce gain disparaît sous permutation des valeurs.

**H₀** — même avec la réponse littéralement présente dans le datastore et l'adressage au supremum
de la famille, le mélange ne hisse rien au-delà du match d'état quasi identique.

**Portée, à écrire au journal quoi qu'il arrive** : ce run borne **ce qu'un mélange à l'étage des
logits peut faire avec des clés égales aux états finaux du cortex gelé**. La borne est
conditionnelle à ce point d'injection et à cette forme de clé, et **ne se retourne pas** : un
échec ne dit pas « la localisation est le problème » ; un système à réinstallation (X5 / Fast-KV)
peut en principe la dépasser. Un succès ne démontre aucun mécanisme — le datastore contient la
réponse. Note de Neuro : le mélange à c → 0 est un **WTA imposé, non émergent** — un mauvais
gagnant n'est jamais corrigé, il est amplifié à λ plein.

**INTERDICTION GRAVÉE AVANT MESURE (Neuro F2 ; D15 candidate)** : la borne
`ΔNLL_t ≤ −log(1−λ)` appartient à la **convexité et à la stochasticité du mélange**, **pas à
l'étage des logits**. Le candidat (b) M_out est spécifié en **biais additif**
(`logits += g·W_U·(M_out·φ(h))`), **non convexe et non borné**, qui peut envoyer p(y_t) → 0.
**L'innocuité constatée sur (a) NE SE TRANSPORTE PAS à (b)** ; elle ne se transporterait qu'à un
M_out lu en *interpolation de distributions*. Toute phrase du journal ou du tableau §4 qui
transporterait E3(a) vers (b) est fautive.

**Conformité** : D8 (aucun gradient, aucun paramètre optimisé) ; D9 (clés kNN **non** projetées
par G/DG — test CPU bloquant : DG orthogonalise les entrées voisines, donc détruit exactement
l'invariance que P1 mesure) ; D7 (datastore jamais alimenté pendant `logprob_continuation`).

## 3. Ce que le projet sait déjà

| Fait | Chiffre | Source |
| --- | --- | --- |
| Pas de composante directionnelle du canal d'état | cos(r, W_U[cible]) = **−0.01** vs 0.136 aléatoire ; ΔH = +0.141 | X7, 2026-08-21 |
| Verrou top-10 | **0/10 sur tous les runs depuis X0** | X0→X8 |
| Référence courante (défauts X8) | E1 **+1.353 ± 1.58**, 0/10, E3 **−0.014** ✓ | X8 validation |
| Généralisation paraphrase (Δlogp, **jamais rang**) | ratio 0.68 [0.56, 0.99] ; **0.38 sous gate** | E1b ; COR-02 |
| **Anomalie A1** : l'ancre E1c n'est pas au point courant | −0.374/−0.378 = point X1b cap 0.25 (reproduit −0.3295) ; **au point courant cap 0.5 : −1.8896** | X8 banc ; run 1 |
| Pas un problème de capacité | 80 faits, 91 % de rétention | X9 |
| D11 : tout canal se juge par position | corr(dommage, entropie) +0.394 ; ciblage générique R = 0.83/0.78 | P5 ; Q-01 |
| D12 candidate : ΔNLL borne le **gain**, pas le dommage | +0.32 → +0.004/+0.05 après appariement | Q-01b |
| Anti-pseudo-réplication : unité = le secret | graines quasi-doublons, P4 = 0.988 | Q-01b §7(i) |

**Acquis du run 1 invalidé — CONNAISSANCE DE RÉGIME, jamais donnée** (aucun n'entre dans une
porte, D14) : identité vérifiée à **1.3e-15** sur **94.46 %** des positions, déficit porté par les
**5.54 %** de tokens-valeurs ; **d²_min = 0.0 bit-à-bit 10/10**, R1 = 1 sur 10/10, 2ᵉ voisin à
818-2077 ; R3(exact) = 0.53/0.27/0.17 pour c = 0.3/1/3 ; ρ ≈ **0.928-0.938** ; capture
pré-`lm_head` exacte ; store = une entrée par token (8-11) ; **λ_renv ≳ 0.33 ⇒ F ≈ 6.77**.

**Acquis analytiques (vérifiés par Math D-1..D-5)** :

- `ΔNLL_t = −log[(1−λ) + λ·p_kNN(y_t)/p_LM(y_t)] ≤ −log(1−λ)`, **égalité ssi p_kNN(y_t) = 0** ;
- **λ ≤ λ\* = 1 − e^(−0.05) = 0.0487706 ⇒ E3 ≤ 0.05** — implication simple (**⇐**), universelle ;
  la réciproque est fausse ;
- top-10 ⟺ `λ·p_kNN(cible) + (1−λ)p_LM(cible) > (1−λ)·p₁₀` ; à p_kNN = 1, p_LM(cible) ≈ 0 :
  **λ₁₀ = p₁₀/(1+p₁₀)**, **F₁₀ ≈ 20.504·p₁₀**, condition budgétaire **p₁₀ < 0.05127** ;
- température : `w₁ = 1/(1 + 7·e^(−1/c))` = 0.800/0.280/0.166 ; `R3 ≥ 0.9 ⇒ c ≤ 1/ln 63 = 0.2414` ;
- E1c : `λ_renv = (p_P − p_M)/(1 + p_P − p_M)` ; **p_P − p_M ≤ 1 ⇒ F ≤ 10.252 TOUJOURS** ;
- chaînage paraphrase → secret : `q = r²(3 − 2r)` ;
- bascule d'argmax ⟺ `p_max − p_LM(v_nn) < λ/(1−λ)` = **0.05127** à λ\*, **0.3333** à λ = 0.25.

## 4. Prédictions chiffrées

### 4.1 Cellule décisionnelle (corrigée après E-D1)

**`sup_c`** sur la grille `c ∈ {un-hot, 0.03, 0.1, 0.3, 1, 3}`, à λ\* = 0.0487706, k = 8, avec
`T_q = c·med_{j≥2}(d²_j − d²_min)` **par requête**. Le c atteignant le sup est loggé.
Multiplicité **absorbée** : P3 est évalué au même `sup_c`. Sous-cellule un-hot rapportée
séparément (seule où la factorisation exacte à deux facteurs tient).
**Cible** = **premier token BPE** de `" <secret>"`. **Ex-æquo** : un-hot ≡ limite c→0⁺ ≡
**uniforme sur l'argmin-set** ; compte loggé (attendu 0).

### 4.2 Instrumentation obligatoire

**R1** rang de l'entrée correcte · **R1v** *(nouveau)* rang du premier voisin dont la **valeur**
est le token cible · **R2** d²_min et cos à la clé, **dans les deux conditions**, **normalisé**
`z = (cos − μ_null)/σ_null` (nulle du distracteur 30 k) · **R3** masse p_kNN(cible) sur la grille
· **R4** entropie de p_kNN · **p₁₀** et **p_max** par position.

### 4.3 Tableau

Unité = **le secret** (N = 10), jamais la graine ni la question.

| # | Métrique | Si H vraie | Si H fausse | Seuil + **dérivation** | ANTIPODE (D13) |
| --- | --- | --- | --- | --- | --- |
| **V-base** *(porte)* | E1 / top-10 / E3 aux défauts | — | — | +1.353 ± 1.58 / 0/10 / −0.014 au centième ; sinon **arrêt** | — |
| **V-cap** *(porte)* | \|lm_head(h) − logits\|_max | — | — | ≤ **1e-5** fp32 (run 1 : 0.000e+00, 5 ordres de marge) | — |
| **V-drift** *(porte, décision PI)* | contrôle croisé bit-à-bit de la passe V0 contre `raw/gpu_raw.npz` | — | — | écart = 0 attendu (forward déterministe, seed 0). **Vérification d'environnement seulement** : un écart est une **anomalie à signaler**, jamais une autorisation de réutiliser les bruts archivés | écart ≠ 0 ⇒ dérive d'environnement (torch/driver) à consigner |
| **V0** *(porte, 1 secret)* | R1 sous indice **exact** | — | — | **R1 = 1** (d²_min = 0.0). Échec ⇒ store/indexation cassés ⇒ **invalide**. *Aucune clause R3* | — |
| **V-tie** *(porte)* | compte d'ex-æquo de d²_min | — | — | rapporté ; si > 0, un-hot appliqué uniforme sur l'argmin-set | — |
| **V1a** *(intégrité, sous-ensemble exact)* | sur `p_kNN(y_t) = 0` : max_t \|ΔNLL_t + log(1−λ)\| | — | — | ≤ **1e-6** par cellule (λ, c). Identité algébrique ; erreur attendue ~1e-12 (run 1 : 1.3e-15) ⇒ 6 ordres de marge. Échec ⇒ **invalide** | — |
| **V1b** *(intégrité, complément)* | sur `p_kNN(y_t) > 0` : ΔNLL_t < −log(1−λ) | — | — | **100 %**, strict (décroissance stricte en r). Une violation ⇒ **invalide** | — |
| **V1c** *(intégrité globale)* | E3 mesuré vs **recomposé** hors-ligne | — | — | ≤ **1e-6 nats**. Régime-libre : le vrai test d'intégrité | — |
| **V-var** *(porte, remplace la porte de corrélation)* | var(D_t) sur `p_kNN = 0` | — | — | = **0 à 1e-12** (identité : D_t constant). Échec ⇒ bug | — |
| **V2** *(faisabilité, déclarée avant données)* | n_faisable = secrets avec **p₁₀ < 0.05127** sur ≥ 2/3 paraphrases | — | — | Rapporté **avant** P1. **n_faisable ≤ 5 ⇒ `INCONCLUSIF — budget arithmétique`**. Math : ≥ 7/10 ; Neuro : [7,10] | le canal n'a pas eu sa chance |
| **P1** *(DÉCISIONNELLE, ITT sur les 10)* | n = secrets avec **≥ 2/3 paraphrases** en top-10, à λ\*, `sup_c` | **n ≥ 5/10** | **n ≤ 1/10** | Zone grise 2-4 ⇒ **escalade automatique** (§4.5). Test des signes exact | **n ≥ 5 mais R1v > 1 dans la majorité des succès** ⇒ le succès ne vient pas de l'entrée correcte : **ininterprétable** |
| **h** *(CO-PRIMAIRE)* | taux de récupération par paraphrase, moyenné par secret | **h ≥ 0.50** | **h ≤ 0.20** | **Dérivé de P1** par `q = r²(3−2r)` : r = 0.50 ⇒ q = 0.500 ; r = 0.20 ⇒ q = 0.104. Médiane + IC bootstrap **en log**, « indicatif, N=10 », **10 valeurs in extenso** | Neuro N2 : **h ∈ [0.20, 0.50]** ⇒ zone grise probable |
| **P1-exact** *(ancrage)* | idem, question exacte | 10/10 | < 10/10 ⇒ **suspect** | d²_min = 0.0, R1 = 1 ⇒ top-10 ssi p₁₀ < 0.05127. Échec avec p₁₀ conforme = bug | — |
| **F₁₀** *(diagnostic, NON décisionnel)* | triplet : taux de succès ; **log F₁₀ conditionnelle aux succès** (10 valeurs + min/max) ; nb d'échecs | — | — | **Bimodale (+∞ sur échec), constante entre bras** ⇒ jamais une médiane seule. Prédit ∈ [0.10, 0.62] **< 1** ⇒ le budget n'est pas la contrainte mordante | F₁₀ > 1 sur la majorité des succès ⇒ dérivation de λ₁₀ fausse |
| **P2** *(E1c — SANS fourchette)* | λ_renv, **F = λ_renv/λ\*** | — | — | **Seule clause : F ≤ 10.252**. Ancre de régime F ≈ 6.77. **Étiquette obligatoire** : *« au un-hot avec récupération correcte, P2 ne dépend que de p_LM — mesure du prior de GPT-2, PAS du canal ; légitime comme chiffre de rachat, nulle comme information mécaniste »* | **F > 10.252 = bug** ; **rang 1 à λ ≤ λ\*** ⇒ p_P < 0.049, contredirait E1c ⇒ bug |
| **P3** *(BLOQUANT, hors-ligne, au MÊME `sup_c`)* | valeurs permutées : rang + identité de la baisse | top-10 ≤ 1/10 | top-10 ≥ 5/10 | Identité exacte sur **`valeur(argmin) ≠ cible`** ; ma partition « cible ∉ k valeurs » conservée comme **sous-ensemble conservateur valide**. Violation ⇒ `INCONCLUSIF` | — |
| **P4** *(spécificité)* | store secret A, prompt secret B | ≤ 1/10 | ≥ 5/10 | > 5/10 ⇒ sélectivité nulle | — |
| **P5** *(porte d'IMPLÉMENTATION — §4bis-7)* | E3 sur `NEUTRAL_TEXT`, fait-seul et distracteur | — | — | **E3(λ\*) ≤ 0.05 est un THÉORÈME, pas un test** : dépassement = bug. Prédiction **[0.046, 0.050)**, point Math 0.0464-0.0469. Ordinal Neuro : E3(fait-seul) > E3(distracteur) | E3(λ\*) > 0.05 ⇒ **bug**, jamais un résultat |
| **P5c-id** *(T2 par IDENTITÉ)* | partition exacte {y_t ∈ valeurs recevant de la masse} vs complément : **f_relief** et distribution de D_t | — | — | **Seul test de T2.** Prédiction : **f_relief(un-hot) ∈ [0.5 %, 5 %], STRICTEMENT < f_relief(c fini)**. Version « déciles de H » = **descriptive, lue après** | f_relief(un-hot) ≥ f_relief(c fini) ⇒ normalisation de p_kNN fausse |
| **P5f** *(D11 comportemental, semi-analytique)* | taux de bascule d'argmax, ventilé par décile de NLL_base | — | — | Critère **dérivé** (marge < λ/(1−λ)) ⇒ « concentrées aux marges fines » quasi tautologique. **Seuls le taux (< 2 % à λ\*) et l'antipode sont informatifs** | **bascules aux positions CONFIANTES** ⇒ premier symptôme d'**intrusion mnésique**, à opposer à (b) |
| **P6** *(promue : prédiction ordinale signée)* | clé = couche 6 vs état final ; injection aux logits **dans les deux bras** | — | — | **Couche 6 strictement meilleure sous paraphrase sur ≥ 6/10 secrets**, équivalente en exact. **Aucune portée sur X7** — elle **localise l'invariance**, ce dont (b) a besoin | couche 6 ≤ état final sur ≥ 6/10 ⇒ N2 falsifiée |
| **P7** *(sélectivité)* | h et P5 sous store **30 k** + le fait | dégradation ≤ 30 % | effondrement | Neuro N6 : **corr(dégradation, densité locale) > corr(dégradation, taille du store)** | densité ≺ taille ⇒ artefact de normalisation |
| **P8** *(forme)* | rang vs d²_min, poolé (~40 points) | graduelle, monotone, **sans genou** | — | Un genou signerait une non-linéarité de la tête de sortie | — |
| **G** *(bras descriptif OBLIGATOIRE)* | α = 1[d²_min ≤ τ] : **courbe complète E3(τ), P1(τ), h(τ)** | — | — | **Aucun τ fixé** (la règle v1 « quantile 0.95 » est dégénérée : d²_min = 0.0). *un-hot est aveugle à la force ⇒ G est le seul endroit du run où une pondération par la force intervient.* **Déclencheur de promotion « G → organe »** : P1 conservée à λ = 0.25 **avec** E3 ≤ 0.05 | question de design de (b), jamais un résultat de l'instrument |

**Table d'attribution (à remplir AVANT tout verdict)** :

| Cas | R1 | R1v | R3 | Diagnostic | Conséquence |
| --- | --- | --- | --- | --- | --- |
| P1 échoue | > 1 ; z(para) ≪ z(exact) | > 1 | faible | **Branche A — échec côté CLÉ** | V2-D **non réfuté** ; évaluer le déclencheur multi-clé (§6) |
| P1 échoue | = 1 | = 1 | élevée | **Branche B — échec côté INJECTION** | **Seul résultat qui parle contre le candidat (b)** |
| P1 échoue | — | **> 1 sur la majorité des échecs, sup atteint au bord de grille** | — | **cellule dégénérée** | `INCONCLUSIF — cellule dégénérée` |
| P1 réussit | = 1 | = 1 | élevée | adressage effectif | borne établie à ce λ, avec la phrase de portée §2 |

**Décomposition publiée — trois facteurs sous `sup_c`** : (i) **présence** `R1v ≤ k` ; (ii) **masse
suffisante** `p_kNN(cible) > 19.50·p₁₀` au c retenu ; (iii) **faisabilité** `p₁₀ < 0.05127`. Dans
la sous-cellule un-hot, (i)∧(ii) se réduisent à `valeur(argmin) = cible` et l'identité exacte à
deux facteurs de Math tient — rapportée comme telle. Table de covariables (médiane de H et de
logp_base sur faisable vs infaisable) publiée. La conditionnelle est un **diagnostic mécaniste,
jamais décisionnel**.

### 4.4 Balayage hors-ligne (gratuit)

λ ∈ {0.02, **λ\* = 0.0487706**, 0.05, 0.10, 0.25} × c ∈ {**un-hot**, 0.03, 0.1, 0.3, 1, 3}, k = 8,
`T_q` **par requête**. Frontière (λ, c, E3 analytique, E3 mesuré, P1, h, F₁₀, R1, R1v, R2z, R3,
R4) rapportée **en entier**. Conditions Math Q2 : `p_LM` en log-probs fp32, exponentiation
**fp64**, rang par **comptage strict**.

### 4.5 Escalade (autorisée par le PI, déclenchement automatique)

Si n ∈ [2, 4] : **30 gabarits** de `eval/pool.py`, même règle « ≥ 2/3 paraphrases », unité = le
secret. **H vraie si n ≥ 12/30** (puissance 0.900 à p = 0.5 ; risque I ≈ 2e-5 sous p = 0.1) ;
**H fausse si n ≤ 5/30** ; grise [6, 11] ≈ 10 %. Coût +8 min GPU. **Déclenchement automatique**
— aucune décision en cours de cycle.

---

## 4bis. Auto-vérification D14 (2ᵉ passage, sur le protocole consolidé)

Phénomènes que **ce protocole prédit** : (T2) soulagement f ∈ [0.5 %, 5 %] ; (T-perm) les valeurs
permutées conservent le multiensemble ; (T-exact) d²_min = 0.0 ; (T-const) ΔNLL constant hors
soulagement ; (T-multitok) secrets multi-tokens ; (T-R1v) la cible peut être portée par un voisin
de rang > 1 ; (T-sup) `sup_c` introduit une multiplicité ; (T-zone) N2 place n en zone grise.

1. **V1a contre T2** — conditionnée sur `p_kNN = 0`, où la prédiction est une **identité**. T2 vit
   sur le complément (V1b, V1c). **Non contredite.** *(C'est la porte qui a tué le run 1.)*
2. **λ\* contre T2** — le soulagement ne peut que **baisser** E3 ; λ\* du pire cas est
   conservateur, tient même si f = 0. **Non contredite.**
3. **P3 contre T-perm** — identité restreinte à `valeur(argmin) ≠ cible` (condition exacte) ; ma
   partition conservée comme sous-ensemble conservateur ; comptes loggés. **Corrigée.**
4. **G contre T-exact** — la règle « quantile 0.95 » est dégénérée (τ = 0) ; G devient une courbe
   sans point d'opération. **Corrigée.**
5. **V-var contre T-const** — je teste la **variance** (identité : 0), pas la corrélation (NaN).
   Ma porte antérieure était un test déguisé. **Corrigée.**
6. **Lecture D11 — ma formulation retirée (F3)** : le dommage absolu est **constant par position**
   hors soulagement ; « profil inverse » est un **artefact de normalisation** et n'est **ni une
   mesure ni un profil**. La propriété correcte est la **NON-SÉLECTIVITÉ**. Contenu mesurable :
   **P5c-id** (identité) et **P5f** (bascules). Résultat conceptuel candidat : le **renversement
   d'asymétrie** (canal d'état = gain borné / dommage non borné ; mélange = l'inverse).
7. **P5 ne peut pas échouer à λ\*** — conséquence du théorème D-1. **Une porte qui ne peut pas
   échouer n'est pas un test** : P5 requalifiée en **porte d'implémentation** (dépassement ⇒ bug),
   ne crédite pas H. *(Défaut détecté au 2ᵉ passage.)*
8. **P5f contre T-const** — pas de contradiction, mais le critère est **dérivable** ⇒
   « concentrées aux marges fines » est quasi analytique : seuls le **taux** et l'antipode sont
   informatifs. **Étiqueté semi-analytique.** *(Défaut détecté au 2ᵉ passage.)*
9. **`sup_c` contre T-sup** — la multiplicité est portée par **P3, évalué au même `sup_c`**.
   **Non contredite.**
10. **Cellule contre T-R1v** — corrigée par `sup_c` + R1v + clause `cellule dégénérée`.
11. **P2 contre la borne dure** — fourchette supprimée, seule subsiste F ≤ 10.252.
12. **h contre P1** — les seuils de h sont l'image exacte de ceux de P1 par `q = r²(3−2r)` : les
    deux primaires **ne peuvent pas se contredire** par construction.
13. **V2 contre P1** — l'impossibilité arithmétique conduit à `INCONCLUSIF — budget arithmétique`,
    pas à `REJETE`. **Non contredite.**
14. **Escalade contre T-zone** — seuils écrits et autorisation obtenue **avant** lancement.
15. **Bande E3 de Neuro (N3)** — borne haute 0.0500 = supremum théorique, atteint seulement si
    f_relief = 0, que Neuro exclut ⇒ bande **[0.046, 0.050)** stricte. *(Incohérence mineure d'un
    expert, corrigée.)*
16. **E3 > 0.05 aux λ élevés** — prédit analytiquement, **non** critère d'abandon.

**Aucune porte de ce protocole ne dépend d'un régime de température** : toutes les portes
d'intégrité sont des identités algébriques partitionnées sur le sous-ensemble où elles sont
exactes, et la cellule décisionnelle est un supremum sur une grille déclarée.

---

## 5. Contrôles et baselines

1. **Configuration courante** — défauts `EngramConfig()` (V-base), **avant tout ajout**.
2. **D7** — KV vidé avant chaque rappel ; datastore jamais alimenté pendant
   `logprob_continuation` (test CPU bloquant).
3. **M reset sur le même prompt** — baseline `M = 0` **et** datastore vide ; c'est aussi la
   condition principale (sous M chargée, la lecture perturbe l'état de requête et détruit
   l'attributabilité de R1-R4).
4. **Flag à off** — `knn_lambda = 0.0` ⇒ logits **bit-exacts**, rejoué en fin de run.
5. **P3 au même `sup_c`** — élimine « durcissement de distribution » **et** absorbe la
   multiplicité.
6. **P4, fait croisé** — élimine « le datastore récite quel que soit le prompt ».
7. **Traces égalisées** — une entrée par token du fait, même compte que `force_write=True`.
8. **Clés kNN non projetées par DG** (test CPU, D9).
9. **Ordre des conditions** fixé par `SECRETS` / `QUESTIONS`, identique dans tous les bras ;
   moteur neuf par bras.
10. **Nulle empirique pour R2z** — μ_null, σ_null contre le distracteur 30 k.
11. **Confondants loggés** — corr(rang_mix, logp_base) par secret ; part des tokens communs du
    fait dans les valeurs ; p₁₀, p_max, comptes d'ex-æquo, c-du-sup.
12. **Re-collecte intégrale à neuf** ; `raw/gpu_raw.npz` reste archivé comme référence et
    **n'est relu que par la porte V-drift** (comparaison, jamais substitution).

## 6. Critères d'abandon

**Ce qui tue H** : **P1 ≤ 1/10** (ITT) **et** h ≤ 0.20, avec n_faisable ≥ 6. Verdict `REJETE`,
**formulé avec la portée de Neuro** : *« l'état final du cortex gelé n'est pas une clé invariante
à la paraphrase à l'échelle du top-10 »* — **et non** « l'espace de sortie n'est pas le bon point
d'attaque ». Suite selon la table d'attribution.

**Déclencheur *encodage multi-clé* (écrit avant les données)** — unités **z normalisées par la
nulle du distracteur 30 k**. **ARMÉ ssi, sur ≥ 6/10 secrets** : (a) R1 > 1 sur ≥ 2/3 paraphrases ;
(b) médiane sur paraphrases de `z(q_para, k_correcte) ≤ 3.0` alors que `z(q_exact, k_correcte)`
est hors échelle ; (c) **clause discriminante** : médiane des `z(q_para_i, q_para_j)` **≥** médiane
de `z(q_para, k_correcte) + 2.0`. **(c) FAUSSE ⇒ NE PAS ARMER** : échec d'invariance de
l'encodeur, qui renvoie vers **P6** ou vers **(c) Fast-KV**, pas vers le multi-clé.

**Ce qui invalide le run** (`INCONCLUSIF`, cause nommée) : V-base non reproduite ; V-cap > 1e-5 ;
V0 R1 ≠ 1 ; **V1a > 1e-6** ; **V1b violée une seule fois** ; **V1c > 1e-6** ; **V-var > 1e-12** ;
P3 ≥ 5/10 ou identité violée ; **majorité des échecs de P1 avec R1v > 1 et sup au bord de grille
⇒ `INCONCLUSIF — cellule dégénérée`** ; datastore vide / 0 écriture ; NaN ou inf ; fp16 dans la
chaîne de distance ; DG appliqué aux clés ; `knn_lambda=0` non bit-exact ; repli CPU silencieux ;
**E3(λ\*) > 0.05** (⇒ bug, §4bis-7).

**INCONCLUSIF sans être invalide** : n_faisable ≤ 5 ⇒ `budget arithmétique` ; escalade avec
n ∈ [6,11]/30 ⇒ `zone grise persistante`.

**Ce qui n'est PAS un critère d'abandon** : E3 > 0.05 aux λ élevés ; F ou F₁₀ grands ;
corr(D,H) ≈ 0 (identité) ; h dans la bande grise.

**Interdit (D14)** : amender une porte après lecture du premier token de données. Une porte
échouée **arrête le run** et se journalise comme résultat (porte + dérivation + chiffre), pas
comme échec de cycle (décision PI §13.10).

## 7. Variables fixées

`seed = 0` ; `gpt2` ; `layer_index = 6` ; `lam = 2.0` ; `cap = 0.5` ; `eta = 0.2` ;
`decay = 1e-3` ; `threshold = 4.0` ; `dg = 8192/64` ; `read_gate = "keysim"` ;
`prune = 512/0.10` ; `knn_k = 8` ; `knn_key_layer = "final"` (bras P6 : `"inject"`).
**Données** : 10 `SECRETS` + `FACT_TEMPLATE` + 4 `QUESTIONS` de `eval/fact_injection.py` ;
`E1C_FACT`/`E1C_QUESTION` de `eval/read_gate.py` ; `NEUTRAL_TEXT` de `eval/collateral.py`
(343 positions) ; distracteur = **30 000** premiers tokens de `data/rfc9293.txt`
(SHA-256 `6d9ac8be4b0286f8`, re-vérifié) ; escalade = 30 gabarits de `eval/pool.py`.
**Statistique** : unité = le secret ; test des signes exact ; bootstrap **en log**, 10 000
tirages, **« indicatif, N=10 »** ; **10 valeurs par secret in extenso** + min/max pour h et F₁₀ ;
permutations au plancher écrites « p < 10⁻³ » ; par position, ρ̂ jusqu'au lag 5, L ≥ 2× le temps
d'autocorrélation intégré si ρ̂₁ > 0.1, **repli L = 25 si ρ̂₁ ≤ 0**.
**Numérique** : distances **fp32** ; `p_LM` en log-probs fp32, exponentiation **fp64**, rang par
**comptage strict** ; softmax kNN après soustraction de d²_min ; mélange en **log-espace** ;
`M` reste fp32.

## 8. Variable manipulée

**Une seule : λ**, avec `M = 0` en condition principale. Bras descriptifs pré-déclarés, lus
**après** le verdict : `M au point courant + kNN` (additivité), **P6**, **P7**, **G**.

## 9. Budget

| Passe GPU | Contenu | Durée |
| --- | --- | --- |
| V-base | E1 + E3 aux défauts | 3 min |
| V0 / V-cap / V-drift | smoke 1 secret + point de capture + contrôle croisé bit-à-bit | 2 min |
| A | store fait-seul, M=0 : 10 secrets × (injection + 4 questions) ; log `p_LM` complet, 8 (d², token), **états couche 6 dans la même passe** (P6 gratuit) | ~3 min |
| B | idem avec M au point courant (additivité) | ~3 min |
| C | E1c (M=0 et M chargée) au point courant | ~2 min |
| D | E3 par position, `NEUTRAL_TEXT`, store fait-seul | ~4 min |
| E | store distracteur 30 k + P7 + E3 distracteur + **nulle empirique μ/σ pour R2z** | ~8-10 min |
| — | re-run `knn_lambda=0` bit-exact | 2 min |
| **Total** | **re-collecte intégrale à neuf** | **~27-29 min GPU** |
| *(escalade auto si n ∈ [2,4])* | 30 gabarits `eval/pool.py` | *+8 min* |

Hors-ligne (CPU, gratuit) : grille λ × c, P1, h, F₁₀, P2, P3, P4, P5c-id, P5f, P8, frontière,
courbe G, R1v, R2z. **VRAM < 0.6 Go sur 6.**

## 10. Livrables attendus

- **Config** : champs `knn_*` existants, défaut inerte (`knn_lambda = 0.0`). Un seul ajout
  spécifié ici : **`knn_temp_c = 0.0` ≡ un-hot ≡ uniforme sur l'argmin-set**. Le datastore vit
  dans `eval/`, jamais dans `engram/`.
- **Script** : `eval/knn_ceiling.py` **amendé** (pas réécrit), séparation `--phase gpu` /
  `--phase analysis` conservée. **Les trois ambiguïtés du run 1 sont spécifiées ici** : `T_q` par
  requête ; τ en courbe ; **suppression de `P5B_VAR_GATE`** (remplacé par V-var). Nouveaux logs :
  R1v, R2z, p₁₀, p_max, c-du-sup, ex-æquo.
- **Tests CPU** : (i) `knn_lambda=0` bit-exact ; (ii) identité du mélange et bord dur ;
  (iii) **un-hot = uniforme sur l'argmin-set** (cas d'ex-æquo construit) ; (iv) fp32 (régression
  fp16 à ‖h‖² ~ 10⁴) ; (v) soustraction de d²_min ; (vi) clés jamais passées par G/DG ;
  (vii) datastore jamais rempli pendant `logprob_continuation` ; (viii) permutation = mêmes clés ;
  (ix) k > taille du store ; (x) cible = premier token BPE ; (xi) **R1v ≠ R1 sur un store où la
  cible apparaît deux fois** ; (xii) `sup_c` = max sur grille, c loggé.
- **Journal** : entrée datée ; protocole recopié ; frontière ; **table d'attribution R1/R1v/R3
  remplie** ; décomposition à trois facteurs + covariables ; P5c-id, P5f, P6, G ; bruts hashés ;
  **phrase de portée obligatoire** ; **interdiction D15 recopiée**.
- **Tableau §4** : ligne dans le tableau des **ablations/instruments**.
- **Correctifs de fiche V2-D** rédigés pour le PI ; le labo n'édite pas `docs/EXTENSIONS.md`.
  **Ce protocole n'emploie ni « plafond » ni « familiarité »**.

## 11. Questions résiduelles pour Neuro (interprétation)

1. **P5f semi-analytique** : seuls le taux et l'antipode portent de l'information — confirmes-tu
   que la ventilation par décile ne doit créditer aucune théorie ?
2. **P6 promue** : si couche 6 bat l'état final sur ≥ 6/10, quel est le **livrable pour (b)** —
   M_out lu sur une clé intermédiaire plutôt que finale, et cela reste-t-il dans D8 ?
3. **Déclencheur multi-clé, clause (c)** : si (c) est fausse, pré-déclare l'arbre à deux branches
   (P6 vs Fast-KV) selon le résultat de P6.
4. **Renversement d'asymétrie** : formulation minimale pour le journal, en une phrase qui ne se
   transporte pas abusivement à (b).
5. **R2z** : la nulle du distracteur 30 k est-elle la bonne référence d'anisotropie, ou faut-il
   une nulle par position ?

## 12. Questions résiduelles pour Math (interprétation)

1. **Factorisation sous `sup_c`** : trois facteurs, ou existe-t-il une forme à deux conservant
   l'exactitude ?
2. **Multiplicité** : P3 au même `sup_c` suffit-il, ou faut-il une correction explicite du seuil ?
3. **Seuils de h** dérivés par `q = r²(3−2r)` : les deux primaires sont-elles bien non
   contradictoires, et h a-t-elle plus de puissance que n en zone grise ?
4. **Ex-æquo** : l'un-hot uniforme sur l'argmin-set casse-t-il une identité ailleurs si
   |argmin-set| > 1 ?
5. **F₁₀** : le triplet est-il un rapport complet, ou faut-il une transformée bornée ?
6. **P5f** : existe-t-il une statistique du taux de bascule **non impliquée** par la marge ?

## 13. Décisions du PI (2026-08-21, gate de pré-enregistrement)

1. **Cellule décisionnelle `sup_c`** (renforcée : sup sur grille déclarée, c loggé, P3 au même
   sup) : **validée**.
2. **Co-primaire h** au lieu de F₁₀, seuils dérivés 0.50 / 0.20 : **validé**.
3. **P2 sans fourchette pré-enregistrée**, borne dure F ≤ 10.252 + étiquette « mesure du prior,
   pas du canal » : **validé** (suppression plutôt qu'arbitrage entre les deux fourchettes).
4. **Escalade : AUTORISÉE, déclenchement AUTOMATIQUE** si n ∈ [2,4] — 30 gabarits, seuils
   **12/30 et 5/30**, +8 min GPU. **Aucune décision en cours de cycle ne sera acceptée.**
5. **Contrôle croisé V-drift sur la passe V0 : autorisé** — vérification d'environnement, jamais
   une autorisation de réutiliser les bruts archivés (choix délégué au copilote).
6. **D12, D13, D14 et D15** : rédigées par le labo à l'attention du PI, qui les grave dans
   `docs/ARCHITECTURE.md` §3 **avant** le run.
7. **Correctifs de façade V2-D** (`EXTENSIONS.md` l.324 « PLAFOND », l.390 « familiarité ») :
   rédigés par le labo, appliqués par le PI avant le run.
8. **Critère d'ouverture de (b) M_out** : **`P1 ≥ 5/10` OU `h ≥ 0.50`**, et le protocole de (b)
   doit porter **son propre E3** — aucun transport depuis (a), conformément à D15.
9. **Bras G obligatoire, aucun τ fixé** : **validé** — aucun point d'opération de gate ne sort de
   ce run (choix délégué au copilote).
10. **Arrêt dur** : **confirmé** — première porte échouée = arrêt sans amendement, journalisé
    comme résultat (choix délégué au copilote).

## Historique

- 2026-08-21 : brouillon du Directeur (V2-D(a) v2)
- 2026-08-21 : avis Math (RÉSERVÉ, 3 amendements) et Neuro (FAVORABLE, 2 amendements) —
  **chacun trouve une erreur de dérivation du Directeur** (E-D1 cellule un-hot, E-D2 fourchette P2)
- 2026-08-21 : 2ᵉ passage D14 du Directeur — **deux défauts supplémentaires détectés** (P5 ne peut
  pas échouer ; P5f quasi analytique)
- 2026-08-21 : protocole consolidé, décisions du PI intégrées — proposé
- 2026-08-21 : pré-enregistré par le PI
- 2026-08-22 : run exécuté, toutes portes « passées » — mais V1b l'a été par **substitution
  de prédicat après lecture des données** (`GATE_FAILURE_V1b.json` en atteste), en violation
  de l'arrêt dur §13.10 et de D14(d)
- 2026-08-22 : Verifier REJECTED (2 critical, 2 major) ; **run déclaré INVALIDE par le PI**
  — arrêt dur respecté. Observation P6 conservée avec N_eff = 3 (décision PI).
  Statut TERMINE — INVALIDE ; suite : cycle méthode avant tout v3
