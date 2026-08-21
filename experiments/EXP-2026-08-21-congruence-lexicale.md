# EXP — Q-01b : le terme signé de la lecture est-il porté par un canal de congruence ?

Statut : TERMINE — REJETE

**Origine** : section « Suite » du journal 2026-08-21 (entrée Q-01). Anomalie ouverte :
le renversement de signe de P6 entre textes A et B. Brouillon du Directeur soumis à
Neuro et Math le 2026-08-21 ; arbitrage ci-dessous ; décisions du PI en §13.

---

## Arbitrage

Les deux avis convergent sur un point que le brouillon ne voyait pas : **le protocole
tel qu'écrit ne pouvait pas trancher**, pour deux raisons indépendantes et
cumulatives — Math : la décisionnelle F est structurellement sous-puissancée
(décomposer un écart entre 2 unités de réplication) ; Neuro : le régresseur choisi
est le mauvais, et son biais d'atténuation est orienté vers le bas, donc un F bas
aurait été sur-lu. Le protocole change de cible : **il établit un canal, il n'arbitre
plus un écart.**

### Avis Math (RÉSERVÉ → FAVORABLE si A1 adopté)

| Remarque | Sort | Raison |
| --- | --- | --- |
| **Q1 bloquante** — SE(F) ≤ 0.128 exigerait σ_pos ≤ 0.17 nats, sous le plancher plausible [0.3, 0.8] ; probabilité de verdict décisif ≤ 15-25 % même si F_vrai = 0.5 ; Δ_raw = −0.095 n'a que 2-4 SE ; il faudrait ~2400 positions confiantes ≈ 12 textes ≈ 1-1.5 h GPU | **Intégrée, sans réserve** | C'est l'objection qui refait le protocole. Cohérente avec H2 déjà déclarée non testable : deux unités de réplication ne décomposent pas un écart. |
| **A1** — inverser les rôles : P1 décisionnelle unique, F rétrogradée en estimation (point + IC, tronquée [−1, 2]) ; vocabulaire « canal établi, part expliquée estimée F = x [IC] » | **Intégrée** | Ossature de §4. §1 et §2 réécrites en conséquence. |
| **A2** — blocs tirés sur la série PLEINE ; h_median et nombre de bins fixés ; sous-ensemble confiant, quintiles, Δ_raw, Δ_adj, E, F recalculés dans chaque réplicat ; IC de F invalidée si > 5 % des réplicats ont \|Δ_raw\| < 0.03 | **Intégrée telle quelle** | §4 (mécanique bootstrap) et livrables. |
| **A3** — placebo d'estimateur : pipeline complet sur impair_iid → null empirique de F̂, IC devant contenir 0 | **Intégrée, fusionnée dans V5** | Calibre tout biais mécanique de la chaîne ; coût nul. |
| **Q2** — OLS/model-based en primaire d'estimation, post-stratification en sensibilité | **Intégrée — le Directeur concède** | Voir « conflit d'estimateur ». |
| **Q3** — seeds quasi-doublons ; moyenne de série d'abord ; intersection-union pour la décisionnelle, Fisher-z poolé en descriptif | **Intégrée** | §7. |
| **Q4** — bug de forme : ⌈3/ρ₁⌉ décroît quand la dépendance croît ; règle correcte L = max(25, ⌈3/(−ln ρ̂₁)⌉ arrondi à la dizaine sup.), plafonnée à N_full/10, ρ̂₁ = max(ρ₁(impair), ρ₁(logfreq)) | **Intégrée — erreur du brouillon reconnue** | Avec ρ₁ ≈ 0.06 → **L = 25** partout. La règle du brouillon donnait L = 50, soit ~3.4 blocs sur 170 positions : bootstrap invalide. |
| **Q5** — pas de modèle borné en primaire ; le plancher atténue la pente (anti-conservateur) ; sensibilité impair/(NLL_base + 0.5), signe seulement | **Intégrée** | §4 sensibilités. |
| **Q6** — un seul estimateur pré-désigné ; ordre de repli pré-déclaré ; **le prior du cortex est ENDOGÈNE** (corr(W_U·r, logfreq) = +0.484 EST le canal) donc strictement descriptif | **Intégrée ; l'option (b) est RETIRÉE** | Régresser sur le prior du cortex, c'est tester le mécanisme contre lui-même. |
| **Q7** — stratification 2D sur H licite mais r(H, NLL_base) ≈ 0.6-0.8 et 15-35 positions/cellule → sensibilité, signe pas amplitude | **Intégrée avec la restriction** | §4. |
| Confondants ajoutés (masse au plancher add-1 ; P3 en porte de l'estimation ; assert d'alignement x_t = logfreq(ids[t]) ; Δ_raw recalculé depuis les séries brutes ; chevauchement corpus/textes) | **Intégrés tous les cinq** | V-gates et §5. |
| **Exigence préalable** : mesurer σ_pos (30 s CPU) avant lancement ; si σ_pos < 0.17, la réponse Q1 s'inverse | **Intégrée en porte V0, avec règle adaptative pré-enregistrée** | Seule adaptation autorisée : sur une variance, pas sur un effet ; seuil et direction fixés avant mesure. |
| Confirmation : D_full des bras ±r̄ sérialisées (l. 589) ; `sum_r` ajouté après l'appel (l. 622/671) → r̄ perdu | **Enregistrée** | Fonde P8b (§13.3). |

### Avis Neuro (FAVORABLE, 2 amendements, coût GPU nul)

| Remarque | Sort | Raison |
| --- | --- | --- |
| Vérification arithmétique des 4 chiffres publiés (pair ± impair reproduisent −0.092/+0.366 sur A, idem B) | **Intégrée en V1** | L'objet de Q-01b existe ; la décomposition est saine. |
| **Q1 — le régresseur est faux** : impair_t ≈ −(s[cible_t] − E_{p_t}[s]) ; le terme de centrage est la moitié du mécanisme et **la moitié qui dépend du texte** ; r² ≈ 0.23 entre logfreq et le régresseur exact → **F biaisé vers 0 par construction** | **Intégrée — devient la clause d'asymétrie de §4** | Sans elle, un F ≤ 0.25 aurait été lu « pas de canal » alors qu'il ne falsifie que le proxy. |
| **Q2 — l'identité qui débloque** : E_p[log q] = −H − KL → impair_t ≈ −b·(logfreq + H_t + KL_t) ; **H_t est déjà sérialisé** | **Intégrée** | Le régresseur correct à coût zéro est **logfreq + H**. Interdit associé : jamais de bivarié impair ~ KL. |
| **N-1** — ajouter **P2′** sur les quintiles de (logfreq + H) ; F′ − F estime la part portée par l'entropie ; sur `impair ~ logfreq + H + logfreq×H` : deux pentes négatives de rapport [0.5, 2], interaction ≈ 0 | **Intégrée** | P2′ et P5 fusionnent avec le modèle unique déjà pré-enregistré (une seule interaction). |
| **N-2** — régresseur EXACT sans r̄ et sans GPU : directions du bras *fixe* reproductibles par `draw_eps`, W_U chargeable en CPU sans forward, normes sérialisées → s_t = rnorm_t·(W_U[cible_t]·u_j) | **Intégrée en porte d'interprétation (P8)** | H1a testée dans sa forme exacte + calibration de l'atténuation du proxy + contrôle positif de V5. Rôle limité à l'interprétation pour préserver la décisionnelle unique. |
| **Q3 — modérateur de repli unique : taux de répétition en contexte** ; rejet argumenté des trois autres | **Intégrée telle quelle** | §4, activé seulement si P1 échoue. |
| **Q4** — signe négatif signé ; « même mécanisme, bilan différent » signé **sous réserve testable** : vérifier le signe aux positions INCERTAINES | **Intégrée** | N-P1b devient descriptive obligatoire. |
| **Q5** — H2 mal étiquetée : clause de sélection sur l'estimateur, pas hypothèse de mécanisme ; lecture gratuite disponible (4 blocs) | **Intégrée, avec correction d'étiquette** | §2 réécrite ; lecture en blocs en descriptive obligatoire (D2). |
| N-P1..N-P5, dont **N-P3 : signe de Δ logfreq(A−B) incertain (55/45), maillon faible** | **Intégrées en colonne « prédiction ordinale »** | Raison honnête pour laquelle F restera probablement muette. |
| Deux prédictions de forme (pente atténuée aux confiantes ≠ échec ; régresseur censuré par le bas) | **Intégrées** | Justifie de porter la décisionnelle sur **toutes** les positions valides. |
| Note corpus : si `pnp_narrative.txt` est un extrait de *Pride and Prejudice*, **pg1342 pondère Austen deux fois** | **Intégrée en porte V3b** | Règle de dé-duplication pré-déclarée. |
| Façades : « logfreq = le prior » ; « congruence » comme propriété du token seul ; **« biais de critère » → BIAIS DE PRIOR ADDITIF SUR LES LOG-ODDS** ; H2 comme « hypothèse rivale » | **Intégrées ; vocabulaire imposé** | Correction de l'entrée Q-01 approuvée par le PI (§13.6). |
| Mécanisme manquant (ablation de la direction de fréquence ; Vinogradova, Lisman & Grace, Carandini & Heeger ; Mu & Viswanath) — note sous condition | **Enregistrée ; fiche EXTENSIONS approuvée par le PI (§13.7)** | Ne devient une proposition que si N-P5 ou P8/P8b établit que l'impair est porté par la congruence. |

### Écartés du brouillon

1. **F comme décisionnelle** (Math Q1) — rétrogradée en estimation.
2. **La règle L = max(25, ⌈3/ρ₁⌉)** (Math Q4) — fausse dans sa forme, corrigée.
3. **L'option « prior inconditionnel du cortex »** (Math Q6) — retirée, endogène.
4. **La post-stratification en primaire d'estimation** (Math Q2) — rétrogradée en sensibilité.
5. **Trois des quatre modérateurs de repli** (Neuro Q3) — un seul reste, nommé avant mesure.

### Conflit d'estimateur — tranché

Le brouillon plaçait la post-stratification en décisionnelle ; Math ré-inverse vers
OLS. **Le Directeur concède à Math** : sous A1, F n'est plus un verdict — l'argument
de fidélité littérale perd son poids dès lors que la question décisionnelle devient
l'existence du canal ; et le coefficient OLS est réutilisable pour la calibration du
plafond de F par P8, ce que la post-stratification ne fournit pas.

**Hiérarchie finale** : verdict = ρ de Spearman (robuste au plancher et à la censure
add-1) ; estimation de F = OLS/model-based sur toutes les positions valides ;
sensibilité = post-stratification par quintiles (mécanique A2 intégrale).

**Corollaire — la décisionnelle porte sur TOUTES les positions valides**, pas sur les
confiantes seules : Math Q2 (√2 de puissance) ; Neuro forme (i) (pente atténuée aux
confiantes par restriction d'étendue — y placer la décisionnelle, c'est se pénaliser
deux fois) ; et A1, qui déplace la cible vers l'existence du canal, laquelle n'est pas
une propriété des seules confiantes. MDE recalculé : SE(ρ) ≈ 1/√(0.887·N) = 0.057 (A)
/ 0.050 (B) ; pour 80 % de puissance jointe en intersection-union (≈ 89,4 % par test),
**MDE ≈ −0.19** contre −0.29 sur les confiantes. Le seuil ρ̂ ≤ −0.20 devient atteignable.

### Changement de cible, énoncé sans détour

Le cycle ne répondra très probablement **pas** à la question de son titre d'origine.
Issue la plus probable, pré-écrite : **canal établi, part expliquée non résolue.**
Acquis en échange : existence, signe, forme (logfreq vs logfreq + H) et atténuation du
canal, plus le chiffrage d'une réponse à l'écart (Q-01c). **Troc validé par le PI
(§13.1).**

---

## 1. Question

**Décisionnelle** : le terme signé de la lecture (impair) est-il porté par un canal de
congruence entre la direction injectée et la cible — canal détectable via la
log-fréquence unigramme de la cible sur les bruts ±r̄ existants ?

**Subordonnée, rapportée en estimation seulement** : quelle part de l'écart de bilan
entre textes A et B aux positions confiantes (Δ_raw = −0.095) ce canal explique-t-il ?

## 2. Hypothèse

**H1 (canal de congruence, biais de prior additif sur les log-odds)** : la lecture
ajoute une poussée quasi constante sur les log-odds, orientée vers l'axe de fréquence ;
elle aide d'autant plus que la cible attendue est haute sur cet axe **relativement à ce
que le cortex attend déjà**. Au premier ordre,
impair_t ≈ −b·(logfreq(cible_t) + H_t + KL_t) (identité Neuro Q2).

- **H1a (le canal existe)** — décisionnelle : à l'intérieur de chaque texte, impair_t
  décroît avec logfreq(cible_t).
- **H1b (le canal rend compte de l'écart)** — **estimation, pas verdict** : rapportée
  point + IC, tronquée à [−1, 2] ; jamais convertie en verdict, sauf réactivation par V0.

**H2 — reclassée.** Ce n'est pas une hypothèse rivale de mécanisme mais une **clause de
sélection sur l'estimateur** (A est le texte de découverte de P5) : elle ne prédit rien
sur la structure interne des textes, elle affirme que Δ_raw est surestimé. Non testable
à N = 2 textes ; lecture gratuite en D2. Si H1b passait, H2 deviendrait superflue par
parcimonie, jamais réfutée.

**Vocabulaire imposé** (Neuro, façades) : « décalage de distribution de la log-fréquence
unigramme », jamais « le prior » ; **« biais de prior additif sur les log-odds »**,
jamais « biais de critère » ; « congruence » n'est jamais une propriété du token cible
seul — elle est **contrastive** (cible contre attente).

## 3. Ce que le projet sait déjà

- **Q-01, RETENU** (journal 2026-08-21) : ciblage entropique générique, R = 0.826
  [0.706, 0.946] (A) / 0.779 [0.547, 1.011] (B) ; V1 ✓ (+0.13536) ; ρ₁ ≈ 0.06,
  N_eff ≈ N ; saturation du cap 0.26 % / 0.11 %.
- **Anomalie ouverte** : readM confiantes −0.0646 (A, 10/10) vs +0.0370 (B, 10/10),
  t apparié = 20.6 ; reproduite **sans M** par le bras +r̄ (−0.0918 → +0.0230).
- **Décomposition** (vérifiée arithmétiquement par Neuro) : impair_conf −0.229 (A) /
  −0.134 (B) ; pair_conf +0.137 / +0.157 ; pair + impair = −0.092 / +0.023 et
  pair − impair = +0.366 / +0.291 reproduisent les deux bras. Effet de bilan.
- **P7** : corr(profil +r̄, profil readM) = 0.9932 / 0.9927. **P4** : similarité
  inter-secrets 0.988 / 0.986 vs null 0.658 / 0.626.
- **cos(r̄_A, r̄_B) = 0.9999** : la direction est un invariant du modèle — l'écart ne
  peut venir que du texte.
- **Diagnostic 2** (`eval/marginal_pull.py`) : corr(W_U·r, log-fréquence unigramme) =
  **+0.484 ± 0.002**. Mesure déjà faite, citée, non refaite. Elle fonde l'hypothèse ;
  prise au carré (r² ≈ 0.23) elle chiffre l'atténuation du proxy et **interdit** d'en
  faire un régresseur (endogène).
- **X7** : aplatissement = coût fixe +0.141 nats, cos ≈ 0 — quatrième symptôme du même
  mode commun.
- **COR-02** : un ratio de deux moyennes bruitées sans incertitude n'est pas un
  résultat. C'est la leçon que A1 applique à F.

## 4. Prédictions chiffrées (hiérarchie pré-enregistrée)

Convention fixée avant mesure : impair_t = [D_t(+r̄) − D_t(−r̄)]/2 ; **impair négatif =
le bras +r̄ aide**. H1 prédit une pente **négative**.

### Porte préalable V0 (AVANT tout calcul de corrélation)

| # | Mesure | Règle pré-enregistrée |
| --- | --- | --- |
| **V0** | σ_pos = écart-type des impair_t aux positions confiantes, par texte (30 s CPU) | Si **σ_pos > 0.17** sur au moins un texte (attendu) : F reste une **estimation**. Si **σ_pos ≤ 0.17 sur les deux textes** : F est **promue co-décisionnelle** avec les bandes d'origine (H1b vraie si F ≥ 0.50 IC excl. 0.25 ; fausse si F ≤ 0.25 IC excl. 0.50). Seuil, direction et conséquence fixés ici, avant toute inspection d'effet. |

### Portes de validité (échec = analyse INVALIDE, pas négative)

| # | Contrôle | Bande |
| --- | --- | --- |
| V1 | Reproduction **depuis les séries brutes** (jamais depuis les chiffres arrondis) : impair_conf, pair_conf, readM_conf, D(+r̄)_conf, et les identités pair ± impair | ± 0.001 sur les 8 valeurs ; identités exactes à 1e-9 |
| V2 | Manifeste SHA-256 écrit **avant** analyse ; `len(D_full) == len(ids)` ; **assert x_t = logfreq(ids[t]) et non ids[t−1]** ; `config` identique dans tous les JSON ; `writes == 0` sur iid/fixe/±r̄ et `writes > 0` sur les 20 readM (D7) ; **compter les JSON `fixe-*` réellement présents** (10 ou 20 : la règle adaptative a pu se déclencher) | toutes vraies |
| V3 | Couverture lexicale du corpus hors A/B | ≥ 0.95 ; repli pré-déclaré complet → narratif → **STOP** ; < 0.80 après repli ⇒ non concluant |
| V3b | **Dé-duplication** : `pnp_narrative.txt` est-il un extrait de `pg1342.txt` (n-grammes de 10 tokens) ? aucun passage partagé avec A ou B ? | si chevauchement : ne garder que `pg1342.txt`, documenter. Résultat publié quel qu'il soit |
| V3c | Masse au plancher add-1 : fraction des cibles à compte ≤ 2 | rapportée ; si > 5 %, sensibilité obligatoire les excluant |
| V4 | ≥ 100 positions confiantes par texte ; sd(logfreq) > 0 par strate ; ≥ 15 positions par bin après fusion | toutes vraies |
| **V5** | **Placebo d'estimateur** : pipeline COMPLET (P1 + F, OLS et post-strat) sur impair_iid = [D(+ε) − D(−ε)]/2, 10 seeds × 2 textes | ρ_null : \|r̄\| ≤ 0.10, IC contenant 0 ; **IC de F_null contenant 0**. Une pente négative robuste ici = biais mécanique de chaîne ⇒ **analyse invalide**. Contrôle positif : la pente lexicale du bras *fixe* doit être centrée sur 0 **entre** tirages tout en étant non nulle **par** tirage |
| **V6** | **Porte du replay r̄ (§13.3)** : le rejeu des 20 streams readM doit reproduire V1 de Q-01 — dommage moyen readM texte A | **+0.1354 ± 0.005** ; hors bande ⇒ le replay est invalide, P8b tombe, P8 (bras fixe) reste |

### Décisionnelle unique — P1

ρ de Spearman entre impair_t (série moyennée sur les 10 seeds **avant** calcul de
statistique) et logfreq(cible_t), sur **toutes les positions valides**, par texte,
IC 95 % par block bootstrap circulaire, test **intersection-union** sur A et B.

| Verdict | Règle |
| --- | --- |
| **Canal établi (H1a soutenue)** | ρ̂ ≤ −0.20 avec IC 95 % excluant 0, **sur A et sur B** |
| **Canal absent au sens du proxy** | les deux IC contiennent 0 avec \|ρ̂\| < 0.10, ou ρ̂ ≥ +0.10 sur un texte |
| **Inconclusif** | sinon — cause nommée : puissance (MDE ≈ −0.19) |

### Estimation rapportée — P2 / P2′ (jamais un verdict, sauf V0)

- **P2** : F sur logfreq. **P2′** : F′ sur le régresseur mécaniquement correct
  **logfreq + H**, bandes et bins identiques. L'écart F′ − F estime la part portée par
  le terme d'entropie.
- Estimateur primaire : OLS `impair ~ logfreq (+ H)` sur toutes les positions valides,
  F = β̂ · Δ(régresseur, confiantes, A−B) / Δ_raw. Sensibilité : post-stratification.
- **Mécanique bootstrap (A2)** : blocs circulaires de longueur L tirés sur la **série
  pleine** ; h_median et nombre de bins **fixés** hors réplicat ; sous-ensemble
  confiant, bornes de quintiles, Δ_raw, Δ_adj, E et F **recalculés dans chaque
  réplicat** ; graine 777, B = 2000 ; **IC de F invalidée si > 5 % des réplicats ont
  \|Δ_raw\| < 0.03** (rapporter alors Δ_raw, Δ_adj, E séparément).
- **P3, porte de l'estimation** : \|Δ_z(régresseur, confiantes, A−B)\| ≥ 0.15 SD. En
  dessous, F est un 0/0 : **non rapportée**.
- **Longueur de bloc** : L = max(25, ⌈3/(−ln ρ̂₁)⌉ arrondi à la dizaine supérieure),
  plafonné à N_full/10, ρ̂₁ = max(ρ₁(impair moyennée-seeds), ρ₁(logfreq)) sur la série
  pleine. Avec ρ₁ ≈ 0.06 : **L = 25**.

### Portes d'interprétation — P8 (bras fixe, CPU) et P8b (r̄ réel, GPU)

**P8** — reconstruction des directions du bras *fixe* par
`draw_eps(SEED_BASE["fixe"][text] + j, 1, d_model)` (Generator dédié, déterministe),
chargement de W_U en CPU **sans forward**, normes lues dans `metrics.rnorm` →
s_t = rnorm_t · (W_U[cible_t] · u_j). Régression `D_fixe ~ s_t + H_t`, 10 ou 20 runs à
directions **aléatoires indépendantes**.

**P8b (approuvé par le PI, §13.3)** — rejeu des 20 streams readM avec **r̄ persisté**
(`sum_r` / `rbar_unit` sérialisés), puis s*_t = rnorm_t · (W_U[cible_t] · r̄_unit) :
la congruence sur la **vraie direction de lecture**, pas sur des directions aléatoires.
Régression `impair_t ~ s*_t + H_t` sur les bruts ±r̄ existants. Soumis à la porte V6.

| Résultat | Lecture imposée du verdict P1 |
| --- | --- |
| P1 établi ET pente négative sur s (P8/P8b) | canal de congruence établi, **et** lexical au sens du proxy — lecture forte |
| P1 absent ET pente négative sur s | **le proxy est insuffisant, pas le canal** — « la fréquence marginale d'un corpus local ne capture pas la congruence », pas « il n'y a pas de canal » |
| P1 absent ET pente nulle sur s | H1a falsifiée dans les deux opérationnalisations — résultat fort, modérateur de repli activé |
| P1 établi ET pente nulle sur s | anomalie de chaîne à nommer (le proxy ferait mieux que l'exact) — analyse suspecte |

Plafond de F : rapport des R² (logfreq vs s*) → facteur d'atténuation à opposer aux
bandes de F. **Limite pré-déclarée de P8** : *fixe* n'a pas de bras antithétique, D_fixe
mélange pair et impair (H en covariable absorbe l'essentiel de la courbure) — test de
canal, pas mesure de F. **P8b n'a pas cette limite** (impair est antithétique par
construction), ce qui en fait la porte d'interprétation principale si V6 passe.

### Clause d'asymétrie (Neuro Q1) — porte l'interprétation de tout résultat bas

Le régresseur exact est contrastif (cible **contre** attente) et son terme de centrage
est celui qui dépend du texte ; logfreq n'en capture que ~23 % de la variance (r² ≈ 0.23).
**F et ρ sont donc biaisés vers 0 par construction.** Conséquence pré-enregistrée : un
ρ̂ ou un F élevé est une preuve forte ; **un ρ̂ ou un F bas ne falsifie pas H1 — il
falsifie « le canal est lexical au sens de la fréquence marginale d'un corpus local »**.
Aucune reformulation de cette clause ne sera admise après mesure.

### Descriptives obligatoires (rapportées quel que soit le verdict)

| # | Métrique | Prédiction Neuro (signée, pré-mesure) |
| --- | --- | --- |
| N-P1 | ρ_S(impair, logfreq), confiantes | −0.35 (A) / −0.30 (B), bande [−0.55, −0.15] ; **une pente plus plate qu'à toutes positions n'est PAS un échec** |
| **N-P1b** | impair aux positions **incertaines** vs confiantes | incertaines ≥ confiantes sur les deux textes, probablement > 0 ; point +0.15, bande [0.00, +0.35]. **Si impair était négatif partout, le récit « même mécanisme, bilan différent » perd son support** |
| N-P2 | F | ≈ 0.30, bande [0.10, 0.60] |
| N-P2′ | F′ | F′ − F ≈ +0.20 ; F′ ∈ [0.45, 0.85] |
| N-P3 | Δ logfreq(A − B) aux confiantes | **signe incertain (≈ 55/45), \|Δ\| petit — maillon faible reconnu** |
| N-P4 | corr(pair, logfreq) et corr(pair, H) | \|corr(pair, logfreq)\| < 0.15 **et** corr(pair, H) > +0.30 |
| N-P5 | modèle `impair ~ logfreq + H + logfreq×H` (une seule interaction) | deux pentes **négatives**, rapport dans [0.5, 2], **interaction ≈ 0** |
| D1 | décomposition du contraste conf/incertaines : β_logfreq·Δlogfreq vs β_H·ΔH | arbitre la tension N-P5 / N-P1b |
| D2 | **lecture H2 gratuite** : 4 blocs de ~110 positions par texte, dispersion **intra**-texte contre l'écart **inter**-textes | l'écart A−B excède-t-il la variabilité d'échantillonnage d'items intra-texte ? |
| D3 | Fisher-z poolé des deux textes | descriptif seulement |
| D4 | fraction de cibles à compte ≤ 2 ; sensibilité les excluant | Neuro, forme (ii) |

### Modérateur de repli, activé seulement si P1 échoue

**Taux de répétition en contexte** : fraction des positions confiantes dont le token
cible est déjà apparu plus tôt dans le même stream (fenêtre = tout le préfixe), coût
nul. Prédiction signée : une cible répétée est rare sous la distribution marginale mais
prédite avec confiance par le cortex (copie/induction) → logfreq + H très négatif →
**impair_t > 0**. Base-rate locale contre base-rate long terme. Les trois autres
candidats (entropie moyenne, mots-outils, burstiness) sont **rejetés par écrit** et ne
pourront pas être ressortis après coup.

### Sensibilités pré-déclarées (rapportées, jamais décisionnelles)

Post-stratification (quintiles, ESS de Kish) ; corpus narratif seul si V3 le commande ;
repondération jointe (H, logfreq) 2D — **signe seulement** ; variante
impair/(NLL_base + 0.5), **signe seulement** ; exclusion des cibles à compte ≤ 2.

**Diagnostics explicitement non décisionnels** : toute statistique conditionnée sur
NLL_base (D en est défini) ; **toute statistique utilisant le prior du cortex comme
régresseur** (endogène). **Interdit** : le bivarié impair ~ KL.

### E3

**Sans objet.** Aucune lecture modifiée, aucun défaut d'`EngramConfig` touché.
Tout changement de défaut consécutif — au premier chef l'ablation de la direction de
fréquence — passera par un protocole séparé avec **E3 ≤ +0.05 nats/token**.

## 5. Contrôles et baselines

1. **Configuration des bruts** : défauts `EngramConfig` sauf `read_gate="none"`, écart
   hérité de Q-01 et assumé. Aucun défaut modifié.
2. **M reset sur le même prompt (D7)** : re-vérifié depuis les JSON — bras synthétiques
   à `writes == 0`, `clear_context` asserté, zéro write sous hook.
3. **Placebo d'estimateur (V5)** : le pipeline entier sur impair_iid.
4. **Contrôle positif** : pente lexicale du bras *fixe*, centrée sur 0 entre tirages,
   non nulle par tirage.
5. **Contrôle « courbure »** : le terme pair ne doit pas porter le canal (N-P4).
6. **Contrôle « dérive ordinale »** : corr(impair, t) et corr(logfreq, t) ; P1
   recalculée après détrend linéaire en t (Δρ ≤ 0.05 attendu).
7. **Contrôle d'autocorrélation** : permutation en blocs des étiquettes logfreq.
8. **Contrôle de circularité lexicale** : A et B **exclus** du corpus — écart explicite
   et documenté vis-à-vis de `eval/marginal_pull.py::unigram_logfreq`, qui inclut A.
9. **Dé-duplication (V3b)** et **immutabilité** (manifeste SHA-256, §13.5).
10. **Porte du replay (V6)** : le rejeu readM doit reproduire l'ancre de Q-01.

## 6. Critères d'abandon

- **Tue H1a au sens du proxy** : les deux IC de ρ contiennent 0 avec \|ρ̂\| < 0.10, null
  par permutation non franchi. **Lecture obligatoire sous la clause d'asymétrie** :
  combiné à P8/P8b, cela falsifie soit le proxy, soit le canal.
- **Tue H1b** : F ≤ 0.25 avec IC excluant 0.50 — **uniquement si V0 a promu F**.
- **Invalide l'analyse** : V0 non mesurée ; V1 hors ±0.001 ou identités fausses ;
  V2 faux ; V3 < 0.80 après repli ; V4 faux ; **V5 violé** ; NaN. (V6 n'invalide que
  P8b, pas l'analyse.)
- **INCONCLUSIF nommé** : P1 en zone grise → cause = puissance (MDE ≈ −0.19). Mesure
  qui lèverait le doute : **Q-01c, ~2400-2500 positions confiantes ≈ 12 textes ≈
  1-1.5 h GPU**, qui départage du même coup la clause de sélection H2.
- **Interdits explicites** : ne pas redéfinir « confiantes » ; ne pas changer de corpus
  hors de l'ordre de repli ; ne pas tester une seconde interaction ; ne pas ressortir un
  modérateur rejeté ; ne pas réinterpréter la clause d'asymétrie.

## 7. Variables fixées

**Héritées des bruts (données, non modifiables)** : **gpt2**, layer **6**, seed **0**,
λ = **2.0**, cap = **0.5**, η = **0.2**, decay = **1e-3**, thr = **4.0**, dg =
**8192/64**, prune 512/0.10, **`read_gate="none"`** ; A = `NEUTRAL_TEXT`, B = `TEXT_B` ;
`FACT_TEMPLATE` + 10 `SECRETS` ; `SEED_BASE` = iid {A:1000, B:2000}, fixe {A:3000,
B:4000} ; partage confiantes/incertaines **à la médiane de H par texte**. Positions :
342 (A) / 450 (B) valides ; ~170 / ~225 confiantes ; N_eff ≈ 0.887·N.

**Fixées par ce protocole** : logfreq = unigramme **au niveau token BPE GPT-2**, compté
sur `data/rfc9293.txt` + `data/pnp_narrative.txt` + `data/pg1342.txt` **après
dé-duplication V3b**, **A et B exclus**, add-1 ; **un seul estimateur pré-désigné pour
tout le décisionnel**, repli complet → narratif → STOP ; quintiles de la logfreq poolée
des confiantes A ∪ B, fusion des bins < 15 ; B = 2000, graine 777, L par la règle
corrigée ; agrégation des 10 seeds **par moyenne de série avant statistique**.

**Clauses statistiques non négociables après coup** : (i) les 10 seeds sont des
quasi-doublons (P4 : 0.988) — leur σ ne servira **jamais** d'erreur standard ; (ii)
moyenne de série d'abord ; (iii) **intersection-union** pour la décisionnelle, Fisher-z
poolé en descriptif.

**Le multi-token est sans objet** : la cible d'une position **est** un token BPE ; les
tokens de continuation rares sont légitimement rares — propriété consignée, pas défaut.

## 8. Variable manipulée

**Une seule** : la congruence de la cible avec la direction injectée, sous **un**
estimateur pré-désigné (logfreq du corpus complet dé-dupliqué), avec **un** modèle à
**une** interaction. P2′ (logfreq + H), P8 (s_t, directions aléatoires) et P8b (s*_t,
direction réelle r̄) ne sont pas des variables supplémentaires : ce sont des
raffinements du **même** régresseur, tous dérivés de l'identité
impair ≈ −b·(logfreq + H + KL) fixée avant mesure. Le contraste A vs B est fixé par les
données existantes.

## 9. Budget

- **GPU : ~5 min** (décision PI §13.3) — rejeu des 20 streams readM + 20 injections,
  déterministes, pour persister r̄ ; VRAM < 2 Go. C'est le seul poste GPU.
- **CPU** : V0 (σ_pos) ~30 s ; re-tokenisation de A et B (`HF_HUB_OFFLINE=1`) ~5 s ;
  V3b ~30 s ; comptage unigramme ~10 s ; P1/P2/P2′ + bootstrap + permutations ~4-6 min ;
  **P8/P8b : chargement de W_U en CPU sans forward (~600 Mo transitoires) + régressions
  ~2-3 min**. **Total ~10-15 min CPU, RAM < 2 Go.**
- **Contingences chiffrées, NON autorisées par ce budget** : *Q-01c* (décisionnelle sur
  F et discriminateur de la clause H2) ~12 textes, 2400-2500 positions confiantes →
  **1-1.5 h GPU** — **pré-enregistrement demandé par le PI (§13.8), protocole séparé** ;
  *re-run complet de Q-01* si V1/V2 échouent : 143 streams, 607 s.

## 10. Livrables attendus

- **Flag `EngramConfig`** : **aucun**. Moteur `engram/` strictement intact.
- **Correctif de sérialisation** : `eval/perturb_position.py` doit persister `sum_r` /
  `rbar_unit` **avant** `save_run_json` (dette identifiée par Math et Neuro : le bug qui
  a rendu r̄ irrécupérable). Correction minimale, sans changer le comportement mesuré —
  V6 le prouve en reproduisant l'ancre.
- **Script** : `eval/lexical_congruence.py` (nouveau, SPDX AGPL-3.0-or-later), **lecture
  seule** sur `experiments/results/specificite-dommage-incertaines/runs/*.json` ;
  options `--runs-dir --corpus --texts --bins --boot --with-instrument --rbar-file` ;
  refuse de tourner si V0/V1/V2 échouent ; docstring portant l'interdiction d'instancier
  `EngramEngine` et l'interdiction du régresseur « prior du cortex » (endogène) ;
  sorties dans `experiments/results/congruence-lexicale/{manifest_sha256.json,
  analysis.json, summary.csv}`.
- **Tests CPU** (sans réseau ; P8 testé sur un W_U factice) : (1) identité impair + pair
  = D(+r̄) ; (2) post-stratification exacte — F = 1.00 ± 1e-6 quand A et B ne diffèrent
  QUE par la distribution du régresseur, F = 0 quand elles sont identiques ; (3)
  déterminisme du bootstrap (graine 777) et **recalcul effectif des bornes de quintiles
  dans le réplicat** ; (4) **règle de bloc** : ρ₁ = 0.06 → L = 25, ρ₁ = 0.9 → L croît,
  plafond N/10 respecté ; (5) comptage unigramme reproductible **et** exclusion effective
  de A/B ; (6) détection de chevauchement V3b ; (7) **assert d'alignement** : un décalage
  volontaire de la cible d'un cran fait échouer V2 ; (8) reproduction bit-exacte des
  directions du bras *fixe* par `draw_eps` ; (9) échec attendu de V2 sur un JSON tronqué ;
  (10) `rbar_unit` sérialisé et rechargeable, norme 1 à 1e-6.
- **Entrée de journal** au format standard, avec V0-V6 / P1 / P2-P2′ / P8-P8b /
  N-P1..N-P5 / D1-D4 telles qu'approuvées, et le **vocabulaire imposé**.
- **Ligne pour `docs/EXTENSIONS.md` §4**, adossée à la ligne Q-01 — rédigée en
  interprétation, appliquée par le PI.
- **Correction de l'entrée Q-01** (« biais de critère » → « biais de prior additif sur
  les log-odds ») : **approuvée par le PI (§13.6)**, rédigée pour lui.

## 11. Questions résiduelles pour Neuro (à traiter en interprétation)

1. **Tension interne** : N-P5 prédit β_H < 0 tandis que N-P1b prédit impair(incertaines)
   ≥ impair(confiantes), probablement > 0 — compatibles seulement si le terme logfreq
   domine le contraste entre strates. D1 est pré-enregistrée pour l'arbitrer : quelle
   valeur du rapport confirme ta théorie plutôt qu'elle ne la sauve ?
2. Si P1 échoue mais que P8/P8b établit le canal, le modérateur de repli reste-t-il le
   bon candidat, ou la lecture devient-elle « le proxy est trop grossier » ?
3. KL est omis du régresseur : quel indice observable dirait que l'omission n'est pas
   anodine ?
4. Si N-P4 échoue (le terme **pair** porte aussi la logfreq) : canal contaminé ou
   artefact d'échelle ?
5. Sous quelle condition exacte l'ablation de la direction de fréquence est-elle
   *déclenchée* : N-P5 seule, P8b seule, ou les deux ?
6. Le garde-fou (a) — « si le gain E2 est lui-même de l'amorçage vers le fréquent,
   l'ablation le détruira » — se teste-t-il sur les bruts existants, ou exige-t-il E2 ?
7. Confirmer la grille §4 telle que rédigée.

## 12. Questions résiduelles pour Math (à traiter en interprétation)

1. **Vérifier le recalcul de MDE** : SE(ρ) ≈ 1/√(0.887·N) = 0.057 (A) / 0.050 (B) →
   MDE ≈ −0.19 sur toutes les positions. Le seuil ρ̂ ≤ −0.20 tient-il, ou faut-il le
   rehausser ?
2. Le passage de « confiantes » à « toutes positions valides » change-t-il la règle de
   bloc ou l'un des garde-fous de A2 ?
3. P8 agrège 10-20 pentes non appariées à directions indépendantes : quelle inférence,
   et faut-il corriger le partage des normes `rnorm_t` avec le bras readM ?
4. Si le nombre de runs *fixe* est 20 plutôt que 10, quelque chose change-t-il dans P8
   ou V5 ?
5. Le rapport des R² (logfreq vs s*) comme « plafond de F » : estimateur défendable ou
   ordinal seulement ?
6. Si V0 rend σ_pos entre 0.17 et 0.20 : règle binaire, ou bande d'indécision à
   pré-enregistrer maintenant ?
7. **P8b** : `impair ~ s*_t + H_t` sur la vraie direction — quelle inférence exacte, et
   le partage de r̄ entre les 10 secrets (une seule direction, 10 profils de norme)
   crée-t-il une dépendance à traiter ?
8. Q-01c : ~2400 positions confiantes depuis ~12 textes courts ou moins de textes plus
   longs ? La réponse change le budget et la structure de réplication.

## 13. Décisions du PI (2026-08-21, gate de pré-enregistrement)

1. **Changement de cible : VALIDÉ.** Q-01b est lancé en connaissance de l'issue la plus
   probable (« canal établi, part expliquée non résolue »).
2. **Corpus** : comptage BPE hors A/B sur les corpus locaux, **complet dé-dupliqué**
   (V3b) en primaire, repli complet → narratif → STOP ; aucun autre estimateur
   décisionnel. L'option « prior du cortex » reste retirée (endogène). Écart assumé et
   documenté vis-à-vis de `eval/marginal_pull.py::unigram_logfreq`.
3. **Chargement de W_U en CPU : AUTORISÉ** (P8). **Et les ~5 min GPU sont ACCORDÉES**
   pour persister r̄ et tester la congruence sur la **vraie direction de lecture**
   (P8b) — le cycle n'est donc plus « GPU zéro » mais « ~5 min GPU », sous la porte V6.
   Bénéfice durable : r̄ cesse d'être perdu pour les cycles suivants.
4. *(fusionnée dans 3)*
5. **Immutabilité** : **manifeste SHA-256** consigné dans l'entrée de journal, avant
   analyse. `.gitignore` inchangé.
6. **Vocabulaire** : l'entrée de journal Q-01 sera **corrigée** (« biais de critère vers
   le prior » → « biais de prior additif sur les log-odds »). Le labo rédige la
   correction, le PI l'applique.
7. **Ablation de la direction de fréquence** : le **déclencheur conditionnel est écrit
   dans `docs/EXTENSIONS.md`**, armé par un verdict positif de N-P5 ou P8/P8b, avec les
   deux garde-fous de Neuro (risque de détruire le gain E2 ; ne pas survendre le résidu
   readM − (+r̄)). Reste orthogonal aux trois candidats V2-D.
8. **Feuille de route** : Q-01b est **inscrit dans `AUDIT-lab.md`** comme question
   suivie ; il **passe avant Q-03** ; **Q-01c est à pré-enregistrer maintenant**
   (protocole séparé, ~12 textes, 1-1.5 h GPU) ; **Q-05 attend le verdict de Q-01b**
   avant sa requalification (déjà pré-approuvée).

## Historique

- 2026-08-21 : brouillon du Directeur
- 2026-08-21 : avis Neuro (favorable, 2 amendements) et Math (réservé → favorable si A1)
- 2026-08-21 : protocole consolidé, décisions du PI intégrées — proposé
- 2026-08-21 : pré-enregistré par le PI
- 2026-08-21 : arbitrages PI en cours de cycle (résolution d'ambiguïté) — V5 partiellement
  échouée → P8 requalifié descriptif non fiable, P8b seule porte ; V3 non franchie →
  analyse poursuivie, proxy affaibli consigné
- 2026-08-21 : terminé, verdict REJETE (H1a falsifiée dans ses deux opérationnalisations ;
  le signe positif est une propriété de la borne ΔNLL, pas un canal inverse)
