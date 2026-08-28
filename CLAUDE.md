# engram — PoC hippocampe/néocortex (fast weights à test-time)

## Ce que c'est

PoC de recherche personnelle (Jean) : une architecture à deux étages inspirée des
*complementary learning systems* —

- **Néocortex** : un petit LLM gelé (GPT-2 124M pour itérer vite, SmolLM2-360M ensuite).
- **Hippocampe** : une matrice de fast weights `M` (d×dg_dim avec la projection
  gyrus denté, le défaut ; d×d en mode dense), branchée par hook PyTorch à une
  couche intermédiaire, lue à travers un gate keysim (X8), mise à jour
  **pendant l'inférence** par delta rule (hebbien corrigé,
  zéro backprop), avec gating par surprise (NLL en ligne), decay, élagage top-k, et reset.

Objectif : montrer qu'un module plastique minuscule permet à un modèle gelé de retenir de
l'information **hors du cache KV** (apprentissage pendant l'épisode, pas du in-context).
Ce n'est PAS un produit — c'est une expérience falsifiable sur un laptop RTX 3060 (6 Go).

## Lire en premier

- `docs/ARCHITECTURE.md` — le design complet : maths, choix de conception, pièges connus.
- `docs/EXTENSIONS.md` — la méthode d'évolution (pas-à-pas mesuré) : échelle des
  mécanismes X0→X5, déclencheurs, tableau des poids. Rien n'entre dans le code sans ça.
- `docs/JOURNAL.md` — journal d'expériences (à tenir à jour : chaque run notable y va).
- `README.md` — vue d'ensemble + protocole d'évaluation.
- `docs/VISION.md` — cas d'usage cibles, chacun avec prérequis (numéros
  d'expérience) et conditions d'échec. Licence : AGPL-3.0-or-later (LICENSE,
  SPDX dans chaque .py — à reporter sur tout nouveau fichier source).
- `docs/POSSIBLE_APPROACH.md` — notes brutes de Jean (mécanismes bio) ; formalisées
  dans EXTENSIONS.md, ne pas modifier.
- `docs/AFTER_v1_THOUGHTS.md` — relecture critique post-v1 (écrite AVANT v1.1, ne pas
  modifier). Son pari « Hebb = E3 catastrophique » a été falsifié par v1.1 ; ses deux
  pistes reprises dans la feuille de route : E2 narratif (n-grammes vs adaptation) et
  I1 (prédicteur d'échec par similarité de clés). Point mis en avant : le ratio de
  généralisation 0.68 est le chiffre-titre du PoC (mémoire associative, pas un
  grep) — recalculé avec incertitude le 2026-08-21 (COR-02) : IC 95 % [0.56,
  0.99], médiane par secret 0.59 [0.45, 0.75] ; à citer « ~0.6–0.7, N=10 ».

## État du projet (2026-08-26)

- Squelette v1 posé et **validé sur GPU** (torch 2.13+cu126 ; GPT-2, SmolLM2 et
  Qwen2.5-1.5B en cache). Parcours complet dans JOURNAL.md : X0 → X1 (gyrus
  denté) → E3 → X1b → E1b/E2 → ablations v1.1 → E2n → I1 → v1.2 SmolLM2 → X7 →
  X9 → X8 (banc + validation) → E4/E4s → campagne Qwen → X8.1/X8.1b/P5.
- **Point de fonctionnement courant = défauts de config (verdict X8,
  2026-08-21)** : dg=8192/64, **read_gate=keysim, λ=2.0, cap=0.5**, η=0.2 →
  GPT-2 : E1 +1.353 ± 1.58, E3 −0.014 ✓ ; SmolLM2 (layer 16) : E1 +0.755 ± 0.94,
  E3 +0.003 ✓. L'ancien point conforme sans gate (λ=1.0, cap=0.25 → E1 +0.740,
  E3 +0.023) est la **référence d'ablation `read_gate="none"`**.
- Acquis v1 (détail : JOURNAL, tableau EXTENSIONS §4) : E1 rappel post-vidage KV
  sur trois modèles ; E1b généralisation 0.68 au point X1b (IC 95 % [0.56, 0.99],
  médiane par secret 0.59 — recalcul COR-02, journal 2026-08-21 ; 0.38 sous gate —
  coût de sélectivité, calibration `gate_keysim_mid` par modèle à faire) ; E2
  adaptation réelle (−0.055 RFC, −0.030 fiction ; depuis X8, la métrique honnête
  est le ΔNLL absolu par moitié) ; E3 sous seuil. Toujours 0/10 top-10 : de
  l'amorçage, pas du rappel fonctionnel.
- Ablations v1.1 : le gating porte ~92 % de l'effet E2 ; DG +57 % ; l'avantage
  Hebb sur E2 RFC s'évapore sur la fiction (E2n) → delta rule par défaut, D5
  quasi close (runs uniques, variance non estimée — Q-02/Q-09 de l'audit).
- I1/X9 : keysim = jauge de saturation de RÉGIME ; pas de falaise de capacité à
  80 faits (91 % de rétention) — d² est un problème théorique ; prédicteur par
  fait mort (corrélations par fait n.s. à N=10).
- X7 (mesuré) : aplatissement = coût FIXE (+0.141 nats d'entropie), lecture SANS
  composante directionnelle (cos ≈ 0) → rappel par recomputation indirecte.
  C'est LE mur commun : top-10, E1c, E4/E4s.
- X8.1 → X8.1b → P5 : anomalie entropie résolue — le dommage de lecture vit aux
  positions incertaines du cortex (corr +0.394) ; **loi 2 : gater côté mémoire,
  jamais côté détresse du cortex** ; two_factor enterré. **Q-01 (2026-08-21,
  /lab-run, RETENU)** : ce ciblage est GÉNÉRIQUE — un bruit de norme appariée le
  reproduit à R ≈ 0.8 sur deux textes ; la direction quasi constante de la
  lecture (invariant du modèle, orientée prior) fixe le signe aux positions
  confiantes. Décision **D11** (ARCHITECTURE §3) : tout canal s'évalue en
  perturbation aux positions incertaines. Référence bio : Salzman, Britten &
  Newsome 1990 — la façade Hasselmo est tombée (Hasselmo ne vaut plus que pour
  X3/Q-05, à requalifier avant lancement).
- Famille conventions : E4/E4s = critère non atteint (pas de préférence
  token-niveau) ; E4-dur RETIRÉ (il mesure la fluidité) ; Qwen : E4s signes
  conformes n.s., E1 fumée +4.18 (N=2, régime X8 — non comparable aux points
  historiques, Q-08 de l'audit).
- **Audit externe 2026-08-21 : voir `AUDIT-lab.md`** — 22 corrections traitées,
  6 questions priorisées prêtes pour /lab-run (+ 2 en réserve). **v2 : chantier
  prioritaire = V2-D**, course à trois candidats : kNN-LM nu (instrument, **fait**)
  → M_out sur les logits (principal, **non ouvert**) → Fast-KV — fiches et
  écartements dans EXTENSIONS.md ; sommeil/LoRA basse priorité actée (verdict X9).
- **Cycle méthode D14-ext (2026-08-22) — `H_méthode` REJETÉE.** Le banc de
  satisfiabilité `eval/gate_bench.py` (D14-S) est né et devient **obligatoire à
  vie** : sur sept défauts trouvés avant toute mesure, **deux n'étaient décidables
  que par exécution** et **deux ont échappé aux trois passes d'auto-vérification
  ET au premier tour d'experts**. Aucune relecture n'aurait suffi ; le banc coûte
  ~6 s de CPU. Fait le plus réutilisable : **deux fois de suite, la porte qui a
  trouvé le défaut est celle qui n'avait besoin d'aucune donnée** (`V-slot` a tué
  l'ancienne para1, `V-ident` a tué `fact_pairs(30)`).
- **V2-D(a) v3 (2026-08-22) — `INCONCLUSIF`, confondant d'instrument.** `knn_k = 8`
  pour un store de 8-11 entrées ⇒ la cible est dans les 8 voisins sur 90/90
  requêtes : **pas d'étape d'adressage dans le circuit mesuré**. P1 sort à 28/30
  (F) et 30/30 (L6) contre un seuil de succès de 12/30, et **ne veut rien dire** :
  le plancher sous **clé de bruit** vaut 12/30 et 23/30, et en L6 la clé d'une
  **autre** unité rend 30/30 avec un **vecteur de succès identique** (contribution
  de la clé exactement nulle). Arrêt formel conforme par P3[L6] (médiane 4/30).
  **Deux acquis, indépendants de la clé** : (i) **le budget E3 n'est PAS le verrou
  du top-10** — V2 30/30 et P1-exact 30/30, une hypothèse explicative du 0/10 tenu
  depuis X0 est éliminée ; (ii) **divergence des deux nulles** — détruire la
  liaison clé→valeur rend 3-4/30, détruire l'adresse en gardant le contenu rend
  12-23/30 : l'effet est porté par **l'appartenance de la cible au store**, pas par
  la récupération. **Rien** n'est acquis sur H, sur l'étage de clé, sur la couche 6,
  ni sur V2-D — la branche B de la table d'attribution est logiquement inaccessible.
  Le critère d'ouverture du candidat (b) **n'est pas franchi au sens de son intention**.
- **Décisions gravées le 2026-08-22** (ARCHITECTURE §3) : **D12** (la ΔNLL borne le
  gain, pas le dommage) · **D13** (toute prédiction signée énumère son antipode) ·
  **D14** (clôture des portes) · **D14-S** (satisfiabilité machine, le banc) ·
  **D14-R** (provenance des chiffres) · **D15** (la borne de dommage appartient à la
  convexité du mélange, pas à l'étage des logits) · **D16** (*toute primaire est une
  différence, jamais un taux* — et « la couche 6 bat l'état final » devient
  **interdite**) · **D17** (une nulle bloquante par maillon) · **D18** (partition
  exhaustive des clauses à seuils ; **E vaut sur le banc seul, pas sur le run**) ·
  **D19** (couloir de faisabilité `C-null`) · **D20** (une clause d'audit exige un
  plancher à produire, jamais un suspect à innocenter).
- **v4-matériel — TERMINÉ le 2026-08-23. Matériau qualifié `RETENU` ; primaire
  `C-ind` par SATURATION ; `ORD-3` ×3.** Protocole
  `experiments/EXP-2026-08-23-v4-materiel.md`. Cellule conjointe **`N-b × C-ind`**,
  l'issue modale gravée d'avance : *« l'instrument est sain, la question posée est
  restée sans réponse »*. `R1 = 1.0000` sur 180/180 × 3 modèles ⇒ **`Σ m_q = 0`** :
  un seul fait, deux affichages — `P(m_q = 0 ∀q) ≈ 10⁻⁹⁴` sous échangeabilité,
  **≈ 1** sous le transcript lexical. `ΔR1_inv = 1 − R1_null` **exactement** :
  aucun dépassement du plafond `36/37`. **Trois acquis** : `B0′ = 0` (tige et
  suffixe **physiquement séparables**) ; `A4` (le porteur du domaine **s'évanouit**
  quand il quitte le préfixe causal) ; et le **théorème d'incompatibilité** —
  *sur tout matériau où l'entité cible figure verbatim dans le préfixe causal de la
  requête, `m_q = 0` est un théorème* : composition des intrusions et identité du
  préfixe sont **incompatibles**, ce qui **ferme une famille entière de protocoles**.
  **58 défauts fermés (0-46 … 0-103) avant qu'un octet de matériau n'existe**, dont
  10 + 9 + 6 + 2 par **quatre passes d'audit indépendant**.
- **Cycle « recouvrement des supports de `topk(G·h)` » — TERMINÉ le 2026-08-26.
  Verdict `REJETE` ; maillon centrage `NON ATTRIBUÉ` (0-155).**
  (`experiments/EXP-2026-08-23-recouvrement-supports.md`.)
  Analyse **CPU pure** sur les états conservés de v4 : `M` jamais instanciée, aucun
  forward, `engram/` non modifié. Primaire = **`Δ* = O_plac − O_type`** (nulle **0
  par construction**), niveau = `Λ = O_type − 2n` sur **corridor absolu**.
  *Historique de l'arrêt initial, conservé —* **le run s'est d'abord arrêté à la deuxième porte**,
  en **9,1 s de CPU**, sans une mesure :
  **`V-G` FAIL** ⇒ **`A3` n'a jamais été calculé avec la `G` du projet** (`numpy`
  contre `torch`, `corrcoef ≈ 0.011`), et le §3 **nommait la mauvaise colonne**
  (« 0.41-0.46 » = la **compression**, pas le cosinus ; le vrai vaut **0.147-0.337**).
  **39 défauts fermés (0-104 … 0-142)**, aucune mesure conduite. Correction de
  provenance autorisée sous **D30**, traçabilité au **§15**. Borne serrée **`p_sym`**
  adoptée sous **D30 alinéa 2** (§16) : **`C-sep` re-fermée sur 12/12, avec la bonne
  `G`**. **`Q-M6` livrée**, puis **trois tours de mesure, tous `APPROVED`** (ré-exécution bit-à-bit,
  ré-implémentation indépendante par un troisième chemin, table `cum` recoupée par un quatrième ;
  banc `dgov` **47 clauses / 119 cas / `E = 0`** ; **< 30 min CPU, 0 GPU** ; défauts **0-104 …
  0-156**). **Résultat** : classes `N` identiques 3/3 (S3 `HAUT`, S2 `BAS`, S1 `ind_L`, S0 `BAS`)
  ⇒ **`C-strat` réalisé, la lecture forte d'`A3` tombe une SECONDE fois**, indépendamment de 0-135.
  `O ≥ p_sym` 12/12 est une **identité au poids probant nul** ; l'excès libre passe de
  **+2.44 ± 0.34** indice à **+0.51 ± 0.38** sous retrait par type (**facteur 4.8**). `Core-G`
  = 1/1, 1/1, **`SANS OBJET`** ; les **12 indices les plus partagés sont dans `top64(G·μ_global)`
  sur 3/3**. `auto ≈ aucun` **partout** ⇒ **l'explication LayerNorm/RMSNorm de `C-mod` est morte**
  et le seul retrait calculable **en ligne** ne mord pas. `N7` **fermée** (`f` enrichie de 6-10 %
  seulement). **`¬c-mec` et `¬c-anti` attribués sous toute règle licite (D33)** ; l'attribution
  `c-cent`/`ind_Δ` **suspendue** par **défaut de protocole** (D31). **`σ±` sous le hasard sur les S0 :
  `NON EXPLIQUÉ`** — *l'ampleur en σ publiée à l'origine est **rectifiée** le 2026-08-27 (0-170,
  D30 al. 1) : la nulle binomiale supposait des essais indépendants, le `N` effectif n'est pas
  établi. Aucune phrase qualifiant l'ampleur avant la porte `V-grappe`.* Le **+57 % d'E2 de X1 reste acquis et intact** (0-75) ; son
  **attribution** ne s'instruira **pas** par des mesures de support à ce locus. **Suite priorité 1 :
  Étape A CPU pure (`P-Base` + `P-σ`).**
- **File d'attente (ordre ratifié le 2026-08-26)** : **priorité 1 — Étape A**, CPU pure, mêmes
  hashes, `< 10 min` : **`P-Base`** (`|A∩B∩top64(G·μ_global)|/|A∩B| ≥ 0.50`, 4 strates × 3 modèles ;
  antipode **`C-Base-mort`** `< 0.25` sur ≥ 2/3) et **`P-σ`** (antipode `C-σ-mort`). `P-Base`
  **produit le plancher que D20 exige pour D32**. **Priorité 2 — `Q-N3`** : re-spécifier la primaire
  `Δ*` (permutation exacte, ou placebo à **directions indépendantes par état**) sous
  pré-enregistrement neuf ; rendement **plafonné par (xx)**. **Priorité 3 — `Q-quant`**, clause
  d'attente **levée**, éligible. **`V2-D(b)`** (`M_out` sur les logits) s'ouvre **après le verdict
  d'Étape A**. **`I3` reste conditionné à `P11`** — la purge écrite des deux réserves d'`I2`.
- **Instrument I2 (`layer_profile`) — `PROPOSE`, deux avis RÉSERVÉ non traités.**
  Profilage par couche (score contrastif d'invariance + entropie matricielle), un
  forward instrumenté, aucune injection. Défauts à corriger avant implémentation :
  le score en **ratio mesure l'anisotropie** autant que l'invariance (remplacer par
  l'**AUC par couche**, invariante par transformation monotone) ; l'inférence
  « P-A falsifiée ⇒ P6 falsifiée » est **illégitime** ; le contrôle `NEUTRAL_TEXT`
  n'est **pas apparié**. Protocole : `experiments/EXP-2026-08-22-layer-profile.md`.
- Runs : SmolLM2 `--model HuggingFaceTB/SmolLM2-360M --layer 16` ; Qwen
  `--model Qwen/Qwen2.5-1.5B --layer 14`.

## Registre des formulations interdites (en vigueur)

*Les vocabulaires interdits vivent éparpillés dans les protocoles ; **ce fichier est leur point de
rassemblement** — c'est ici qu'un agent qui rédige cherche son registre. Chaque entrée nomme sa
source ; le protocole d'origine fait foi sur le détail.*

- **Plafond lexical (xiii, v4)** — `ΔR1_inv ≈ 36/37` **ne se lit JAMAIS** *« le modèle retrouve
  l'unité »*. Formulation obligatoire : ***« compatible avec un encodage de surface ; le canal
  identité n'est pas adressé par cette primaire »***. *(Un plafond atteint n'est pas un succès.)*
- **« Sémantique » (xv, réserve `lab-neuro`)** — le mot est **interdit**. Maximum licite :
  ***« appartenance au sous-vivier déclaré »***. La mesure ne distingue pas le domaine sémantique de
  l'identité lexicale de l'ensemble dont les tokens sont tirés (défaut 0-67 : `C7` **crée** cette
  corrélation).
- **Interdit d'étage (xiv, 0-75)** — **rien sur le chemin d'ÉCRITURE depuis une mesure de LECTURE**.
  Le **+57 % d'E2 de X1** vit sur l'écriture ; il **reste acquis** et aucune mesure de lecture ne le
  confirme ni ne l'entame. Seule son **attribution** peut être en jeu.
- **Interdit d'ordre (0-63)** — les modèles **ne sont pas des réplicats** : forwards déterministes,
  un seul tirage de matériau. **La borne d'une conjonction est le `min`, jamais le produit** ; **tout
  produit de p-valeurs par modèle rend le run invalide**. Toute conjonction passe par un
  **bootstrap joint** sur le **même** rééchantillon de clusters.
- **« Séparation de patterns » (ix, 0-59)** — interdit d'appliquer la formule **à un run** qui mesure
  `h` brut sans instancier `M`. Réservée à la **motivation historique de X1**.
- **Le centrage (xviii, `lab-neuro`)** — **aucune désignation biologique** : ni « inhibition
  tonique », ni « normalisation divisive », ni « retrait de mode commun par les interneurones ». Le
  centrage est un **instrument statistique** ; formulation licite : ***« retrait d'une composante
  affine estimée hors ligne, en leave-one-out, sur les états du même type »***.
- **Support vide (D23, 0-74)** — une quantité sans domaine de définition se publie ***`SANS OBJET`,
  jamais `0`***. Un support vide **ne licencie aucune borne**, pas même une équivalence TOST.
- **Classe d'équivalence (D28)** — elle dit ***« effet borné par 2 × la résolution »***, **jamais**
  « pas d'effet ».
- **Indécidable (0-71, 0-85)** — *« augmenter la résolution »* n'est pas une suite par défaut : la
  **cause doit être nommée** (puissance, support, précision) avant la suite.
- **Bin dur (A-5)** — interdit **sous toute forme, y compris adverbiale** : « tendance »,
  « suggère », « va dans le sens de ».

## Environnement

- Windows 11, laptop RTX 3060 6 Go, Python 3.12.
- venv dans `.venv/` ; deps dans `requirements.txt` (torch **cu126** via extra-index
  pytorch.org — pas cu124, voir le journal du 2026-08-20 : repli CPU silencieux).
- transformers v5 : utiliser `dtype=` (pas `torch_dtype=`) dans `from_pretrained`.
- Installer : `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt -e .`
- Tests (CPU, sans téléchargement HF) : `.venv\Scripts\python -m pytest tests/ -q`

## Conventions du projet

- PyTorch pur + `transformers` HF pour le cortex. Pas de framework d'entraînement :
  il n'y a **aucun backprop** dans ce projet (v1) — c'est le point.
- `M` reste en fp32 même si le cortex est en fp16 (stabilité des mises à jour).
- Tout hyperparamètre passe par `EngramConfig` (`engram/config.py`) — pas de constantes
  en dur dans le code.
- Chaque expérience → une entrée datée dans `docs/JOURNAL.md` (config, résultat, conclusion).
- Les décisions de design non triviales se documentent dans `docs/ARCHITECTURE.md`,
  section « Décisions », avec leur justification — ce fichier est la mémoire du projet.

## Vocabulaire

- **cortex** : le LLM gelé. **hippocampe / M** : la matrice de fast weights.
- **write** : une mise à jour delta rule de M. **read** : l'injection `λ·M·φ(h)` dans le
  flux résiduel. **surprise** : NLL du token observé sous les logits courants.
- **clear_context** : vider le cache KV en gardant M — l'opération clé des évals.
- **reset** : M ← 0 (« espace latent presque vierge »).
