# Architecture — engram v1

Ce document est la référence de design du PoC. Toute décision non triviale doit y être
consignée avec sa justification (section « Décisions »). Si le code et ce document
divergent, c'est un bug de documentation à corriger immédiatement.

## 1. Vue d'ensemble

```text
                    ┌─────────────────────────────────────────┐
                    │  Cortex (LLM gelé, fp16, aucun gradient) │
  token ──────────▶ │  blocs 0..L-1                            │
                    │      │ h_t  (flux résiduel, couche L)    │
                    │      ▼                                   │
                    │  ┌────────────── hook ────────────────┐  │
                    │  │ stash h_t (pré-injection)           │  │
                    │  │ h_t ← h_t + λ·g · M · φ(h_t) (READ) │  │
                    │  └─────────────────────────────────────┘  │
                    │      │                                   │
                    │  blocs L..N-1  →  logits                 │
                    └─────────────────────────────────────────┘
                           │
                           ▼  token suivant observé
                    surprise = NLL(token | logits)
                    si surprise > seuil :
                        M ← M + η·(v − M·k)·kᵀ        (WRITE, delta rule)
                        avec k = φ(h_{t-1}),  v = h_t   (tous deux PRÉ-injection)
                    M ← (1−δ)·M                        (decay, à chaque write)
                    périodiquement : élagage top-k     (sparsification)
```

- La boucle est **token par token** (cache KV actif) : lent mais uniforme, et c'est ce
  qui rend l'apprentissage réellement « en ligne ». Le débit n'est pas un objectif v1.
- `M` est d×key_dim : **d×dg_dim avec la projection gyrus denté, le défaut**
  (GPT-2 : 768×8192 ≈ 6,3M paramètres, ~25 Mo fp32), ou pleine d×d en mode dense
  X0 (768×768 ≈ 590k, 2,4 Mo). C'est le seul état mutable du système en dehors
  du cache KV.
- La lecture est **gatée** (X8, défaut `read_gate="keysim"`) : g ≈ 0 hors
  domaine, g ≈ 1 sur un match mémoire — voir §2.1 et la décision D10.

## 2. Les maths, pièce par pièce

### 2.1 Lecture (rappel associatif, gatée)

`h ← h + clip(λ · g · M · φ(h))`

- `φ(h)` : clé de requête, normalisée L2 — `h/‖h‖` en mode dense,
  `normalize(topk(G·h))` avec la projection gyrus denté (§2.2b, le défaut).
  Rôle : borner l'énergie des clés, rendre η interprétable (les écritures ont
  toutes la même échelle côté clé).
- `g ∈ [0, 1]` : **gate de lecture X8** (défaut `read_gate="keysim"`) — sigmoïde
  raide sur le cos max entre φ(h) et le buffer de clés I1 : la lecture ne s'ouvre
  que si la requête ressemble à quelque chose que M a réellement écrit
  (pertinence côté mémoire). Verdict X8 (journal 2026-08-21) : E1 intact, E3
  éliminé, régime agressif rouvert (λ=2, cap=0.5) ; coût : ratio paraphrases
  0.68 → 0.38. Le facteur entropie est disqualifié et le two_factor enterré
  (X8.1/X8.1b/P5) : le dommage de lecture est concentré aux positions
  incertaines du cortex — un gate déclenché par l'incertitude lit exactement là
  où lire coûte. **Loi 2 : gater côté mémoire, jamais côté détresse du cortex.**
- `λ` (`cfg.lam`) : force d'injection dans le flux résiduel. Trop grand → on écrase le
  signal du cortex et le modèle divague ; trop petit → pas d'effet mesurable.
- `clip` : la norme du vecteur injecté est plafonnée à `cfg.max_read_norm` fois la norme
  de h. Garde-fou de stabilité n°1 : même si M devient énorme, l'injection reste bornée.
  Mesuré le 2026-08-20 (journal, X1b) : le cap agit comme **gating doux** — les
  récupérations pertinentes le saturent, les parasites restent dessous ; λ contrôle le
  bruit, le cap le signal. Sous le gate X8, le cap reste en **plancher de
  sécurité** (décision D10) : en domaine, g ≈ 1 laisse passer λ=2 plein — le cap
  borne l'injection dans tous les cas. Écho biologique corrigé (audit
  2026-08-21) : l'étiquette « acétylcholine » posée ici était une analogie de
  façade — dans le modèle cholinergique (Hasselmo), l'ACh haute en régime de
  nouveauté favorise l'encodage et SUPPRIME le rappel récurrent : son analogue
  mesuré est la loi 2 (X8.1b/P5), pas le cap.

### 2.2 Écriture (delta rule, PAS Hebb pur)

`M ← M + η · (v − M·k) · kᵀ` avec `k = φ(h_{t-1})`, `v = h_t`

- Le terme correctif `−M·k` est ce qui distingue la delta rule du Hebb pur (`+η·v·kᵀ`) :
  elle n'écrit que **l'erreur de prédiction de la mémoire**, pas la corrélation brute.
  Conséquences : pas d'accumulation divergente quand la même association revient, et
  capacité de **réécriture** (nouvelle valeur pour une clé connue remplace l'ancienne au
  lieu de s'y additionner). C'est la réponse au défaut « corrélation ≠ utilité » du
  hebbien, sans introduire de backprop : la règle reste locale et bon marché.
- Sémantique **prédictive** (clé = état précédent, valeur = état courant) : M apprend des
  transitions d'états latents. Au rappel, un état courant ressemblant à un ancien
  h_{t-1} injecte un « souvenir » de ce qui avait suivi — c'est le mécanisme qui peut
  changer les prédictions après vidage du cache KV.

### 2.2b Projection gyrus denté (X1, optionnelle — `dg_dim` > 0)

`φ_dg(h) = normalize(topk(G·h, dg_topk))` avec `G` (dg_dim × d) aléatoire **gelée**,
seedée par `cfg.seed`. M devient rectangulaire d × dg_dim (768×8192 ≈ 25 Mo fp32).

- **Pourquoi** : en mode dense, deux états latents proches (cos ≈ 0.9+, fréquent dans
  un même registre) écrasent mutuellement leurs associations — c'est l'explosion de
  variance observée au balayage X0. La projection haute dimension + top-k orthogonalise
  brutalement : deux clés proches en dimension d partagent peu de composantes après
  sparsification. C'est la séparation de patterns du gyrus denté biologique.
- **Bonus** : capacité théorique ~dg_dim associations (au lieu de ~d), writes creux
  (l'outer product ne touche que dg_topk colonnes).
- **Coût** : un produit G·h par lecture/écriture — négligeable à cette échelle.

### 2.3 Gating par surprise

`surprise_t = NLL(token_t | logits_t)` — gratuit, on a déjà les logits.

On n'écrit que si `surprise > cfg.surprise_threshold`. C'est le « je me rends compte que
ça ne colle pas, j'apprends maintenant ». Repères : NLL en nats ; GPT-2 sur de l'anglais
courant tourne autour de 3–4 nats/token, un token vraiment imprévisible est > 6–8.
Le seuil est un hyperparamètre central des ablations (gating vs toujours-écrire).

### 2.4 Oubli : decay + élagage

- **Decay** : `M ← (1−δ)·M` appliqué à chaque write (pas à chaque step : une mémoire
  qui n'écrit rien ne doit pas s'évaporer pendant une longue lecture facile).
- **Élagage** : tous les `cfg.prune_every` writes, on ne garde que la fraction
  `cfg.prune_keep` des coefficients de plus grande magnitude (le reste ← 0).
  C'est la version pauvre de la consolidation pendant le sommeil : on ne garde que les
  traces fortes. (La vraie consolidation — distiller M dans un LoRA — est v2.)

### 2.5 Coût et capacité de M

Le scaling de M est en k·d² (clés en dimension k·d via DG, valeurs en d) et c'est
**assumé** : à d = 4096 et k = 4, M ferait ~64M paramètres / ~270 Mo fp32 —
négligeable devant le cortex correspondant. La capacité EST la raison d'être du
quadratique, exactement comme pour le gyrus denté biologique (l'expansion
dimensionnelle est le mécanisme, pas un accident d'implémentation). Corollaire acté
dans EXTENSIONS §X9 : pas de factorisation de M (block-diagonal, Kronecker, rang
faible) par défaut — elles plafonnent la capacité que le quadratique achète, et les
updates rang-1 de la delta rule ne vivent pas sur la variété de Kronecker.

Sortie de secours documentée si d ET le nombre de souvenirs persistants explosaient
un jour : la mémoire à slots (lignée **kNN-LM / Memorizing Transformers** —
attribution corrigée à l'audit 2026-08-21 : Titans n'est PAS une mémoire à slots
mais une mémoire neuronale à gradient test-time, le plus proche parent de M, et le
contrôle naturel de D6 : surprise par gradient vs NLL). Compromis connu à ne pas
oublier : les
slots perdent la **superposition** — or le ratio de généralisation ~0.68 (E1b) est
probablement une propriété de la superposition distribuée ; une mémoire à slots
rappellerait mieux l'exact et moins bien la paraphrase.

### 2.6 Reset

`M ← 0`. Le cache KV et M sont indépendants : `clear_context()` vide le cache en gardant
M (c'est l'op des évals), `reset_memory()` fait l'inverse.

## 3. Décisions de design (et leurs raisons)

| # | Décision | Raison |
| --- | --- | --- |
| D1 | Clés/valeurs = états latents **pré-injection** | Éviter la boucle de rétroaction M→h→M : si on écrivait les états post-injection, M apprendrait ses propres sorties (divergence quasi garantie). |
| D2 | M en fp32, cortex en fp16 | Les incréments `η·(v−Mk)kᵀ` sont petits ; en fp16 ils partent en underflow et la mémoire n'apprend rien. 2,4 Mo, le coût est nul. |
| D3 | Couche d'insertion L au milieu du réseau (défaut : 6/12 pour GPT-2) | Trop tôt : représentations pas assez sémantiques (quasi lexicales). Trop tard : plus assez de blocs après L pour intégrer l'injection dans les logits. À balayer empiriquement — `cfg.layer_index`. |
| D4 | Boucle token-par-token, pas de prefill parallèle | L'écriture au pas t dépend de l'état de M au pas t-1 : le traitement parallèle d'un prompt casserait la causalité des writes. Uniforme et honnête, au prix du débit (acceptable : PoC). |
| D5 | Delta rule plutôt que Hebb pur | Voir 2.2. Hebb pur gardé comme ablation (`cfg.hebbian_only`). **Nuance (2026-08-21, journal)** : Hebb bat la delta sur E2 (−0.095 vs −0.055, et ce n'est pas un effet de η) car le terme correctif éteint les writes répétés — delta reste le défaut pour son bornage et sa capacité de réécriture ; Hebb = régime de forte adaptation, arbitrage final sur E2 long horizon. |
| D6 | Gating par NLL plutôt que par norme d'erreur latente | La NLL est le seul signal qui mesure une erreur **de tâche** (prédire le token) et elle est gratuite. La norme de `v−Mk` mesure la nouveauté pour M, pas l'utilité — gardée comme critère secondaire possible. |
| D7 | `eval/` vide le cache KV avant le rappel | Sans ça, impossible de distinguer la contribution de M du in-context learning ordinaire. C'est LE contrôle qui rend le PoC falsifiable. |
| D8 | Pas de backprop nulle part en v1 | C'est l'hypothèse testée : une règle locale + un cortex gelé suffisent-ils à un effet mesurable ? Introduire du gradient brouillerait la réponse. |
| D9 | G (gyrus denté) aléatoire gelée, jamais apprise, seedée par cfg.seed | L'orthogonalisation ne demande aucun apprentissage (Johnson-Lindenstrauss fait le travail) ; une G apprise exigerait du gradient (contredit D8) ; la seed fixe garantit les mêmes clés d'un run à l'autre (comparabilité). Top-k par magnitude et non ReLU+top-k : conserve l'information de signe, deux fois plus de motifs distincts. |
| D10 | Le cap `max_read_norm` reste en **plancher de sécurité sous le gate X8** (pas remplacé par lui) | En domaine, g ≈ 1 laisse passer λ=2 plein : le cap borne l'injection dans tous les cas, gate ouvert ou fermé. Tranche le « point de vigilance » noté dans EXTENSIONS §X8 ; référencée par `config.py` et `hippocampus.py`. Actée à l'implémentation X8 (2026-08-21) ; entrée ajoutée à l'audit (COR-06). |
| D11 | La détresse du cortex (entropie, incertitude) est **proscrite comme signal d'ouverture** d'une lecture ou d'une injection ; tout canal, actuel ou futur (V2-D compris), s'évalue comme **perturbation aux positions incertaines** (dommage par position, pas seulement en moyenne) | Q-01 (journal + protocole pré-enregistré, 2026-08-21) : le ciblage du dommage aux positions incertaines est générique — un bruit de norme appariée le reproduit à R ≈ 0.8 sur deux textes — et la direction quasi constante de la lecture (invariant du modèle, orientée prior) en fixe le signe aux positions confiantes (+r̄ reproduit le profil à 0.993). Remplace l'ancrage Hasselmo de la loi 2 par Salzman, Britten & Newsome 1990, *Nature* 346 (6280), 174–177 (microstimulation de MT : biais des jugements vers la direction encodée par le site stimulé, décalage de courbe psychométrique — effet maximal sur le choix près du seuil ; traitement quantitatif : Salzman et al. 1992, *J. Neurosci.* 12(6), 2331–2355). |
| D12 | La métrique ΔNLL **borne le gain** (D_t ≥ −NLL_base) **mais pas le dommage**. Toute association entre un effet ΔNLL et une covariable corrélée à NLL_base (\|r\| > 0.3) doit, avant toute lecture mécaniste, survivre à l'appariement par décile de NLL_base **et** à la restriction à la zone non mordante (NLL_base ∈ [1, 3] nats). | Q-01b (journal 2026-08-21) : l'écart répétées/nouvelles s'effondre de +0.32 à +0.004/+0.05 sous appariement, et la pente positive disparaît en zone non mordante. Pendant de D11, côté métrique. |
| D13 | Toute **prédiction signée** pré-enregistrée énumère explicitement la **cellule de l'antipode** (le résultat de signe opposé, robuste) et la conséquence qui lui est attachée, avant mesure. | Q-01b : la grille §4 n'énumérait que la cellule nulle et la confirmatoire — le run est passé par le trou. Sans cellule d'antipode, un résultat inverse se retrouve sans lecture pré-enregistrée, donc exposé à l'improvisation. |
| D14 | **Clôture des portes.** (a) Toute porte ou seuil pré-enregistré porte sa **dérivation chiffrée** et son hypothèse de régime. (b) Après tout amendement changeant un régime (paramétrage, mécanisme prédit, distribution attendue), **chaque** porte antérieure est **re-dérivée numériquement** sous le nouveau régime, et la re-dérivation figure dans l'arbitrage. (c) Une porte d'**intégrité** se conditionne sur le sous-ensemble où sa prédiction est **exacte**, jamais sur un agrégat qui mélange l'identité et un phénomène prédit. (d) **Aucun amendement de porte après le premier token de données lu** — deux portes incohérentes ⇒ run invalide, on re-pré-enregistre. | V2-D(a) run 1 (journal 2026-08-21) : deux portes calibrées sous un régime écarté par la consolidation ont invalidé le run, dont une qui comptait comme preuve de non-mélange un phénomène que le protocole prédisait. *Un pré-enregistrement n'est pas une somme de clauses, c'est un système : le tableau d'arbitrage garde la trace de ce qui entre, rien ne garde celle de ce que chaque entrée périme.* |
| D14-S | **Satisfiabilité machine des portes.** Toute porte pré-enregistrée est accompagnée de la preuve qu'elle est **satisfiable ET falsifiable dans l'arithmétique effective** (fp32/fp64 selon la chaîne) : le protocole exhibe (i) une entrée synthétique qui la fait passer, (ii) une entrée synthétique qui la fait échouer, (iii) l'analyse ULP aux points où la quantité testée peut coller à son bord. Une porte strictement stricte (« 100 %, < ») sur une quantité qui atteint son bord à un ULP près est **interdite d'écriture**. Le banc (i)+(ii) est un test CPU livré AVANT le run. | V2-D(a) v2 (journal 2026-08-22) : la porte V1b, dérivée correctement du théorème, était **logiquement vraie et machinalement fausse** — insatisfiable en fp64 dès que le décrément réel est sous un ULP. *Le run 1 est mort d'un seuil sans dérivation ; le run 2 d'une dérivation sans arithmétique.* |
| D14-R | **Provenance des chiffres.** Tout chiffre cité dans un protocole — porte, prédiction **ou justification de design** — porte l'une de deux étiquettes : *re-dérivé* (la dérivation figure dans le protocole) ou *re-mesuré depuis les bruts* (commande + hash). Un chiffre recopié d'une **analyse** antérieure est proscrit ; un chiffre issu d'un **run invalidé** l'est doublement. Les acquis de régime se re-calculent depuis `raw/`, jamais depuis le journal. | V2-D(a) v2 : l'acquis « d²_min = 0.0 bit-à-bit » du run 1 était faux (réel : 0.00449220464) et a contaminé une justification de design (courbe G) sans passer par une porte — la règle §3 protégeait les portes, pas les justifications : trop étroite d'un cran. |
| D15 | La **borne de dommage** d'un canal de sortie appartient à la **convexité et à la stochasticité du mélange**, pas à l'étage d'injection. Une innocuité mesurée sur un canal **interpolant** (kNN-LM : `p = (1−λ)p_LM + λp_kNN`, d'où `ΔNLL ≤ −log(1−λ)`) **ne se transporte à aucun canal additif non borné** — au premier chef M_out en biais de logits (`logits += g·W_U·(M_out·φ(h))`), qui peut envoyer p(y_t) → 0. Tout candidat porte **son propre** E3. | V2-D(a) v2, avis Neuro (F2) : propriété manquante = la stochasticité du mélange. Sans cette clause, un succès d'innocuité de (a) serait transporté à (b) par abus, et le chantier v2 partirait sur une garantie qui n'existe pas. |
| D16 | **Primaire en différence, jamais en taux.** Toute primaire de récupération s'écrit `Δ = métrique(clé correcte) − métrique(clé nulle appariée)`, mesurée dans le même run : un **taux de détection sans taux de fausse alarme** est interdit comme statistique décisionnelle. Le plancher sous clé nulle se dérive **analytiquement avant le run** depuis la géométrie mesurée du store et doit être strictement inférieur au seuil avec marge — **porte bloquante**, sinon le run ne part pas. **Corollaire de vocabulaire** (remplace et durcit l'interdiction §2 (ii) de v3) : *un écart entre bras n'est pas un écart de mémoire tant qu'il survit au retrait de la clé.* La phrase « la couche 6 bat l'état final » est **interdite** ; seule formulation autorisée : « sous cet instrument, le bras L6 accepte davantage d'entrées du store que le bras F sous toute clé, y compris nulle — un **écart de discrimination** (familiarité non diagnostique, Norman & O'Reilly 2003), pas de récupération ». | V2-D(a) v3 (journal 2026-08-22) : P1 = 28/30 et 30/30 au-dessus du seuil de succès 12/30, alors que le plancher sous **clé de bruit de norme appariée** vaut 12/30 (F) et 23/30 (L6) — et qu'en L6 la clé d'une **autre unité** rend 30/30 avec un **vecteur de succès identique**, soit une contribution de la clé exactement nulle (d′ = 0). |
| D17 | **Une nulle bloquante par maillon.** Tout protocole de mesure identifie sa chaîne causale (données → clé → sélection → pondération → mélange → mesure) et attache une **nulle bloquante à chaque maillon**. Un maillon sans nulle est un **défaut de protocole**, pas une économie. | V2-D(a) v3 : le protocole avait une nulle pour les **données** (V-para/V-slot/V-ident) et une pour la **liaison clé→valeur** (P3) ; aucune pour les maillons **« clé »** et **« sélection des k voisins »** — c'est exactement là que le run est mort. Les deux nulles divergent d'un ordre de grandeur sur ce run (3-4/30 vs 12-23/30) : preuve interne qu'elles ne sont **pas substituables**. |
| D18 | **Partition exhaustive des clauses à seuils.** Toute clause à seuils définit une partition **exhaustive** de son espace d'issues, chaque classe nommée avec son verdict (PASS / INCONCLUSIF-cause / invalide), et le banc D14-S exécute **un cas par classe** : la **totalité de la fonction** est un item de couverture, pas une évidence. | V2-D(a) v3 : la médiane de P3[L6] est tombée dans la région **[4, 11]**, que ni le §4.4 ni le §6 ne spécifiaient et qu'aucun contre-exemple du banc n'exerçait. Et `P1-cellule-dégénérée`, conditionnée à l'ensemble des échecs de P1, est **vacuée par ensemble vide** quand P1 = 30/30 — la définition exacte de **E**, sur un banc certifié `E = 0`. **E vaut sur le banc seul, pas sur le run.** |
| D19 | **Couloir de faisabilité (`C-null`).** Un test de H déclare **ses deux bornes** avant la passe décisionnelle : faisabilité **supérieure** (`p₁₀ < λ/(1−λ)` — le canal *peut* réussir à pleine masse) **et** séparation **inférieure** (`p₁₀ > (λ/(1−λ))·m/s`, soit `s_i ≥ 4·λ·m_i/((1−λ)·p₁₀,i)` — le canal *ne peut pas* réussir sans clé). Les deux sont décidables **sans donnée nouvelle**. **Réduire `k` ne suffit pas** : à clé nulle, l'espérance de masse sur la cible reste ~`m/s` ; le levier est `s`, ou l'exclusion du membre plat de la cellule décisionnelle. | V2-D(a) v3 : V2 vérifiait la borne **supérieure** (30/30 sous 0.0512711) et **aucune clause** la borne inférieure. Avec `knn_k = 8` pour un store de 8-11 entrées, le gain sous clé aveugle vaut `0.0512711/8 = 0.00641` contre une médiane p₁₀ de **0.005806** : ~la moitié des unités passent **sous n'importe quelle clé**. Il aurait fallu **s ≈ 36**. |
| D20 | **Une clause d'audit exige un plancher à produire, jamais un suspect à innocenter.** Forme correcte : « si la primaire dépasse tel niveau, le run n'est pas interprétable avant que la même primaire ait été mesurée sous la **version nulle de la variable manipulée**, à cellule, λ, store et code identiques. » Aucun nom de cause. **Une clause d'audit qui peut innocenter est plus dangereuse qu'une absence de clause.** | V2-D(a) v3, auto-consigné par Neuro : sa clause N11 (`n_L6 ≥ 25/30` ⇒ audit de fuite) s'est déclenchée et a prescrit un audit **lexical** — qui rend 0 violation, donc un **blanc-seing erroné**. *On audite ce dont on vient de souffrir* : le cycle venait de passer deux amendements sur le recouvrement lexical. Sans l'initiative du Verifier, N11 aurait légitimé le résultat. |
| D21 | **Tout dtype de la chaîne de mesure est épinglé au pré-enregistrement, forward inclus.** Un dtype non écrit est une **variable manipulée par le matériel**, pas une constante. | I2 (journal 2026-08-23) : le §7 fixait cosinus/Gram en fp32 et valeurs propres en fp64, **mais pas le forward** ; le code a choisi fp16 sur CUDA. **La seule variable numérique libre du protocole est exactement celle qui a déclenché la clause d'invalidation** — 1920 NaN dans le bloc final sous noyau SDPA batché, reproductibles bit-à-bit, absents en fp32. |
| D22 | **Les critères d'abandon sont une porte EXÉCUTABLE, pas une section en prose.** Chaque clause du §6 a un test machine qui **bloque l'écriture du rapport** ; un drapeau d'invalidation **calculé puis ignoré** est lui-même un motif d'invalidation du pipeline. | I2 : le drapeau `nan_ou_inf: true` a été correctement calculé et écrit dans les bruts, **puis ignoré** ; `rapport.json` ne contient **aucune** section d'invalidation et le pipeline a rendu un verdict de bande comme si la clause n'existait pas. Les portes du §4.7 étaient exécutables, celles du §6 ne l'étaient pas. |
| D23 | **Les réductions statistiques sont NaN-strictes, et la portée des clauses binaires se grave avant mesure.** Toute réduction (AUC, IC, bootstrap) **propage les non-finis ou s'arrête** : l'absorption silencieuse est proscrite. Toute clause binaire d'invalidation précise **sa portée** (états capturés / quantités intermédiaires / quantités publiées) au pré-enregistrement. | I2 : `auc_par_couche` avalait les NaN (comparaison `NaN > x` → `False` ⇒ paire comptée comme défaite), ce qui a **permis au run de continuer** ; `ic_du_max` filtrait les échantillons bootstrap non finis en rapportant `B` inchangé. Et la clause « NaN / inf » du §6, non qualifiée, a dû être lue **après coup** — inconditionnellement, faute de portée gravée. |
| D24 | **Toute nulle publie son cardinal de séquences distinctes en porte bloquante avant run.** Une nulle qui perd sa variabilité ne borne plus rien : elle a l'apparence d'un plancher sans en être un. | I2 : la nulle de cadre s'est réduite à **5 séquences distinctes sur 80 lignes** pour para1 (verbe global + deux slots de contenu remplacés), avec des lignes **byte-identiques**. C'est le mode de mort du défaut 0-8, que le §5 avait été **réécrit pour réparer**, réapparu sur un autre type — **deuxième occurrence : c'est un motif de porte, plus un incident**. |
| D24-b | **Tout cardinal de séquences se calcule sur le PRÉFIXE CAUSALEMENT VISIBLE à la quantité mesurée, jamais sur la séquence entière.** Toute nulle **et toute condition de contraste** publient leur cardinal de séquences distinctes **tronquées au point de capture** ; et **tout contraste décisionnel** (variante, type, identité) doit être **réalisé avant ce point** dans tous les cadres — propriété du matériau vérifiée au banc (D25), jamais convention d'usage. *En LM causal, ce qui suit la capture n'existe pas pour l'état capturé : un cardinal calculé sur la séquence entière est une **fiction**.* | Cadrage v4-matériel (journal 2026-08-23) : la variante intra-type proposée était un **adjoint circonstanciel**, naturellement **postposé** ⇒ les deux variantes d'un même état auraient été **bit-identiques au point de capture**, donnant `R1_ident = 1` **par arithmétique et non par représentation**. **C'est le mode de mort de D24 sous une forme que D24 ne voit pas** — D24 compte les séquences complètes, et elles étaient bien distinctes. Trouvé avant qu'un seul octet de matériau n'existe. |
| D25 | **Un matériau est qualifié pour une classe d'instruments, jamais « bon » en soi — et la qualification se DÉRIVE de propriétés vérifiées, jamais de l'usage historique.** Tout jeu de données porte, **dans sa source**, la **table des propriétés testées par le banc** (indépendance longueur/unité ; existence de paires intra-unité intra-type ; diversité de type ; survie à l'effacement de casse ; absence de période sur tout slot et tout couple de slots ; cardinal des nulles ; …), chacune avec son résultat et sa porte. Un instrument **déclare ses prérequis** ; la compatibilité matériau × instrument se **vérifie mécaniquement** contre cette table, et une incompatibilité est un **arrêt**, pas un avertissement. **Une liste blanche d'instruments autorisés, entretenue à la main, est proscrite** : elle vieillit comme un acquis de régime recopié sans re-dérivation. | Cycle I2 / v4 (journal 2026-08-22 et 2026-08-23) : `pool.fact_pairs`, **conçu pour le rappel** (E1, X9) où ses symétries combinatoires sont inoffensives, a coûté **quatre tours de portes et aucune mesure propre** en géométrie représentationnelle, où **les mêmes symétries fuient par tous les canaux** — période 20 sur (entity, verb) ; dominance de type rendant la bande M **structurellement inaccessible** ; appariement de longueur **insatisfaisable 16/16** ; verbe **parfaitement confondu** avec une strate ; et les **240 paires intra-unité toutes inter-types**, confondant identité et invariance **1:1 à tout N**. **Durcissement du PI (2026-08-23)** : *« la leçon des quatre tours n'est pas "`fact_pairs` était mauvais", c'est "**personne n'avait écrit ce que `fact_pairs` garantissait**" — D25 doit graver les garanties ; la liste des instruments n'en est que l'ombre portée. »* |
| D26 | **La difficulté ne se supprime pas, elle MIGRE — donc toute purification du matériau se mesure DES DEUX CÔTÉS de la purification, jamais d'un seul.** Chaque fois qu'une condition est assainie (pool purifié, strate écartée, canal fermé), le confondant ne disparaît pas : il se déplace **vers l'endroit que le protocole ne regarde pas encore**. La seule réponse stable est la **double mesure** : la quantité purifiée **et** son homologue non purifié, chacune avec sa nulle et son `K` (D17). Une primaire seule, du côté propre, mesure la propreté et non le phénomène. | Cinquième occurrence de la même structure dans le projet, et à la cinquième elle cesse d'être une astuce locale : **signal/bruit de X7** ; **les deux nulles de v3** ; **la scission `N-a`/`N-b`** (un verdict unique recouvrait deux mondes dont l'un était faux) ; **les deux pools de v4** (`C5` purifié / surface-apparié) ; **le plancher lexical `V-lex`**. Le cas générateur est `C5` (protocole v4-matériel, défaut 0-45) : la nulle était inexacte par **dominance de surface dans le pool** ; purifier le pool a restauré l'exactitude **et rendu la cible accessible lexicalement** — la difficulté avait migré de la nulle vers le contrôle, en un seul pas. *Au même rang méthodologique que l'arrêt dur (D22) et le banc de satisfiabilité (D14-S).* |
| D27 | **Quand un confondant résiste à N purifications, cesser de le chasser : SCINDER l'instrument en un CALIBRATEUR qui l'absorbe et une QUESTION définie HORS de son domaine.** Le calibrateur a le droit d'être dominé par le confondant — **il ne décide rien** ; il ne fournit que la nulle exacte, le plancher, et la santé de l'instrument. La question, elle, se pose sur une quantité **indéfinie là où le confondant règne** : son plancher n'est pas *dominé*, il est **hors du domaine de définition** — ce qui est mathématiquement plus fort que toute borne. *C'est la solution GÉNÉRALE de la loi de migration (D26) : après six occurrences où la difficulté chassée d'un endroit se logeait dans le suivant, la réponse stable n'est pas une purification de plus, c'est une scission des fonctions.* | Cycle v4-matériel (journal 2026-08-23). Six migrations en un cycle : `C5` purifie le pool ⇒ la difficulté passe au **contrôle** ; `V-lex` attrape la migration ⇒ **`V-lex` est saturée par la même construction** (`ΔR1_lex = 36/37` **est le plafond**, pas un plancher comparable) ; la composition remplace `V-lex` ⇒ le **canal suffixe** y devient structurellement invisible. Formulation de Neuro sur le cas extrême : *« la difficulté ne migre pas vers un endroit qu'on regarde, elle migre vers un endroit qu'on ne PEUT PAS regarder »*. **Réalisation dans v4** : la primaire 1 (`ΔR1_inv`, plafonnée à `36/37` par simple lecture lexicale) devient **calibrateur** et perd toute bande ; la composition des intrusions (`X\|m ~ Hypergéom(36, 5, m)`) porte **seule** la question — et sous le lecteur lexical à 0 forward, `m_q = 0` pour **toute** requête, donc l'excès est **indéfini**. *« Un maillon qui mesure le confondant, un maillon qui mesure l'effet »* (PI) : D17 assumée jusqu'au bout. |
| D28 | **Un couloir d'équivalence réglé sur l'enveloppe nulle de son PROPRE estimateur est structurellement inatteignable — ET AUGMENTER LA RÉSOLUTION EN ÉLOIGNE.** L'inclusion `IC ⊂ [−c, +c]` exige `\|estimé\| ≤ c − hw` ; si `c` est l'enveloppe nulle et `hw` la demi-largeur du **même** estimateur, le seuil vaut **≈ 0** et la classe n'est atteinte que par **sous-estimation bruitée de `σ̂`** — un artefact. Et, `c` et `hw` étant tous deux `∝ σ/√K`, **`P(classe d'équivalence)` tend vers 0 quand `K` croît**. **Règle** : marge de **significativité** = 1× l'enveloppe nulle ; couloir d'**équivalence** = **2×**. Et une classe d'équivalence ne dit **jamais** « pas d'effet » : elle dit **« effet borné par 2× la résolution »** (lecture de type TOST). | **La seconde moitié est la découverte, et c'est elle qui coûte cher : la sortie intuitive est la mauvaise direction.** Le réflexe universel devant un test qui ne conclut pas — *plus de données* — est **précisément le geste qui enfonce**. Un protocole ainsi réglé n'a **pas de mode de succès négatif** : son monde nul rend « indécidable » au lieu de « rien de détectable ». Défaut 0-81 du cycle v4 (journal 2026-08-23), trouvé par audit indépendant puis dérivé par `lab-math` : sous la spec initiale, `C-ind` était **modal à 54-67 % sous la nulle** ; sous le schéma 1×/2×, la classe d'équivalence redevient modale (**92.5-93.2 %**, vérifié par simulation) et **stable en `K`**. Coût du durcissement, **chiffré et déclaré, jamais absorbé** : le seuil de détection du seul maillon décisionnel passe de ~26 % à ~52 % d'enrichissement relatif. |
| D29 | **La CORRECTION d'un défaut est une ÉCRITURE, et elle produit des défauts au même taux que l'écriture initiale.** Corollaire exécutable : **tout correctif de plus de 10 lignes repasse le circuit de relecture COMPLET, pas le circuit allégé** — et en deçà du seuil, **la relecture ciblée reste obligatoire, jamais nulle**. Corollaire de personnel : **le relecteur ne doit pas avoir écrit l'objet relu** — l'auteur qui se relit trouve les défauts qu'il savait chercher. | **Chiffré sur le cycle v4** (journal 2026-08-23) : **19 lignes réécrites → 9 défauts ; 14 → 6 ; 11 → 2.** Le folklore (« le fix introduit le bug suivant ») devient un taux mesuré. Et ce ne sont pas des broutilles : **deux défauts nés de correctifs portaient sur une décision d'ARRÊT** — dont `B0′`, où **le seul détecteur de fuite capable de mordre était absent de la porte exécutable**. *Justification formelle du relecteur indépendant* : sur un protocole déjà validé par deux experts et par la session principale, l'auteur relisant son propre texte avait trouvé **3** défauts ; le lecteur n'ayant pas écrit les tables en a trouvé **10 de plus, dont 4 critiques**. **Seuil de 10 lignes** : politique calibrée sur trois points, **pas une loi** — le décrochage observé n'est pas le taux mais la **gravité** (les défauts nés des 11 lignes n'affectaient aucun routage ; ceux nés des 14 et des 19 comprenaient des décisions d'arrêt). |

## 4. Pièges connus (à surveiller dès les premiers runs)

1. **Divergence auto-entretenue** : M s'auto-renforce via ses lectures. Défenses en
   place : D1 (pré-injection), clip de lecture, decay, η petit. Symptôme : perplexité
   qui explose, génération qui boucle. Premier réflexe : baisser λ puis η.
2. **Mémoire fantôme** : un Δlog-prob positif en fact injection qui viendrait d'un
   artefact (tokenisation du secret, position, prior du modèle sur X). Défense : le
   protocole compare M actif vs M reset **sur le même prompt exact**, et tourne sur
   plusieurs secrets/formulations tirés au hasard.
3. **Gating muet** : seuil trop haut → zéro write, résultats trivialement nuls. Les
   évals loggent le nombre de writes ; un run avec 0 write est invalide, pas négatif.
4. **fp16 silencieux** : si un tenseur fp16 se glisse dans la chaîne d'update, la
   mémoire n'apprend rien sans erreur visible. Les casts sont explicites dans
   `hippocampus.py` ; ne pas « simplifier » ça.

## 5. Évaluations (protocole exact)

### E1 — Injection de fait (`eval/fact_injection.py`)

1. Choisir un fait arbitraire : « The password is ⟨secret⟩. » (secret = mot rare tiré
   d'une liste, pour minimiser le prior du cortex).
2. `reset_memory()` ; streamer le fait avec écriture **forcée** (le gating est un
   mécanisme de production ; pour l'éval d'injection on veut mesurer la capacité de M,
   pas la politique de gating — celle-ci a sa propre ablation).
3. `clear_context()` — le cache KV disparaît, M survit.
4. Streamer la question « The password is » et relever log-prob(⟨secret⟩).
5. Mesurer la même log-prob après `reset_memory()` (même prompt, M=0).
6. Répéter sur N secrets. Métrique : Δlog-prob moyen ± écart-type, et le taux de cas
   où le secret devient top-1/top-10.

### E1b — Variante paraphrase (extension de E1)

Même protocole que E1, mais la question de rappel est une reformulation du fait
injecté (« The secret code happens to be », etc.). Mesure le rappel par indice
*partiel* — c'est le symptôme déclencheur de l'extension X2 (CA3) : si
E1-exact ≫ E1b-paraphrase, la mémoire fait du par-cœur, pas de l'association.

### E2 — Dérive de domaine (`eval/domain_drift.py`)

1. Prendre un long texte technique homogène (> 4k tokens), hors du registre de
   pré-entraînement courant.
2. Le streamer 2 fois dans 2 conditions : M actif (gating normal) vs M gelé à zéro.
3. Métrique : NLL moyenne par moitié de document, et surtout l'**interaction**
   (baisse 2ᵉ moitié avec M) − (baisse 2ᵉ moitié sans M). Le cache KV étant borné ou
   vidé entre chunks, la baisse attribuable à M est isolée.

**Note de métrique (X8, 2026-08-21 ; répercutée à l'audit, COR-14)** : au point X8
(gate keysim, régime agressif), le coût d'échauffement de la 1ʳᵉ moitié disparaît —
M aide dès le début et l'interaction tombe mécaniquement à ~0. Rapporter AUSSI le
ΔNLL **absolu** par moitié vs contrôle : l'interaction était conçue pour une
mémoire qui paie avant de gagner.

### E3 — Dommage collatéral (`eval/collateral.py`)

Le scénario d'échec le plus probable n'est pas « ça ne marche pas », c'est « ça
marche mais ça rend le modèle plus bête ailleurs » — le dilemme stabilité/plasticité
mesuré en direct. Ni E1 ni E2 ne le voient : E3 comble cet angle mort.

1. Charger M via le protocole E1 (injection d'un fait, write forcé).
2. Mesurer la NLL moyenne sur un texte **neutre** (registre générique, sans rapport
   avec le fait), lecture active, écriture coupée.
3. Comparer à la même NLL avec M = 0 (même texte, même contexte).
4. Métrique : Δ NLL/token. Un coût > ~0,05 nats/token est un signal d'alarme ; le
   reporter dans le tableau des poids (EXTENSIONS.md §4) pour chaque mécanisme.

## 6. Extensions et suite de la feuille de route

La méthode d'évolution du PoC (pas-à-pas, un mécanisme à la fois, poids chiffré par
ajout) et l'échelle complète des mécanismes candidats — projection gyrus denté,
lecture itérée CA3, alternance theta, reset néogenèse, substrat épisodique kNN —
sont formalisées dans **`docs/EXTENSIONS.md`**. Les notes brutes d'origine sont dans
`docs/POSSIBLE_APPROACH.md`.

Ordre convenu (2026-08-20) : **run baseline naïve d'abord** (X0, les chiffres de
référence), puis X1 (gyrus denté) + E3, chacun benchmarké séparément.

## 7. Hors scope v1 (notes pour v2)

- **Rappel directionnel (V2-D, LE chantier v2 prioritaire — acté 2026-08-21)** :
  la réponse au mur X7 (zéro composante directionnelle : top-10, E1c, E4s),
  devenu une **course à trois candidats** (2026-08-21) — kNN-LM nu en
  instrument de plafond d'abord, M_out sur les logits (candidat principal),
  Fast-KV (contexte fantôme KV) — chacun soumis au contrat zéro-gradient (D8)
  et à D11 (évaluation en perturbation aux positions incertaines). Fiches
  d'intention et options écartées : EXTENSIONS.md, entrée V2-D.
- **Sommeil / consolidation — basse priorité actée (verdict X9 : pas de
  falaise)** : distiller périodiquement le contenu de M dans un LoRA
  du cortex, puis reset de M. Spec retenue : replay **génératif depuis M** (pas les
  données brutes) — voir EXTENSIONS.md, entrée « V2 — Replay / sharp-wave ripples ».
  C'est là que l'oubli catastrophique revient.
- **Projections apprises** W_k, W_v (au lieu de l'identité) — demanderait du gradient.
- **M multi-têtes / par bloc** (plusieurs mémoires à des couches différentes).
- **Sparsité structurée de M** (blocs, pour préfigurer un mapping hardware réaliste).
