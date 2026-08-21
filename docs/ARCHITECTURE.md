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
