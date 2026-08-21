# Extensions — mécanismes candidats et protocole d'ablation

Formalisation de `POSSIBLE_APPROACH.md` (notes brutes conservées telles quelles).
Ce document définit **comment on avance** : une modification à la fois, benchmarkée
contre la version précédente, avec un « poids » chiffré par ajout. Rien n'entre dans
le cœur du code sans être passé par ce protocole.

## 1. Méthode : le pas-à-pas mesuré

1. **Baseline d'abord** : la version naïve (X0) tourne sur toute la suite d'évals
   avant tout ajout. Ses chiffres sont la référence absolue du tableau §4.
2. **Un mécanisme à la fois** : jamais deux ajouts dans le même run. Chaque extension
   est un flag de config (défaut = désactivé) pour rester ablatable à vie.
3. **Suite d'évals fixe** : E1 + E2 + E3 (voir ARCHITECTURE.md §5), mêmes seeds,
   mêmes hyperparamètres que la baseline sauf le mécanisme testé.
4. **Poids d'un ajout** = ses deltas sur les trois évals, consignés dans le tableau §4
   et dans le journal. On garde un mécanisme si son gain sur E1/E2 justifie sa
   complexité ET qu'il n'aggrave pas E3 (le dommage collatéral).
5. **Un symptôme avant un remède** : les mécanismes marqués « repli » ne s'implémentent
   que si leur symptôme déclencheur est observé. Pas d'ingénierie spéculative.

## 2. L'échelle des extensions

Chaque entrée : origine biologique → problème visé → déclencheur → esquisse → coût.

### X0 — Baseline naïve (référence)

Delta rule pleine matrice, φ = normalisation L2, gating NLL, decay + top-k, reset.
C'est l'état actuel du code. Aucun flag.

### X1 — Projection gyrus denté (séparation de patterns) — *assurance v1, pas un repli*

- **Bio** : le gyrus denté projette l'entrée dans un espace beaucoup plus grand et
  très sparse avant écriture hippocampique — orthogonalisation pour que deux
  expériences proches ne s'écrasent pas.
- **Problème visé** : interférence entre clés corrélées (mode d'échec le plus
  probable de E2 : un document homogène produit des états latents très proches).
- **Déclencheur** : aucun — prévu d'office juste après le run baseline (coût quasi nul).
- **Esquisse** : `φ_dg(h) = topk(G·h, k)` avec `G` (d → D, D ≈ 8–16×d) **aléatoire
  gelée** (seed fixée, jamais apprise). M devient rectangulaire d×D. Bonus : capacité
  théorique ~D associations au lieu de ~d, et writes creux.
- **Config** : `dg_dim` (0 = désactivé), `dg_topk`. M : 768×8192 ≈ 25 Mo fp32 — trivial.

### X2 — Lecture itérée CA3 (complétion de patterns) — *repli*

- **Bio** : CA3 est un réseau auto-associatif récurrent (un Hopfield, littéralement) :
  un fragment d'indice suffit à reconstruire le pattern complet.
- **Problème visé** : rappel par indice partiel — E1 qui marche avec le prompt exact
  mais échoue en paraphrase.
- **Déclencheur** : ajouter une variante paraphrase à E1 ; si exact ≫ paraphrase.
  **Testé le 2026-08-20 : NON observé** (ratio paraphrases/exact = 0.68, dégradation
  graduelle par recouvrement sémantique — voir JOURNAL). X2 reste en poche.
  **Mise à jour (audit 2026-08-21)** : sous le gate keysim (défaut X8), le ratio
  tombe à **0.38** — le déclencheur se réévalue donc au point courant. Seuil
  acté : ratio < 0.3 persistant APRÈS calibration du gate (`gate_keysim_mid`,
  Q-06 de l'audit) — on ne compense pas par CA3 un coût réglable du gate.
- **Esquisse** : itérer la lecture 2–3 fois : `h' = h + λ·M·φ(h)`, relire avec h',
  converger vers l'attracteur. Config : `read_iters` (défaut 1 = comportement actuel).
  **Caveat (audit 2026-08-21)** : telle quelle, l'esquisse est INOPÉRANTE sous le
  gate keysim — g est évalué sur φ(h) avant toute injection : une première passe
  gatée à ~0 ne converge vers rien. Une lecture itérée devra itérer aussi le gate
  (ou l'outrepasser explicitement).

### X3 — Alternance theta encode/retrieve — *repli*

- **Bio** : l'oscillation theta (~8 Hz) sépare temporellement écriture (plasticité
  haute) et lecture (plasticité coupée) — une lecture ne contamine jamais la mémoire.
- **Problème visé** : instabilité / divergence de M malgré les garde-fous existants.
- **Déclencheur** : perplexité qui explose ou génération qui boucle avec M active.
- **Déjà à moitié en place** : l'engine interdit structurellement l'écriture pendant
  `generate` et pendant le rappel d'éval (règle absolue, engine.py). La version
  complète alternerait des micro-phases read-only/write-only pendant le stream.
- **Note (audit 2026-08-21)** : la moitié en place protège l'ÉCRITURE de la
  lecture ; ce que la séparation theta protège d'abord (Hasselmo), c'est le
  RAPPEL pendant l'encodage — précisément le côté que X8.1b/P5 a mesuré (le
  dommage de lecture vit aux positions incertaines, celles où l'écriture
  déclencherait). Cette moitié-là est aujourd'hui couverte de fait par la loi 2
  (gate keysim) ; si un X3 complet se justifie un jour, c'est ce côté qu'il
  formalisera.

### X4 — Néogenèse / reset de lignes — *repli*

- **Bio** : la néogenèse du gyrus denté efface activement d'anciens souvenirs en
  recâblant les circuits — l'oubli est un mécanisme entretenu, pas une défaillance.
- **Problème visé** : saturation de M sur les longs streams (E2 qui plafonne puis
  régresse).
- **Déclencheur** : courbe NLL de E2 en U (amélioration puis dégradation).
  **Note (audit 2026-08-21)** : déclencheur NON OBSERVABLE en l'état —
  `domain_drift.py` collecte les NLL par token mais n'imprime que deux moitiés ;
  sortir la courbe par chunk (donnée déjà collectée) avant de pouvoir statuer.
- **Esquisse** : tracker l'utilisation des lignes en lecture (EMA de |contribution|),
  réinitialiser périodiquement les moins utilisées. Version riche de notre top-k.

### X5 — Substrat épisodique explicite (indexing theory / table vectorielle) — *repli diagnostique*

- **Bio** : l'hippocampe ne stockerait pas le contenu mais un *index* — un pointeur
  vers le pattern cortical à rallumer (hippocampal indexing theory).
- **Problème visé** : échec complet de E1 après balayage honnête de λ/η/couche.
- **Déclencheur** : E1 nul partout. C'est le repli de dernier recours, et il est
  *diagnostique* : lignée prouvée (kNN-LM, Memorizing Transformers) — si même
  l'injection kNN exacte ne bouge pas les logits, le coupable n'est pas le substrat
  de stockage mais le **mécanisme d'injection** (couche, λ), et on sait où chercher.
- **Esquisse** : remplacer M par un store (K, V) explicite + recherche par similarité,
  même point d'injection. Interface commune avec FastWeightMemory pour swap propre.
- **Renvoi (2026-08-21)** : le candidat V2-D « Fast-KV » (contexte fantôme KV)
  est X5 arrivé par la porte de l'ingénierie — même théorie de l'indexation,
  substrat = paires KV rejouées dans les couches basses au lieu d'un store
  (K, V) lu en couche n/2. Si ce chantier s'ouvre un jour, arbitrer
  explicitement la fusion X5/Fast-KV plutôt que d'entretenir deux entrées.

### I1 — Prédicteur d'échec par similarité de clés — *instrumentation, pas un mécanisme*

- **Origine** : AFTER_v1_THOUGHTS — la variance E1 (±0.8 même avec DG) dit que
  l'orthogonalisation est inégale selon les faits : certaines clés tombent bien
  séparées, d'autres non.
- **Idée** : logger, à chaque write, la similarité cosinus max entre la nouvelle clé
  φ(h) et les clés déjà écrites (ou leur approximation via les colonnes actives de M).
  Si la variance E1 par secret corrèle avec cette similarité, on a un **prédicteur de
  quand M va échouer** — presque aussi précieux qu'une réparation, et un critère de
  gating d'écriture potentiel (refuser d'écraser une clé trop proche).
- **Coût** : instrumentation pure, aucun changement de mécanisme. Bon candidat v1.1+.
- **FAIT (2026-08-21, journal)** : hypothèse falsifiée telle que formulée (corr
  intra-régime positive), mais keysim est une **jauge de saturation calibrée** :
  0.23/0.45/0.78 → rappel +0.74/+0.77/+0.03. La capacité de M n'est pas la
  contrainte, la distinctivité des indices l'est. Débouché ouvert : X6, gating
  d'écriture par similarité (refuser/alerter quand cos max > ~0.6).
- **Décision X6 (audit 2026-08-21)** : « X6 » a désigné deux idées distinctes
  dans le journal — (a) un hybride Hebb/delta (A1b) et (b) le gating d'ÉCRITURE
  par keysim (ci-dessus). Aucune n'est « absorbée par X8 », qui est un gate de
  LECTURE. Statut acté : (a) suit l'arbitrage D5 (E2 long horizon — Q-09 de
  l'audit) ; (b) est GELÉE sans mesure — X9 a montré les collisions génériques
  inoffensives, et le symptôme visé (écrasement par collision d'indices
  discriminants) n'a été observé qu'en gabarit commun artificiel. Réouverture :
  ce symptôme en usage réaliste.

### X7 — Loi prior/rappel et hypothèse d'aplatissement — *MESURÉ (2026-08-21)*

- **Origine** : convergence de trois observations indépendantes — tambourine négatif
  partout, corr(Δlogp, prior) = −0.50, taxe E3 sur texte confiant — vers une seule
  hypothèse : l'injection `λ·M·φ(h)` agit en partie comme un **aplatisseur de
  distribution** (aide la queue, casse le sommet). Si la loi tient en haut de la
  distribution, le mécanisme actuel ne peut PAS atteindre le rappel top-10 : il
  dégrade précisément le régime où le rappel devrait aboutir.
- **Mesure décisive avant tout mécanisme** (`eval/flattening.py`) : décomposer la
  lecture r = λ·M·φ(h) par projection sur la direction d'unembedding du token cible
  (logit lens) — fraction alignée = rappel, fraction orthogonale = bruit. Répondre :
  « r contient X % de signal, Y % de bruit, et le bruit domine quand le prior est
  haut » plutôt que « M aplatit ». Complément : entropie de sortie avec/sans M.
- **Protocole de la courbe** : ~24 cibles échantillonnées agressivement dans les
  rangs 1–500 (aujourd'hui, tambourine est l'UNIQUE témoin du régime probable — la
  loi en corr −0.50 est ajustée sur un nuage au flanc vide) + sous-ensemble
  multi-tokens dont seul le token de tête est probable (l'aplatissement frappe-t-il
  au point d'entrée seulement ?).
- **Mécanisme découlant de la mesure** : voir X8 (gate de lecture à deux facteurs).
- **Portée** : « quand écouter sa mémoire plutôt que son modèle du monde » est la
  question de tout système à mémoire externe (RAG compris, qui la tranche à la
  hache en concaténant tout). Une réponse continue calibrée incertitude × pertinence,
  mesurable et ablatable, dépasse le cadre du PoC.
- **X8.1/X8.1b/P5 (2026-08-21, journal)** : le two_factor est ENTERRÉ par une
  chaîne de trois interventions — binarisation (réfute « g intermédiaire »),
  créneaux forcés (réfute « volatilité temporelle »), analyse par position
  (confirme : dommage concentré aux positions incertaines, corr +0.394,
  confiantes −0.099 vs incertaines +0.364). **Loi finale : un gate de lecture
  déclenché par l'incertitude du cortex est adversarial par construction — gater
  côté mémoire (pertinence), jamais côté détresse du cortex.** Le cas
  « confiance erronée » relève de V2-D, pas d'un gate.
- **MESURÉ (2026-08-21, journal)** : hypothèse raffinée par les données — (1)
  l'aplatissement est un **coût fixe** (+0.141 nats d'entropie, uniforme) ; (2) la
  lecture n'a **aucune composante directionnelle** vers la cible (cos ≈ 0 vs 0.136
  de base) : le rappel opère par recomputation indirecte des couches aval ; (3) pas
  de régime de pénalité active — le gain tend vers 0 aux rangs bas sans devenir
  négatif (tambourine reste un outlier inexpliqué par le prior seul). **Feu vert
  renforcé pour le gate** : coût fixe + gain nul en régime confiant = tout à gagner.
  Le verrou top-10 est requalifié : absence de signal directionnel, pas pénalité —
  piste v2+ : valeurs dans l'espace d'unembedding ou tête de lecture apprise.

### X8 — Gate de lecture (keysim RETENU) — *dépend de X7 (fait) ; distinct du X6 d'écriture (voir I1)*

**Verdict (2026-08-21) : gate keysim RETENU, nouveaux défauts**
(`read_gate="keysim"`, λ=2.0, cap=0.5 — le cap reste en plancher de sécurité,
décision D10, ARCHITECTURE §3). E1 GPT-2 +1.353 ± 1.58 / SmolLM2 +0.755 ± 0.94 ;
E3 −0.014 / +0.003 ✓ ; E2 : bénéfice absolu GPT-2 (−0.055 sur les deux moitiés),
échec au régime agressif SmolLM2 intra-domaine (voir la note de métrique, §4).
Le facteur entropie est disqualifié (anomalie), puis le two_factor ENTERRÉ par
l'arc X8.1 → X8.1b → P5 : le dommage de lecture vit aux positions incertaines du
cortex — gater côté mémoire, jamais côté détresse. Coût mesuré : ratio
paraphrases 0.68 → 0.38 (`gate_keysim_mid`, calibration par modèle à faire —
Q-06 de l'audit). Détail : journal du 2026-08-21. Protocole d'origine ci-dessous.

- **Origine** : X7 a chiffré le feu vert — l'injection coûte un aplatissement FIXE
  (+0.141 nats d'entropie) et ne rapporte quasi rien quand le cortex est confiant
  (Δ→0 aux rangs bas). Couper la lecture en régime confiant ne sacrifie ~rien et
  économise ~tout.
- **Esquisse** : remplacer le gain de lecture constant par
  `g = f(entropie du cortex, keysim de lecture)` — où le keysim de lecture est le
  cos max entre la clé de requête φ(h) et les clés du buffer I1 (côté lecture,
  distinct du keysim d'écriture). Angle mort nommé AVANT implémentation : la
  **confiance erronée** — le cortex peut être sûr ET avoir tort (fait périmé) ; un
  gate par entropie seule ferait taire M au moment exact où il détient la mise à
  jour. Le facteur keysim répond : un match très fort force le passage même à basse
  entropie (« non, attends, je *sais* que ça a changé »). L'idée keysim d'I1 est reprise
  ici côté LECTURE ; le gating d'écriture (X6-b) reste un mécanisme distinct,
  gelé — voir la décision X6 dans l'entrée I1.
- **Éval dédiée E1c (correction de fait)** : rendre X fortement prédit (l'injecter,
  ou choisir X à rang naturellement bas), puis injecter « The password has changed.
  It is now Y. » ; mesurer le rappel de Y. Critère : le gate à deux facteurs rappelle
  Y là où le gate entropie-seule échoue (c'est le test discriminant entre les deux).
- **Benchmark contre trois baselines** : cap global actuel, gating entropie seule,
  gating keysim seul. **Cibles chiffrées (échec explicite si manquées)** :
  E3 ≤ 0.05 ; E1 > le point conforme actuel (+0.494 SmolLM2 / +0.740 GPT-2) ;
  E2 SmolLM2 ≥ −0.025 (récupérer au moins la moitié du −0.050 perdu à cap 0.1).
- **Point de vigilance (pas un conflit tranché)** : X1b a acté le cap comme gating
  doux et point de conformité (ARCHITECTURE §2.1). X8 le remplace-t-il ou le
  garde-t-il en plancher de sécurité ? À décider à l'implémentation, en le notant
  dans les Décisions.

### X9 — Courbe de capacité — *MESURÉ (2026-08-21) : pas de falaise à 80 faits*

**Verdict** : +0.710 ± 0.61 à N=80 (91 % de la référence N=5, 71/80 positifs) — la
falaise n'existe pas à cette échelle, d² est un problème théorique, factorisations
écartées définitivement. Prédicteur par fait mort (keysim max sature à ~1.0 sur les
mots-outils partagés — collisions inoffensives, la delta rule converge sur clés
identiques). Détail : journal du 2026-08-21. Protocole d'origine ci-dessous.

- **Origine** : revue externe post-v1.2. I1 a montré que 10 faits à indices
  distincts tiennent sans coût et que keysim est une jauge de RÉGIME ; X9 étend en
  courbe : E1-multi varié à 5 / 10 / 20 / 40 / 80 faits, d fixe, défauts.
- **Objectif** : localiser la **falaise d'interférence** (critère : le rang N où le
  Δlogp moyen tombe sous 50 % du mono-fait), et tester le prédicteur par fait :
  corrélation entre échec de rappel et cos max de sa clé vs les clés déjà écrites.
  Réserve honnête posée avant mesure : I1 a trouvé cette corrélation POSITIVE
  intra-régime (+0.2/+0.5 hors saturation) — X9 la re-teste en charge graduée ;
  si elle reste non-négative, le prédicteur par fait est mort, seule la jauge de
  régime survit.
- **Contrainte de protocole** : il faut ≥ 80 gabarits à contextes distincts (le pool
  actuel en a 10) — les étendre AVANT de mesurer, sinon les collisions de gabarit
  confondent la mesure de capacité (leçon de I1).
- **Décision par défaut, actée ici** : aucune restructuration de M (block-diagonal,
  Kronecker, rang faible) avant ce résultat — et probablement pas après : la
  capacité d'une mémoire matricielle est bornée par sa dimension, les
  factorisations la PLAFONNENT, et les updates rang-1 de la delta rule ne vivent
  pas sur la variété de Kronecker. Ce chiffre décide si d² est un problème réel ou
  théorique.

### X10 — Comparatif de kernels d'adressage — *GELÉ (2026-08-21)*

**Gel (audit 2026-08-21)** : sa métrique principale était « la position de la
falaise X9 » — or X9 a conclu qu'il n'y a PAS de falaise à cette échelle (80
faits, 91 % de rétention). Déclencheur de réouverture : une falaise observée en
charge réelle, ou un 3ᵉ modèle où l'adressage DG régresse. La branche « DG
apprise » (conflit D8/D9, ci-dessous) est RETIRÉE tant que le gel tient —
l'arbitrage D8/D9 n'a plus d'objet. Protocole d'origine ci-dessous.

- **Origine** : revue externe. La projection DG est déjà un kernel d'adressage ; la
  question est lequel repousse la falaise de X9 le plus loin, gratuitement.
- **Protocole** : à taille de M ÉGALE, comparer (1) DG actuelle (aléatoire + top-k),
  (2) DPFP (Schlag et al. 2021), (3) DG apprise offline. Métriques : position de la
  falaise X9 + E1/E3 standard.
- **CONFLIT À SIGNALER (non tranché)** : la variante « DG apprise offline » heurte
  D8 (aucun backprop en v1) et D9 (G aléatoire gelée, jamais apprise —
  justification : JL suffit, une G apprise exigerait du gradient). Une G apprise
  hors ligne puis gelée au test respecte la lettre de « pas de gradient à
  l'inférence » mais introduit l'entraînement dans le projet. À arbitrer
  explicitement (amendement de D8/D9 ou variante hors-protocole) avant de lancer
  cette branche de X10 ; les branches (1)-(2) ne posent aucun conflit.

### E4 — Éval « conventions de projet » — *MESURÉ (2026-08-21) : critère non atteint, juge en cause*

**Verdict** : gain de discrimination non significatif (+0.01/+0.02, 6/10 paires) sur
GPT-2 ET SmolLM2 — mais les deux cortex ont un baseline INVERSÉ (ils préfèrent les
phrases violantes, −0.13) : un juge qui ne sait pas juger ne peut pas montrer de
gain de jugement. Spécificité et E3 tiennent (en gating ; le mode force PERD la
spécificité — le gating préserve la spécificité, écho A2). Les paires quasi-verbatim
gagnent fort (+0.09 à +0.18), les normatives perdent : mémoire associative, pas
inférence. **Vision par-projet ni validée ni tuée : en attente d'un juge valide.**
À rejouer en E4b : scoring au token décisif + cortex multilingue (Qwen2.5-1.5B).
Détail : journal du 2026-08-21. Protocole d'origine ci-dessous.

- **Origine** : vision d'usage (README §« Vision d'usage ») — M comme cache chaud du
  contexte diffus d'un projet. E4 teste exactement ça, rien d'autre.
- **Protocole** (à lancer dès que X8 tient ; NE PAS lancer avant) :
  1. Streamer un vrai document de conventions dans M, écriture active, config
     conforme post-X8 — le CLAUDE.md d'engram lui-même.
  2. Vider le cache KV (comme E1 : élimine l'explication « c'est le contexte »).
  3. Mesurer la NLL du cortex sur (a) du texte/code CONFORME aux conventions,
     (b) du texte/code qui les VIOLE — paires minimales autant que possible
     (même fonction, style conforme vs non conforme).
  4. Discrimination = NLL(violant) − NLL(conforme), avec M chargée vs M vierge.
- **Critère de succès** : la discrimination augmente significativement avec M
  chargée, à E3 toujours ≤ 0.05 sur texte neutre.
- **Critère d'échec explicite** : pas de gain de discrimination, ou gain payé en
  E3 — la vision par-projet est alors réfutée à cette échelle, et on le note.
- **Contrôle de spécificité** : même mesure avec les conventions d'un AUTRE projet
  chargées dans M — la discrimination ne doit PAS augmenter (élimine l'effet
  « M pleine »).

### V2-D — Rappel directionnel — *LE chantier v2 prioritaire ; course à trois candidats (2026-08-21)*

- **Origine** : le mur mesuré en X7 (cos(r, W_U[cible]) ≈ −0.01, sous la base
  aléatoire 0.136 — le chemin direct est activement orthogonal à la cible) et ses
  quatre visages : E1c, verrou top-10, E4, E4s. M actuelle fait de l'**amorçage**
  (priming) : elle teinte le calcul (d'où E1/E2, l'adaptation, la paraphrase) mais
  ne peut pas **nommer** un token. **Q-01 (2026-08-21) est la 5ᵉ observation** :
  la lecture d'état est une perturbation quasi générique, de direction invariante
  du modèle — le rappel précis exige un canal qui désigne des tokens.
- **Contrat d'admission (critère de la course)** : cortex gelé, règle locale, en
  ligne, **zéro module entraîné par gradient** (D8) — c'est ce qui rend chaque
  résultat attribuable. Tout candidat qui y déroge est hors course (voir les
  options écartées en fin d'entrée).
- **Règle transversale (D11, issue de Q-01)** : aucun canal ne s'ouvre sur la
  détresse du cortex, et chaque candidat s'évalue comme **perturbation aux
  positions incertaines** (dommage par position, pas seulement en moyenne) — les
  logits et le cache KV ont leurs positions fragiles comme le flux résiduel.
- **Ordre de la course** : (a) kNN-LM nu d'abord (instrument de plafond), puis
  (b) M_out (candidat principal), (c) Fast-KV. Un candidat = une future entrée
  pré-enregistrée avec critères chiffrés le moment venu — ici, fiches
  d'intention seulement, aucun numéro d'expérience n'est promis.

**Candidat (a) — kNN-LM nu (baseline-instrument, À FAIRE EN PREMIER)** :
datastore brut de paires (état caché → token suivant) interpolé directement dans
la distribution de sortie (Khandelwal et al. 2020 — son résultat historique est
précisément le rappel des tokens rares que le modèle seul ne hisse jamais). Pas
un candidat produit (un datastore se compte en dizaines de Go — l'esprit
contraire du projet) mais **l'instrument qui mesure le PLAFOND** : si le
datastore brut hisse les rangs sur E1/E1c là où M échoue, l'espace de sortie est
confirmé comme bon point d'attaque avant d'écrire une ligne du mécanisme
compressé ; s'il échoue aussi, V2-D est mal parti et on le saura pour trois fois
rien. Ligne D11 : l'interpolation sera elle aussi regardée par position.

**Candidat (b) — M_out sur les logits (candidat principal)** : une seconde
matrice M_out qui stocke `φ(h) → u_token` (valeurs = lignes de l'unembedding W_U
du token OBSERVÉ, pas l'état caché), écrite par la même delta rule au même
signal de surprise, lue en **biais additif sur les logits** :
`logits += g·W_U·(M_out·φ(h))` — soit une correction linéaire en ligne de la
tête de sortie. Même taille que M (d×dg_dim), même gate keysim (organe déjà
validé par X8), aucun backprop : la version compressée et en ligne de kNN-LM,
la plus simple conforme au contrat. Ligne D11 : une distribution de sortie a
aussi ses positions fragiles — le biais additif sera évalué en dommage par
position, pas seulement en E3 moyen.

**Candidat (c) — Fast-KV (contexte fantôme — nouveau, 2026-08-21)** : M pilote
l'injection de paires KV **virtuelles** dans le cache des couches basses — le
fait n'est plus dans la fenêtre, son ombre KV y est. Rationale : l'attention
native sait DÉSIGNER des tokens (induction heads), contrairement au flux
résiduel en n/2 (le mur X7) — on utilise la machinerie de rappel du cortex au
lieu de la contourner. Bio : théorie de l'indexation hippocampique
(l'hippocampe stocke de quoi RÉINSTALLER le pattern cortical, pas le contenu) —
c'est **X5 arrivé par la porte de l'ingénierie** (renvoi croisé dans l'entrée
X5). Question de design centrale, OUVERTE : construire les paires KV sans
entraînement ; candidat naturel — **rejouer les KV réels capturés au moment de
l'écriture** (l'hippocampe comme enregistreur d'états d'attention). À
protocoler le jour venu, pas ici : (1) E3 dédié obligatoire, par position
(D11 — un contexte fantôme peut distraire l'attention comme M distrait le flux,
et Q-01 dit que TOUT canal coûte aux positions fragiles) ; (2) coût mémoire des
KV stockés à chiffrer — n_couches_basses × 2 × d par token fantôme (GPT-2 :
6 couches ≈ 37 Ko fp32 par token → ~1 000 tokens fantômes ≈ 35-40 Mo ; à
vérifier contre l'esprit « dizaines de Mo ») ; (3) **point de vigilance D7
(signalé, non tranché)** : `clear_context()` est LE contrôle du PoC — il devra
vider AUSSI le contexte fantôme, la réinstallation devant être re-déclenchée
par M au moment du rappel ; sinon l'éval ne distingue plus la mémoire du
in-context et D7 perd son objet.

**Options examinées et écartées (2026-08-21)** — même statut que les
factorisations de M écartées par X9 :

- **Pseudo-inverse de W_U** (Δh = W_U†·P_cible injecté dans le flux résiduel) :
  fabrique une direction de logits puis la re-projette dans le résiduel pour que
  l'unembedding la re-extraie — deux projections pour revenir au point de
  départ, pseudo-inverse mal conditionnée (d × vocab), et le vecteur reconstruit
  retraverse les couches qui le RECOMPUTENT (le mur X7, réimporté). Si le but
  est de biaiser les logits : les biaiser directement. Écartée comme « M_out en
  moins bien ».
- **Cross-attention traductrice pré-entraînée** (~1M params entraînés une fois
  sur dataset générique) : viole le contrat zéro-gradient (D8) — chaque succès
  deviendrait inattribuable (M ou le traducteur ?) — et ne résout PAS X7 : le
  problème du point d'injection n/2 n'est pas une transformation mal apprise
  mais l'absence de chemin direct vers les tokens (cos ≈ −0.01, mesuré).
  Écartée par défaut ; si un jour toutes les options sans entraînement sont
  épuisées, la rouvrir sous son vrai nom (« passage au gradient »), pas comme
  couche d'interfaçage.

- **Architecture résultante — deux canaux de lecture** : un canal d'ÉTAT
  (l'existant : diffus, généralisant, ratio 0.38–0.68) et un canal de SORTIE
  (directionnel, précis, pour le rappel exact — b ou c selon la course). C'est
  la hiérarchie registre/cache dessinée pour l'usage par-projet, mais À
  L'INTÉRIEUR du mécanisme. Rapprochement familiarité/recollection corrigé
  (audit 2026-08-21) : la carte naïve est inversée — la **recollection**
  (réinstallation d'un pattern via un index) est le processus hippocampique,
  dont l'analogue ici est X5 (et donc Fast-KV) ; un canal de sortie
  item-spécifique type kNN-LM est plus proche de la **familiarité**. X5 et V2-D
  partagent donc la même justification bio (indexing theory), par deux chemins
  différents.
- **Évals de verdict (communes aux candidats)** : E1 top-10 (le canal de sortie
  doit débloquer des rangs que l'amorçage ne peut pas atteindre), E1c (renverser
  « Paris »), E4s (la préférence token-niveau), chacune avec le canal d'état
  seul / sortie seul / les deux — E3 toujours ≤ 0.05, ET par position (D11).
- **Principe consigné (dépasse engram)** : dans une mémoire test-time, **les
  gates ne sont pas des optimisations, ce sont les organes qui rendent le
  mécanisme viable** — écriture gatée = spécificité (E4 mode force), lecture
  gatée = innocuité (E3 +0.57 sans gate), et le rappel précis exige un troisième
  organe, directionnel.
- **Priorité inchangée** : cette entrée prépare le terrain, elle n'ouvre pas le
  chantier — les corrections d'audit et la file Q (Q-01b, requalification Q-05,
  Q-03) passent avant toute ouverture V2-D.

### V2 — Replay / sharp-wave ripples (consolidation M → LoRA) — *hors v1, basse priorité actée*

- **Bio** : replay de séquences compressées (jusqu'à 20×), parfois inversées, parfois
  *recombinées* — de la planification autant que de la consolidation.
- **Spec retenue pour la v2** : la distillation vers le LoRA ne rejouera pas les
  données brutes mais des échantillons **générés depuis M** (échantillonner des clés,
  générer avec/sans M, distiller la différence). C'est ici que le dilemme
  stabilité/plasticité revient en grand.
- **Le LoRA n'est PAS une amélioration générique** : c'est la réponse à UN problème —
  le débordement de M sur la durée de vie d'un projet (usage engram-par-projet).
  Déclencheur défini : la falaise X9. **Verdict X9 (2026-08-21) : falaise lointaine**
  (aucune dégradation à 80 faits) → **basse priorité actée**. M seule couvre
  vraisemblablement la durée de vie d'un projet ; le cycle veille/sommeil
  (M déborde → distillation → reset) attendra un signal de débordement réel.
- **Prérequis absolus avant toute implémentation** : X8 tient ET X9 mesuré (fait).
  Contribution visée si un jour lancé : la version EN LIGNE nourrie automatiquement
  par M — le LoRA-par-projet statique existe déjà ailleurs.
- **Question ouverte (pas une spec)** : l'écriture involontaire. En usage réel, M
  apprend aussi les erreurs et les pistes abandonnées — le gating par surprise dit
  « c'est nouveau », pas « c'est validé ». La consolidation devra filtrer par un
  signal de **valence** encore à définir.

## 3. Correspondance mécanisme ↔ piège

Chaque extension répond à un piège identifié dans ARCHITECTURE.md §4 — le cerveau a
rencontré exactement ces bugs et a shippé des fixes :

| Piège / mode d'échec | Fix biologique | Extension |
| --- | --- | --- |
| Interférence entre clés proches | Séparation de patterns (gyrus denté) | X1 |
| Rappel partiel impossible | Complétion de patterns (CA3) | X2 |
| Lecture qui contamine l'écriture | Rythme theta | X3 |
| Saturation / oubli mal ciblé | Néogenèse, oubli actif | X4 |
| Substrat matriciel inadapté | Indexation plutôt que stockage | X5 |
| Oubli catastrophique à la consolidation | Replay SWR recombinant | V2 |

## 4. Tableau des poids (à remplir au fil des runs)

Référence : config par défaut (`EngramConfig`), seeds fixes, GPT-2 124M.
Les deltas se lisent **contre la ligne précédente retenue**, pas contre X0.
Depuis X8, l'interaction E2 seule ne suffit plus (le coût d'échauffement
disparaît, elle tombe à ~0 mécaniquement) : rapporter aussi le **ΔNLL absolu
par moitié** — voir ARCHITECTURE §5 E2, note de métrique (audit, COR-14).

| Étape | E1 Δlogp (nats) | E1 top-10 | E2 interaction | E3 dégradation | Verdict |
| --- | --- | --- | --- | --- | --- |
| X0 naïve (défauts initiaux : λ=0.5, η=0.05) | +0.219 ± 0.271 (N=10, 9/10 > 0) | 0/10 | *(à venir)* | *(éval à écrire)* | signal faible mais directionnel |
| X0 naïve, point retenu (layer=6, λ=1.0, η=0.1) | +0.735 ± 0.886 (N=10) | 0/10 | *(à venir)* | +0.1991 ± 0.0241 (> seuil) | référence pour X1 — variance = interférence, déclencheur X1 observé |
| X1 gyrus denté (dg=8192/64, λ=2.0, η=0.2) | +1.361 ± 1.588 (N=10) | 0/10 | *(à venir)* | +0.1354 ± 0.0147 (> seuil ×2.7) | mécanisme retenu (+0.63 E1, −0.06 E3 vs X0) mais point NON conforme E3 |
| **X1b compromis λ×cap (dg=8192/64, λ=1.0, η=0.2, cap=0.25)** | **+0.740 ± 0.799** (N=10) | 0/10 | **−0.0551 ✓** (writes 23 %) | **+0.0228 ± 0.0056 ✓** | **référence GPT-2** — premier point conforme, tableau complet ; cap = gating doux (λ contrôle le bruit, cap le signal) |
| **v1.2 SmolLM2-360M (layer=16, cap=0.1, reste identique)** | **+0.494 ± 0.245** (N=10) ; multi varié +0.817 ± 0.214 (cap 0.25) | 0/10 | −0.0500 (cap 0.25) ; **−0.0170 ✓ au point conforme** | **+0.0415 ± 0.0029 ✓** | **le signal passe l'échelle** — mécanisme universel, calibration (λ, cap) par-modèle ; couche tardive nocive ; corr keysim −0.4 : n.s. à N=10, indication seulement (requalifiée à l'audit 2026-08-21) |
| **X8 gate keysim (nouveaux défauts : gate=keysim, λ=2.0, cap=0.5)** | GPT-2 **+1.353 ± 1.58** ; SmolLM2 **+0.755 ± 0.94** (N=10) | 0/10 | interaction ~0 mais **absolu −0.055 sur les deux moitiés** (GPT-2) ; SmolLM2 : échec au régime agressif (parité au mieux) | GPT-2 **−0.014 ✓** ; SmolLM2 **+0.003 ✓** | **RETENU (2026-08-21)** — E3 éliminé, régime agressif rouvert ; coût : ratio paraphrases 0.68 → 0.38 ; entropie/two_factor enterrés (X8.1 → P5) |

Ablations mesurées au point X1b (détails : journal des 2026-08-20/21) — chacune
retire un mécanisme de la référence, aucune n'est retenue comme nouveau défaut :

| Ablation | E1 exact | E2 interaction | E3 | Enseignement |
| --- | --- | --- | --- | --- |
| Hebb pur (sans terme correctif) | +0.693 | **−0.0948** (RFC) / −0.0335 (narratif) | +0.0331 ✓ | son avantage RFC était du n-gramme : égalité avec delta sur fiction (E2n, 2026-08-21) — **arbitrage D5 quasi clos, delta confirmée par défaut** |
| toujours-écrire (thr=0) | — | −0.0045 | — | **le gating porte ~92 % de l'effet E2** |
| clés denses (dg off) | — | −0.0352 | — | DG apporte +57 % d'interaction |
| delta η=0.4 | — | −0.0488 | — | contrôle C1 : l'avantage Hebb ≠ pas plus grand |
| Q-01 perturbations appariées (diagnostic, gate none, 2 textes, 2026-08-21) | — | — | read-M +0.135 (réf.) ; iid_pair reproduit le ciblage à R = 0.83 (A) / 0.78 (B) | **H générique RETENUE (composite)** — le ciblage entropie n'est pas spécifique à M ; la direction prior (+r̄ ≈ read-M à 0.993) fixe le signe aux confiantes ; aucune interférence de contenu (P4 0.988 vs null 0.658) — protocole `experiments/EXP-2026-08-21-specificite-dommage-incertaines.md` |
| Q-01b congruence lexicale (diagnostic CPU sur bruts Q-01, gate none, 2026-08-21) | — | — | sans objet (aucune lecture modifiée) | **H1a REJETÉE** — ρ_S +0.18/+0.22 et β_s* +0.045/+0.028 (signe OPPOSÉ à la prédiction, p<10⁻³, deux opérationnalisations dont la direction réelle r̄) ; le signe positif s'effondre sous appariement NLL_base (+0.32 → +0.004/+0.05) et en zone non mordante ⇒ **propriété de la borne D_t ≥ −NLL_base, pas un canal inverse** ; D2 : |Δ_raw| 0.095 < sd intra-texte 0.27/0.13 — **l'anomalie A/B de Q-01 était de la pseudo-réplication** ; F non rapportée (P3 : Δ_z ≈ 0) ; ablation direction de fréquence **NON ARMÉE** (condition β_s* < 0 ; mesuré positif) — protocole `experiments/EXP-2026-08-21-congruence-lexicale.md` |

*(les lignes suivantes du tableau principal s'ajoutent quand leur déclencheur est observé)*
