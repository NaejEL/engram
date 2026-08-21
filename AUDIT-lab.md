# Audit du labo — engram

Généré par /ai-lab-audit le 2026-08-21. Labo : présent. Factory : absente (corrections en liste seule, pas de specs).
Lentilles : fidelite ; validite ; fuites ; bio ; route. Audit complet. Journal audité jusqu'à l'entrée du 2026-08-21 (« X8.1b + P5 : anomalie résolue »).

Bilan : 39 findings bruts, 31 confirmés après validation adversariale, 8 rejetés/dédupliqués.
Aucune fuite dans le banc E1/E2/E3 : le contrôle D7 (clear_context / reset) tient, les résultats
du tableau des poids ([docs/EXTENSIONS.md](docs/EXTENSIONS.md) §4) ne sont pas invalidés.
Arbitrage (Jean, 2026-08-21) : critère « fiabiliser d'abord », 6 questions retenues, 2 en réserve.

## Corrections à faire

Tri : sévérité décroissante puis taille croissante. Pas de specs (factory absente) — à traiter à la main.

| ID | Finding | Lentille | Sévérité | Taille | Dépend de | Spec | Statut |
| --- | --- | --- | --- | --- | --- | --- | --- |
| COR-01 | Journal 2026-08-21 (Qwen) : la « série +0.74 → +0.85 → +4.18, l'amorçage s'amplifie » mélange trois régimes (GPT-2 +0.74 = λ1/cap0.25 sans gate ; Qwen = défauts X8, où GPT-2 fait +1.353) et repose sur N=2 secrets (« fumée ») — requalifier l'entrée en fumée tant que les 10 secrets au même régime ne sont pas mesurés | validite | critical | S | — | — | FAIT (2026-08-21) |
| COR-02 | Ratio de généralisation 0.68 (« chiffre-titre », CLAUDE.md) = ratio de deux moyennes bruitées (dénominateur +0.740 ± 0.799, N=10) sans propagation d'incertitude — recalculer par secret (médiane + IC bootstrap) sur les 40 valeurs déjà mesurées (E1b, 2026-08-20) | validite | critical | XS | — | — | FAIT (2026-08-21) |
| COR-03 | CLAUDE.md « État du projet » annonce les défauts λ=1.0/cap=0.25 alors que `engram/config.py` (l.23-24, 49) porte λ=2.0, cap=0.5, read_gate="keysim" depuis le verdict X8 — l'ancien point devient « référence d'ablation read_gate=none » | route | major | XS | — | — | FAIT (2026-08-21) |
| COR-04 | CLAUDE.md (l.76-80) et README (l.122-127) listent X8/X9 « à faire » avec l'ordre X9→X8→X10 alors que X8, X8.1, X9, E4, E4s et la campagne Qwen sont mesurés — mettre l'état et la feuille de route au 2026-08-21 | route | major | XS | COR-03 | — | FAIT (2026-08-21) |
| COR-05 | Tableau des poids EXTENSIONS.md §4 : aucune ligne X8 malgré le verdict « RETENU, nouveaux défauts » (E1 +1.353 ± 1.58, E3 −0.014 GPT-2 ; +0.755/+0.003 SmolLM2) ; ligne vide « X1 gyrus denté » orpheline (l.329) à supprimer | route | major | XS | — | — | FAIT (2026-08-21) |
| COR-06 | Décision D10 (cap = plancher de sécurité sous le gate) citée par config.py l.46 et hippocampus.py l.125 mais absente du tableau ARCHITECTURE.md §3 ; §2.1 décrit une lecture sans terme de gate g — zéro occurrence « gate/keysim » dans LA référence de design (violation de sa propre règle l.4-5) | route | major | XS | — | — | FAIT (2026-08-21) |
| COR-07 | Journal v1.2 : « pari perdu dans le bon sens » alors que le pari pré-enregistré Q3 (« < +0.740 ») est TENU au point conforme (+0.494 à cap 0.1) ; +0.852 est mesuré sans σ à cap 0.25 non conforme (E3 +0.079) — reformuler | validite | major | XS | — | — | FAIT (2026-08-21) |
| COR-08 | « Loi 2 contre-intuitive (ne lisez pas quand le modèle hésite) » : c'est la prédiction du modèle cholinergique de Hasselmo (ACh haute en nouveauté supprime le rappel récurrent) — requalifier pour le papier ; X3 « à moitié en place » l'est du mauvais côté (la moitié lecture est réalisée de fait par keysim+P5) ; « acétylcholine » pour le cap (ARCHITECTURE §2.1) est une analogie de façade | bio | major | XS | — | — | FAIT (2026-08-21) |
| COR-09 | Corrélations keysim à N=10 promues en conclusions : « le signe AFTER_v1 émerge à l'échelle » (−0.34/−0.40, n.s., p≈0.25) et « hypothèse falsifiée dans les trois régimes » (+0.2/+0.5, n.s.) — assortir de « n.s., N=10 » dans journal, CLAUDE.md et EXTENSIONS §4 | validite | major | S | — | — | FAIT (2026-08-21) |
| COR-10 | E4s Qwen titrée « première double dissociation correctement signée » pour +0.018/−0.010 (N=10, sans σ ; deux signes conformes = 1 chance sur 4 au hasard) — titrer « signes conformes, non significatifs » | validite | major | S | — | — | FAIT (2026-08-21) |
| COR-11 | Priorité v2 contradictoire : CLAUDE.md l.81, README l.133 et ARCHITECTURE §7 désignent le sommeil/LoRA quand EXTENSIONS (V2-D « LE chantier prioritaire », replay « basse priorité actée ») et le journal disent l'inverse | route | minor | XS | — | — | FAIT (2026-08-21) |
| COR-12 | X10 toujours planifié avec pour métrique « la position de la falaise X9 » alors que X9 a conclu « pas de falaise » — requalifier (geler avec déclencheur, ou re-motiver) et trancher la branche « DG apprise » (conflit D8/D9) ou la retirer | route | minor | XS | — | — | FAIT (2026-08-21) |
| COR-13 | « X6 » désigne deux mécanismes différents (hybride Hebb/delta, journal l.259 ; gating d'écriture par keysim, l.312/370) et « X8 absorbe X6 » est faux (X8 = gate de LECTURE) — le gating d'écriture est abandonné sans décision écrite | route | minor | XS | — | — | FAIT (2026-08-21) |
| COR-14 | Métrique E2 requalifiée par X8 (« la métrique honnête devient le bénéfice absolu ») non répercutée dans ARCHITECTURE §5 E2 ni dans l'en-tête « E2 interaction » du tableau §4 — rapporter interaction ET ΔNLL absolu par moitié | route | minor | XS | COR-05 | — | FAIT (2026-08-21) |
| COR-15 | VISION.md périmée : « X8 en validation finale », « E4-dur Qwen en attente » alors qu'E4-dur a tourné, échoué et été retiré au profit d'E4s ; la survie du cas d'usage a) repose désormais sur V2-D — le dire | route | minor | XS | — | — | FAIT (2026-08-21) |
| COR-16 | ARCHITECTURE §1 l.34, §2.1 l.43 et D2 décrivent M « pleine d×d, 2,4 Mo » et φ = h/‖h‖ alors que le défaut est DG d×8192 (~25 Mo) ; CLAUDE.md dit « M (d×d) » — présenter les deux modes, DG en défaut | fidelite | minor | XS | — | — | FAIT (2026-08-21) |
| COR-17 | Déclencheur X2 (CA3) figé sur le ratio pré-gate 0.68 alors que le défaut courant donne 0.38 sous keysim, sans seuil quantitatif pour « exact ≫ paraphrase » ; l'esquisse X2 (itérer la lecture) est inopérante sous gate (g évalué sur φ(h) avant toute injection : une 1ʳᵉ passe gatée à ~0 ne converge vers rien) | bio | minor | XS | — | — | FAIT (2026-08-21) |
| COR-18 | Déclencheur X4 « courbe E2 en U » inobservable avec l'instrument actuel : domain_drift.py collecte les NLL par token mais n'imprime que deux moitiés — sortir la courbe par chunk (donnée déjà collectée) ou consigner « déclencheur non observable en l'état » | bio | minor | XS | — | — | FAIT (2026-08-21) |
| COR-19 | ARCHITECTURE §2.5 : « mémoire à slots (lignée Titans) » — attribution inexacte, Titans est une mémoire neuronale à gradient test-time (le plus proche parent de M, et le contrôle naturel de D6 : surprise par gradient vs NLL) ; la lignée slots = kNN-LM / Memorizing Transformers | bio | minor | XS | — | — | FAIT (2026-08-21) |
| COR-20 | EXTENSIONS V2-D : « familiarité vs recollection » cartographiée à l'envers (la recollection est hippocampique, par réinstallation via index = X5 ; le canal de sortie kNN-LM-like est côté familiarité item-spécifique) — reformuler, et noter que X5 et V2-D partagent la même justification bio | bio | minor | XS | — | — | FAIT (2026-08-21) |
| COR-21 | eval/flattening.py l.116-120 : la branche de repli de `random_cos_baseline` streame le fait sans `clear_context()` préalable — la base de cosinus aléatoire (0.136, citée dans X7 et CLAUDE.md) est calculée hors protocole ; diagnostic seulement, cos cible ≈ −0.01 non affecté | fuites | minor | XS | — | — | FAIT (2026-08-21) |
| COR-22 | tests/test_hippocampus.py : aucun test du mode par défaut keysim seul, ni de « buffer vide après reset() ⇒ lecture nulle » (propriété qui fonde le contrôle D7 sous gate), ni de l'ordre update→decay — trois tests CPU à ajouter | fidelite | minor | S | — | — | FAIT (2026-08-21) |

## Questions de recherche retenues (ordre recommandé — critère « fiabiliser d'abord »)

### Q-01 — Spécificité de la loi « dommage aux positions incertaines »
- **Question** : la corrélation dommage/entropie de +0.394 (P5) est-elle spécifique à la lecture de M, ou toute perturbation de même norme aux positions incertaines la reproduit-elle ?
- **Motivation** : VAL-06 — P5 (journal 2026-08-21) conclut « un gate déclenché par l'incertitude est adversarial par construction » sans contrôle par vecteur aléatoire de norme ‖r‖ ni M chargée d'un fait sans rapport ; la sensibilité de la NLL à une perturbation croît mécaniquement avec l'entropie.
- **Ce qu'elle trancherait** : la forme finale de la loi 2 (propriété de la lecture de M vs propriété du cortex) — conclusion de l'arc X8.1, chapitre méthode du papier.
- **Coût estimé** : XS — **Dépend de** : —
- **Lancer** : `/lab-run "La corrélation dommage/entropie +0.394 (P5) est-elle spécifique à la lecture de M, ou toute perturbation de même norme aux positions incertaines la reproduit-elle ?"`
- **Statut** : A_FAIRE

### Q-03 — E3 keysim « éliminé » avec incertitude
- **Question** : le E3 keysim « −0.0076, éliminé (devient négatif) » et la non-monotonie cap 0.25→0.1 à λ=1 (+0.023 vs +0.031) tiennent-ils une fois l'écart-type rapporté ?
- **Motivation** : VAL-10 — banc X8 (journal 2026-08-21) sans σ alors que les E3 antérieurs en ont un (σ par secret ≈ 0.006 : −0.0076 est compatible avec zéro) ; la non-monotonie contredit localement « le cap contrôle le signal » et n'est commentée nulle part.
- **Ce qu'elle trancherait** : la formulation de la ligne X8 du tableau §4 (« E3 éliminé » vs « E3 ≈ 0 ») et la robustesse locale de la théorie du cap.
- **Coût estimé** : XS — **Dépend de** : —
- **Lancer** : `/lab-run "Le E3 keysim de −0.0076 (banc X8) et la non-monotonie cap 0.25 → 0.1 à λ=1 survivent-ils au rapport de l'écart-type par secret ?"`
- **Statut** : A_FAIRE

### Q-02 — Variance de la famille E2
- **Question** : l'interaction E2 (−0.0551) et ses ablations (« gating 92 % », « DG +57 % », Hebb « nettement » −0.0948) survivent-elles à une estimation de la variance inter-documents et inter-découpages ?
- **Motivation** : VAL-03 — toute la famille E2 (journal 2026-08-20/21) repose sur des runs uniques par condition sur un document par registre ; E2n admet « égalité (runs uniques) » pour un écart de 0.003 mais « nettement » qualifie un écart de 0.04 sans plus de données ; le déterminisme d'E2 n'est attesté par aucune reproduction.
- **Ce qu'elle trancherait** : la ligne E2 du tableau §4, les pourcentages d'ablation, et l'arbitrage D5 (delta vs Hebb).
- **Coût estimé** : M — **Dépend de** : —
- **Lancer** : `/lab-run "L'interaction E2 de −0.0551 et ses ablations (gating 92 %, DG +57 %, Hebb −0.0948) survivent-elles à une estimation de variance inter-documents et inter-découpages ?"`
- **Statut** : A_FAIRE

### Q-04 — L'élagage, confondant non ablaté de D5
- **Question** : E2 avec `prune_every=0` donne-t-il la même interaction (à ±0.005) pour la delta rule et pour Hebb pur ?
- **Motivation** : BIO-05 — l'élagage (512 writes, keep 10 %) est une lésion discrète jamais ablatée ; E2 RFC = 1401 writes (2 élagages), les runs delta/Hebb comparés ont des comptes de writes différents donc des élagages différents ; l'homéostasie de Tononi & Cirelli est multiplicative, pas un seuil dur.
- **Ce qu'elle trancherait** : un confondant du débat D5 et une ligne d'ablation manquante du tableau §4.
- **Coût estimé** : S — **Dépend de** : —
- **Lancer** : `/lab-run "E2 avec prune_every=0 donne-t-il la même interaction à ±0.005 pour la delta rule et pour Hebb pur ?"`
- **Statut** : A_FAIRE

### Q-06 — Calibration du gate et coût de généralisation
- **Question** : un balayage de `gate_keysim_mid` ∈ {0.4, 0.5, 0.6, 0.7} ramène-t-il le ratio paraphrases de 0.38 vers ≥ 0.6 sans faire repasser E3 au-dessus de 0.05, sur GPT-2 et SmolLM2 ?
- **Motivation** : ROAD-12 — le coût du mécanisme RETENU (ratio 0.68 → 0.38 sous gate, journal X8) n'apparaît dans aucun tableau ; `gate_keysim_mid=0.6` est calibré sur GPT-2 (keysim d'écriture ≈ 0.14 sur Qwen vs ~0.23 GPT-2) et son commentaire config.py (« calibré par X9 ») est inexact.
- **Ce qu'elle trancherait** : le point d'opération du gate par modèle et le compromis sélectivité/généralisation du mécanisme par défaut.
- **Coût estimé** : S — **Dépend de** : —
- **Lancer** : `/lab-run "Un balayage de gate_keysim_mid entre 0.4 et 0.7 ramène-t-il le ratio paraphrases de 0.38 vers au moins 0.6 sans E3 au-dessus de 0.05, sur GPT-2 et SmolLM2 ?"`
- **Statut** : A_FAIRE

### Q-05 — Lecture et écriture aux mêmes positions
- **Question** : le dommage E3 par position est-il plus grand aux positions où l'écriture aurait déclenché (surprise > 4 nats) qu'aux autres ?
- **Motivation** : BIO-02 — P5 découpe par entropie de base, pas par surprise du token observé ; si lecture et écriture se disputent les mêmes positions, le système fait l'inverse de la séparation theta (encoder ET rappeler au même pas, là où le rappel coûte) et le coût initial E2 (+0.049 en 1ʳᵉ moitié) a un mécanisme candidat.
- **Ce qu'elle trancherait** : le statut de X3 (alternance encode/retrieve) et la lecture bio de la loi 2 (COR-08).
- **Coût estimé** : XS — **Dépend de** : Q-01
- **Lancer** : `/lab-run "Le dommage E3 par position est-il plus grand aux positions où l'écriture aurait déclenché (surprise > 4 nats) qu'aux autres ?"`
- **Statut** : A_FAIRE

## Réserve / file d'attente (hors plafond, prêtes pour /lab-run)

| ID | Question | Raison |
| --- | --- | --- |
| Q-08 | Sur Qwen2.5-1.5B, E1 à 10 secrets et layer ∈ {7, 14, 21}, au même régime que GPT-2/SmolLM2, reproduit-il le profil « milieu > tardif nocif » et la série d'amplification (+0.74 → +0.85 → +4.18) ? (ROAD-09 + VAL-01 ; trancherait D3 et requalifierait COR-01 ; coût S) — `/lab-run "Sur Qwen2.5-1.5B, E1 à 10 secrets et layer dans {7, 14, 21}, au même régime que GPT-2 et SmolLM2, reproduit-il le profil milieu > tardif nocif et la série d'amplification ?"` | au-delà du plafond (6) — sélectionnée par l'utilisateur, mise en réserve par le critère « fiabiliser d'abord » |
| Q-09 | Sur ≥ 20k tokens aux défauts X8, Hebb pur diverge-t-il (‖M‖ sans plateau, NLL finale > delta) ou reste-t-il à parité avec la delta rule ? (ROAD-08 ; clôturerait D5 — sinon la fermer par écrit sans ce run ; coût S) — `/lab-run "Sur au moins 20k tokens aux défauts X8, Hebb pur diverge-t-il ou reste-t-il à parité avec la delta rule ?"` | au-delà du plafond (6) — sélectionnée par l'utilisateur, mise en réserve par le critère « fiabiliser d'abord » |

## Questions non retenues dans ce cycle

| ID | Question | Raison |
| --- | --- | --- |
| Q-07 | À writes égaux sur E4s Qwen, une trace d'éligibilité post-surprise (écrire les n tokens suivant chaque pic) augmente-t-elle le gain Zephyr/Boreas là où un seuil abaissé ne le fait pas ? (BIO-08, coût M) | exclue par l'utilisateur (ronde de sélection, 2026-08-21) |

## Findings rejetés à la validation

| ID | Finding | Raison |
| --- | --- | --- |
| FID-01 | Défauts config vs CLAUDE.md/tableau §4 | dédupliqué — fusionné dans COR-03/COR-04/COR-05 (ROAD-01/02/03) |
| FID-02 | D10 absente de §3, §2.1 sans gate | dédupliqué — fusionné dans COR-06 (ROAD-04) |
| VAL-08 | Incohérences défauts/statuts/tableau entre documents | dédupliqué — même contenu que ROAD-01/02/03 |
| FID-04 | Lecture silencieusement nulle si keysim sans track_keys | rejeté — le défaut est déjà keysim donc `__post_init__` force track_keys=True avant toute mutation ; le scénario décrit n'existe nulle part dans le code actuel |
| LEAK-01 | E4 : NLL moyenne diluée sur phrases de longueurs différentes | rejeté (doublon) — déjà noté au journal E4 du 2026-08-21 comme affinement (a), et E4s (token décisif) l'a corrigé |
| VAL-09 | Winner's curse du balayage X0 ; « variance ÷1.6 » | rejeté — 0.886/0.566 = 1.57 arrondit correctement à 1.6 ; secrets communs donc comparaison appariée par construction ; retenir le meilleur point durcit la barre pour X1 |
| VAL-11 | X9 : corrélation poolée sur prédicteur saturé | rejeté — l'entrée X9 diagnostique elle-même la saturation et en tire la conclusion prudente (« prédicteur par fait mort ») |
| VAL-12 | Médiane et comptage de signes absents des résultats-titre E1 | rejeté — tambourine est suivi et commenté dans quasiment chaque entrée ; moyenne ± σ suffit, remarque cosmétique |
