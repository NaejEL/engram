---
name: lab-run
description: Lance un cycle du laboratoire IA (Director + experts Neuro/Math → protocole pré-enregistré avec gate humaine → Builder implémente et exécute → Verifier adversarial avec boucle de correction → interprétation croisée → journal) sur une question de recherche, le mot-clé "suivant", ou un protocole approuvé.
disable-model-invocation: true
argument-hint: "<question de recherche | suivant | chemin d'un protocole approuvé> [--models \"role=modele;...\"]"
---

# /lab-run — un cycle du laboratoire `engram`

Tu es la session principale : tu orchestres, tu écris les fichiers, tu parles au PI
(Jean). Les agents `lab-director`, `lab-neuro`, `lab-math` ne modifient rien ;
`lab-builder` touche au code et aux résultats ; `lab-verifier` n'y touche pas. Tu
n'exécutes ni tests ni évals toi-même hors des étapes prévues : tu délègues.

Repères du projet : journal `docs/JOURNAL.md` (chronologique ; entrée modèle en
tête du fichier ; la dernière entrée du fichier est `## 2026-08-20 — v0 : squelette
posé`, conservée en pied — les nouvelles entrées s'insèrent **juste avant elle**) ;
feuille de route et tableau des poids `docs/EXTENSIONS.md` §4 ; architecture
`docs/ARCHITECTURE.md` (jamais éditée par le labo) ; figés `docs/POSSIBLE_APPROACH.md`
et `docs/AFTER_v1_THOUGHTS.md` (jamais édités) ; tests
`.venv\Scripts\python -m pytest tests/ -q` ; interpréteur `.venv\Scripts\python` ;
GPU RTX 3060 6 Go. Date du jour : utiliser la date système.

Argument reçu : `$ARGUMENTS`.

## 0. Affectation des modèles

Lire `.claude/lab-models.json` (absent → tous les rôles en `inherit`). Clés :
`director.cadrage`, `director.interpretation`, `neuro`, `math`, `builder`,
`verifier`, `validator`. Valeurs : `inherit`, `opus`, `fable`, `sonnet`, `haiku` — toute clé absente ou valeur
inconnue vaut `inherit`.

Si `$ARGUMENTS` contient `--models "role=modele;..."`, appliquer ces paires **pour ce
cycle uniquement**, sans réécrire le fichier ; une valeur hors de la liste ci-dessus
est refusée avec un message explicite, jamais ignorée en silence.

À chaque lancement d'un sous-agent, passer le paramètre `model` de l'outil Agent avec
la valeur du rôle correspondant — **sauf si elle vaut `inherit`**, auquel cas ne pas
passer le paramètre du tout (l'agent suit alors le modèle de la session).

Annoncer l'affectation retenue en une ligne (rôle → modèle, et sa provenance : fichier
ou `--models`) **avant la première dépense**, pour que le PI puisse interrompre.

**Économie du modèle le plus cher.** Tout rôle affecté à autre chose qu'`inherit` est
une ressource rare : ne lui passer que le protocole, le compte rendu et les entrées de
journal réellement citées — jamais `docs/JOURNAL.md` entier ni les fichiers de
`experiments/results/` (ils restent au Builder et au Verifier) ; ne jamais le relancer
pour une reformulation cosmétique ; l'interprétation croisée (étape 6) n'a lieu
**qu'une fois par cycle**, après approbation du Verifier — jamais à chaque itération de
la boucle de correction.

## 1. Entrée

`$ARGUMENTS` privé de l'option `--models` :

- Si `$ARGUMENTS` est le chemin d'un fichier existant sous `experiments/` dont une
  ligne vaut exactement `Statut : PRE-ENREGISTRE` → aller directement à l'étape 4.
- Si ce fichier existe mais porte `Statut : PROPOSE` → reprendre à l'étape 3 (gate).
- Si ce fichier porte `Statut : TERMINE — …` → s'arrêter : « protocole déjà
  terminé », proposer `/lab-run suivant`.
- Sinon, `$ARGUMENTS` est une question de recherche, ou le mot-clé `suivant`.

## 2. Phase Cadrage

1. Lancer `lab-director` (Agent, subagent_type `lab-director`, modèle du rôle
   `director.cadrage`) en **mode cadrage** avec la question (ou `suivant`).
   Récupérer le brouillon de protocole.
2. Lancer **en parallèle, dans le même message**, `lab-neuro` et `lab-math` (modèles
   des rôles `neuro` et `math`), chacun avec le brouillon complet et les questions
   qui lui sont adressées. Récupérer les deux avis.
3. Relancer `lab-director` (rôle `director.cadrage`) avec son brouillon et les deux avis ; il rend le
   **protocole consolidé** (section « Arbitrage » + sections 1-13 + `Statut : PROPOSE`).

## 3. Gate humaine — pré-enregistrement

1. Poser au PI les *Questions pour le PI* du protocole consolidé (AskUserQuestion,
   une question par point, options concrètes quand c'est possible). Intégrer les
   réponses dans le protocole (texte, pas d'interprétation).
2. Écrire `experiments/EXP-<AAAA-MM-JJ>-<slug>.md` (slug : minuscules, tirets,
   ≤ 40 caractères, issu de la question) avec en première ligne après le titre
   `Statut : PROPOSE`, puis le protocole consolidé intégral, puis une section
   `## Historique` (`- <date> : proposé`).
3. Présenter au PI un résumé : question, hypothèse, tableau des prédictions chiffrées,
   critères d'abandon, budget (runs, durée, VRAM), livrables. Demander une
   **approbation explicite** (AskUserQuestion : « Approuver et pré-enregistrer » /
   « Modifier » / « Abandonner »).
4. Si « Modifier » : recueillir la demande, relancer `lab-director` avec le
   protocole et la demande, réécrire le fichier, reposer la question.
5. Si « Approuver » : remplacer `Statut : PROPOSE` par `Statut : PRE-ENREGISTRE`,
   ajouter `- <date> : pré-enregistré par le PI` à l'historique. **À partir de là,
   les sections Prédictions chiffrées et Critères d'abandon ne sont plus jamais
   modifiées, par personne.**
6. Si « Abandonner » : passer le statut à `ABANDONNE`, s'arrêter proprement.
7. **Headless** : si aucune interaction n'est possible (pas de réponse aux
   AskUserQuestion, ou exécution via `ci/lab.sh` / `ci/lab.ps1`), s'arrêter avec le
   message d'erreur explicite : `Cycle headless : un protocole PRE-ENREGISTRE est
   requis en argument ; aucune mesure sans pré-enregistrement.` Ne jamais mesurer
   sans protocole pré-enregistré.

## 4. Phase Build & Run

Lancer `lab-builder` (contexte frais, modèle du rôle `builder`) avec : le chemin du protocole pré-enregistré,
le rappel « exécuter toutes les conditions et contrôles, enregistrer les bruts dans
`experiments/results/<slug>/`, rendre un compte rendu brut sans interprétation ».
Récupérer le compte rendu brut. Ne pas lire les fichiers bruts toi-même : le compte
rendu suffit à la session.

## 5. Phase Verify — boucle de correction, 3 itérations maximum

Initialiser `iteration = 1`.

1. Lancer un **nouveau** `lab-verifier` (contexte vierge, modèle du rôle `verifier`) avec le chemin du protocole
   et le compte rendu brut courant. Récupérer son bloc JSON final.
2. Si `verdict == "APPROVED"` **et** `tests_passed`, `protocol_followed`,
   `rerun_consistent` valent tous `true` → sortir de la boucle.
3. Sinon : relancer `lab-builder` avec le chemin du protocole, **la liste complète
   des `issues`** (file, severity, kind, description) et la consigne « corriger chaque
   issue sans régresser, relancer les tests, ré-exécuter uniquement les runs
   affectés et les marquer rerun ». Récupérer le nouveau compte rendu brut ;
   `iteration += 1` ; revenir en 1.
4. Si `iteration > 3` sans approbation : **échouer explicitement**. Publier les issues
   restantes au PI ; rédiger (via `lab-director` en mode interprétation, verdict
   forcé `INCONCLUSIF — expérience invalidée`) une entrée de journal « expérience
   invalidée — <cause> » ; la soumettre à la gate de l'étape 7 ; passer le statut du
   protocole à `TERMINE — INVALIDE`. Ne jamais approuver par épuisement.

## 6. Phase Interprétation croisée

1. Lancer **en parallèle** `lab-neuro` et `lab-math` (modèles des rôles `neuro` et
   `math`) avec : le protocole, le compte
   rendu brut vérifié, le JSON du Verifier. Math recalcule au moins un chiffre clé ;
   Neuro confronte les courbes à ce que la biologie prédisait.
2. Lancer `lab-director` en **mode interprétation** (modèle du rôle
   `director.interpretation`) avec le protocole, le compte
   rendu brut, le JSON du Verifier et les deux avis. Récupérer : le verdict
   (`RETENU` / `REJETE` / `INCONCLUSIF`), le tableau prédit-vs-mesuré, l'entrée de
   journal rédigée, la ligne de tableau proposée, l'éventuelle décision D<n> candidate
   pour le PI, la prochaine expérience suggérée.

## 7. Gate humaine — consignation

1. Présenter au PI : verdict, tableau prédit-vs-mesuré, entrée de journal intégrale,
   ligne de tableau. AskUserQuestion : « Consigner » / « Modifier la rédaction » /
   « Ne pas consigner ».
2. Sur « Consigner » :
   - compléter l'entrée par une dernière ligne `*Modèles : director.interpretation
     <modèle>, math <modèle>, neuro <modèle>, builder <modèle>, verifier <modèle>.*`
     — quel modèle a rendu le jugement fait partie des conditions de reproductibilité,
     au même titre que les seeds ;
   - insérer l'entrée dans `docs/JOURNAL.md` **juste avant** la ligne
     `## 2026-08-20 — v0 : squelette posé` (ordre chronologique respecté, format
     exact de l'entrée modèle) ; créer le fichier avec l'en-tête du modèle s'il
     n'existait pas ;
   - ajouter la ligne proposée au tableau des poids (ou des ablations) de
     `docs/EXTENSIONS.md` §4, sans toucher au reste du document ;
   - passer le statut du protocole à `TERMINE — <verdict>` et ajouter
     `- <date> : terminé, verdict <verdict>` à son historique ;
   - **ne toucher à aucun autre document** : `docs/ARCHITECTURE.md`, `README.md`,
     `CLAUDE.md`, `docs/VISION.md`, `docs/POSSIBLE_APPROACH.md`,
     `docs/AFTER_v1_THOUGHTS.md` restent intacts ; si le Director recommande une
     décision D<n> ou une mise à jour d'« État du projet », la reporter dans le
     rapport final pour que le PI la fasse.
3. Sur « Modifier la rédaction » : recueillir la demande, relancer `lab-director`
   pour réécrire l'entrée (le verdict ne change pas sans nouveau chiffre), reposer
   la question.
4. **Headless** : écrire l'entrée de journal sans gate (le pré-enregistrement en
   était l'autorisation), en ajoutant à la fin de l'entrée la ligne
   `*(consignée automatiquement — à relire)*`.

## 8. Rapport final

Afficher, sans commiter :

- question et verdict ;
- chiffres clés vs prédictions pré-enregistrées (tableau) ;
- fichiers modifiés (`git status --short`) ;
- résultat de la commande de test (exit code) ;
- nombre d'itérations Builder/Verifier ;
- affectation des modèles effectivement utilisée (rôle → modèle, provenance) ;
- emplacement des résultats bruts (`experiments/results/<slug>/`) ;
- décisions/documents que le PI doit mettre à jour lui-même, le cas échéant ;
- prochaine expérience suggérée avec la commande exacte :
  `/lab-run "<question>"` ou `/lab-run suivant`.

Ne jamais commiter à la place du PI.
