# Vision — cas d'usage cibles

Ce document décrit où le mécanisme pourrait aller SI les verrous tombent. Chaque
usage cite ses prérequis (numéros d'expérience) et ce qui le tuerait — même
honnêteté que le journal : des mécanismes et des conditions, pas de marketing.
État des verrous au 2026-08-21 : X9 mesuré (pas de falaise à 80 faits), X8 en
validation finale (E1/E3 conformes sur GPT-2 et SmolLM2, E2 SmolLM2 en cours),
E4 rédigée en attente du verdict X8.

## Principe directeur

Ce qui distingue M n'est pas la performance, ce sont ses propriétés d'**artefact** :

- **apprentissage sans backprop** — viable sur hardware faible (tout ce repo tourne
  sur un laptop RTX 3060) ;
- **mémoire diffuse** — sémantique, pas exacte (ratio paraphrases 0.68, E1b) ;
- **sérialisable** — un fichier de quelques Mo (25 Mo fp32 pour GPT-2, moins en fp16) ;
- **effaçable totalement et vérifiablement** — M ← 0 est une opération, pas une
  promesse ;
- **additive** — M vit dans un espace linéaire : deux mémoires s'additionnent.

Les usages en découlent. Ceux qui exigeraient une mémoire **exacte** sont hors
périmètre : le rappel flou y devient un générateur d'hallucinations plausibles —
pour l'exact, c'est RAG ou contexte (hiérarchie registre/cache/disque, README
§« Vision d'usage »).

## a) Engram-par-projet

Une M sérialisée par projet, chargée/déchargée au changement de repo (le reset
devient un switch). Porte le contexte **diffus** — conventions de code, décisions
d'architecture récurrentes, corrections répétées, vocabulaire du domaine — qu'on
re-paye aujourd'hui en préambule type CLAUDE.md à chaque session, et qui ne sert
pas par son exactitude mais parce qu'il teinte le modèle (`h ← h + λ·M·φ(h)`,
littéralement).

- **Prérequis** : X8 (fait) + E4. **Statut (2026-08-21, E4 puis E4s)** : sévèrement
  dégradé. E4s (conventions arbitraires, anglais simple, token décisif — l'excuse
  du juge éliminée) échoue dans ses 8 variantes : la teinte diffuse ne produit pas
  de préférence token-niveau pour la convention (trace spécifique ~+0.05, noyée
  sous une dérive non spécifique). Racine identifiée : l'absence de composante
  directionnelle de la lecture (X7) — le même mur que E1c et le top-10. Ce qui
  survit : l'association quasi-verbatim et le régime rappel/adaptation (E1/E2).
  Dernier test en attente : E4-dur sur Qwen. Le sauvetage plausible passe par le
  rappel directionnel (v2).
- **Tué par** : échec E4-dur sur Qwen ET absence de rappel directionnel en v2 —
  la falaise X9 est, elle, déjà écartée (rien à 80 faits).

## b) Personnalisation edge respectueuse de la vie privée

Adaptation sur l'appareil : pas de backprop donc pas de GPU d'entraînement, pas de
télémétrie. L'argument central est **M ← 0 comme droit à l'oubli mécanique** — une
opération dont l'effet est total et vérifiable, pas une promesse de politique de
confidentialité. La mémoire est un fichier local que l'utilisateur inspecte,
sauvegarde ou détruit. Aucun fine-tuning ne peut offrir ça.

- **Prérequis** : X8, et un cortex quantifié qui tient sur la cible (à mesurer :
  le comportement de M sur un cortex int4/int8 n'a pas de numéro d'expérience —
  en créer un avant toute promesse).
- **Tué par** : un E3 qui ne tient pas sur cortex quantifié, ou un coût de la
  boucle token-par-token rédhibitoire sur la cible.

## c) Fusion et diff d'engrams

M vit dans un espace linéaire : additionner deux matrices superpose deux mémoires.
Ouvre : M d'équipe par merge, mémoires versionnées dans git, diff entre M-lundi et
M-vendredi pour voir ce que la semaine a appris. Une mémoire d'IA manipulable comme
un artefact — mergeable, diffable, auditable — est une propriété qu'aucun fine-tune
n'a.

- **Prérequis** : E5 (candidate, à rédiger) — rappel après merge de deux M chargées
  indépendamment, comparé aux deux M séparées. Aucune promesse avant ce numéro.
- **Limite connue** : la falaise d'interférence borne le nombre de contributeurs.
  X9 donne le premier ordre de grandeur (80 faits sans dégradation sur GPT-2) mais
  n'a testé qu'une M unique — le merge superpose aussi les *decay/prune* divergents,
  ce qu'E5 devra regarder.
- **Tué par** : un E5 où le merge détruit le rappel des deux sources (interférence
  additive), ou ne survit pas à des historiques d'élagage différents.

## d) Agent qui n'insiste pas

Un échec en cours de session pénalise la piste morte dans M immédiatement — sans
re-prompt, sans contexte gonflé. C'est le motif originel du projet (« je me rends
compte que ça ne marche pas, j'apprends maintenant »), et le plus dur : il exige un
signal de **valence**. Le gating par surprise dit « c'est nouveau », pas « c'est
faux » — pénaliser demande de savoir que la piste a échoué, pas juste qu'elle a
surpris.

- **Prérequis** : la question ouverte « valence » (notée dans EXTENSIONS §V2,
  écriture involontaire) doit devenir une spec avec son numéro d'expérience. Rien
  n'existe aujourd'hui.
- **Tué par** : l'absence de signal de valence extractible en ligne, ou E1c — la
  correction d'un fait que le cortex croit connaître a déjà échoué dans tous les
  modes (limite directionnelle X7) ; « désapprendre une piste » pourrait buter sur
  le même mur. À tester avant d'y croire.

## e) Instrument de recherche

E1–E4 comme petit benchmark reproductible sur laptop pour comparer des mécanismes
de mémoire test-time : stabilité/plasticité (E3), capacité (X9), arbitrage mémoire
vs modèle du monde (X8/E1c). L'usage le plus sobre et peut-être le plus réaliste :
tout le contenu de ce repo — protocoles pré-enregistrés, critères d'échec chiffrés,
ablations une-par-une — est déjà cet instrument.

- **Prérequis** : aucun nouveau. C'est l'état actuel du repo.
- **Tué par** : rien de spécifique — au pire, il reste un négatif propre et
  documenté.
