# Archive DÉGRADÉE — `Qwen3.8-Flash-Next`, table d'embeddings n-grammes

> **À DÉPLACER dans `report/data/qwen38-flash-next/ARCHIVE-DEGRADEE.md` et à hasher** — la règle
> `deny` de permission (défaut **0-221**) interdit à la session principale d'écrire sous `report/data/`.

> **CE N'EST PAS UNE COPIE DU DOCUMENT.** Sous **D38**, une source externe se cite par **copie
> archivée et hashée**. Cet artefact n'en est pas une : il porte des **extraits verbatim** obtenus par
> requête HTTP, **traités par un modèle de langage** avant restitution. Le document brut n'a pas été
> capturé, **aucun SHA-256 de la page d'origine n'est disponible**, et la mise en page, les tableaux et
> le contexte éditorial sont **perdus**.
>
> **Statut : `descriptif`. Étiquette : `externe-archivé-dégradé`.** Cette archive **satisfait moins que
> la lettre de D38** et **davantage qu'une URL vivante** — elle fige la formulation exacte, sa source et
> sa date. **Décision PI du 2026-09-04, prise en connaissance de cette limite.**
>
> **Un lecteur qui veut la source primaire doit ouvrir l'URL ci-dessous et constater lui-même** ; si
> elle a changé ou disparu, **cet artefact ne permet pas de le prouver**. C'est exactement la faiblesse
> que D38 a été gravée pour interdire, et elle est ici **déclarée, non masquée**.

## Provenance

| Champ | Valeur |
| --- | --- |
| **URL** | `https://www.marktechpost.com/2026/08/26/alibabas-qwen-team-releases-qwen3-8-flash-next-a-125b-multimodal-moe-with-6b-active-parameters-previewing-the-qwen4-architecture/` |
| **Date de publication de la source** | 2026-08-26 |
| **Date de consultation** | 2026-09-05 |
| **Méthode** | requête HTTP suivie d'une extraction par modèle de langage — **pas une capture brute** |
| **Consulté par** | session principale (coordinateur), cycle de rédaction |
| **Sources concordantes NON archivées** | LMSYS/SGLang · vLLM Recipes · dépôt `QwenLM/Qwen3.8-Flash-Next` — **citées pour mémoire, non capturées, non hashées** |

## Extraits VERBATIM

> **« A 20,000,000-entry bigram/trigram table at layer 2 adds capacity through deterministic
> lookups. »**

> **« It can be offloaded to host memory with asynchronous prefetch — though offload currently runs
> only on NVIDIA devices. »**

## Chiffres, tels que rendus par l'extraction

| `id` proposé | Valeur | Remarque |
| --- | --- | --- |
| `N-45` | **180 B** sur disque | décomposé ci-dessous |
| `N-46` | **125 B** backbone | |
| `N-47` | **51 B** embeddings n-grammes | |
| `N-48` | **4 B** multi-token prediction | *composante que la première transmission au laboratoire avait OMISE* |
| `N-49` | **6 B** actifs par token | |
| `N-50` | **20 000 000** entrées | verbatim ci-dessus |
| `N-51` | **couche 2** | verbatim ci-dessus |

## CE QUE LA SOURCE NE DIT PAS — et qui borne la section 6

**L'article n'indique PAS** si la table est **entraînée puis figée**, ou **inscriptible à l'inférence**.
L'extraction le déclare explicitement : *« The article does not explicitly state whether the N-gram
embedding table is trained during initial training, remains fixed afterward, or is writable at
inference time. »*

**Conséquence, à porter dans le même bloc que toute citation de cette source :**

- Formulation **INTERDITE** : *« la table est figée à l'inférence »* — **la source ne l'établit pas**.
- Formulation **LICITE, et seule licite** : ***« la source ne se prononce pas sur la mutabilité de la
  table à l'inférence ; la question test-time-inscriptible reste ouverte — et elle reste ouverte parce
  que RIEN N'EST DIT, non parce que quelque chose serait établi. »***

## Les deux bornes obligatoires — interdit **(xxxi)**, dans le même paragraphe

> **Borne 1.** La table est à la **couche 2** ; `engram` s'accroche aux couches **6, 16 et 14**. Un
> étage de lookup **précoce** adressé par n-grammes et un étage plastique en **milieu de pile** ne sont
> **pas au même endroit du réseau** : la convergence est réelle et **ne dit rien du locus de capture**.
>
> **Borne 2.** Portée exacte de ce que cette source établit : ***« un étage de lookup adressé par la
> surface est industriellement utile »***. Elle **n'établit pas** que le locus d'`engram` est écrasé par
> la surface — cela est établi par les **trois instruments du laboratoire, et par eux seuls** (v4
> `R1 = 1.0000` sur 180/180 × 3 ; recouvrement du 2026-08-26 ; Étape A `R-nul` 24/24). **Confondre les
> deux échangerait une mesure contre un argument d'autorité.**
