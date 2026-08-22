# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pool de faits pour X9 (courbe de capacité) : 80 gabarits à contextes distincts
+ 80 secrets. Pré-requis posé par I1 : sans contextes distincts, les collisions de
gabarit confondent la mesure de capacité.

Génération combinatoire déterministe : OWNERS(16) × ENTITIES(20) × VERBS(5),
indexées par i mod n — lcm(16, 20, 5) = 80, donc les 80 triplets sont distincts.
"""

from __future__ import annotations

OWNERS = [
    "The captain's", "Her", "His", "Grandma's", "The mayor's", "Our", "Their",
    "The wizard's", "The neighbor's", "My uncle's", "The museum's", "The village's",
    "The gardener's", "The company's", "The school's", "The pirate's",
]
ENTITIES = [
    "ship", "cat", "dog", "boat", "racehorse", "parrot", "robot", "band",
    "bakery", "sailboat", "dragon", "hamster", "yacht", "chess club", "greenhouse",
    "windmill", "lighthouse", "observatory", "tavern", "puppet",
]
VERBS = ["is called", "is named", "was christened", "answers to", "goes by the name"]

SECRETS_80 = [
    "swordfish", "obsidian", "marmalade", "zeppelin", "catapult", "lighthouse",
    "porcupine", "avalanche", "tambourine", "nebula", "bumblebee", "cathedral",
    "driftwood", "eucalyptus", "flamingo", "gargoyle", "harmonica", "iceberg",
    "jackal", "kaleidoscope", "labyrinth", "mandolin", "nightingale", "obelisk",
    "pendulum", "quicksand", "rhubarb", "saxophone", "telescope", "volcano",
    "walrus", "xylophone", "zucchini", "asteroid", "bagpipes", "chandelier",
    "dandelion", "espresso", "foghorn", "gondola", "hammock", "igloo",
    "jukebox", "kayak", "lantern", "mongoose", "narwhal", "octopus",
    "papyrus", "quiver", "raccoon", "sundial", "tornado", "unicorn",
    "vulture", "wheelbarrow", "anchovy", "boomerang", "cactus", "dolphin",
    "eclipse", "falcon", "glacier", "harpoon", "jasmine", "kangaroo",
    "lobster", "meteor", "nutmeg", "oregano", "python", "quartz",
    "radish", "sapphire", "thunderbird", "urchin", "velvet", "wombat",
    "mahogany", "periscope",
]
assert len(SECRETS_80) == 80
assert len(set(SECRETS_80)) == 80


def fact_pairs(n: int) -> list[tuple[str, str]]:
    """n paires (fait_avec_{secret}, question) à contextes distincts (n ≤ 80)."""
    assert n <= 80
    pairs = []
    for i in range(n):
        owner = OWNERS[i % len(OWNERS)]
        entity = ENTITIES[i % len(ENTITIES)]
        verb = VERBS[i % len(VERBS)]
        pairs.append((f"{owner} {entity} {verb} {{secret}}.", f"{owner} {entity} {verb}"))
    assert len(set(pairs)) == n
    return pairs


# =========================================================================
#  AJOUTS PUREMENT ADDITIFS — V2-D(a) v3
#  Protocole : experiments/EXP-2026-08-22-knn-borne-logits-v3.md §7
#
#  RIEN AU-DESSUS DE CETTE LIGNE N'EST MODIFIÉ. `OWNERS`, `ENTITIES`, `VERBS`,
#  `SECRETS_80` et `fact_pairs` sont GELÉS : `SECRETS_80` gèle la courbe de
#  capacité X9, le modifier casserait la reproductibilité d'un run publié
#  (décision PI §13.14).
# =========================================================================

POOL_UNITS_N = 30                 # §4.2 : N = 30 unités par bras

# ---------------------------------------------------------------- OWNER_OBJ
# §7, para3. Table GELÉE de 16 entrées : chaque OWNER vers sa forme OBJET, pour
# le génitif postnominal « the <entité> that belongs to <OWNER_OBJ> ». 16
# décisions, toutes FORCÉES par la grammaire anglaise (possessif prénominal →
# groupe nominal objet ; pronoms possessifs → pronoms objets : Her→her, His→him,
# Our→us, Their→them). Aucune n'est un choix de rédaction.
# Règle appliquée aux formes en 's : retrait de « 's » + minusculisation
# mécanique du premier caractère (§7 : « aucun nom propre dans OWNERS »).
OWNER_OBJ = {
    "The captain's": "the captain",
    "Her": "her",
    "His": "him",
    "Grandma's": "grandma",
    "The mayor's": "the mayor",
    "Our": "us",
    "Their": "them",
    "The wizard's": "the wizard",
    "The neighbor's": "the neighbor",
    "My uncle's": "my uncle",
    "The museum's": "the museum",
    "The village's": "the village",
    "The gardener's": "the gardener",
    "The company's": "the company",
    "The school's": "the school",
    "The pirate's": "the pirate",
}
assert len(OWNER_OBJ) == 16
assert set(OWNER_OBJ) == set(OWNERS)

# ------------------------------------------------------- table d'unités v3
# §7 : `SECRETS_80[i]` pour i ≠ 5, et `secret(5) := SECRETS_80[30]`.
# MOTIF : `SECRETS_80[5] == ENTITIES[16] == "lighthouse"` — le secret de l'unité 5
# est un mot présent dans les quatre indices de l'unité 16 (vérifié par exécution
# le 2026-08-22), ce qui rendrait un succès de l'unité 5 attribuable au datastore
# de l'unité 16 (porte V-tok).
# RÈGLE DÉTERMINISTE qui a produit la substitution : *le premier `SECRETS_80[j]`,
# j ≥ 30, qui passe V-tok* — soit `SECRETS_80[30] == "walrus"`.
POOL_UNIT_SECRET_SUBSTITUTIONS = {5: 30}
POOL_UNIT_SECRETS = [
    SECRETS_80[POOL_UNIT_SECRET_SUBSTITUTIONS.get(i, i)] for i in range(POOL_UNITS_N)
]
assert POOL_UNIT_SECRETS[5] == "walrus"
assert SECRETS_80[5] == ENTITIES[16] == "lighthouse"
assert len(set(POOL_UNIT_SECRETS)) == POOL_UNITS_N


# ------------------------------------------------- POOL_PARAPHRASES (§7)
# Règles de génération GELÉES (Neuro, tour 2). Principe : *l'arbitraire doit être
# global (un choix appliqué 30 fois), jamais local (30 choix)* — sinon la
# covariable C3 mesure le talent du rédacteur.
#
#   para1 — substitution du verbe attributif par un VERBE GLOBAL UNIQUE, hors
#           `VERBS` : `"bears the codename"` (§15, A-1). La rotation
#           `+1 mod 5` est SUPPRIMÉE : elle logeait dans l'indice de l'unité i le
#           matériel identifiant de l'unité i+1 (fuite STRUCTURELLE — la
#           vérité-terrain était détruite, pas seulement bruitée).
#   para2 — cadre + marqueur temporel : préfixe unique gelé
#           `"Years later, everyone still remembered that "` + owner minusculisé
#           (minusculisation mécanique du PREMIER caractère) + entité + verbe
#           D'ORIGINE (un seul facteur manipulé par type).
#   para3 — changement de cadre syntaxique et de registre :
#           `"So what's the name of the " + entity + " that belongs to "
#            + OWNER_OBJ[owner] + "? It's "`
#           Seul intrant par unité : `OWNER_OBJ`.
#
# Ces trois lignes sont RECOPIÉES À LA LETTRE du §7 du protocole ; elles ne sont
# pas « améliorées » ici (l'espace final de para3 y compris).
# §15, A-1 — verbe attributif GLOBAL aux 30 unités, hors des cinq tables
# indexées par l'unité. Vérifié par exécution le 2026-08-22 : aucune VALEUR DE
# LIGNE de `OWNERS`, `ENTITIES`, `VERBS`, `SECRETS_80`, `OWNER_OBJ` n'y apparaît
# verbatim (la règle A-4 porte sur les valeurs de ligne, pas sur les mots isolés :
# l'article « the » ne compte pas). Premier candidat conforme de la liste de repli
# déterministe `"bears the codename"` → `"translates as"` → `"carries the tag"`
# → `"denotes"` : c'est donc le premier qui est retenu, sans choix à la main.
PARA1_VERB = "bears the codename"
PARA1_VERB_FALLBACKS = ("bears the codename", "translates as",
                        "carries the tag", "denotes")
PARA2_PREFIX = "Years later, everyone still remembered that "
PARA3_HEAD = "So what's the name of the "
PARA3_MID = " that belongs to "
PARA3_TAIL = "? It's "
POOL_PARAPHRASE_TYPES = ("para1", "para2", "para3")


def _lower_first(s: str) -> str:
    """Minusculisation MÉCANIQUE du premier caractère (§7 : règle totale, aucun
    nom propre dans OWNERS)."""
    return s[:1].lower() + s[1:]


def pool_paraphrases(n: int = POOL_UNITS_N) -> list[tuple[str, str, str]]:
    """n × 3 indices paraphrasés, par les règles déterministes gelées du §7.

    Chaque indice est un PRÉFIXE (la cible ` <secret>` est la continuation) —
    même convention que la question exacte de `fact_pairs`.
    """
    assert n <= 80
    out = []
    for i in range(n):
        owner = OWNERS[i % len(OWNERS)]
        entity = ENTITIES[i % len(ENTITIES)]
        verb = VERBS[i % len(VERBS)]
        para1 = f"{owner} {entity} {PARA1_VERB}"        # §15 A-1 : verbe global
        para2 = f"{PARA2_PREFIX}{_lower_first(owner)} {entity} {verb}"
        para3 = f"{PARA3_HEAD}{entity}{PARA3_MID}{OWNER_OBJ[owner]}{PARA3_TAIL}"
        out.append((para1, para2, para3))
    return out


POOL_PARAPHRASES = pool_paraphrases(POOL_UNITS_N)
assert len(POOL_PARAPHRASES) == POOL_UNITS_N
assert all(len(p) == 3 for p in POOL_PARAPHRASES)


def pool_paraphrases_pre_amendment(n: int = POOL_UNITS_N) -> list[tuple[str, str, str]]:
    """ANCIENNE para1 (rotation `+1 mod 5`), conservée UNIQUEMENT comme
    contre-exemple ÉCHOUANT obligatoire de la porte `V-slot` (§15, A-4).

    Elle n'est utilisée par aucune mesure : `V-slot` doit échouer dessus de façon
    déterministe et sans aucun run — c'est la porte qui aurait tué le défaut
    avant que 150 triples ne soient comptés.
    """
    assert n <= 80
    out = []
    for i in range(n):
        owner = OWNERS[i % len(OWNERS)]
        entity = ENTITIES[i % len(ENTITIES)]
        verb = VERBS[i % len(VERBS)]
        verb_rot = VERBS[(i + 1) % len(VERBS)]          # ANCIENNE règle : +1 mod 5
        para1 = f"{owner} {entity} {verb_rot}"
        para2 = f"{PARA2_PREFIX}{_lower_first(owner)} {entity} {verb}"
        para3 = f"{PARA3_HEAD}{entity}{PARA3_MID}{OWNER_OBJ[owner]}{PARA3_TAIL}"
        out.append((para1, para2, para3))
    return out


def unit_slots(i: int, n: int = POOL_UNITS_N) -> dict[str, str]:
    """Les **valeurs de ligne** des cinq tables indexées par l'unité `i` (§15, A-4).

    Une valeur de ligne d'une unité `j ≠ i` apparaissant verbatim dans
    `para_t(i)` est une **fuite structurelle** : l'indice de `i` porte alors le
    matériel qui identifie `j`, et la vérité-terrain est détruite.
    """
    owner = OWNERS[i % len(OWNERS)]
    return {"OWNERS": owner,
            "ENTITIES": ENTITIES[i % len(ENTITIES)],
            "VERBS": VERBS[i % len(VERBS)],
            "SECRETS_80": SECRETS_80[POOL_UNIT_SECRET_SUBSTITUTIONS.get(i, i)],
            "OWNER_OBJ": OWNER_OBJ[owner]}


def all_row_values() -> list[str]:
    """Toutes les valeurs de ligne des cinq tables indexées par l'unité."""
    out = set(OWNERS) | set(ENTITIES) | set(VERBS) | set(SECRETS_80)
    out |= set(OWNER_OBJ.values())
    return sorted(out, key=lambda s: (-len(s), s))


def unit_table(n: int = POOL_UNITS_N) -> list[dict]:
    """Table d'unités de v3 : fait, indice exact, 3 paraphrases, secret substitué.

    `fact_pairs` n'est pas modifiée ; la substitution du secret 5 vit ICI.
    """
    pairs = fact_pairs(n)
    paras = pool_paraphrases(n)
    secrets = [SECRETS_80[POOL_UNIT_SECRET_SUBSTITUTIONS.get(i, i)] for i in range(n)]
    return [
        {"i": i,
         "owner": OWNERS[i % len(OWNERS)],
         "entity": ENTITIES[i % len(ENTITIES)],
         "verb": VERBS[i % len(VERBS)],
         "fact_template": pairs[i][0],
         "fact_no_secret": pairs[i][0].replace(" {secret}", ""),
         "exact": pairs[i][1],
         "paraphrases": list(paras[i]),
         "slots": unit_slots(i, n),
         "secret": secrets[i]}
        for i in range(n)
    ]
