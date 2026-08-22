# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pool de faits pour X9 (courbe de capacité) : 80 gabarits à contextes distincts
+ 80 secrets. Pré-requis posé par I1 : sans contextes distincts, les collisions de
gabarit confondent la mesure de capacité.

Génération combinatoire déterministe : OWNERS(16) × ENTITIES(20) × VERBS(5),
indexées par i mod n — lcm(16, 20, 5) = 80, donc les 80 triplets sont distincts.
"""

from __future__ import annotations

from functools import lru_cache

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


# =========================================================================
#  §16 — JEU D'UNITÉS DE v3 (amendement du 2026-08-22)
#
#  Cause : dans `fact_pairs`, `entity = i mod 20` et `verb = i mod 5`, et 5
#  divise 20 ⇒ le couple (entity, verb) a une PÉRIODE DE 20 : à N = 30 il
#  n'existe que 20 couples distincts, donc 10 collisions inévitables. Pour ces
#  paires l'owner est le seul discriminateur, et il est EFFACÉ en BPE dans deux
#  des trois types (para2 minuscule le premier caractère, para3 passe en
#  `OWNER_OBJ`). ⇒ `fact_pairs(30)` ne peut pas porter 30 unités identifiables.
#
#  Conditions de design dérivées (§16 C) :
#    C-1  (owner, entity) deux à deux distincts ;
#    C-2  (entity, verb)  deux à deux distincts ;
#    C-3  pour chaque owner retenu, sa forme para2 (minusculisée) ET sa forme
#         OWNER_OBJ partagent AU MOINS UN TOKEN BPE avec sa forme du fait —
#         **évaluée par exécution du tokenizer, jamais par jugement**.
#
#  Règle de sélection (§16 D), déterministe et pré-déclarée : les 30 PREMIERS
#  triplets (owner, entity, verb) dans l'ordre d'énumération
#      for e in range(len(ENTITIES)): for o in range(len(OWNERS)):
#          for v in range(len(VERBS))
#  qui satisfont C-1, C-2 et C-3. Premier conforme, aucun choix à la main. Si
#  moins de 30 triplets conformes existent : ARRÊT (fait sur le pool, à
#  consigner ; aucune condition n'est relâchée).
#
#  ADDITIF : `OWNERS`, `ENTITIES`, `VERBS`, `SECRETS_80`, `fact_pairs` restent
#  INTOUCHÉS (X9 reproductible). La table d'unités de v3 est cette fonction-ci.
# =========================================================================


@lru_cache(maxsize=1)
def _gpt2_encode():
    """Tokenizer GPT-2 seul (CPU, cache local) — AUCUN modèle n'est chargé.

    C-3 est « évaluée par exécution du tokenizer » (§16 D) : le jeu d'unités est
    donc une donnée gelée DÉFINIE par le BPE de GPT-2. Chargement PARESSEUX :
    `eval/capacity.py` n'importe que `SECRETS_80`/`fact_pairs` et ne doit pas
    payer transformers.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    return lambda s: [int(t) for t in tok.encode(s)]


def _tk(tokenize=None):
    return _gpt2_encode() if tokenize is None else tokenize


def owner_forms(owner: str) -> dict[str, str]:
    """Les trois formes de l'owner, DANS LA POSITION où chacune apparaît.

    - `fait`  : début de phrase (`"The captain's ship is called ..."`) ;
    - `para2` : précédée d'une espace (préfixe gelé qui finit par `"that "`) ;
    - `obj`   : précédée d'une espace (`" that belongs to "` + OWNER_OBJ).

    La position est portée par l'espace initiale parce que le BPE de GPT-2 en
    dépend : `"Her"` → `[9360]`, `" her"` → `[607]`.
    """
    return {"fait": owner,
            "para2": " " + _lower_first(owner),
            "obj": " " + OWNER_OBJ[owner]}


def owner_c3(owner: str, tokenize=None) -> tuple[bool, dict]:
    """C-3 pour un owner, PAR EXÉCUTION du tokenizer (§16 C)."""
    tk = _tk(tokenize)
    f = owner_forms(owner)
    t_fait = set(tk(f["fait"]))
    t_para2 = set(tk(f["para2"]))
    t_obj = set(tk(f["obj"]))
    det = {"owner": owner,
           "bpe_fait": sorted(t_fait), "bpe_para2": sorted(t_para2),
           "bpe_obj": sorted(t_obj),
           "partage_para2": sorted(t_fait & t_para2),
           "partage_obj": sorted(t_fait & t_obj)}
    det["ok"] = bool(t_fait & t_para2) and bool(t_fait & t_obj)
    return det["ok"], det


def v3_unit_triples_stats(n: int = POOL_UNITS_N, tokenize=None) -> dict:
    """Sélection §16 D, avec le compte de triplets EXAMINÉS et les motifs de
    rejet. `arret = True` si moins de `n` triplets conformes existent."""
    tk = _tk(tokenize)
    c3 = {o: owner_c3(OWNERS[o], tk)[0] for o in range(len(OWNERS))}
    triples: list[tuple[int, int, int]] = []
    seen_oe: set[tuple[int, int]] = set()
    seen_ev: set[tuple[int, int]] = set()
    examines = 0
    rejets = {"C-1": 0, "C-2": 0, "C-3": 0}
    for e in range(len(ENTITIES)):
        for o in range(len(OWNERS)):
            for v in range(len(VERBS)):
                if len(triples) == n:
                    break
                examines += 1
                if (o, e) in seen_oe:
                    rejets["C-1"] += 1
                    continue
                if (e, v) in seen_ev:
                    rejets["C-2"] += 1
                    continue
                if not c3[o]:
                    rejets["C-3"] += 1
                    continue
                triples.append((o, e, v))
                seen_oe.add((o, e))
                seen_ev.add((e, v))
            if len(triples) == n:
                break
        if len(triples) == n:
            break
    return {"n_demande": n, "triplets": triples, "n_retenus": len(triples),
            "triplets_examines": examines, "rejets": rejets,
            "owners_conformes_c3": sorted(o for o in c3 if c3[o]),
            "owners_non_conformes_c3": sorted(o for o in c3 if not c3[o]),
            "arret": len(triples) < n,
            "enumeration": "for e in ENTITIES: for o in OWNERS: for v in VERBS"}


def v3_unit_triples(n: int = POOL_UNITS_N, tokenize=None) -> list[tuple[int, int, int]]:
    """Les `n` triplets (indice owner, indice entity, indice verb) du §16 D.

    ARRÊT si moins de `n` triplets conformes existent : c'est un FAIT SUR LE
    POOL, à consigner — aucune condition n'est relâchée pour atteindre `n`.
    """
    st = v3_unit_triples_stats(n, tokenize)
    if st["arret"]:
        raise RuntimeError(
            f"§16 D — ARRÊT : seulement {st['n_retenus']} triplets conformes "
            f"C-1 ∧ C-2 ∧ C-3 sur {st['triplets_examines']} examinés, "
            f"{n} demandés. Fait sur le pool, à consigner.")
    return st["triplets"]


# ------------------------------------------------------- secrets de la table
# §16 D : la substitution du secret (§15 A-1) est RECALCULÉE sur la nouvelle
# table par la même règle — *le premier `SECRETS_80[j]`, j ≥ 30, qui passe
# V-tok*. Elle peut changer d'unité, ou disparaître.
# V-tok (§4.3), recopiée : (a) les 30 premiers tokens BPE des secrets deux à
# deux distincts ; (b) aucun ne coïncide avec le 1ᵉʳ token BPE d'un mot du pool ;
# (c) aucun secret n'est un mot du pool. Le « pool » est `OWNERS ∪ ENTITIES ∪
# VERBS` lu MOT À MOT — les tables sont gelées, donc la lecture est la même que
# celle de la porte (`eval/gate_bench.py::pool_words`), qui n'est pas modifiée.

def _pool_words() -> list[str]:
    out: set[str] = set()
    for e in list(OWNERS) + list(ENTITIES) + list(VERBS):
        out.add(e)
        out.update(e.split())
    return sorted(out)


def v3_unit_secrets_stats(n: int = POOL_UNITS_N, tokenize=None) -> dict:
    """Secrets de la table v3 + substitutions, par la règle du §16 D."""
    tk = _tk(tokenize)
    words = _pool_words()
    wordset = set(words)
    first = lambda s: int(tk(" " + s)[0])           # noqa: E731
    word_first = {first(w) for w in words}

    def conforme(s: str, pris: set[int]) -> bool:
        t = first(s)
        return (s not in wordset) and (t not in word_first) and (t not in pris)

    secrets, subs, pris = [], {}, set()
    for i in range(n):
        s = SECRETS_80[i]
        if conforme(s, pris):
            secrets.append(s)
            pris.add(first(s))
            continue
        j = n
        while j < len(SECRETS_80) and not conforme(SECRETS_80[j], pris):
            j += 1
        if j >= len(SECRETS_80):
            raise RuntimeError(f"aucun secret de remplacement conforme pour i={i}")
        subs[i] = j
        secrets.append(SECRETS_80[j])
        pris.add(first(SECRETS_80[j]))
    return {"secrets": secrets, "substitutions": subs,
            "regle": "premier SECRETS_80[j], j ≥ 30, qui passe V-tok"}


def v3_unit_secrets(n: int = POOL_UNITS_N, tokenize=None) -> list[str]:
    return v3_unit_secrets_stats(n, tokenize)["secrets"]


# ------------------------------------------------------ indices de la table

def _fact_and_exact(owner: str, entity: str, verb: str) -> tuple[str, str]:
    """MÊME gabarit de surface que `fact_pairs` (non modifiée) : seule
    l'indexation des trois slots change (§16 D)."""
    return (f"{owner} {entity} {verb} {{secret}}.", f"{owner} {entity} {verb}")


def pool_paraphrases(n: int = POOL_UNITS_N, tokenize=None) -> list[tuple[str, str, str]]:
    """n × 3 indices paraphrasés, par les règles déterministes gelées du §7 /
    §15 A-1, sur le jeu d'unités du §16.

    Chaque indice est un PRÉFIXE (la cible ` <secret>` est la continuation) —
    même convention que la question exacte de `fact_pairs`.
    """
    out = []
    for o, e, v in v3_unit_triples(n, tokenize):
        owner, entity, verb = OWNERS[o], ENTITIES[e], VERBS[v]
        para1 = f"{owner} {entity} {PARA1_VERB}"        # §15 A-1 : verbe global
        para2 = f"{PARA2_PREFIX}{_lower_first(owner)} {entity} {verb}"
        para3 = f"{PARA3_HEAD}{entity}{PARA3_MID}{OWNER_OBJ[owner]}{PARA3_TAIL}"
        out.append((para1, para2, para3))
    return out


def pool_paraphrases_pre_amendment(n: int = POOL_UNITS_N,
                                   tokenize=None) -> list[tuple[str, str, str]]:
    """ANCIENNE para1 (rotation `+1 mod 5`), conservée UNIQUEMENT comme
    contre-exemple ÉCHOUANT obligatoire de la porte `V-slot` (§15, A-4).

    Elle n'est utilisée par aucune mesure : `V-slot` doit échouer dessus de façon
    déterministe et sans aucun run — c'est la porte qui aurait tué le défaut
    avant que 150 triples ne soient comptés.
    """
    out = []
    for o, e, v in v3_unit_triples(n, tokenize):
        owner, entity, verb = OWNERS[o], ENTITIES[e], VERBS[v]
        verb_rot = VERBS[(v + 1) % len(VERBS)]          # ANCIENNE règle : +1 mod 5
        para1 = f"{owner} {entity} {verb_rot}"
        para2 = f"{PARA2_PREFIX}{_lower_first(owner)} {entity} {verb}"
        para3 = f"{PARA3_HEAD}{entity}{PARA3_MID}{OWNER_OBJ[owner]}{PARA3_TAIL}"
        out.append((para1, para2, para3))
    return out


def unit_slots(i: int, n: int = POOL_UNITS_N, tokenize=None) -> dict[str, str]:
    """Les **valeurs de ligne** des cinq tables indexées par l'unité `i` (§15, A-4).

    Une valeur de ligne d'une unité `j ≠ i` apparaissant verbatim dans
    `para_t(i)` est une **fuite structurelle** : l'indice de `i` porte alors le
    matériel qui identifie `j`, et la vérité-terrain est détruite.
    """
    o, e, v = v3_unit_triples(n, tokenize)[i]
    owner = OWNERS[o]
    return {"OWNERS": owner,
            "ENTITIES": ENTITIES[e],
            "VERBS": VERBS[v],
            "SECRETS_80": v3_unit_secrets(n, tokenize)[i],
            "OWNER_OBJ": OWNER_OBJ[owner]}


def all_row_values() -> list[str]:
    """Toutes les valeurs de ligne des cinq tables indexées par l'unité."""
    out = set(OWNERS) | set(ENTITIES) | set(VERBS) | set(SECRETS_80)
    out |= set(OWNER_OBJ.values())
    return sorted(out, key=lambda s: (-len(s), s))


def unit_table(n: int = POOL_UNITS_N, tokenize=None) -> list[dict]:
    """Table d'unités de v3 (§16) : triplet sélectionné, fait, indice exact,
    3 paraphrases, secret (substitution recalculée).

    `fact_pairs` n'est pas modifiée ; le jeu d'unités de v3 vit ICI.
    """
    triples = v3_unit_triples(n, tokenize)
    paras = pool_paraphrases(n, tokenize)
    secrets = v3_unit_secrets(n, tokenize)
    out = []
    for i, (o, e, v) in enumerate(triples):
        owner, entity, verb = OWNERS[o], ENTITIES[e], VERBS[v]
        tpl, exact = _fact_and_exact(owner, entity, verb)
        out.append(
            {"i": i,
             "triplet": [o, e, v],
             "owner": owner,
             "entity": entity,
             "verb": verb,
             "fact_template": tpl,
             "fact_no_secret": tpl.replace(" {secret}", ""),
             "exact": exact,
             "paraphrases": list(paras[i]),
             "slots": {"OWNERS": owner, "ENTITIES": entity, "VERBS": verb,
                       "SECRETS_80": secrets[i], "OWNER_OBJ": OWNER_OBJ[owner]},
             "secret": secrets[i]})
    return out


# ------------------------------------------------------------------ PEP 562
# Les constantes de la table v3 sont PARESSEUSES : leur valeur dépend du
# tokenizer GPT-2 (C-3), qu'aucun importateur de `SECRETS_80`/`fact_pairs` ne
# doit payer. `from pool import POOL_PARAPHRASES` continue de fonctionner.
_LAZY = {
    "POOL_PARAPHRASES": lambda: pool_paraphrases(POOL_UNITS_N),
    "POOL_UNIT_TRIPLES": lambda: v3_unit_triples(POOL_UNITS_N),
    "POOL_UNIT_SECRETS": lambda: v3_unit_secrets(POOL_UNITS_N),
    "POOL_UNIT_SECRET_SUBSTITUTIONS":
        lambda: v3_unit_secrets_stats(POOL_UNITS_N)["substitutions"],
}


def __getattr__(name):
    if name in _LAZY:
        value = _LAZY[name]()
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
