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
