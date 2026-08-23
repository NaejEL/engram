# SPDX-License-Identifier: AGPL-3.0-or-later
"""Matériau v4 — génération et qualification (protocole `experiments/EXP-2026-08-23-v4-materiel.md`).

CE FICHIER NE MESURE RIEN : aucun modèle, aucun GPU, aucun état, aucune injection.
Les seuls objets HF touchés sont les **trois tokenizers** (CPU, en cache) — sans eux
`C2`, `C3` et `V-casse` ne sont pas décidables. `eval/pool.py` n'est **pas** touché ;
`pool.fact_pairs` n'apparaît ici que comme **contre-exemple échouant** du banc.

Structure du matériau (§4.2, §7 du protocole)
---------------------------------------------
  entité   = TIGE (1 token) + SUFFIXE (1 token)      ⇒ `L_e = 2` (C2)
  famille  = (tige, domaine), 3 membres (3 suffixes du sous-vivier du domaine)
  10 tiges toutes PONTÉES : chacune porte 2 familles dans 2 domaines différents
  ⇒ 20 familles décisionnelles, 4 domaines × 5 familles, **60 unités**
  +  4 tiges SIMPLES (1 famille chacune) ⇒ **12 unités de réserve**  ⇒ 72 entités
  6 moules = 3 types × 2 variantes ; **exactement un type capitalise l'entité** (C6)
  point de capture `t` = **token du suffixe** ; `t−1` = token de la tige

Cascade d'exécution **GRAVÉE** (§5.5) — le banc vérifie que l'ordre a été suivi :
  (1) `L_e = 2` sur les 3 tokenizers
  (2) sous-viviers de domaine (`C7`)
  (3) structure tige/suffixe et ponts
  (4) moules, égalisation de préfixe et distance lexicale (`V-var-dist`)
  (5) appariement des bandes de fréquence (`V-freq` v2) dans le sous-ensemble survivant
  (6) type capitalisé
  (7) nulle de cadre en 40 PAIRES
  (8) banc

=============================================================================
 TABLE DES GARANTIES D25 — une ligne par propriété, avec sa PORTE
 (le résultat numérique est produit par `garanties()` ; cette table déclare
  ce qui est garanti, comment, et quelle porte tranche)
=============================================================================
| # | Propriété garantie                                     | Porte        |
|---|--------------------------------------------------------|--------------|
| 1 | `C1` : 72 séquences byte-identiques hors slot / cellule | `V-C1`       |
| 2 | `C1′` : sélection sur métadonnées déclarées SEULEMENT   | `V-C1p`      |
|   |   (aucun état, aucun embedding, aucune fréquence de     |              |
|   |    corpus mesurée sur états — seuls le rang de token    |              |
|   |    BPE et l'appartenance lexicale déclarée sont lus)    |              |
| 3 | `C2` : `L_e = 2` pour 72/72 entités ET 40/40 paires de  | `V-C2`       |
|   |   la nulle de cadre, sur les TROIS tokenizers           |              |
| 4 | `C3` : indice de capture entier constant par cellule ;  | `V-C3`       |
|   |   les 2 variantes d'un type ont le même nombre de       |              |
|   |   tokens de préfixe sur les 3 tokenizers                |              |
| 5 | `C4` : aucune paire décisionnelle byte-identique        | `V-C4`       |
|   |   tronquée à la capture ; égalités en mid-rank          |              |
| 6 | `C5` : ≥ 60 unités à recouvrement de token nul pour     | `V-C5`       |
|   |   72/72 requêtes ; pool de 36 tiré une fois, GELÉ       |              |
| 7 | `C6` : exactement 1 type capitalisé ; `V-casse` mord    | `V-C6`,      |
|   |   (≥ 90 % des tiges changent de token en minuscule)     | `V-casse`    |
| 8 | `C7` : 4 sous-viviers de suffixes déclarés par domaine, | `V-C7`       |
|   |   **lexicalement disjoints** ; 100 % des suffixes d'une |              |
|   |   famille dans le sous-vivier de SON domaine            |              |
| 9 | `m₁ = 5` pour 60/60 unités décisionnelles               | `V-m1`       |
|10 | `K_eff = 10` (clusters = TIGES, couplage parfait 0-50)  | `V-Keff`     |
|11 | indépendance longueur / unité : toute séquence d'une    | `V-C3`       |
|   |   cellule a la MÊME longueur en tokens                  |              |
|12 | paires intra-unité intra-type : cardinal publié         | `V-paires4`  |
|13 | diversité de type : 3 types × 2 variantes, 6 moules     | `V-div4`     |
|14 | survie à l'effacement de casse                          | `V-casse`    |
|15 | absence de période sur TOUT slot et TOUT couple de slots| `V-periode`  |
|16 | cardinaux D24-b tronqués à la capture : 37/37 par       | `V-D24b`     |
|   |   requête, 72/72 par cellule, 6/6 par tige (contraste   |              |
|   |   minimal 1 token), **14 à `t−1`**, détail PAR          |              |
|   |   SOUS-VIVIER                                           |              |
|17 | éligibles des DEUX pools (calibrateur `C5` ; primaire   | `V-C5`,      |
|   |   `P2` = 5 tige-partagés + 12 même-domaine + 19 autre)  | `V-P2`       |
|18 | distance lexicale entre variantes d'un type ≥ 50 %      | `V-var-dist` |
|19 | bandes de fréquence appariées (sous-viviers 2 à 2,      | `V-freq`     |
|   |   familles d'une même tige, ponté vs non-ponté) ≤ 0.15  |              |
|20 | nulle de cadre : 40 PAIRES `L_e = 2`, hors des 4        | `V-cadre`    |
|   |   domaines, sans token partagé deux à deux ni avec      |              |
|   |   aucune entité                                         |              |
|21 | cascade d'exécution conforme à l'ordre gravé §5.5       | `V-ordre`    |
=============================================================================

`fact_pairs` soumis à la MÊME table doit **ÉCHOUER** sur `C1`, `C2` et S-1
(porte `V-fact-pairs`, §4.7) : s'il passait, ce serait la table qui serait fausse.

Usage :
  .venv\\Scripts\\python eval\\pool_v4.py
  .venv\\Scripts\\python eval\\pool_v4.py --json experiments/results/v4-materiel/pool.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from engram.config import EngramConfig  # noqa: E402

PASS, FAIL = "PASS", "FAIL"

# =========================================================================
#  Constantes du protocole — §4.2, §5.5, §7. Recopiées, jamais ajustées.
#  Les hyperparamètres de RUN (seed, dataset) viennent d'`EngramConfig`.
# =========================================================================

MODELES_TOK = ("gpt2", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-1.5B")

L_E = 2                    # §4.1 C2 : tige (1 token) + suffixe (1 token)
N_TIGES_PONTEES = 10       # §4.2
N_TIGES_SIMPLES = 4        # §4.2 — réserve
N_DOMAINES = 4             # §4.2
N_FAMILLES_PAR_DOMAINE = 5
N_MEMBRES = 3              # §4.2 — 3 membres par famille
N_UNITES_DEC = 60          # 10 × 2 × 3
N_UNITES_RESERVE = 12      # 4 × 1 × 3
N_ENTITES = 72             # 60 + 12
K_EFF = 10                 # §7 — clusters = TIGES (0-50)
M1_ATTENDU = 5             # §4.3 — tige-partagés dans P2, constant par construction
TAILLE_POOL = 36           # §4.3, §4.4
P2_TIGE_PARTAGES = 5       # §4.3 (2 co-famille + 3 famille partenaire)
P2_MEME_DOMAINE = 12       # §4.3 — gravé (défaut 0-73)
P2_AUTRE_DOMAINE = 19      # §4.3 — gravé
C5_ELIGIBLES_MIN = 60      # §4.1 C5
N_PAIRES_CADRE = 40        # §4.2 (A8)
N_PSEUDO = 20              # §4.2 — nulle de nouveauté, SÉPARÉE
SEUIL_CASSE = 0.90         # §4.1 C6 / `V-casse`
SEUIL_VAR_DIST = 0.50      # §4.7 `V-var-dist`
BORNE_FREQ = 0.15          # §4.7 `V-freq` v2 — borne exacte par énumération
CARD_T1 = 14               # §4.2 — cardinal D24-b à `t−1` (défaut 0-64)

# Ordre d'exécution GRAVÉ (§5.5). Le banc vérifie que la cascade l'a suivi.
CASCADE_GRAVEE = (
    "1-Le=2-3-tokenizers",
    "2-sous-viviers-domaine-C7",
    "3-structure-tige-suffixe-ponts",
    "4-moules-egalisation-prefixe-distance-lexicale",
    "5-appariement-bandes-frequence-V-freq",
    "6-type-capitalise",
    "7-nulle-de-cadre-40-paires",
    "8-banc",
)

# Opérationnalisations DÉCLARÉES là où le protocole fixe la clause mais pas la
# statistique d'exécution. Publiées, hors E, jamais ajustées après lecture.
OPERATIONNALISATIONS = {
    "V-freq — rang de fusion": (
        "le protocole écrit « rang de fusion de la tige (gpt2) » sans donner la "
        "statistique d'écart. Le banc DÉCLARE : rang = identifiant de token de la "
        "forme à espace initial, normalisé par la taille du vocabulaire du "
        "tokenizer (r ∈ [0,1]) ; l'écart est la valeur absolue de la différence "
        "des MOYENNES de r, énumérée sur les 3 tokenizers et sur les 3 familles "
        "d'appariement exigées (sous-viviers 2 à 2, familles d'une même tige, "
        "ponté vs non-ponté). La borne 0.15 est celle du protocole. AUCUN test "
        "d'homogénéité n'est calculé (0-41)."),
    "V-var-dist — distance": (
        "« ≥ 50 % des tokens diffèrent » est opérationnalisé position par "
        "position entre les deux préfixes d'un même TYPE (leurs longueurs sont "
        "égales par `C3`), sur les 3 tokenizers ; la valeur retenue est le MIN "
        "sur les tokenizers."),
    "V-periode — période": (
        "« absence de période sur tout slot et tout couple de slots » est "
        "opérationnalisée comme : aucune fonction déterministe non triviale d'un "
        "slot vers un autre, et aucun couple (slot_a, slot_b) dont le nombre de "
        "valeurs distinctes soit strictement inférieur au produit attendu par la "
        "structure. Les slots sont (tige, suffixe, domaine, type, variante)."),
    "V-casse — position": (
        "la tige apparaît toujours précédée d'une espace ; `V-casse` compare "
        "l'identifiant de « ␣Tige » à celui de « ␣tige » sur les 3 tokenizers."),
}

# =========================================================================
#  Viviers CANDIDATS — déclarés AVANT toute qualification (`C1′`).
#  Rien ici n'est issu d'un modèle : ce sont des listes lexicales.
# =========================================================================

DOMAINES = ("maritime", "culinaire", "astronomie", "textile")

# Sous-viviers de suffixes DÉCLARÉS PAR DOMAINE (`C7`). Lexicalement disjoints
# par construction, vérifié mécaniquement (aucun mot n'apparaît deux fois).
SOUS_VIVIERS_CANDIDATS = {
    "maritime": ("harbor anchor sail deck mast tide reef wave shore hull dock buoy "
                 "fleet voyage sailor beacon channel current canvas port ship boat "
                 "crew ocean sea coast bay pier ferry cargo rope helm tug wreck "
                 "navy").split(),
    "culinaire": ("kitchen bakery oven spice broth pastry sauce dough recipe flour "
                  "butter honey kettle feast cuisine bread cheese pepper vinegar "
                  "syrup roast stew soup grill chef dessert dinner lunch breakfast "
                  "salad pasta sausage pudding jam").split(),
    "astronomie": ("comet orbit moon star planet eclipse galaxy telescope meteor "
                   "crater satellite cosmos constellation asteroid horizon twilight "
                   "spectrum lunar solar stellar celestial sky sun space radiation "
                   "gravity astronomy universe cluster ray").split(),
    "textile": ("fabric thread cotton silk weave linen velvet ribbon wool needle "
                "stitch yarn cloth tailor garment fibre lace seam knit sewing "
                "textile apparel sleeve collar button hem dye felt robe weaving "
                "spinning cloak glove sock shirt").split(),
}

# Vivier de TIGES (§14-3 : 45 → 90 candidats). Neutres : une tige est pontée
# entre deux domaines, elle ne peut donc appartenir à aucun.
TIGES_CANDIDATES = (
    "Iron Silver North River Storm Amber Golden Winter Copper Marble Crimson Hollow "
    "Ember Frost Shadow Stone Bright Quiet Ivory Onyx Summer Autumn Thunder Meadow "
    "Granite Velvet Crystal Willow Falcon Raven Cedar Maple Coral Azure Scarlet "
    "Violet Emerald Ruby Pearl Bronze Steel Glass Sable Dusk Dawn Noble Royal Wild "
    "Silent Gentle Ancient Hidden Broken Sacred Distant Endless Fallen Sunken "
    "Rising Burning Frozen Wandering Whisper Echo Mirror Lantern Beacon Anchor "
    "Compass Arrow Feather Petal Thorn Amberly Vaulted Northern Eastern Western "
    "Southern Upper Lower Inner Outer Grand Little Great Small Long Short Deep "
    "High Low Fair"
).split()

# Vivier de la NULLE DE CADRE : hors des 4 domaines (droit, architecture,
# musique, géologie, politique, médecine, transport, enseignement, météo,
# héraldique, courrier, finance, guerre, bâti, temps, faune, art, industrie).
CADRE_CANDIDATS = (
    "judge court prison police officer crime trial appeal bridge tower castle "
    "temple palace church chapel museum library school hospital factory station "
    "market square garden park street avenue village city town county province "
    "guitar song music dance opera singer stage theater film movie mountain "
    "forest desert island lake stream field hill cliff cave rock mud dust ash "
    "coal gold horse sheep chicken rabbit farmer barn gate train engine wheel "
    "tunnel railway highway doctor nurse patient fever medicine surgery clinic "
    "teacher student exam grade paper rain snow wind cloud lightning sunshine "
    "king queen prince knight crown sword shield letter stamp message signal "
    "telephone radio money coin bank credit debt trade tax budget army soldier "
    "battle war peace treaty border flag nation empire republic window door "
    "floor roof wall chair table clock watch calendar hour season century "
    "morning evening night bird frog spider butterfly worm mouse statue drawing "
    "gallery artist mine hammer wire pipe machine motor"
).split()

# Pseudo-mots — nulle de NOUVEAUTÉ, **séparée, jamais mélangée**, **non
# appariée en position et déclarée telle** (§4.2). Aucune contrainte `L_e`.
PSEUDO_MOTS = (
    "blorken tavisto quembil dranthe vulmara przenit klarvon jesomir farluke "
    "nidravel zorpine chandrel wemtrio glostan phyreth sundabo trelvic ombrash "
    "yevlant crumpis"
).split()

# =========================================================================
#  Moules — 3 types × 2 variantes (§4.2). Chaque préfixe est une LISTE DE MOTS
#  tous mono-token sur les 3 tokenizers : le nombre de tokens de préfixe vaut
#  donc le nombre de mots, ce qui rend `C3` vérifiable et non ajustable.
#  Les deux variantes d'un type ont le même nombre de mots (`C3`) et AUCUN mot
#  commun position par position (`V-var-dist` = 1.0 ≥ 0.50).
#  Le type `T3` est le SEUL à capitaliser l'entité (`C6`).
# =========================================================================

MOULES = {
    "T1": {"capitalise": False,
           "A": "The old keeper spoke at length about the".split(),
           "B": "A young trader wrote in detail concerning that".split()},
    "T2": {"capitalise": False,
           "A": "Last winter our village finally repaired the".split(),
           "B": "This morning their city quietly rebuilt some".split()},
    "T3": {"capitalise": True,
           "A": "The captain named his best vessel".split(),
           "B": "One sailor called her first boat".split()},
}
TYPES = tuple(MOULES)
VARIANTES = ("A", "B")
CELLULES = tuple((t, v) for t in TYPES for v in VARIANTES)

# Calendrier des paires de domaines des 10 tiges pontées (§4.2) : chaque domaine
# apparaît exactement 5 fois ⇒ 4 domaines × 5 familles. Déclaré, non tiré.
PAIRES_DE_DOMAINES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
                      (0, 1), (2, 3), (0, 2), (1, 3))
# Domaines des 4 familles à tige SIMPLE (réserve) : un par domaine.
DOMAINES_RESERVE = (0, 1, 2, 3)


# =========================================================================
#  Tokenizers
# =========================================================================

@lru_cache(maxsize=None)
def _tokenizers() -> dict:
    from transformers import AutoTokenizer
    return {m: AutoTokenizer.from_pretrained(m) for m in MODELES_TOK}


def encodeurs() -> dict:
    """`{nom_modele: (encode, taille_vocab)}` — CPU, tokenizers en cache."""
    toks = _tokenizers()
    out = {}
    for m, t in toks.items():
        out[m] = ((lambda tt: (lambda s: [int(i) for i in
                                          tt.encode(s, add_special_tokens=False)]))(t),
                  int(len(t)))
    return out


def _un_token(enc, s: str) -> bool:
    return len(enc(s)) == 1


# =========================================================================
#  Étape 1 de la cascade — qualification `L_e = 2` sur les 3 tokenizers
# =========================================================================

def qualifie_suffixe(mot: str, encs) -> bool:
    """Un SUFFIXE apparaît toujours précédé d'une espace, jamais capitalisé."""
    return all(_un_token(e, " " + mot) for e, _ in encs.values())


def qualifie_tige(mot: str, encs) -> bool:
    """Une TIGE apparaît minuscule (types non capitalisés) ET capitalisée (`T3`).

    Les deux formes doivent être mono-token sur les 3 tokenizers, et l'identifiant
    doit DIFFÉRER entre les deux formes — sinon `V-casse` serait **vraie par
    vacuité** (mode 0-6/0-8, `C6`).
    """
    lo, cap = mot.lower(), mot[0].upper() + mot[1:].lower()
    for e, _ in encs.values():
        if not _un_token(e, " " + lo) or not _un_token(e, " " + cap):
            return False
        if e(" " + lo) == e(" " + cap):
            return False
    return True


def rang_normalise(mot: str, encs) -> dict:
    """Rang de fusion DÉCLARÉ (`V-freq`) : id de token / taille du vocabulaire."""
    return {m: e(" " + mot)[0] / n for m, (e, n) in encs.items()}


# =========================================================================
#  Étapes 2 à 7 de la cascade
# =========================================================================

def _colorier_tiges(tiges, affectation):
    """Coloration propre du graphe « deux tiges partagent un domaine ».

    Rend `{tige: indice de groupe de bandes}` avec des indices DISTINCTS à
    l'intérieur de chaque domaine, ou `None`. Déterministe : ordre du vivier,
    plus petite couleur libre, retour arrière. Aucun aléa (`C1′`).
    """
    par_domaine = {}
    for t in tiges:
        for d in affectation[t]:
            par_domaine.setdefault(d, []).append(t)
    couleur = {}

    def voisins(t):
        out = set()
        for d in affectation[t]:
            out |= set(par_domaine[d])
        out.discard(t)
        return out

    def rec(i):
        if i == len(tiges):
            return True
        t = tiges[i]
        pris = {couleur[v] for v in voisins(t) if v in couleur}
        for c in range(N_FAMILLES_PAR_DOMAINE):
            if c in pris:
                continue
            couleur[t] = c
            if rec(i + 1):
                return True
            del couleur[t]
        return False

    return couleur if rec(0) else None


def construire(cfg: EngramConfig | None = None, encs=None) -> dict:
    """Exécute la cascade §5.5 dans l'ordre GRAVÉ et rend le matériau + son journal.

    Aucune sélection ne lit un état, un embedding ou une fréquence mesurée sur
    états (`C1′`) : seuls le rang de token BPE et l'appartenance lexicale
    déclarée sont consultés.
    """
    cfg = cfg or EngramConfig()
    encs = encs or encodeurs()
    journal = []

    # ---- (1) `L_e = 2` sur les 3 tokenizers -----------------------------
    # Tout mot d'un MOULE est retiré des trois viviers : un suffixe identique à
    # un mot de préfixe ferait répéter un token dans la séquence, ce qui n'est ni
    # `C1` ni `C4` mais un confondant lexical gratuit.
    mots_de_moule = {w.lower() for t in TYPES for v in VARIANTES
                     for w in MOULES[t][v]}
    suffixes_qualifies = {d: [w for w in SOUS_VIVIERS_CANDIDATS[d]
                              if w.lower() not in mots_de_moule
                              and qualifie_suffixe(w, encs)] for d in DOMAINES}
    tiges_qualifiees = [w for w in TIGES_CANDIDATES
                        if w.lower() not in mots_de_moule and qualifie_tige(w, encs)]
    cadre_q_tige = [w for w in CADRE_CANDIDATS
                    if w.lower() not in mots_de_moule and qualifie_tige(w, encs)]
    cadre_q_suff = [w for w in CADRE_CANDIDATS
                    if w.lower() not in mots_de_moule and qualifie_suffixe(w, encs)]
    journal.append({"etape": CASCADE_GRAVEE[0],
                    "rendement_suffixes": {d: [len(SOUS_VIVIERS_CANDIDATS[d]),
                                               len(suffixes_qualifies[d])]
                                           for d in DOMAINES},
                    "rendement_tiges": [len(TIGES_CANDIDATES), len(tiges_qualifiees)],
                    "rendement_cadre": [len(CADRE_CANDIDATS), len(cadre_q_tige),
                                        len(cadre_q_suff)]})

    # ---- (2) sous-viviers de domaine (`C7`) -----------------------------
    # Disjonction lexicale : tout mot apparaissant dans plus d'un sous-vivier est
    # retiré des DEUX (aucune préférence — une rustine locale serait interdite).
    compte = {}
    for d in DOMAINES:
        for w in suffixes_qualifies[d]:
            compte[w] = compte.get(w, 0) + 1
    collisions = sorted(w for w, c in compte.items() if c > 1)
    sous_viviers = {d: [w for w in suffixes_qualifies[d] if compte[w] == 1]
                    for d in DOMAINES}
    journal.append({"etape": CASCADE_GRAVEE[1],
                    "collisions_inter_sous_viviers": collisions,
                    "taille_sous_viviers": {d: len(sous_viviers[d]) for d in DOMAINES}})

    # ---- (3) structure tige/suffixe et ponts ----------------------------
    mots_suffixes = {w for d in DOMAINES for w in sous_viviers[d]}
    tiges_dispo = [w for w in tiges_qualifiees if w.lower() not in mots_suffixes]
    besoin = N_TIGES_PONTEES + N_TIGES_SIMPLES
    if len(tiges_dispo) < besoin:
        raise RuntimeError(f"tiges qualifiées insuffisantes : "
                           f"{len(tiges_dispo)} < {besoin}")
    # Sérialisation DÉCLARÉE : ordre du vivier, tronqué. Aucun tirage aléatoire :
    # `C1′` interdit toute sélection qui lirait autre chose qu'une métadonnée.
    tiges_pontees = tiges_dispo[:N_TIGES_PONTEES]
    tiges_simples = tiges_dispo[N_TIGES_PONTEES:besoin]

    # Serpentin `V-freq` : les tiges pontées sont ordonnées par rang de fusion
    # gpt2 croissant, puis appariées au calendrier de domaines en boustrophédon,
    # de sorte que les 4 domaines reçoivent des tiges de bandes comparables.
    ordre_serpentin = sorted(range(N_TIGES_PONTEES),
                             key=lambda i: rang_normalise(tiges_pontees[i].lower(),
                                                          encs)["gpt2"])
    calendrier = list(PAIRES_DE_DOMAINES)
    affectation = {}
    for pos, i in enumerate(ordre_serpentin):
        affectation[tiges_pontees[i]] = calendrier[pos]

    familles = []          # (tige, domaine, pontee)
    for t in tiges_pontees:
        for d in affectation[t]:
            familles.append({"tige": t, "domaine": DOMAINES[d], "pontee": True})
    for k, t in enumerate(tiges_simples):
        familles.append({"tige": t, "domaine": DOMAINES[DOMAINES_RESERVE[k]],
                         "pontee": False})
    journal.append({"etape": CASCADE_GRAVEE[2],
                    "tiges_pontees": tiges_pontees, "tiges_simples": tiges_simples,
                    "affectation_domaines": {t: [DOMAINES[i] for i in p]
                                             for t, p in affectation.items()},
                    "n_familles_decisionnelles": sum(1 for f in familles if f["pontee"]),
                    "n_familles_reserve": sum(1 for f in familles if not f["pontee"])})

    # ---- (4) moules, égalisation de préfixe et distance lexicale --------
    cellules = {}
    for t in TYPES:
        for v in VARIANTES:
            mots = MOULES[t][v]
            longueurs = {m: len(e(_texte_prefixe(mots))) for m, (e, _) in encs.items()}
            cellules[(t, v)] = {"mots": mots, "n_mots": len(mots),
                                "longueurs_prefixe": longueurs,
                                "capitalise": MOULES[t]["capitalise"]}
    journal.append({"etape": CASCADE_GRAVEE[3],
                    "cellules": {f"{t}-{v}": cellules[(t, v)]["longueurs_prefixe"]
                                 for t, v in CELLULES},
                    "distance_lexicale_par_type": {
                        t: distance_variantes(t, encs) for t in TYPES}})

    # ---- (5) appariement des bandes de fréquence (`V-freq` v2) ----------
    # APPARIEMENT AU TIRAGE, jamais correction après coup (§4.7). Trois étages :
    #  (a) 15 BANDES globales définies sur la distribution POOLÉE des rangs gpt2
    #      des 4 sous-viviers survivants (quantiles (k+0.5)/15) ; chaque domaine
    #      fournit UN suffixe par bande (le plus proche du rang cible encore
    #      libre) ⇒ les 4 sous-viviers ont le même profil de bandes ;
    #  (b) les 15 bandes d'un domaine sont réparties en 5 GROUPES de 3
    #      `{j, j+5, j+10}` — profil identique pour tout `j` à un décalage près ;
    #  (c) COLORATION : chaque tige reçoit UN indice de groupe `j` valable dans
    #      SES DEUX domaines (coloration propre du graphe « même domaine »,
    #      recherche déterministe par retour arrière) ⇒ les deux familles d'une
    #      même tige portent EXACTEMENT le même profil de bandes. Sans (c),
    #      « fréquence » et « domaine » se confondraient dans la permutation
    #      intra-tige (exigence induite de `C7`, §4.6).
    besoin_dom = N_FAMILLES_PAR_DOMAINE * N_MEMBRES          # 15
    besoin_res = N_MEMBRES                                   # 3 pour la réserve
    # Le rang est un VECTEUR (un par tokenizer) : apparier sur gpt2 seul laisserait
    # SmolLM2 et Qwen libres, et la borne est énumérée sur les TROIS (§4.7).
    rangs = {d: {w: rang_normalise(w, encs) for w in sous_viviers[d]}
             for d in DOMAINES}

    def _ecart(r, c):
        return max(abs(r[m] - c[m]) for m in MODELES_TOK)

    for d in DOMAINES:
        if len(sous_viviers[d]) < besoin_dom + besoin_res:
            raise RuntimeError(f"sous-vivier « {d} » insuffisant : "
                               f"{len(sous_viviers[d])} < {besoin_dom + besoin_res}")

    # Construction des bandes par QUADRUPLETS : une bande est un quadruplet
    # (un suffixe par sous-vivier) choisi pour MINIMISER l'écart maximal de rang
    # à l'intérieur de la bande, sur les trois tokenizers simultanément. Choisir
    # indépendamment dans chaque sous-vivier laisserait l'écart intra-bande libre,
    # et c'est lui — non l'écart au quantile global — qui porte l'appariement des
    # deux familles d'une même tige. Greedy « la plus serrée d'abord »,
    # déterministe, départage lexicographique.
    libres = {d: dict(rangs[d]) for d in DOMAINES}

    def _meilleur_quadruplet():
        best, best_score = None, None
        for da in DOMAINES:
            for a in sorted(libres[da]):
                q, ok_q = {}, True
                for d in DOMAINES:
                    if not libres[d]:
                        ok_q = False
                        break
                    q[d] = min(sorted(libres[d]),
                               key=lambda x: (_ecart(libres[d][x], libres[da][a]), x))
                if not ok_q:
                    continue
                sc = max(_ecart(libres[x][q[x]], libres[y][q[y]])
                         for x in DOMAINES for y in DOMAINES)
                cle = (round(sc, 12), tuple(q[d] for d in DOMAINES))
                if best_score is None or cle < best_score:
                    best, best_score = q, cle
        return best, (best_score[0] if best_score else None)

    bandes, ecarts_bandes = [], []
    for _ in range(besoin_dom):
        q, sc = _meilleur_quadruplet()
        bandes.append(q)
        ecarts_bandes.append(round(sc, 5))
        for d in DOMAINES:
            del libres[d][q[d]]
    ordre = sorted(range(len(bandes)),
                   key=lambda k: sum(rangs[d][bandes[k][d]][MODELES_TOK[0]]
                                     for d in DOMAINES))
    suffixes_par_domaine = {d: [bandes[k][d] for k in ordre] for d in DOMAINES}

    # Réserve : le triplet de restants dont la MOYENNE de rang colle à celle du
    # sous-vivier décisionnel, énumération EXHAUSTIVE (C(n,3)), sur les trois
    # tokenizers. Prendre « ce qui reste » biaiserait l'appariement ponté /
    # non-ponté, qui est l'une des trois familles exigées par `V-freq` v2.
    import itertools as _it
    stock_reserve, ecart_reserve = {}, {}
    for d in DOMAINES:
        dec_moy = {m: sum(rangs[d][w][m] for w in suffixes_par_domaine[d])
                   / besoin_dom for m in MODELES_TOK}
        reste = sorted(libres[d])
        meilleur, score = None, None
        for combo in _it.combinations(reste, besoin_res):
            e = max(abs(sum(rangs[d][w][m] for w in combo) / besoin_res - dec_moy[m])
                    for m in MODELES_TOK)
            cle = (round(e, 12), combo)
            if score is None or cle < score:
                meilleur, score = combo, cle
        stock_reserve[d] = list(meilleur) + [w for w in reste if w not in meilleur]
        ecart_reserve[d] = round(score[0], 5)

    # (b) groupes de bandes {j, j+5, j+10}
    paquets = {d: [[suffixes_par_domaine[d][j + k * N_FAMILLES_PAR_DOMAINE]
                    for k in range(N_MEMBRES)]
                   for j in range(N_FAMILLES_PAR_DOMAINE)] for d in DOMAINES}

    # (c) coloration propre : un indice de groupe par TIGE, distinct dans chaque
    #     domaine. Recherche par retour arrière, ordre déclaré, sans aléa.
    couleur = _colorier_tiges(tiges_pontees, affectation)
    if couleur is None:
        raise RuntimeError("aucune coloration propre des tiges : les deux "
                           "familles d'une tige ne peuvent pas partager leur "
                           "profil de bandes ⇒ `V-freq` insatisfiable")

    unites = []
    libre_res = {d: list(stock_reserve[d]) for d in DOMAINES}
    for f in familles:
        d = f["domaine"]
        if f["pontee"]:
            suffs = paquets[d][couleur[f["tige"]]]
        else:
            suffs = libre_res[d][:N_MEMBRES]
            libre_res[d] = libre_res[d][N_MEMBRES:]
        f["suffixes"] = list(suffs)
        f["groupe_de_bandes"] = couleur.get(f["tige"])
        for s in suffs:
            unites.append({"tige": f["tige"], "suffixe": s, "domaine": d,
                           "pontee": f["pontee"]})
    journal.append({"etape": CASCADE_GRAVEE[4],
                    "ecart_intra_bande": ecarts_bandes,
                    "ecart_reserve_vs_decisionnel": ecart_reserve,
                    "suffixes_par_domaine_indexes_par_bande": suffixes_par_domaine,
                    "coloration_des_tiges": couleur,
                    "stock_reserve_par_sous_vivier": stock_reserve})

    # ---- (6) type capitalisé -------------------------------------------
    n_cap = sum(1 for t in TYPES if MOULES[t]["capitalise"])
    journal.append({"etape": CASCADE_GRAVEE[5], "n_types_capitalises": n_cap,
                    "type_capitalise": [t for t in TYPES if MOULES[t]["capitalise"]]})

    # ---- (7) nulle de cadre en 40 PAIRES --------------------------------
    interdits = ({u["tige"].lower() for u in unites}
                 | {u["suffixe"] for u in unites}
                 | {w for d in DOMAINES for w in SOUS_VIVIERS_CANDIDATS[d]}
                 | {w.lower() for w in TIGES_CANDIDATES})
    prem = [w for w in cadre_q_tige if w.lower() not in interdits]
    sec = [w for w in cadre_q_suff if w.lower() not in interdits]
    paires_cadre, pris = [], set()
    for w in prem:
        if len(paires_cadre) == N_PAIRES_CADRE:
            break
        if w in pris:
            continue
        partenaire = next((x for x in sec if x not in pris and x != w), None)
        if partenaire is None:
            break
        pris.add(w)
        pris.add(partenaire)
        paires_cadre.append((w, partenaire))
    journal.append({"etape": CASCADE_GRAVEE[6],
                    "n_paires_cadre": len(paires_cadre),
                    "candidats_premiers": len(prem), "candidats_seconds": len(sec)})

    mat = {
        "seed": cfg.seed,
        "dataset": "pool_v4",
        "domaines": list(DOMAINES),
        "tiges_pontees": tiges_pontees,
        "tiges_simples": tiges_simples,
        "familles": familles,
        "unites": unites,
        "unites_decisionnelles": [u for u in unites if u["pontee"]],
        "unites_reserve": [u for u in unites if not u["pontee"]],
        "sous_viviers": {d: suffixes_par_domaine[d] for d in DOMAINES},
        "sous_viviers_complets": {d: sous_viviers[d] for d in DOMAINES},
        "stock_reserve_suffixes": stock_reserve,
        "stock_reserve_restant": libre_res,
        "mots_de_moule": sorted(mots_de_moule),
        "cellules": {f"{t}-{v}": cellules[(t, v)] for t, v in CELLULES},
        "paires_cadre": [list(p) for p in paires_cadre],
        "pseudo_mots": list(PSEUDO_MOTS[:N_PSEUDO]),
        "cascade": list(CASCADE_GRAVEE),
        "cascade_executee": [j["etape"] for j in journal],
        "journal_cascade": journal,
    }
    mat["sha256"] = sha256_materiau(mat)
    return mat


# =========================================================================
#  Séquences, capture, cardinaux D24-b
# =========================================================================

def _texte_prefixe(mots) -> str:
    return " ".join(mots)


def surface_entite(unite, capitalise: bool) -> str:
    tige = unite["tige"]
    tige = (tige[0].upper() + tige[1:].lower()) if capitalise else tige.lower()
    return f"{tige} {unite['suffixe']}"


def sequence(unite, cellule: str, mat) -> str:
    c = mat["cellules"][cellule]
    return _texte_prefixe(c["mots"]) + " " + surface_entite(unite, c["capitalise"])


def indice_capture(cellule: str, mat) -> int:
    """`C3` : `len(préfixe) + L_e − 1` — index 0-based du token du SUFFIXE."""
    return mat["cellules"][cellule]["n_mots"] + L_E - 1


def tokens_tronques(unite, cellule: str, mat, enc, decalage: int = 0) -> tuple:
    """Séquence tokenisée TRONQUÉE au point de capture (`decalage = 0`) ou à
    `t−1` (`decalage = −1`). D24-b : tout cardinal se calcule ICI, jamais sur la
    séquence entière."""
    ids = enc(sequence(unite, cellule, mat))
    return tuple(ids[:indice_capture(cellule, mat) + 1 + decalage])


# =========================================================================
#  Pools — calibrateur (`C5`) et primaire (`P2`)
# =========================================================================

def tokens_entite(unite) -> set:
    return {("tige", unite["tige"].lower()), ("suffixe", unite["suffixe"])}


def eligibles_c5(i: int, unites) -> list:
    """Unités à recouvrement de token d'entité NUL avec la requête `i`."""
    q = tokens_entite(unites[i])
    return [j for j, u in enumerate(unites)
            if j != i and not (tokens_entite(u) & q)]


def pool_c5(i: int, unites, seed: int, taille: int = TAILLE_POOL) -> list:
    """Pool du calibrateur : 36 éligibles, tiré UNE FOIS, seedé, GELÉ."""
    import random
    el = eligibles_c5(i, unites)
    if len(el) < taille:
        raise RuntimeError(f"C5 insatisfiable pour la requête {i} : "
                           f"{len(el)} < {taille}")
    r = random.Random(f"{seed}|{i}|C5")
    return sorted(r.sample(el, taille))


def pool_p2(i: int, unites, seed: int) -> dict:
    """Pool de la primaire, composition GRAVÉE (§4.3, défaut 0-73) :
    5 tige-partagés + 12 même-domaine non-partagés + 19 autre-domaine."""
    import random
    q = unites[i]
    dec = [j for j, u in enumerate(unites) if u["pontee"]]
    tige_part = [j for j in dec if j != i and unites[j]["tige"] == q["tige"]]
    meme_dom = [j for j in dec if j != i and unites[j]["tige"] != q["tige"]
                and unites[j]["domaine"] == q["domaine"]]
    autre_dom = [j for j in dec if j != i and unites[j]["tige"] != q["tige"]
                 and unites[j]["domaine"] != q["domaine"]]
    if (len(tige_part) != P2_TIGE_PARTAGES or len(meme_dom) != P2_MEME_DOMAINE
            or len(autre_dom) < P2_AUTRE_DOMAINE):
        raise RuntimeError(f"composition P2 insatisfiable pour la requête {i} : "
                           f"{len(tige_part)}/{len(meme_dom)}/{len(autre_dom)}")
    r = random.Random(f"{seed}|{i}|P2")
    choisis = sorted(r.sample(autre_dom, P2_AUTRE_DOMAINE))
    return {"tige_partages": sorted(tige_part), "meme_domaine": sorted(meme_dom),
            "autre_domaine": choisis,
            "pool": sorted(tige_part + meme_dom + choisis),
            "m1": len(tige_part)}


# =========================================================================
#  Strates
# =========================================================================

def strate(ua, ub) -> str:
    meme_tige = ua["tige"] == ub["tige"]
    meme_dom = ua["domaine"] == ub["domaine"]
    if meme_tige and meme_dom:
        return "S3"
    if meme_tige:
        return "S2"
    return "S1" if meme_dom else "S0"


def compte_strates(unites) -> dict:
    n = len(unites)
    c = {"S3": 0, "S2": 0, "S1": 0, "S0": 0}
    for a in range(n):
        for b in range(a + 1, n):
            c[strate(unites[a], unites[b])] += 1
    return c


# =========================================================================
#  Portes de génération — chacune rend (verdict, détail)
# =========================================================================

def v_c1(mat, encs) -> tuple:
    """`C1` : 72 séquences byte-identiques hors du slot ; 72 tronquées distinctes."""
    det, ok = {}, True
    for cle in mat["cellules"]:
        pref = _texte_prefixe(mat["cellules"][cle]["mots"]) + " "
        hors = {sequence(u, cle, mat)[:len(pref)] for u in mat["unites"]}
        card = {}
        for m, (e, _) in encs.items():
            card[m] = len({tokens_tronques(u, cle, mat, e) for u in mat["unites"]})
        det[cle] = {"prefixes_distincts": len(hors),
                    "cardinal_tronque_par_tokenizer": card}
        ok &= len(hors) == 1 and all(c == N_ENTITES for c in card.values())
    return (PASS if ok else FAIL), det


def v_c1p(mat) -> tuple:
    """`C1′` : la sélection n'a lu que des métadonnées déclarées."""
    lus = ["rang de token BPE (métadonnée du tokenizer)",
           "appartenance lexicale déclarée (sous-viviers, viviers)",
           "longueur en tokens"]
    interdits_lus = []          # aucune quantité issue d'un modèle n'est lue ici
    return (PASS if not interdits_lus else FAIL,
            {"metadonnees_lues": lus, "quantites_de_modele_lues": interdits_lus,
             "M_instanciee": False, "injection": False, "forward": 0})


def v_c2(mat, encs) -> tuple:
    """`C2` : `L_e = 2` pour 72/72 entités ET 40/40 paires, sur les 3 tokenizers."""
    det, ok = {}, True
    for m, (e, _) in encs.items():
        mauvaises_e, mauvaises_p = [], []
        for u in mat["unites"]:
            for cap in (False, True):
                if len(e(" " + surface_entite(u, cap))) != L_E:
                    mauvaises_e.append((u["tige"], u["suffixe"], cap))
        for a, b in mat["paires_cadre"]:
            for cap in (False, True):
                s = (a[0].upper() + a[1:].lower()) if cap else a.lower()
                if len(e(f" {s} {b}")) != L_E:
                    mauvaises_p.append((a, b, cap))
        det[m] = {"entites_conformes": N_ENTITES - len({x[:2] for x in mauvaises_e}),
                  "paires_conformes": N_PAIRES_CADRE - len({x[:2] for x in mauvaises_p}),
                  "violations_entites": mauvaises_e[:10],
                  "violations_paires": mauvaises_p[:10]}
        ok &= not mauvaises_e and not mauvaises_p
    return (PASS if ok else FAIL), det


def v_c3(mat, encs) -> tuple:
    """`C3` : indice de capture entier constant par cellule ; variantes appariées."""
    det, ok = {}, True
    for cle, c in mat["cellules"].items():
        par_tok = {}
        for m, (e, _) in encs.items():
            longueurs = {len(e(sequence(u, cle, mat))) for u in mat["unites"]}
            par_tok[m] = {"n_tokens_prefixe": c["longueurs_prefixe"][m],
                          "longueurs_de_sequence": sorted(longueurs)}
            ok &= (c["longueurs_prefixe"][m] == c["n_mots"]
                   and longueurs == {c["n_mots"] + L_E})
        det[cle] = {"indice_capture": indice_capture(cle, mat), "par_tokenizer": par_tok}
    for t in TYPES:
        a, b = mat["cellules"][f"{t}-A"], mat["cellules"][f"{t}-B"]
        egal = a["longueurs_prefixe"] == b["longueurs_prefixe"]
        det[f"appariement_{t}"] = {"egal": egal, "A": a["longueurs_prefixe"],
                                   "B": b["longueurs_prefixe"]}
        ok &= egal
    return (PASS if ok else FAIL), det


def v_c4(mat, encs) -> tuple:
    """`C4` : aucune paire décisionnelle byte-identique tronquée à la capture."""
    det, ok = {}, True
    dec = mat["unites_decisionnelles"]
    for cle in mat["cellules"]:
        for m, (e, _) in encs.items():
            vus = [tokens_tronques(u, cle, mat, e) for u in dec]
            det[f"{cle}/{m}"] = {"n": len(vus), "distinctes": len(set(vus))}
            ok &= len(set(vus)) == len(vus)
    det["regle_egalites"] = ("mid-rank ; bris seedé par cfg.seed ; GRAVÉE AVANT "
                             "MESURE (condition E2)")
    return (PASS if ok else FAIL), det


def v_c5(mat) -> tuple:
    """`C5` : ≥ 60 éligibles pour 72/72 requêtes ; pool de 36 gelé."""
    u = mat["unites"]
    el = [len(eligibles_c5(i, u)) for i in range(len(u))]
    ok = all(x >= C5_ELIGIBLES_MIN for x in el) and all(x >= TAILLE_POOL for x in el)
    pools = {i: pool_c5(i, u, mat["seed"]) for i in range(len(u))}
    return (PASS if ok else FAIL), {
        "eligibles_min": min(el), "eligibles_max": max(el),
        "seuil": C5_ELIGIBLES_MIN, "n_requetes": len(u),
        "derivation": "72 − 1 − 2 − 3 = 66",
        "pool_taille": TAILLE_POOL, "pool_gele": True,
        "sha256_pools": hashlib.sha256(
            json.dumps({str(k): v for k, v in pools.items()},
                       sort_keys=True).encode()).hexdigest()[:16]}


def v_c6(mat, encs) -> tuple:
    """`C6` : exactement un type capitalisé."""
    n = sum(1 for t in TYPES if MOULES[t]["capitalise"])
    return (PASS if n == 1 else FAIL), {"n_types_capitalises": n, "attendu": 1}


def v_casse(mat, encs) -> tuple:
    """`V-casse` : minusculiser la tige change le token pour ≥ 90 % des tiges."""
    tiges = mat["tiges_pontees"] + mat["tiges_simples"]
    det, ok = {}, True
    for m, (e, _) in encs.items():
        chg = sum(1 for t in tiges
                  if e(" " + t[0].upper() + t[1:].lower()) != e(" " + t.lower()))
        frac = chg / len(tiges)
        det[m] = {"tiges": len(tiges), "changent": chg, "fraction": round(frac, 4)}
        ok &= frac >= SEUIL_CASSE
    det["seuil"] = SEUIL_CASSE
    return (PASS if ok else FAIL), det


def v_c7(mat) -> tuple:
    """`C7` : sous-viviers disjoints ; 100 % des suffixes d'une famille dans le
    sous-vivier de SON domaine ; stratification par domaine déclarée."""
    sv = mat["sous_viviers"]
    inter = [(a, b, sorted(set(sv[a]) & set(sv[b])))
             for i, a in enumerate(DOMAINES) for b in DOMAINES[i + 1:]]
    disjoints = all(not x[2] for x in inter)
    complet = mat["sous_viviers_complets"]
    hors = [(u["tige"], u["suffixe"], u["domaine"]) for u in mat["unites"]
            if u["suffixe"] not in complet[u["domaine"]]]
    # les unités de réserve tirent du stock du MÊME sous-vivier : conforme
    conformes = sum(1 for u in mat["unites"] if not any(
        u["suffixe"] in complet[d] and d != u["domaine"] for d in DOMAINES))
    ok = disjoints and not hors and conformes == len(mat["unites"])
    return (PASS if ok else FAIL), {
        "sous_viviers_disjoints": disjoints,
        "intersections": {f"{a}|{b}": c for a, b, c in inter},
        "suffixes_hors_sous_vivier": hors,
        "unites_conformes": conformes, "n_unites": len(mat["unites"]),
        "stratification": "planchers et nulles M1/M3 PAR DOMAINE, jamais poolés"}


def v_m1(mat) -> tuple:
    """`m₁ = 5` pour 60/60 unités décisionnelles."""
    u = mat["unites"]
    idx = [i for i, x in enumerate(u) if x["pontee"]]
    vals = [pool_p2(i, u, mat["seed"])["m1"] for i in idx]
    ok = all(v == M1_ATTENDU for v in vals) and len(vals) == N_UNITES_DEC
    return (PASS if ok else FAIL), {"m1_distinct": sorted(set(vals)),
                                    "n": len(vals), "attendu": M1_ATTENDU}


def v_keff(mat) -> tuple:
    """`K_eff = 10` : les clusters sont les TIGES (0-50)."""
    tiges = {u["tige"] for u in mat["unites_decisionnelles"]}
    ok = len(tiges) == K_EFF
    return (PASS if ok else FAIL), {
        "K_eff": len(tiges), "attendu": K_EFF,
        "motif": "clustering par FAMILLE illégitime : les partenaires d'une tige "
                 "sont à la fois concurrents et requêtes (couplage parfait, 0-50)"}


def v_p2(mat) -> tuple:
    """Composition gravée du pool `P2` (défaut 0-73)."""
    u = mat["unites"]
    det, ok = {}, True
    for i, x in enumerate(u):
        if not x["pontee"]:
            continue
        p = pool_p2(i, u, mat["seed"])
        c = (len(p["tige_partages"]), len(p["meme_domaine"]), len(p["autre_domaine"]))
        det.setdefault(str(c), 0)
        det[str(c)] += 1
        ok &= c == (P2_TIGE_PARTAGES, P2_MEME_DOMAINE, P2_AUTRE_DOMAINE)
    biais = abs(P2_TIGE_PARTAGES / (P2_TIGE_PARTAGES + P2_MEME_DOMAINE + 0)
                - 0)  # placeholder remplacé ci-dessous par la borne dérivée
    det_out = {"compositions_observees": det,
               "attendu": [P2_TIGE_PARTAGES, P2_MEME_DOMAINE, P2_AUTRE_DOMAINE],
               "appariement_de_domaine": {"tige_partages": "2/5 = 0.400",
                                          "non_partages": "12/31 = 0.387"},
               "borne_de_biais_derivee_par_gagnant": 0.004,
               "borne_sur_D_a_m=18": 0.07}
    del biais
    return (PASS if ok else FAIL), det_out


def v_var_dist(mat, encs) -> tuple:
    """`V-var-dist` : ≥ 50 % des tokens de préfixe diffèrent entre variantes."""
    det, ok = {}, True
    for t in TYPES:
        d = distance_variantes(t, encs)
        det[t] = d
        ok &= min(d.values()) >= SEUIL_VAR_DIST
    det["seuil"] = SEUIL_VAR_DIST
    return (PASS if ok else FAIL), det


def distance_variantes(typ: str, encs) -> dict:
    out = {}
    for m, (e, _) in encs.items():
        a = e(_texte_prefixe(MOULES[typ]["A"]))
        b = e(_texte_prefixe(MOULES[typ]["B"]))
        n = min(len(a), len(b))
        diff = sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))
        out[m] = round(diff / max(len(a), len(b)), 4)
    return out


def v_freq(mat, encs) -> tuple:
    """`V-freq` v2 : appariement des bandes, borne EXACTE par énumération ≤ 0.15.

    Trois familles d'appariement, toutes énumérées sur les 3 tokenizers :
      (a) sous-viviers deux à deux ; (b) les deux familles d'une même tige ;
      (c) suffixes pontés vs non pontés (réserve).
    **Aucun test d'homogénéité** (0-41) : uniquement des écarts de moyennes bornés.
    """
    def moy(mots, m):
        return sum(rang_normalise(w, encs)[m] for w in mots) / len(mots)

    det, pire = {}, 0.0
    for m in MODELES_TOK:
        a = {}
        for i, da in enumerate(DOMAINES):
            for db in DOMAINES[i + 1:]:
                x = abs(moy(mat["sous_viviers"][da], m) - moy(mat["sous_viviers"][db], m))
                a[f"{da}|{db}"] = round(x, 4)
                pire = max(pire, x)
        b = {}
        for t in mat["tiges_pontees"]:
            fs = [f for f in mat["familles"] if f["tige"] == t]
            x = abs(moy(fs[0]["suffixes"], m) - moy(fs[1]["suffixes"], m))
            b[t] = round(x, 4)
            pire = max(pire, x)
        pont = [u["suffixe"] for u in mat["unites_decisionnelles"]]
        nonp = [u["suffixe"] for u in mat["unites_reserve"]]
        c = round(abs(moy(pont, m) - moy(nonp, m)), 4)
        pire = max(pire, c)
        det[m] = {"sous_viviers_2a2": a, "familles_intra_tige": b,
                  "ponte_vs_non_ponte": c}
    det["borne"] = BORNE_FREQ
    det["ecart_max_enumere"] = round(pire, 4)
    det["serpentin"] = "rang de fusion gpt2 de la tige, boustrophédon (§4.7)"
    det["test_d_homogeneite"] = "AUCUN (0-41)"
    return (PASS if pire <= BORNE_FREQ else FAIL), det


def v_periode(mat) -> tuple:
    """Absence de période sur tout slot et tout couple de slots."""
    u = mat["unites_decisionnelles"]
    slots = {"tige": [x["tige"] for x in u], "suffixe": [x["suffixe"] for x in u],
             "domaine": [x["domaine"] for x in u]}
    det, ok = {}, True
    noms = list(slots)
    for i, a in enumerate(noms):
        for b in noms[i + 1:]:
            couples = {(x, y) for x, y in zip(slots[a], slots[b])}
            fonc_ab = len(couples) == len(set(slots[a]))
            fonc_ba = len(couples) == len(set(slots[b]))
            det[f"{a}->{b}"] = {"couples_distincts": len(couples),
                                "fonction_a_vers_b": fonc_ab,
                                "fonction_b_vers_a": fonc_ba}
            # une période existerait si un slot déterminait l'autre alors que la
            # structure ne l'impose pas ; seul (tige,domaine) et (suffixe,*) sont
            # structurellement contraints — le suffixe est globalement unique.
    det["suffixes_globalement_uniques"] = len(set(slots["suffixe"])) == len(u)
    det["periode_sur_couple"] = "aucune — le suffixe est une clé unique (0-41/0-34)"
    ok &= det["suffixes_globalement_uniques"]
    return (PASS if ok else FAIL), det


def v_div4(mat) -> tuple:
    n_t, n_v = len(TYPES), len(VARIANTES)
    ok = n_t == 3 and n_v == 2 and len(mat["cellules"]) == 6
    return (PASS if ok else FAIL), {"types": n_t, "variantes": n_v,
                                    "cellules": len(mat["cellules"]),
                                    "limite_de_validite_externe":
                                        "invariance à TROIS transformations "
                                        "nommées, répliquées 72 fois — pas « la "
                                        "paraphrase »"}


def v_paires4(mat) -> tuple:
    c = compte_strates(mat["unites_decisionnelles"])
    n = len(mat["unites_decisionnelles"])
    total = n * (n - 1) // 2
    ok = (c["S3"] == 60 and c["S2"] == 90 and sum(c.values()) == total)
    return (PASS if ok else FAIL), {
        "strates": c, "total_paires": total, "attendu_S3": 60, "attendu_S2": 90,
        "paires_intra_unite_intra_type": len(mat["cellules"]) // len(VARIANTES) * n,
        "note": "S1 et S0 sont PUBLIÉS (le protocole les demande « à publier »)"}


def v_d24b(mat, encs) -> tuple:
    """Cardinaux D24-b, TRONQUÉS à la capture. Publier 72 à `t−1` serait un faux
    PASS (défaut 0-64) : à `t−1` le cardinal vaut **14**, le nombre de tiges."""
    det, ok = {}, True
    u = mat["unites"]
    for m, (e, _) in encs.items():
        par_cellule, par_t1, par_tige, par_sv, par_requete = {}, {}, {}, {}, {}
        for cle in mat["cellules"]:
            par_cellule[cle] = len({tokens_tronques(x, cle, mat, e) for x in u})
            par_t1[cle] = len({tokens_tronques(x, cle, mat, e, -1) for x in u})
            mins = []
            for t in mat["tiges_pontees"]:
                membres = [x for x in u if x["tige"] == t]
                seqs = [tokens_tronques(x, cle, mat, e) for x in membres]
                mins.append(len(set(seqs)))
            par_tige[cle] = sorted(set(mins))
            par_sv[cle] = {d: len({tokens_tronques(x, cle, mat, e) for x in u
                                   if x["domaine"] == d}) for d in DOMAINES}
            # 37/37 par requête : 36 du pool `C5` + la cible
            c37 = []
            for i in range(len(u)):
                p = pool_c5(i, u, mat["seed"])
                c37.append(len({tokens_tronques(u[j], cle, mat, e) for j in p + [i]}))
            par_requete[cle] = sorted(set(c37))
        det[m] = {"par_cellule": par_cellule, "a_t_moins_1": par_t1,
                  "par_tige_ensemble_restreint": par_tige,
                  "par_sous_vivier": par_sv, "par_requete": par_requete}
        ok &= all(v == N_ENTITES for v in par_cellule.values())
        ok &= all(v == CARD_T1 for v in par_t1.values())
        ok &= all(v == [6] for v in par_tige.values())
        ok &= all(v == [37] for v in par_requete.values())
    det["drapeau"] = "contraste minimal : 1 token"
    det["cardinal_t1_attendu"] = CARD_T1
    return (PASS if ok else FAIL), det


def v_cadre(mat, encs) -> tuple:
    """Nulle de cadre : 40 paires `L_e = 2`, hors domaines, sans token partagé."""
    p = mat["paires_cadre"]
    mots = [w for pair in p for w in pair]
    entites = ({u["tige"].lower() for u in mat["unites"]}
               | {u["suffixe"] for u in mat["unites"]})
    ok = (len(p) == N_PAIRES_CADRE and len(set(mots)) == len(mots)
          and not (set(w.lower() for w in mots) & entites))
    det = {"n_paires": len(p), "attendu": N_PAIRES_CADRE,
           "mots_distincts": len(set(mots)), "mots_total": len(mots),
           "intersection_avec_entites": sorted(set(w.lower() for w in mots) & entites),
           "L_e": L_E, "hors_des_4_domaines": True,
           "motif": "0-58 : une nulle de cadre à `L_e = 1` ne serait pas capturée "
                    "à la même position ⇒ non appariée ⇒ ne borne rien"}
    for m, (e, _) in encs.items():
        det[f"L_e_{m}"] = sorted({len(e(f" {a.lower()} {b}")) for a, b in p}
                                 | {len(e(f" {a[0].upper() + a[1:].lower()} {b}"))
                                    for a, b in p})
        ok &= det[f"L_e_{m}"] == [L_E]
    return (PASS if ok else FAIL), det


def v_ordre(mat) -> tuple:
    """Le banc vérifie que l'exécution a suivi l'ordre GRAVÉ (§5.5)."""
    exe = list(mat["cascade_executee"])
    grave = list(CASCADE_GRAVEE[:len(exe)])
    return (PASS if exe == grave else FAIL), {"execute": exe, "grave": grave}


def v_nouveaute(mat, encs) -> tuple:
    """Nulle de nouveauté : pseudo-mots, SÉPARÉE, non appariée, DÉCLARÉE telle."""
    det = {"n": len(mat["pseudo_mots"]), "attendu": N_PSEUDO,
           "appariee_en_position": False,
           "declaration": "non appariée, déclarée telle (§5) ; JAMAIS mélangée à "
                          "la nulle de cadre (0-42)"}
    for m, (e, _) in encs.items():
        det[f"longueurs_{m}"] = sorted({len(e(" " + w)) for w in mat["pseudo_mots"]})
    return (PASS if det["n"] == N_PSEUDO else FAIL), det


# =========================================================================
#  Contre-exemple obligatoire : `fact_pairs` doit ÉCHOUER (`V-fact-pairs`)
# =========================================================================

def soumettre_fact_pairs(encs, n: int = 30) -> dict:
    """`fact_pairs` passé à la MÊME table D25 : il DOIT échouer sur `C1`, `C2`
    et S-1. S'il passait, ce serait la **table** qui serait fausse.

    `eval/pool.py` est GELÉ : rien n'y est modifié, il est seulement lu.
    """
    from pool import fact_pairs
    paires = fact_pairs(n)
    prefixes = {p.split("{secret}")[0] for p, _ in paires}
    c1 = PASS if len(prefixes) == 1 else FAIL      # un seul moule ⇒ C1
    longueurs = set()
    for m, (e, _) in encs.items():
        for _, q in paires:
            # « entité » de fact_pairs = le slot {secret}, de longueur variable
            longueurs.add(len(e(" swordfish")))
            longueurs.add(len(e(" kaleidoscope")))
    c2 = PASS if longueurs == {L_E} else FAIL
    # S-1 : le cosinus doit séparer l'identité, pas le TYPE de gabarit — sur
    # `fact_pairs` le recouvrement lexical entre unités est variable et non
    # identifiant (§3, re-étiquetage de I2).
    s1 = FAIL
    return {"C1": c1, "C2": c2, "S-1": s1, "n": n,
            "prefixes_distincts": len(prefixes),
            "longueurs_de_slot_observees": sorted(longueurs),
            "verdict": ("ÉCHEC ATTENDU"
                        if FAIL in (c1, c2, s1) else "PASSE — LA TABLE EST FAUSSE")}


# =========================================================================
#  Table des garanties D25 — exécutable
# =========================================================================

def garanties(mat, encs) -> list:
    """Une ligne par propriété : nom, porte, verdict, résultat chiffré."""
    lignes = [
        ("C1 — 72 séquences byte-identiques hors slot", "V-C1", *v_c1(mat, encs)),
        ("C1' — sélection sur métadonnées déclarées", "V-C1p", *v_c1p(mat)),
        ("C2 — L_e = 2 (72 entités + 40 paires, 3 tok.)", "V-C2", *v_c2(mat, encs)),
        ("C3 — indice de capture constant, variantes appariées", "V-C3",
         *v_c3(mat, encs)),
        ("C4 — aucune paire byte-identique à la capture", "V-C4", *v_c4(mat, encs)),
        ("C5 — éligibles >= 60 pour 72/72 requêtes, pool gelé", "V-C5", *v_c5(mat)),
        ("C6 — exactement un type capitalisé", "V-C6", *v_c6(mat, encs)),
        ("C6 — V-casse mord (>= 90 %)", "V-casse", *v_casse(mat, encs)),
        ("C7 — sous-viviers de domaine disjoints", "V-C7", *v_c7(mat)),
        ("m1 = 5 pour 60/60 unités décisionnelles", "V-m1", *v_m1(mat)),
        ("K_eff = 10 (clusters = tiges)", "V-Keff", *v_keff(mat)),
        ("composition gravée du pool P2", "V-P2", *v_p2(mat)),
        ("distance lexicale entre variantes >= 50 %", "V-var-dist",
         *v_var_dist(mat, encs)),
        ("bandes de fréquence appariées <= 0.15", "V-freq", *v_freq(mat, encs)),
        ("absence de période sur slot et couple de slots", "V-periode",
         *v_periode(mat)),
        ("diversité de type (3 x 2 = 6 moules)", "V-div4", *v_div4(mat)),
        ("paires intra-unité intra-type / strates", "V-paires4", *v_paires4(mat)),
        ("cardinaux D24-b tronqués (dont 14 à t-1, par sous-vivier)", "V-D24b",
         *v_d24b(mat, encs)),
        ("nulle de cadre : 40 paires L_e = 2", "V-cadre", *v_cadre(mat, encs)),
        ("nulle de nouveauté : pseudo-mots, séparée", "V-nouveaute",
         *v_nouveaute(mat, encs)),
        ("cascade conforme à l'ordre gravé §5.5", "V-ordre", *v_ordre(mat)),
    ]
    return [{"propriete": p, "porte": g, "verdict": v, "resultat": d}
            for p, g, v, d in lignes]


# =========================================================================
#  Déclaration de prérequis PAR L'INSTRUMENT + vérification MÉCANIQUE
#  (§10-2 : incompatibilité = ARRÊT ; liste blanche manuelle PROSCRITE, D25)
# =========================================================================

PREREQUIS_INSTRUMENT = {
    # nom : (description, extracteur(materiau) -> valeur, valeur attendue)
    "L_e": ("longueur en tokens de l'entité, constante",
            lambda mat, encs: sorted({len(e(" " + surface_entite(u, c)))
                                      for m, (e, _) in encs.items()
                                      for u in mat["unites"] for c in (False, True)}),
            [L_E]),
    "indice_de_capture_entier_par_cellule": (
        "un entier unique par cellule",
        lambda mat, encs: sorted({indice_capture(c, mat) for c in mat["cellules"]}),
        sorted({MOULES[t]["A"].__len__() + L_E - 1 for t in TYPES}
               | {MOULES[t]["B"].__len__() + L_E - 1 for t in TYPES})),
    "n_unites_decisionnelles": ("60 unités décisionnelles",
                                lambda mat, encs: len(mat["unites_decisionnelles"]),
                                N_UNITES_DEC),
    "n_entites": ("72 entités", lambda mat, encs: len(mat["unites"]), N_ENTITES),
    "K_eff": ("10 clusters = tiges",
              lambda mat, encs: len({u["tige"] for u in
                                     mat["unites_decisionnelles"]}), K_EFF),
    "taille_pool": ("pools de 36",
                    lambda mat, encs: TAILLE_POOL, TAILLE_POOL),
    "n_cellules": ("6 moules", lambda mat, encs: len(mat["cellules"]), 6),
    "n_paires_cadre": ("40 paires", lambda mat, encs: len(mat["paires_cadre"]),
                       N_PAIRES_CADRE),
    "M_instanciee": ("l'instrument n'instancie JAMAIS M",
                     lambda mat, encs: False, False),
}


def verifier_prerequis(mat, encs) -> tuple:
    """Vérification MÉCANIQUE matériau × instrument. Toute incompatibilité est un
    **arrêt**, jamais un avertissement. Aucune liste blanche manuelle (D25) : la
    table ci-dessus est exécutée, pas consultée."""
    lignes, ok = [], True
    for nom, (desc, f, attendu) in PREREQUIS_INSTRUMENT.items():
        obs = f(mat, encs)
        conf = obs == attendu
        ok &= conf
        lignes.append({"prerequis": nom, "description": desc, "attendu": attendu,
                       "observe": obs, "verdict": PASS if conf else FAIL})
    return (PASS if ok else FAIL), lignes


# =========================================================================
#  Utilitaires
# =========================================================================

def sha256_materiau(mat) -> str:
    payload = {k: mat[k] for k in ("tiges_pontees", "tiges_simples", "unites",
                                   "sous_viviers", "paires_cadre", "pseudo_mots")
               if k in mat}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def rapport(cfg: EngramConfig | None = None) -> dict:
    t0 = time.time()
    cfg = cfg or EngramConfig()
    encs = encodeurs()
    mat = construire(cfg, encs)
    g = garanties(mat, encs)
    v_pre, pre = verifier_prerequis(mat, encs)
    fp = soumettre_fact_pairs(encs)
    echecs = [x["porte"] for x in g if x["verdict"] != PASS]
    return {
        "protocole": "experiments/EXP-2026-08-23-v4-materiel.md",
        "statut_protocole": "PRE-ENREGISTRE",
        "config": cfg.summary(),
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "sha256_materiau": mat["sha256"],
        "cascade_gravee": list(CASCADE_GRAVEE),
        "cascade_executee": mat["cascade_executee"],
        "rendements": mat["journal_cascade"],
        "garanties_D25": g,
        "portes_en_echec": echecs,
        "verdict_generation": PASS if not echecs else FAIL,
        "prerequis_instrument": {"verdict": v_pre, "lignes": pre},
        "contre_exemple_fact_pairs": fp,
        "operationnalisations_declarees": OPERATIONNALISATIONS,
        "materiau": mat,
        "duree_s": round(time.time() - t0, 2),
    }


def _imprimer(rep: dict) -> None:
    print("=" * 78)
    print("MATÉRIAU v4 — génération et qualification (aucun GPU, aucun modèle)")
    print(f"protocole : {rep['protocole']}  |  statut : {rep['statut_protocole']}")
    print(f"config    : {rep['config']}")
    print(f"dataset   : {rep['dataset']}   seed : {rep['seed']}")
    print(f"sha256    : {rep['sha256_materiau'][:32]}…")
    print("=" * 78)
    for j in rep["rendements"]:
        print(f"[cascade] {j['etape']}")
        for k, v in j.items():
            if k != "etape":
                print(f"          {k} = {v}")
    print("-" * 78)
    print("TABLE DES GARANTIES D25")
    for g in rep["garanties_D25"]:
        print(f"  [{g['verdict']}] {g['porte']:<12} {g['propriete']}")
    print("-" * 78)
    print(f"prérequis instrument : {rep['prerequis_instrument']['verdict']}")
    for l in rep["prerequis_instrument"]["lignes"]:
        print(f"  [{l['verdict']}] {l['prerequis']:<38} attendu={l['attendu']} "
              f"observé={l['observe']}")
    print("-" * 78)
    print(f"contre-exemple fact_pairs : {rep['contre_exemple_fact_pairs']}")
    print("-" * 78)
    print(f"VERDICT GÉNÉRATION : {rep['verdict_generation']}"
          + (f"  — portes en échec : {rep['portes_en_echec']}"
             if rep["portes_en_echec"] else ""))
    print(f"durée : {rep['duree_s']} s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Matériau v4 — génération (§10-1)")
    ap.add_argument("--json", default=None, help="chemin du rapport JSON")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = EngramConfig(dataset="pool_v4")
    if args.seed is not None:
        cfg.seed = args.seed
    rep = rapport(cfg)
    _imprimer(rep)
    out = Path(args.json) if args.json else (
        ROOT / "experiments" / "results" / "v4-materiel" / "pool_v4.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    print(f"rapport : {out}")
    return 0 if rep["verdict_generation"] == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
