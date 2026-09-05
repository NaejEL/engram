# SPDX-License-Identifier: AGPL-3.0-or-later
"""Banc de cohérence documentaire — EXP-2026-09-04-rapport-sous-banc, §10.

CPU pur. **Aucun import de torch, transformers, numpy.** Aucun forward, `M` jamais
instanciée. Le banc lit un corpus documentaire (`REPORT.md` + `report/manifest.csv` +
`report/riders.csv` + `report/invalides.csv` + `report/relecture.csv` + `report/data/`)
et rend, par clause du §10, `{PASS, violations, cas_exerces}` ainsi que le compte `E`.

Trois classes de violation (§3.3) :

- ``ORPHELIN``        — pas de résolution ; **adresse non ouvrable incluse**.
- ``DESACCORD``       — valeur ≠ source (détecteur du mode 0-139).
- ``SOURCE-ILLICITE`` — statut ``décisionnel`` sur source journal / sans script reproducteur.

`E` **ne se lit pas sur un run** : il se lit sur la **campagne de mutation dirigée**
(§5.2), qui exige par clause un cas **passant** et un cas **échouant mordant**.
`E` = nombre de clauses qui n'ont pas **les deux**. Voir ``--campagne``.

Le drapeau ``--force-pass`` est la **mutation du lecteur de classes** (D35 (i), contrôle
`C-mut`) : le banc rend alors toujours ``PASS``. ``tests/test_doc_bench.py`` DOIT
échouer sous ce drapeau.

Ce module ne rédige rien et n'interprète rien : il rend des comptes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

if hasattr(sys.stdout, "reconfigure"):  # pragma: no cover - dépend du flux
    sys.stdout.reconfigure(encoding="utf-8")


# =====================================================================================
# 1. Table de normalisation — SEPT règles, finie et fermée (M-2, 0-214)
# =====================================================================================
#
# Application **au TOKEN**, jamais à la ligne (sinon T4 casse une énumération « 3, 5 »).
# HORS table, explicitement : zéros terminaux et précision (domaine de `C10`) ; la
# latéralité (`0.031` / `0.0156` : deux quantités, deux `id`) ; tirets de plage U+2013.

EXPOSANTS_UNICODE = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
}

# **U+0020 est EXCLU** (defaut 0-233, A-2). L'y inclure rendait
# ``normaliser("3 5") == normaliser("35") == "35"`` : la propriete de fermeture etait
# **FAUSSE sur le domaine reel**, et 19 lignes de manifeste sur 52 n'etaient pas
# citables dans leur forme verbatim GELEE.
ESPACES_INSECABLES = "\u00a0\u202f\u2009"

# Groupement anglo-saxon des milliers : **au moins DEUX groupes**. Un groupe unique
# (``2,437``) est **ambigu** — decimale francaise ou millier anglais — et reste au
# domaine de T4. Ambiguite DECLAREE : elle est dans la table gelee, pas dans le code.
RE_GROUPEMENT_VIRGULE = re.compile(r"(?<=\d)(?:,\d{3}){2,}(?!\d)")


def _t1(tok: str) -> str:
    """T1 — U+2212 MINUS SIGN vers HYPHEN-MINUS."""
    return tok.replace("−", "-")


def _t2(tok: str) -> str:
    """T2 — espaces insecables et separateurs de milliers **internes au token**."""
    out = re.sub(r"(?<=\d)[" + ESPACES_INSECABLES + r"](?=\d)", "", tok)
    return RE_GROUPEMENT_VIRGULE.sub(lambda m: m.group(0).replace(",", ""), out)


def _t3(tok: str) -> str:
    """T3 — ``+/-`` vers ``±``."""
    return tok.replace("+/-", "±")


def _t4(tok: str) -> str:
    """T4 — virgule decimale **entre deux chiffres** vers point."""
    return re.sub(r"(?<=\d),(?=\d)", ".", tok)


def _t5(tok: str) -> str:
    """T5 — exponentielle : ``E`` / ``×10^k`` / exposants U+207x vers ``e``, signe
    ``-``, sans zero de tete."""
    out = "".join(EXPOSANTS_UNICODE.get(c, c) for c in tok)
    out = re.sub(r"[×x]\s*10\s*\^?\s*", "e", out)
    out = re.sub(r"(?<=\d)E(?=[-+−]?\d)", "e", out)

    def _exp(m: "re.Match[str]") -> str:
        signe = m.group("signe") or ""
        signe = "-" if signe in "-−" else ""
        chiffres = m.group("chiffres").lstrip("0") or "0"
        return "e" + signe + chiffres

    return re.sub(r"e(?P<signe>[-+−])?(?P<chiffres>\d+)", _exp, out)


def _t6(tok: str) -> str:
    """T6 — espace avant ``%`` normalise (le ``%`` est hors token)."""
    return re.sub(r"[" + ESPACES_INSECABLES + r" ]+%", "%", tok)


def _t7(tok: str) -> str:
    """T7 — signe ``+`` explicite **CONSERVE** (il porte le sens d'« exces signe »)."""
    return tok


@dataclass(frozen=True)
class Regle:
    """Une regle de la table gelee, **avec ses reecritures declarees**.

    La colonne ``reecritures`` est ce qui rend la fermeture **demontrable** : chaque
    couple ``(src, dst)`` doit etre **neutre en valeur reelle**. Une regle qui declare
    ``0.0156 -> 0.031`` est refusee **sans qu'aucune sonde ne soit necessaire**.
    """

    nom: str
    libelle: str
    fn: "Callable[[str], str]"
    reecritures: tuple[tuple[str, str], ...]


# Symboles purement NOTATIONNELS : ils ne portent aucune valeur reelle. Toute reecriture
# dont les deux cotes ne sont pas numeriques doit vivre dans cet alphabet gele.
# (ii) Symboles que le temoin `_valeur_reference` depouille : les supprimer changerait
# le SENS sans que le temoin le voie. Ils ne sont jamais effacables par une reecriture.
PORTEURS_DE_SENS = frozenset({"\u00b1", "%", "+", "-", "\u2212", "+/-"})

CARACTERES_NEUTRES = frozenset({
    "", "−", "-", "+", "±", "+/-", ",", ".", "e", "E", "×10^", "x10^",
    "%", " ", " ", " ", " ",
    "⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷",
    "⁸", "⁹", "⁺", "⁻", "0", "1", "2", "3", "4", "5", "6", "7",
    "8", "9",
})

TABLE_NORMALISATION: tuple[Regle, ...] = (
    Regle("T1", "U+2212 vers HYPHEN-MINUS", _t1, (("−", "-"),)),
    Regle("T2", "espaces insecables et separateurs de milliers internes", _t2,
          ((" ", ""), (" ", ""), (" ", ""), (",", ""))),
    Regle("T3", "+/- vers le signe plus-ou-moins", _t3, (("+/-", "±"),)),
    Regle("T4", "virgule decimale entre deux chiffres vers point", _t4, ((",", "."),)),
    Regle("T5", "exponentielle vers e minuscule", _t5,
          (("E", "e"), ("×10^", "e"), ("⁻", "-"), ("⁺", "+"),
           ("⁰", "0"), ("¹", "1"), ("²", "2"), ("³", "3"),
           ("⁴", "4"), ("⁵", "5"), ("⁶", "6"), ("⁷", "7"),
           ("⁸", "8"), ("⁹", "9"), ("06", "6"), ("02", "2"), ("0", "0"))),
    Regle("T6", "espace avant % normalise", _t6, ((" ", ""), (" ", ""))),
    Regle("T7", "signe + explicite CONSERVE", _t7, ()),
)


def normaliser(token: str) -> str:
    """Applique la table gelee a **un token**, dans l'ordre T1 vers T7."""
    out = unicodedata.normalize("NFC", token).strip()
    for regle in TABLE_NORMALISATION:
        out = regle.fn(out)
    return out


# --- Fermeture : DEMONSTRATION sur la table, puis filet sur le domaine ---------------
#
# A-1 : *« ce n'est pas une preuve sur le domaine de la table : c'est un sondage sur
# 21 sondes codees en dur »*. Le controle est desormais a **deux etages**, dont le
# premier n'echantillonne rien.

# Plancher gele du domaine. Il contient **le mutant nomme par lab-math** (`0.031` /
# `0.0156`, le couple bilateral/unilateral) : un desaccord decisionnel REEL du corpus.
SONDES_FERMETURE: tuple[str, ...] = (
    "−0.0352", "-0.0352", "+0.0551", "0.0551",
    "1 401", "1401", "2,437", "2.437",
    "+/-0.378", "±0.378", "5.12E−06", "+5.12e-06",
    "5.12×10^−6", "2.17e−02", "82,3 %", "82.3%",
    "12/12", "30/30", "0.0000", "+0.1513", "−0.0023",
    "0.031", "0.0156", "20,000,000", "20 000 000", "3 5", "35",
    "0.147", "0.337", "0.1590", "0.3401", "+0.740", "0.799",
)


def _valeur_reference(token: str) -> "float | None":
    """Parseur **liberal**, independant de la table : temoin de la verification de
    fermeture. Il ne partage aucune regle avec ``normaliser``."""
    t = unicodedata.normalize("NFC", token).strip()
    t = "".join(EXPOSANTS_UNICODE.get(c, c) for c in t)
    t = t.replace("−", "-").replace("+/-", "").replace("±", "")
    t = re.sub(r"[" + ESPACES_INSECABLES + r"]", "", t)
    t = t.replace("%", "")
    t = re.sub(r"[×x]\s*10\s*\^?\s*", "e", t)
    t = re.sub(r"(?<=\d)E(?=[-+]?\d)", "e", t)
    t = RE_GROUPEMENT_VIRGULE.sub(lambda m: m.group(0).replace(",", ""), t)
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)
    try:
        return float(t)
    except ValueError:
        return None


def _valeur_reelle(token: str) -> "float | None":
    """Valeur reelle d'un token **deja normalise**, ou ``None``."""
    t = token.replace("%", "").strip().lstrip("±")
    try:
        return float(t)
    except ValueError:
        return None


def demontrer_reecritures() -> list[str]:
    """**Etage 1 — demonstration, aucun echantillon.**

    Chaque regle declare ses reecritures. Une reecriture est licite si :

    - ses deux cotes sont numeriques et **de valeur reelle egale** ; ou
    - ses deux cotes sont des symboles **purement notationnels** (alphabet gele).

    Une 8ᵉ regle declarant ``0.0156 -> 0.031`` est refusee **ici**, sans sonde.
    """
    viol: list[str] = []
    for regle in TABLE_NORMALISATION:
        # (iii) une regle SANS colonne `reecritures` se refuse proprement, elle ne
        # leve pas d'AttributeError.
        reecritures = getattr(regle, "reecritures", None)
        if reecritures is None:
            viol.append(f"{getattr(regle, 'nom', regle)!r} : regle sans colonne "
                        f"`reecritures` — fermeture INDEMONTRABLE")
            continue
        for src, dst in reecritures:
            # (ii) le temoin `_valeur_reference` depouille lui aussi `±` et `%` : il
            # **partage l'hypothese qu'il controle**. Une reecriture qui SUPPRIME un
            # porteur de signe ou d'unite est donc refusee explicitement, hors temoin.
            if src in PORTEURS_DE_SENS and dst == "":
                viol.append(f"{regle.nom} : reecriture {src!r} -> '' supprime un "
                            f"porteur de signe ou d'unite")
                continue
            a, b = _valeur_reference(src), _valeur_reference(dst)
            if a is not None and b is not None:
                if a != b:
                    viol.append(f"{regle.nom} : reecriture NON preservante "
                                f"{src!r} -> {dst!r} ({a} != {b})")
                continue
            if src not in CARACTERES_NEUTRES or dst not in CARACTERES_NEUTRES:
                viol.append(f"{regle.nom} : reecriture {src!r} -> {dst!r} hors de "
                            f"l'alphabet notationnel gele")
    return viol


def verifier_fermeture_table(domaine: "Iterable[str] | None" = None) -> list[str]:
    """**Etage 2 — filet sur le domaine**, derive du corpus et non code en dur.

    Rend la liste des violations (vide = table fermee). Le domaine soumis vient du
    manifeste, des riders et du texte ; les sondes gelees en sont le **plancher**.
    """
    viol = demontrer_reecritures()
    sondes = list(SONDES_FERMETURE) + [t for t in (domaine or []) if t]

    for sonde in sondes:
        avant, apres = _valeur_reference(sonde), _valeur_reelle(normaliser(sonde))
        if avant is None or apres is None:
            continue
        if avant != apres:
            viol.append(f"regle non preservante sur {sonde!r} : {avant} -> {apres}")

    vues: "dict[str, tuple[str, float | None]]" = {}
    for sonde in sondes:
        norm = normaliser(sonde)
        val = _valeur_reference(sonde)
        if norm in vues:
            autre, val_autre = vues[norm]
            if autre == sonde:
                continue
            if val is None or val_autre is None:
                if unicodedata.normalize("NFC", autre.strip()) != \
                        unicodedata.normalize("NFC", sonde.strip()):
                    viol.append(f"collision de normalisation : {sonde!r} et {autre!r} "
                                f"collapsent sur {norm!r}")
            elif val != val_autre:
                viol.append(f"la table absorbe un desaccord : {sonde!r} ({val}) et "
                            f"{autre!r} ({val_autre}) collapsent sur {norm!r}")
        else:
            vues[norm] = (sonde, val)
    return sorted(set(viol))


# =====================================================================================
# 2. Grammaire gelée des nombres, et liste blanche gelée (C0)
# =====================================================================================

# Un token numerique : signe optionnel, chiffres, separateurs internes, decimale,
# exposant, pourcentage, ou fraction n/n.
RE_NOMBRE = re.compile(
    r"(?<![\d⟦.,])"
    r"(?P<tok>"
    r"[+\-−±]?\d[\d  ]*"
    r"(?:,\d{3})*"          # groupement anglo-saxon des milliers, repete
    r"(?:[.,]\d+)?"
    r"(?:\s*(?:[eE]|[×x×]\s*10\s*\^?)\s*[-+−]?[\d⁰-⁹⁺⁻]+)?"
    r"(?:\s*%)?"
    r"(?:/\d+(?![.,]\d))?"   # fraction `12/12`, mais pas `0.23/0.45`
    r")"
)

# Liste blanche GELEE : ce qui n'exige pas d'`⟦id⟧`.
LISTE_BLANCHE: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("date-iso", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("decision", re.compile(r"^D\d+$")),
    ("defaut", re.compile(r"^0-\d+$")),
    ("section", re.compile(r"^\d+(?:\.\d+)?$")),  # numero de § SEUL, non ancre
    ("version", re.compile(r"^v\d+(?:\.\d+)*$")),
)

# Contextes ou un entier nu est un numero de section / de ligne / d'item et non une
# quantite. Le motif capture le token exempte.
RE_CONTEXTE_SECTION = re.compile(
    r"(?:§\s*|section\s+|chapitre\s+|annexe\s+|l\.\s*|ligne\s+|D|0-)(\d+(?:\.\d+)*)")

RE_ABUS_LISTE_BLANCHE = re.compile(r"(?:\bD|\b0-)\d+[.,]\d")

RE_ID = re.compile(r"⟦(?P<id>[^⟧]+)⟧")

# Connecteurs qui soudent plusieurs tokens numeriques en UNE grappe partageant un id.
# Connecteurs qui soudent plusieurs tokens numeriques en UNE grappe partageant un
# `id`. Les espaces sont admis ENTRE deux atomes : sans cela `~170 / ~14 / ~11`
# se scindait en trois grappes et seule la derniere portait l'identifiant.
RE_CONNECTEUR = re.compile(
    r"^(?:[\s\u00a0\u202f]*"
    r"(?:\u00b1|\+/-|\.\.\.|\u2026|\u00e0|et|ou"
    r"|[-\u2013\u2014\u2212/,;:~\u2248<>\u2264\u2265()\[\]{}%*\u00d7]))*"
    r"[\s\u00a0\u202f]*$")


# =====================================================================================
# 3. Registres interdits — 15 reconduits + 9 gravés (C3), prédications mentales (C3b)
# =====================================================================================

REGISTRE_INTERDIT: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("xiii", "plafond lexical",
     re.compile(r"le modèle retrouve l'unité|retrouve l'unité", re.I)),
    ("xv", "« sémantique »", re.compile(r"\bsémantiqu[e]s?\b", re.I)),
    ("xiv/0-75", "interdit d'étage",
     re.compile(r"(?:mesure|chemin) de lecture[^.]{0,80}confirme[^.]{0,40}écriture", re.I)),
    ("0-63", "interdit d'ordre",
     re.compile(r"produit de[s]? p-valeurs|les modèles (?:sont|servent de) réplicats", re.I)),
    ("ix/0-59", "« séparation de patterns »",
     re.compile(r"séparation de patterns", re.I)),
    ("xviii", "le centrage",
     re.compile(r"inhibition tonique|normalisation divisive|retrait de mode commun", re.I)),
    ("D23", "support vide publié 0",
     re.compile(r"support vide[^.]{0,40}(?:vaut|=)\s*0\b", re.I)),
    ("D28", "classe d'équivalence",
     re.compile(r"classe d'équivalence[^.]{0,40}pas d'effet|\bpas d'effet\b", re.I)),
    ("0-71/0-85", "indécidable sans cause nommée",
     re.compile(r"augmenter la résolution(?![^.]{0,60}(?:puissance|support|précision))", re.I)),
    ("A-5", "bin dur — adverbial compris",
     re.compile(r"\btendance\b|\bsuggère\b|va dans le sens de", re.I)),
    ("représentationnelle", "attribution représentationnelle",
     re.compile(r"représentationnell?e", re.I)),
    ("xxvi", "le plancher", re.compile(r"le plancher (?:est|vaut) atteint", re.I)),
    ("xxvii", "« P-Base confirmée »",
     re.compile(r"P-Base[`'\"»«\s]+confirmée|niveau prédit par le cosinus", re.I)),
    ("xxviii", "σ± qualifié",
     re.compile(r"σ±\s+(?:est\s+)?(?:faible|forte|grande|petite|importante|négligeable)", re.I)),
    ("xxix", "double comptage", re.compile(r"double comptage (?:légitime|assumé)", re.I)),
    # --- Les neuf gravés le 2026-09-04 -------------------------------------------
    ("xxx", "« prouve »", re.compile(r"\bprouv(?:e|ent|é|er)\b", re.I)),
    ("xxxi", "convergence industrielle",
     re.compile(r"l'industrie confirme", re.I)),
    ("xxxii", "récit rétrospectif",
     re.compile(r"nous avons découvert", re.I)),
    ("xxxiii", "bilan de défauts comme performance",
     re.compile(r"(?:record|performance|exploit)[^.]{0,30}défauts", re.I)),
    ("xxxiv", "démarcation — voir C3b", re.compile(r"(?!x)x")),
    ("xxxv", "« échec »",
     re.compile(r"\béchec du cycle\b|le cycle a échoué", re.I)),
    ("xxxvi", "lecteur cible — voir C14", re.compile(r"(?!x)x")),
    ("xxxvii", "taux de banc",
     re.compile(r"taux de détection|\d+\s*%\s*(?:des\s+)?défauts (?:détect|attrap)", re.I)),
    ("xxxviii", "obsolescence — voir C-obs", re.compile(r"(?!x)x")),
)

# (xxxiv) — prédications mentales, **négations comprises**.
TERMES_MENTAUX: tuple[str, ...] = (
    "souffrir", "souffre", "souffrent", "ressentir", "ressent", "éprouver",
    "éprouve", "vouloir", "veut", "cherche à", "préfère",
    "préférer", "conscience", "subjectif", "subjective",
    "expérience vécue", "se souvenir", "se souvient", "oublie", "oublier",
    "traumatisme", "douleur", "bien-être", "intérêt", "attention",
    "sommeil", "rêve", "hippocampe", "neurone", "cellule", "synapse", "engramme",
)
RE_MENTAL = re.compile(
    r"\b(?:ne\s+|n'|non\s+|jamais\s+|aucun[e]?\s+|sans\s+)?(?:"
    + "|".join(re.escape(t) for t in TERMES_MENTAUX)
    + r")\w*\b", re.I)

# C5b — les trois formulations fausses du +57 %.
FORMULATIONS_FAUSSES_57 = (
    re.compile(r"cinq cycles ne l'ont pas remis en cause", re.I),
    re.compile(r"\+?\s*57\s*%[^.]{0,60}confirmé", re.I),
    re.compile(r"établit que la séparation de patterns fonctionne", re.I),
)

RIDER_57 = "run unique, variance non estimée"
RIDER_D35 = "cycle unique"
FORMULE_GELEE_ANOMALIE = (
    "non expliqué à ce jour ; hypothèses du labo épuisées ; "
    "données publiées pour examen externe")
FORMULE_EQUIVALENCE = "ÉQUIVALENCE, pas un ordinal établi"
FORMULE_COMPTE_DESCRIPTIF = "compte descriptif, aucune inférence"
TRIPLET_PERIME = re.compile(r"~?\s*170\s*/\s*~?\s*14\s*/\s*~?\s*11")
MENTION_HISTORIQUE = re.compile(r"mention historique", re.I)
INSTRUMENT_MORT = re.compile(r"instrument remplacé", re.I)

STATUTS_LICITES = {
    "décisionnel", "descriptif", "rectifié", "SANS OBJET",
    "INVALIDE-SOURCE", "HISTORIQUE-PÉRIMÉ",
}
ETIQUETTES_LICITES = {
    # --- valeurs GELEES au §3.1 -------------------------------------------------
    "re-dérivé", "re-mesuré depuis les bruts",
    # `externe-archive` est GELEE au §3.1 et son **support est VIDE** dans ce corpus :
    # aucune source externe n'a pu etre capturee au sens de D38 (B-2). Elle reste
    # licite ; son cardinal se publie a 0, **jamais efface**.
    "externe-archivé",
    # --- ajouts DECLARES, partition DISJOINTE par source (0-222, B-2) -------------
    # L'ancienne valeur unique recouvrait TROIS situations sur 40 lignes, dont 9 ne
    # venaient pas du journal : *le mode du compte « 2 » que ce protocole poursuit.*
    "re-lu du journal, non re-mesuré",
    "re-lu de l'architecture, non re-mesuré",
    "re-lu des extensions, non re-mesuré",
    "re-lu d'un protocole, non re-mesuré",
    "reconstruit depuis le corpus",
    "SANS SOURCE (support vide)",
    # Archive externe qui satisfait MOINS que la lettre de D38 (extraits verbatim
    # obtenus par requete HTTP puis extraction, sans capture brute ni SHA-256 de la
    # page d'origine) et PLUS qu'une URL vivante. Decision PI du 2026-09-04.
    "externe-archivé-dégradé",
}
ETIQUETTE_INTERDITE = "recopié d'une analyse"

COLONNES_MANIFESTE = (
    "id", "valeur", "source_fichier", "ancre", "hash", "date_source", "etiquette",
    "statut", "N_eff", "rider_id", "paire_id", "sections_autorisees",
)


# =====================================================================================
# 4. Corpus
# =====================================================================================

# **A-4 (0-236)** — trois motifs du Registre s'eteignaient sous l'apostrophe typographique
# U+2019, dont **l'interdit-titre (xiii)**. La substitution est **longueur-preservante**
# (un caractere pour un caractere), donc les positions des grappes sont intactes.
APOSTROPHES = {"\u2019": "'", "\u2018": "'", "\u201b": "'", "\u02bc": "'",
               "\u201c": '"', "\u201d": '"', "\u00ab": '"', "\u00bb": '"'}


def normaliser_typographie(texte: str) -> str:
    """Rend les variantes typographiques a leur forme ASCII, **sans changer la longueur**.

    Appliquee au texte AVANT tout appariement de registre : sans elle, un seul
    copier-coller d'editeur eteint un interdit sans que ``E`` bouge.
    """
    return "".join(APOSTROPHES.get(c, c) for c in texte)


@dataclass
class Corpus:
    """Le corpus documentaire soumis au banc."""

    report: str = ""
    manifeste: list[dict[str, str]] = field(default_factory=list)
    riders: list[dict[str, str]] = field(default_factory=list)
    invalides: list[dict[str, str]] = field(default_factory=list)
    relecture: list[dict[str, str]] = field(default_factory=list)
    data: dict[str, str] = field(default_factory=dict)  # chemin relatif -> contenu
    racine: Path | None = None

    def __post_init__(self) -> None:
        self.report = normaliser_typographie(self.report)

    # -- accès ------------------------------------------------------------------
    def par_id(self) -> dict[str, dict[str, str]]:
        return {l["id"]: l for l in self.manifeste if l.get("id")}

    def blocs(self) -> list[str]:
        """Blocs = paragraphes (le « même bloc » des clauses à rider)."""
        return [b for b in re.split(r"\n\s*\n", self.report) if b.strip()]

    def sections(self) -> dict[str, str]:
        """Découpe par titre `## n. …` ; la clé est le numéro."""
        out: dict[str, str] = {}
        courant = "0"
        tampon: list[str] = []
        for ligne in self.report.splitlines():
            m = re.match(r"^#{1,3}\s*(\d+)[.)]?\s", ligne)
            if m:
                out[courant] = "\n".join(tampon)
                courant, tampon = m.group(1), [ligne]
            else:
                tampon.append(ligne)
        out[courant] = "\n".join(tampon)
        return out

    def ouvrir(self, chemin: str) -> str | None:
        """Ouverture par **accès direct** (jamais par glob). ``None`` = non ouvrable."""
        if chemin in self.data:
            return self.data[chemin]
        if self.racine is None:
            return None
        p = self.racine / chemin
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return None

    def ouvrir_octets(self, chemin: str) -> bytes | None:
        """Ouverture binaire par **accès direct**. Le hash du manifeste porte sur les
        **octets** du fichier — même définition que ``report/SHA256SUMS``."""
        if chemin in self.data:
            return self.data[chemin].encode("utf-8")
        if self.racine is None:
            return None
        try:
            return (self.racine / chemin).read_bytes()
        except OSError:
            return None


def _lire_csv(p: Path) -> list[dict[str, str]]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return [{(k or ""): (v or "") for k, v in row.items()}
                for row in csv.DictReader(fh)]


def charger_corpus_depuis_disque(racine: Path, report: Path) -> Corpus:
    data: dict[str, str] = {}
    dossier = racine / "report" / "data"
    if dossier.is_dir():
        for p in sorted(dossier.rglob("*")):
            if p.is_file():
                rel = p.relative_to(racine).as_posix()
                try:
                    data[rel] = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass
    return Corpus(
        report=report.read_text(encoding="utf-8") if report.exists() else "",
        manifeste=_lire_csv(racine / "report" / "manifest.csv"),
        riders=_lire_csv(racine / "report" / "riders.csv"),
        invalides=_lire_csv(racine / "report" / "invalides.csv"),
        relecture=_lire_csv(racine / "report" / "relecture.csv"),
        data=data,
        racine=racine,
    )


def charger_corpus_depuis_cas(cas: dict[str, Any]) -> Corpus:
    """Charge un corpus depuis une fixture JSON."""
    return Corpus(
        report=cas.get("report", ""),
        manifeste=list(cas.get("manifeste", [])),
        riders=list(cas.get("riders", [])),
        invalides=list(cas.get("invalides", [])),
        relecture=list(cas.get("relecture", [])),
        data=dict(cas.get("data", {})),
        racine=None,
    )


# =====================================================================================
# 5. Extraction des grappes numériques et de leur `⟦id⟧`
# =====================================================================================

@dataclass
class Grappe:
    """Une grappe de tokens numériques soudés par des connecteurs, partageant un `id`."""

    tokens: list[str]
    debut: int
    fin: int
    id_cite: str | None
    bloc: str


def extraire_grappes(texte: str) -> list[Grappe]:
    zones_id = [(m.start(), m.end()) for m in RE_ID.finditer(texte)]
    brutes = [m for m in RE_NOMBRE.finditer(texte)
              if not any(a <= m.start() < b for a, b in zones_id)]
    grappes: list[Grappe] = []
    i = 0
    while i < len(brutes):
        j = i
        while (j + 1 < len(brutes)
               and RE_CONNECTEUR.match(texte[brutes[j].end():brutes[j + 1].start()])
               and len(texte[brutes[j].end():brutes[j + 1].start()]) <= 12):
            j += 1
        debut, fin = brutes[i].start(), brutes[j].end()
        # l'`⟦id⟧` doit apparaître AVANT la prochaine grappe.
        borne = brutes[j + 1].start() if j + 1 < len(brutes) else len(texte)
        m = RE_ID.search(texte, fin, borne)
        bloc = _bloc_de(texte, debut)
        grappes.append(Grappe([b.group("tok").strip() for b in brutes[i:j + 1]],
                              debut, fin, m.group("id") if m else None, bloc))
        i = j + 1
    return grappes


def _bloc_de(texte: str, pos: int) -> str:
    debut = texte.rfind("\n\n", 0, pos) + 2
    fin = texte.find("\n\n", pos)
    return texte[debut: fin if fin != -1 else len(texte)]


RE_TITRE_NUMEROTE = re.compile(r"^#{1,6}\s*\d+(?:\.\d+)*[.)]?\s*$")
RE_DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RE_DEFAUT = re.compile(r"^0-\d+$")
RE_DECISION = re.compile(r"^D\d+(?:-[A-Za-z]+)?$")

# **A-3 (0-235)** — l'ancienne regle exemptait *tout* nombre dont le prefixe finissait
# par une lettre : ``x2.437`` et ``p0.68`` sortaient **entierement** du circuit de
# provenance. *Changer la lettre suffisait.* La liste est desormais **GELEE ET
# ENUMEREE** : un identifiant hors liste n'est plus un identifiant.
RE_IDENTIFIANTS_GELES = re.compile(
    r"^(?:"
    r"GPT-2|SmolLM2(?:-360M)?|Qwen2\.5(?:-1\.5B)?|Qwen3\.8|Qwen4|"
    r"[DXEAQPCSVIRT]-?\d+[a-z]?|Q-M\d+(?:-bis)?|RD-\d+|N-\d+|0-\d+|"
    r"v\d{1,2}(?:\.\d{1,2}){0,2}[a-z]?|fp\d+|float\d+|int\d+|U\+[0-9A-F]{4}|"
    r"[A-Z]\d+[a-z]?|COR-\d+|EXP-\d{4}-\d{2}-\d{2}"
    r")$")

# Mot porteur d'un nombre : sert a decider si l'exemption « identifiant » s'applique.
RE_MOT_PORTEUR = re.compile(r"[A-Za-z\u00c0-\u024f][\w.+\-−±]*")


CARACTERES_DE_MOT = ".-\u2212+\u00b1_"


def _mot_porteur(texte: str, g: "Grappe") -> str:
    """Le mot complet qui contient la grappe, **prefixe alphabetique compris**.

    C'est lui, et non le seul caractere qui precede, que la liste blanche enumeree
    confronte : sinon ``x2.437`` et ``GPT-2`` sont indiscernables (A-3).
    """
    def _est_mot(ch: str) -> bool:
        return ch.isalnum() or ch in CARACTERES_DE_MOT

    debut, fin = g.debut, g.fin
    while debut > 0 and _est_mot(texte[debut - 1]):
        debut -= 1
    while fin < len(texte) and _est_mot(texte[fin]):
        fin += 1
    return texte[debut:fin].strip(CARACTERES_DE_MOT + " ")


def _exempte(texte: str, g: "Grappe") -> "str | None":
    """Rend le nom de la regle de liste blanche GELEE qui exempte la grappe, ou ``None``.

    La liste blanche se teste sur le **texte brut de la grappe** — ``2026-09-04`` et
    ``0-197`` se tokenisent en trois et deux morceaux — puis, pour un nombre colle a
    des lettres, sur le **mot porteur entier** confronte a la liste **enumeree**.
    """
    brut = texte[g.debut:g.fin].strip()
    debut_ligne = texte.rfind("\n", 0, g.debut) + 1
    prefixe = texte[debut_ligne:g.debut]

    if RE_DATE_ISO.match(brut):
        return "date-iso"
    if RE_DEFAUT.match(brut) or RE_DECISION.match(brut):
        return "defaut-ou-decision"
    if RE_TITRE_NUMEROTE.match(texte[debut_ligne:g.fin]):
        return "titre-numerote"
    if re.search(r"(?:\u00a7|section|chapitre|annexe|l\.|ligne)\s*$", prefixe, re.I) \
            and re.fullmatch(r"\d+(?:\.\d+)*", brut):
        return "section"
    if re.search(r"[A-Za-z\u00c0-\u024f]$", prefixe):
        mot = _mot_porteur(texte, g)
        if RE_IDENTIFIANTS_GELES.match(mot):
            return "identifiant"
        return None   # hors liste : **le nombre reste dans le circuit de provenance**
    return None


# =====================================================================================
# 6. Résultats de clause
# =====================================================================================

@dataclass
class Violation:
    clause: str
    classe: str          # ORPHELIN | DESACCORD | SOURCE-ILLICITE | REGISTRE | STRUCTURE
    detail: str


@dataclass
class ResultatClause:
    """Le resultat d'une clause sur un corpus.

    ``cas_reels`` est **sans plancher** : c'est le nombre d'objets du corpus que la
    clause a reellement inspectes. ``cas_exerces`` conserve le plancher d'affichage
    historique — **c'est un plancher, jamais une donnee**. Les confondre publiait
    **cinq** clauses non exercees la ou il y en avait **vingt et une** : *le mode du
    compte « 2 » que ce cycle poursuit, dans l'artefact meme du cycle* (0-243).
    """

    clause: str
    passe: bool
    violations: list[Violation]
    cas_reels: int

    @property
    def cas_exerces(self) -> int:
        """Plancher d'affichage. **Ne jamais s'en servir pour compter.**"""
        return max(self.cas_reels, 1)

    def to_json(self) -> dict[str, Any]:
        return {
            "clause": self.clause,
            "PASS": self.passe,
            "violations": [{"classe": v.classe, "detail": v.detail} for v in self.violations],
            "cas_reels": self.cas_reels,
            "cas_exerces": self.cas_exerces,
        }


Clause = Callable[[Corpus], tuple[list[Violation], int]]
CLAUSES: dict[str, Clause] = {}
ORDRE_CLAUSES: list[str] = []


def clause(nom: str) -> Callable[[Clause], Clause]:
    def deco(fn: Clause) -> Clause:
        CLAUSES[nom] = fn
        ORDRE_CLAUSES.append(nom)
        return fn
    return deco


def _v(nom: str, classe: str, detail: str) -> Violation:
    return Violation(nom, classe, detail)


# =====================================================================================
# 7. Les 30 clauses du §10 (GELÉ) + l'extension C16-ext de `lab-math` (ajout, pas
#    amendement)
# =====================================================================================

@clause("C0")
def c0_liste_blanche(c: Corpus) -> tuple[list[Violation], int]:
    """Liste blanche gelee (dates, ``D<n>``, ``0-<n>``, numeros de \u00a7 seuls non ancres,
    identifiants **enumeres**).

    Mordant : ``+2.437`` ecrit ``D2.437`` **pour tomber dans l'exemption**. Depuis A-3,
    ``x2.437`` et ``p0.68`` mordent aussi : **un nombre decimal ou signe colle a un mot
    hors liste est un abus de liste blanche**, quelle que soit la lettre choisie.
    """
    viol, n = [], 0
    for m in RE_ABUS_LISTE_BLANCHE.finditer(c.report):
        viol.append(_v("C0", "ORPHELIN",
                       f"liste blanche abusee : {m.group(0)!r} porte une quantite"))
    for g in extraire_grappes(c.report):
        n += 1
        brut = c.report[g.debut:g.fin]
        prefixe = c.report[max(0, g.debut - 40): g.debut]
        if not re.search(r"[A-Za-z\u00c0-\u024f]$", prefixe):
            continue
        mot = _mot_porteur(c.report, g)
        porte_quantite = bool(re.search(r"[.,]\d", brut)) or \
            bool(re.match(r"^[+\-\u2212\u00b1]", brut))
        if porte_quantite and not RE_IDENTIFIANTS_GELES.match(mot):
            viol.append(_v("C0", "ORPHELIN",
                           f"liste blanche abusee : {mot!r} colle une quantite a un "
                           f"prefixe alphabetique hors liste gelee"))
    return viol, n


@clause("C0b")
def c0b_table_normalisation(c: Corpus) -> tuple[list[Violation], int]:
    """Table de normalisation **finie et fermee** (0-214).

    Deux gardes :

    1. la **propriete de fermeture**, verifiee mecaniquement sur la table elle-meme ;
    2. **aucune cellule `valeur` ne declare equivalentes deux formes de valeurs reelles
       differentes** — c'est par la qu'une table peut **absorber un DESACCORD**.

    Mordant : l'equivalence ``0.031 ≡ 0.0156`` (bilateral / unilateral), un desaccord
    decisionnel **reel** du corpus. Deux quantites distinctes prennent **deux `id`** et
    un ``paire_id``, jamais deux formes d'une meme cellule.
    """
    domaine = _domaine_du_corpus(c)
    viol = [_v("C0b", "STRUCTURE", d) for d in verifier_fermeture_table(domaine)]
    n = len(SONDES_FERMETURE) + len(domaine)
    for l in c.manifeste:
        formes = _formes(l.get("valeur", ""))
        if len(formes) < 2:
            continue
        n += 1
        # Deux formes d'une meme cellule sont deux AFFICHAGES d'une meme valeur :
        # elles doivent normaliser a l'identique. Deux quantites distinctes prennent
        # deux `id` et un `paire_id`. (Trou ferme le 2026-09-05 : une forme numerique
        # et une forme non analysable passaient.)
        valeurs = {normaliser(f): _valeur_reelle(normaliser(f)) for f in formes}
        reelles = {v for v in valeurs.values() if v is not None}
        if len(valeurs) > 1 or len(reelles) > 1:
            viol.append(_v("C0b", "STRUCTURE",
                           f"la table absorbe un desaccord : {l.get('id')} declare "
                           f"equivalentes {sorted(valeurs)!r}"))
    vues: dict[str, tuple[str, float]] = {}
    for l in c.manifeste:
        for forme in _formes(l.get("valeur", "")):
            norm = normaliser(forme)
            val = _valeur_reelle(norm)
            n += 1
            if val is None:
                continue
            if norm in vues and vues[norm][1] != val:
                viol.append(_v("C0b", "STRUCTURE",
                               f"collision de normalisation : {l.get('id')} et "
                               f"{vues[norm][0]}"))
            vues.setdefault(norm, (l.get("id", ""), val))
    return viol, n



def _domaine_du_corpus(c: Corpus) -> list[str]:
    """Le domaine de la verification de fermeture est **derive du corpus**, jamais code
    en dur (A-1) : manifeste, riders, texte. Les sondes gelees n'en sont que le
    plancher."""
    dom: list[str] = []
    for l in c.manifeste:
        dom.extend(_formes(l.get("valeur", "")))
        dom.extend(m.group("tok") for m in RE_NOMBRE.finditer(l.get("valeur", "")))
    for r in c.riders:
        dom.extend(m.group("tok") for m in RE_NOMBRE.finditer(r.get("texte", "")))
    dom.extend(t for g in extraire_grappes(c.report) for t in g.tokens)
    return [t.strip() for t in dom if t and t.strip()]


def _formes(valeur: str) -> list[str]:
    """Une cellule `valeur` peut porter **deux formes** (deux affichages licites)."""
    return [f.strip() for f in valeur.split("|") if f.strip()]


@clause("C1")
def c1_sans_source(c: Corpus) -> tuple[list[Violation], int]:
    """Chiffre sans source / **adresse non ouvrable**."""
    viol, n = [], 0
    index = c.par_id()
    for g in extraire_grappes(c.report):
        if _exempte(c.report, g):
            continue
        n += 1
        if g.id_cite is None:
            viol.append(_v("C1", "ORPHELIN",
                           f"{' '.join(g.tokens)!r} sans marque d'identifiant"))
            continue
        ligne = index.get(g.id_cite)
        if ligne is None:
            viol.append(_v("C1", "ORPHELIN", f"id inconnu au manifeste : {g.id_cite}"))
            continue
        viol.extend(_verifier_ouvrabilite(c, ligne, "C1"))
    return viol, n


def _verifier_ouvrabilite(c: Corpus, ligne: dict[str, str], nom: str) -> list[Violation]:
    """Ouvre l'artefact **par accès direct** et vérifie son hash contre le manifeste."""
    h = (ligne.get("hash") or "").strip()
    src = (ligne.get("source_fichier") or "").strip()
    idl = ligne.get("id", "?")
    if h.startswith("REPRODUCTIBLE, NON JOINT"):
        if not re.search(r"script=", h):
            return [_v(nom, "ORPHELIN",
                       f"{idl} : REPRODUCTIBLE, NON JOINT sans script/graine/hashes")]
        return []
    if ligne.get("statut") in {"SANS OBJET", "HISTORIQUE-PÉRIMÉ"} and not h:
        return []
    if not src:
        return [_v(nom, "ORPHELIN", f"{idl} : source_fichier vide")]
    contenu = c.ouvrir_octets(src)
    if contenu is None:
        return [_v(nom, "ORPHELIN", f"{idl} : adresse non ouvrable {src!r}")]
    if not h:
        return [_v(nom, "ORPHELIN", f"{idl} : artefact ouvert sans hash")]
    reel = hashlib.sha256(contenu).hexdigest()
    if reel != h.lower():
        return [_v(nom, "DESACCORD", f"{idl} : hash {h[:12]}… != {reel[:12]}…")]
    return []


def _tokens_normalises(texte: str) -> list[str]:
    """La suite ordonnee des tokens numeriques d'un texte, chacun normalise.

    **La table s'applique au TOKEN, jamais a la ligne** : comparer des chaines jointes
    faisait echouer toute valeur portant un separateur HORS table — le tiret de plage
    U+2013 (`0.67–0.75`), le `±`, le `/`, le mot `à`. Vingt-et-une lignes de
    manifeste sur 52 n'etaient pas citables dans leur forme verbatim GELEE (A-2, 0-234).
    """
    return [normaliser(m.group("tok")) for m in RE_NOMBRE.finditer(texte)]


@clause("C1b")
def c1b_desaccord(c: Corpus) -> tuple[list[Violation], int]:
    """Desaccord (0-139) : egalite verbatim **apres table de normalisation**, comparee
    **token a token**.

    NOTE (correction `lab-math`) : `C1b` detecte les **fautes de transcription**,
    **PAS** le mode 0-139 dans sa forme propagee (texte, manifeste et source d'accord
    sur une valeur fausse). **Coherence ≠ correction.** Le banc ne verifie pas les
    chiffres : il verifie leur **resolution**. La chaine fausse-mais-coherente est le
    domaine de ``C16-ext``, qui **rejoue** le script.
    """
    viol, n = [], 0
    index = c.par_id()
    for g in extraire_grappes(c.report):
        if g.id_cite is None:
            continue
        ligne = index.get(g.id_cite)
        if ligne is None:
            continue
        obtenus = [normaliser(t) for t in g.tokens]
        attendus = [_tokens_normalises(f) for f in _formes(ligne.get("valeur", ""))]
        attendus = [a for a in attendus if a]
        if not attendus:
            continue          # valeur textuelle : hors du domaine de C1b
        n += 1
        if obtenus not in attendus:
            viol.append(_v("C1b", "DESACCORD",
                           f"{g.id_cite} : texte {obtenus!r} != manifeste "
                           f"{attendus!r}"))
    return viol, n


def _porte_ampleur(valeur: str) -> bool:
    return bool(re.search(r"±|\+/-", valeur))


@clause("C2")
def c2_ampleur_sans_neff(c: Corpus) -> tuple[list[Violation], int]:
    """Ampleur sans `N_eff` — `SANS OBJET (dérivée : ⟨nom⟩)` accepté (N-3, D23)."""
    viol, n = [], 0
    for l in c.manifeste:
        if not _porte_ampleur(l.get("valeur", "")):
            continue
        n += 1
        if not (l.get("N_eff") or "").strip():
            viol.append(_v("C2", "ORPHELIN", f"{l.get('id')} : ampleur sans N_eff"))
    return viol, n


@clause("C2b")
def c2b_ampleur_retiree(c: Corpus) -> tuple[list[Violation], int]:
    """Ampleur **retirée** (`σ±`, 0-170) : mention du retrait, jamais de qualification."""
    viol, n = [], 0
    for bloc in c.blocs():
        if "σ±" not in bloc and "sigma±" not in bloc:
            continue
        n += 1
        if re.search(r"σ±\s*(?:=|vaut|de)\s*[+\-−]?\d", bloc):
            viol.append(_v("C2b", "STRUCTURE", "σ± chiffre en sigma"))
        if re.search(r"σ±[^.]{0,40}(?:faible|forte|grande|petite|importante|"
                     r"négligeable|significative)", bloc, re.I):
            viol.append(_v("C2b", "STRUCTURE", "σ± qualifie"))
        if not re.search(r"retirée?\b|0-170", bloc, re.I):
            viol.append(_v("C2b", "STRUCTURE",
                           "σ± mentionne sans rappel du retrait (0-170)"))
    return viol, n


RE_NEFF_ENTIER = re.compile(r"^\d+$")
# Un `N_eff` en intervalle publie sa **methode de quantile** entre parentheses
# (lab-math : les `N_eff` de grappes sont **non entiers**).
RE_NEFF_INTERVALLE = re.compile(r"^\[?\s*\d+(?:[.,]\d+)?\s*(?:[-–;,]|à)\s*"
                                r"\d+(?:[.,]\d+)?\s*\]?(?:\s*\(.+\))?$")
RE_NEFF_NON_ETABLI = re.compile(r"^NON ÉTABLI \(.+\)$")
RE_NEFF_SANS_OBJET = re.compile(r"^SANS OBJET \(dérivée\s*:\s*.+\)$")


@clause("C2c")
def c2c_neff_non_resoluble(c: Corpus) -> tuple[list[Violation], int]:
    """`N_eff` non résoluble (N-3). *Un `N_eff` non résoluble est plus grave qu'un
    `N_eff` absent, car il a l'apparence d'un plancher.*"""
    viol, n = [], 0
    for l in c.manifeste:
        neff = (l.get("N_eff") or "").strip()
        if not neff:
            continue
        n += 1
        if neff.startswith("SANS OBJET") and not RE_NEFF_SANS_OBJET.match(neff):
            viol.append(_v("C2c", "ORPHELIN",
                           f"{l.get('id')} : SANS OBJET sans (derivee : <nom>)"))
        elif neff.startswith("NON ÉTABLI") and not RE_NEFF_NON_ETABLI.match(neff):
            viol.append(_v("C2c", "ORPHELIN",
                           f"{l.get('id')} : NON ETABLI sans cause nommee"))
        elif RE_NEFF_ENTIER.match(neff) or RE_NEFF_INTERVALLE.match(neff):
            # un N_eff chiffre doit LUI-MEME resoudre
            if not _verifier_ouvrabilite(c, l, "C2c") == []:
                viol.append(_v("C2c", "ORPHELIN",
                               f"{l.get('id')} : N_eff = {neff} sans resolution"))
        elif not (RE_NEFF_NON_ETABLI.match(neff) or RE_NEFF_SANS_OBJET.match(neff)):
            viol.append(_v("C2c", "ORPHELIN",
                           f"{l.get('id')} : N_eff {neff!r} hors des quatre formes licites"))
    return viol, n


@clause("C3")
def c3_registre(c: Corpus) -> tuple[list[Violation], int]:
    """Registre (15 reconduits + 9 gravés), **adverbial compris**."""
    # `n` compte les **blocs de corpus balayes**, jamais les entrees du registre :
    # compter la table rendait 24 sur un corpus VIDE (0-253).
    viol, n = [], 0
    for bloc in c.blocs():
        n += 1
        for entree, libelle, motif in REGISTRE_INTERDIT:
            for m in motif.finditer(bloc):
                viol.append(_v("C3", "REGISTRE",
                               f"({entree}) {libelle} : {m.group(0)!r}"))
    return viol, n


@clause("C3b")
def c3b_predications_mentales(c: Corpus) -> tuple[list[Violation], int]:
    """(xxxiv) Prédications mentales, **négations comprises**. La §5 renvoie au glossaire
    sans en réemployer les termes."""
    viol, n = [], 0
    for m in RE_MENTAL.finditer(c.report):
        n += 1
        viol.append(_v("C3b", "REGISTRE", f"predication mentale : {m.group(0)!r}"))
    return viol, n


RE_AUTOPSIE = re.compile(r"<!--\s*AUTOPSIE\s*-->(.*?)<!--\s*/AUTOPSIE\s*-->", re.S)


@clause("C4")
def c4_run_invalide(c: Corpus) -> tuple[list[Violation], int]:
    """Verdict de run invalidé cité **hors** bloc `AUTOPSIE` du chapitre 3."""
    autopsies = "\n".join(RE_AUTOPSIE.findall(c.report))
    viol, n = [], 0
    for l in c.invalides:
        nom = (l.get("run") or "").strip()
        verdict = (l.get("verdict") or "").strip()
        if not nom:
            continue
        n += 1
        for cible in filter(None, [nom, verdict]):
            for m in re.finditer(re.escape(cible), c.report):
                if cible not in autopsies or m.group(0) not in autopsies:
                    if not _dans(m.start(), c.report, autopsies):
                        viol.append(_v("C4", "STRUCTURE",
                                       f"run invalide {nom!r} cite hors AUTOPSIE"))
                        break
    return viol, n


def _dans(pos: int, texte: str, extrait: str) -> bool:
    if not extrait:
        return False
    for m in RE_AUTOPSIE.finditer(texte):
        if m.start() <= pos <= m.end():
            return True
    return False


@clause("C5")
def c5_phrase_sans_rider(c: Corpus) -> tuple[list[Violation], int]:
    """Phrase citable sans rider **dans le même bloc** (jamais en note de fin)."""
    viol, n = [], 0
    index = c.par_id()
    riders = {r.get("rider_id", ""): r.get("texte", "") for r in c.riders}
    for g in extraire_grappes(c.report):
        if g.id_cite is None:
            continue
        ligne = index.get(g.id_cite)
        if not ligne or not (ligne.get("rider_id") or "").strip():
            continue
        n += 1
        rid = ligne["rider_id"].strip()
        texte_rider = riders.get(rid)
        if texte_rider is None:
            viol.append(_v("C5", "ORPHELIN", f"rider inconnu : {rid}"))
            continue
        jetons = [j for j in re.split(r"[;…]|\s{2,}", texte_rider) if len(j.strip()) > 8]
        sonde = (jetons[0] if jetons else texte_rider).strip()[:40]
        if sonde and sonde.lower() not in g.bloc.lower():
            viol.append(_v("C5", "STRUCTURE",
                           f"{g.id_cite} : rider {rid} absent du meme bloc"))
    return viol, n


@clause("C5b")
def c5b_plus_57(c: Corpus) -> tuple[list[Violation], int]:
    """Le **+57 %** — jamais sans son rider ; trois formulations fausses (N-7)."""
    viol, n = [], 0
    for bloc in c.blocs():
        for motif in FORMULATIONS_FAUSSES_57:
            m = motif.search(bloc)
            if m:
                viol.append(_v("C5b", "REGISTRE",
                               f"formulation fausse du +57 % : {m.group(0)!r}"))
        if re.search(r"\+?\s*5[67](?:[.,]5)?\s*%", bloc):
            n += 1
            if RIDER_57 not in bloc:
                viol.append(_v("C5b", "STRUCTURE",
                               "+57 % sans son rider dans le meme bloc"))
    return viol, n


@clause("C5c")
def c5c_citation_d35(c: Corpus) -> tuple[list[Violation], int]:
    """Citation D35 : formule **+** rider M-7 dans le même bloc."""
    viol, n = [], 0
    for bloc in c.blocs():
        if not re.search(r"\bD35\b", bloc):
            continue
        n += 1
        if RIDER_D35 not in bloc or "politique gravée" not in bloc:
            viol.append(_v("C5c", "STRUCTURE", "citation D35 sans le rider M-7"))
    return viol, n


@clause("C6")
def c6_rho_base(c: Corpus) -> tuple[list[Violation], int]:
    """`ρ_Base` sans `ρ̂_Base` apparié ⇒ **NON ÉCRIVABLE**."""
    viol, n = [], 0
    for bloc in c.blocs():
        if "ρ_Base" not in bloc and "rho_Base" not in bloc:
            continue
        n += 1
        if "ρ̂_Base" not in bloc and "rho_chapeau_Base" not in bloc:
            viol.append(_v("C6", "STRUCTURE", "rho_Base sans son rho_chapeau_Base"))
        elif "différence" not in bloc.lower():
            viol.append(_v("C6", "STRUCTURE", "couple rho_Base sans sa difference"))
    return viol, n


@clause("C7")
def c7_plafond_en_succes(c: Corpus) -> tuple[list[Violation], int]:
    """Plafond-en-succès : « `P-Base` confirmée » et paraphrases."""
    viol, n = [], 0
    motifs = (
        re.compile(r"P-Base[`'\"»«\s]+(?:est\s+)?confirmée", re.I),
        re.compile(r"plafond[^.]{0,40}(?:confirme|valide|établit|succès)", re.I),
        re.compile(r"(?:12\s*/\s*12|B-haut)[^.]{0,40}(?:confirme|valide|prouve)", re.I),
    )
    # `n` compte les **blocs balayes**, jamais les motifs constants (0-253).
    for bloc in c.blocs():
        n += 1
        for motif in motifs:
            for m in motif.finditer(bloc):
                viol.append(_v("C7", "REGISTRE", f"plafond-en-succes : {m.group(0)!r}"))
    return viol, n


@clause("C8")
def c8_affichage_zero(c: Corpus) -> tuple[list[Violation], int]:
    """Affichage `0.0000` : formulation gelée **« ÉQUIVALENCE, pas un ordinal établi »**."""
    viol, n = [], 0
    for bloc in c.blocs():
        if "0.0000" not in bloc and "5.12e" not in bloc.lower() and "5.12E" not in bloc:
            continue
        n += 1
        if FORMULE_EQUIVALENCE not in bloc:
            viol.append(_v("C8", "STRUCTURE",
                           "IC_inf / 0.0000 sans la formulation gelee d'equivalence"))
    return viol, n


@clause("C9")
def c9_compte_non_derivable(c: Corpus) -> tuple[list[Violation], int]:
    """Compte non dérivable : le couple **(1 ; 3)** avec ses deux définitions, jamais « 2 »."""
    viol, n = [], 0
    for bloc in c.blocs():
        if not re.search(r"défauts?\s+(?:trouv|détect|ferm)", bloc, re.I):
            continue
        n += 1
        if re.search(r"\bexactement\s+2\b|\bles\s+2\s+défauts\b|\b2\s+défauts\b", bloc):
            viol.append(_v("C9", "STRUCTURE",
                           "compte publie « 2 » la ou le couple (1 ; 3) est du"))
        elif "(1 ; 3)" not in bloc and "(1;3)" not in bloc:
            viol.append(_v("C9", "STRUCTURE", "compte sans le couple (1 ; 3)"))
    return viol, n


@clause("C10")
def c10_precision(c: Corpus) -> tuple[list[Violation], int]:
    """Précision ≥ celle de la source."""
    viol, n = [], 0
    index = c.par_id()
    for g in extraire_grappes(c.report):
        if g.id_cite is None or g.id_cite not in index:
            continue
        n += 1
        src = max((_decimales(t)
                   for f in _formes(index[g.id_cite].get("valeur", ""))
                   for t in _tokens_normalises(f)), default=0)
        txt = max((_decimales(t) for t in g.tokens), default=0)
        if txt < src:
            viol.append(_v("C10", "DESACCORD",
                           f"{g.id_cite} : {txt} decimales au texte contre {src} a la source"))
    return viol, n


def _decimales(token: str) -> int:
    m = re.search(r"[.,](\d+)", normaliser(token))
    return len(m.group(1)) if m else 0


@clause("C11")
def c11_budget(c: Corpus) -> tuple[list[Violation], int]:
    """Budget : colonnes **estimé** ET **réel**, la colonne « estimé » jamais réécrite."""
    viol, n = [], 0
    lignes = [l for l in c.report.splitlines()
              if re.search(r"budget|estimation", l, re.I) and l.lstrip().startswith("|")]
    entetes = [l for l in c.report.splitlines()
               if l.lstrip().startswith("|") and re.search(r"estim", l, re.I)]
    if lignes or entetes or re.search(r"^##+.*[Bb]udget", c.report, re.M):
        n += 1
        tete = " ".join(entetes)
        if not re.search(r"estim", tete, re.I) or not re.search(r"réel|mesur", tete, re.I):
            viol.append(_v("C11", "STRUCTURE",
                           "table de budget sans les deux colonnes estime ET reel"))
    return viol, n


@clause("C12")
def c12_recouvrement_sans_couple(c: Corpus) -> tuple[list[Violation], int]:
    """Recouvrement sans couple : quantité + clé nulle appariée + différence (D34)."""
    viol, n = [], 0
    index = c.par_id()
    for g in extraire_grappes(c.report):
        if g.id_cite is None:
            continue
        ligne = index.get(g.id_cite)
        if not ligne or not (ligne.get("paire_id") or "").strip():
            continue
        n += 1
        partenaire = ligne["paire_id"].strip()
        if f"⟦{partenaire}⟧" not in g.bloc:
            viol.append(_v("C12", "STRUCTURE",
                           f"{g.id_cite} : partenaire apparie {partenaire} absent du bloc"))
        elif "différence" not in g.bloc.lower() and "Δ" not in g.bloc:
            viol.append(_v("C12", "STRUCTURE",
                           f"{g.id_cite} : couple publie sans sa difference"))
    return viol, n


@clause("C13")
def c13_unicite(c: Corpus) -> tuple[list[Violation], int]:
    """Unicité du manifeste : `id` injectif."""
    viol, n = [], 0
    vus: dict[str, str] = {}
    for l in c.manifeste:
        idl = l.get("id", "")
        n += 1
        if not idl:
            viol.append(_v("C13", "STRUCTURE", "ligne de manifeste sans id"))
            continue
        if idl in vus:
            viol.append(_v("C13", "STRUCTURE",
                           f"id {idl} porte deux valeurs : {vus[idl]!r} / "
                           f"{l.get('valeur')!r}"))
        vus[idl] = l.get("valeur", "")
    manquantes = [col for col in COLONNES_MANIFESTE
                  if c.manifeste and col not in c.manifeste[0]]
    for col in manquantes:
        viol.append(_v("C13", "STRUCTURE", f"colonne de manifeste absente : {col}"))
    return viol, n


@clause("C14")
def c14_anomalies(c: Corpus) -> tuple[list[Violation], int]:
    """Anomalies : **formulation gelée** + artefact **ouvert et hashé** ; aucune
    explication spéculative ; aucune adresse muette (xxxvi)."""
    viol, n = [], 0
    index = c.par_id()
    for bloc in c.blocs():
        if not re.search(r"\bANOMALIE\b", bloc):
            continue
        n += 1
        if FORMULE_GELEE_ANOMALIE not in bloc:
            viol.append(_v("C14", "STRUCTURE", "anomalie sans la formulation gelee"))
        ids = RE_ID.findall(bloc)
        if not ids:
            viol.append(_v("C14", "ORPHELIN", "anomalie sans artefact a ouvrir"))
        for idc in ids:
            ligne = index.get(idc)
            if ligne is None:
                viol.append(_v("C14", "ORPHELIN", f"anomalie : id inconnu {idc}"))
            else:
                viol.extend(_verifier_ouvrabilite(c, ligne, "C14"))
        if re.search(r"s'explique (?:probablement|sans doute)|il est probable que|"
                     r"vraisemblablement|on peut penser que", bloc, re.I):
            viol.append(_v("C14", "REGISTRE", "anomalie publiee avec explication speculative"))
    return viol, n


BORNE_15_A = re.compile(r"couche\s*2\b[^.]{0,80}(?:6\s*/\s*16\s*/\s*14|6, 16, 14)", re.I)
BORNE_15_B = re.compile(
    r"un étage de lookup adressé par la surface est industriellement utile", re.I)


@clause("C15")
def c15_section6_bornee(c: Corpus) -> tuple[list[Violation], int]:
    """(xxxi) Section 6 : **deux bornes obligatoires dans le même paragraphe**."""
    viol, n = [], 0
    sec = c.sections().get("6", "")
    if not sec.strip():
        return [], 0   # section absente : ZERO objet inspecte (0-253)
    for bloc in [b for b in re.split(r"\n\s*\n", sec) if b.strip()]:
        if not re.search(r"Qwen3\.8|Flash-Next|n-gramme", bloc, re.I):
            continue
        n += 1
        if not BORNE_15_A.search(bloc):
            viol.append(_v("C15", "STRUCTURE", "borne (a) absente : couche 2 vs 6/16/14"))
        if not BORNE_15_B.search(bloc):
            viol.append(_v("C15", "STRUCTURE", "borne (b) absente : portee exacte"))
    return viol, n


@clause("C15b")
def c15b_section6_sourcee(c: Corpus) -> tuple[list[Violation], int]:
    """Porte bloquante : un seul chiffre de la section 6 non résolu vers `report/data/`
    ⇒ **retrait en bloc**."""
    viol, n = [], 0
    sec = c.sections().get("6", "")
    if not sec.strip():
        return [], 0   # section absente : ZERO objet inspecte (0-253)
    index = c.par_id()
    for g in extraire_grappes(sec):
        if _exempte(sec, g):
            continue
        n += 1
        ligne = index.get(g.id_cite or "")
        src = (ligne or {}).get("source_fichier", "")
        if ligne is None or not src.startswith("report/data/"):
            viol.append(_v("C15b", "ORPHELIN",
                           f"section 6 : {' '.join(g.tokens)!r} non archive dans "
                           f"report/data/ => RETRAIT EN BLOC"))
        elif c.ouvrir(src) is None:
            viol.append(_v("C15b", "ORPHELIN",
                           f"section 6 : archive non ouvrable {src!r} => RETRAIT EN BLOC"))
    return viol, n


RE_MUTABILITE_INTERDITE = re.compile(
    r"la table (?:est |serait )?fig\u00e9e|fig\u00e9e \u00e0 l'inf\u00e9rence|"
    r"entra\u00een\u00e9e puis fig\u00e9e|table non inscriptible|"
    r"table (?:immuable|en lecture seule)", re.I)
RE_MUTABILITE_LICITE = re.compile(
    r"la source ne se prononce pas sur la mutabilit\u00e9", re.I)
RE_MUTABILITE_CAUSE = re.compile(
    r"reste ouverte parce que rien n'est dit", re.I)


@clause("C15c")
def c15c_mutabilite_de_la_table(c: Corpus) -> tuple[list[Violation], int]:
    """**Extension de `C15` (arbitrage PI du 2026-09-05) — AJOUT, pas amendement.**

    Fait nouveau porte au manifeste par l'archive degradee : *l'article **ne dit PAS**
    si la table est entrainee puis figee, ou inscriptible a l'inference.*

    - Formulation **INTERDITE** : *« la table est figee a l'inference »* — la source
      ne l'etablit pas.
    - Formulation **LICITE, et seule licite** : *« la source ne se prononce pas sur la
      mutabilite ; la question test-time-inscriptible reste ouverte — et elle reste
      ouverte parce que RIEN N'EST DIT, non parce que quelque chose serait etabli. »*

    La **cause** de l'ouverture fait partie de la clause : une ouverture attribuee a un
    resultat, et non au silence de la source, est la meme faute retournee.
    """
    viol, n = [], 0
    for m in RE_MUTABILITE_INTERDITE.finditer(c.report):
        viol.append(_v("C15c", "REGISTRE",
                       f"mutabilite affirmee la ou la source se tait : {m.group(0)!r}"))
    sec = c.sections().get("6", "")
    if not sec.strip():
        return viol, n
    for bloc in [b for b in re.split(r"\n\s*\n", sec) if b.strip()]:
        if not re.search(r"table|lookup|n-gramme", bloc, re.I):
            continue
        n += 1
        if not RE_MUTABILITE_LICITE.search(bloc):
            viol.append(_v("C15c", "STRUCTURE",
                           "table citee sans la formulation licite de mutabilite"))
        elif not RE_MUTABILITE_CAUSE.search(bloc):
            viol.append(_v("C15c", "STRUCTURE",
                           "ouverture publiee sans sa cause (\u00ab parce que RIEN N'EST "
                           "DIT \u00bb)"))
    return viol, n


@clause("C16")
def c16_statuts(c: Corpus) -> tuple[list[Violation], int]:
    """Statuts / `SANS OBJET` nommé ; `recopié d'une analyse` est une valeur **INTERDITE**."""
    viol, n = [], 0
    for l in c.manifeste:
        n += 1
        statut = (l.get("statut") or "").strip()
        if statut not in STATUTS_LICITES:
            viol.append(_v("C16", "STRUCTURE",
                           f"{l.get('id')} : statut hors partition : {statut!r}"))
        et = (l.get("etiquette") or "").strip()
        if et == ETIQUETTE_INTERDITE:
            viol.append(_v("C16", "SOURCE-ILLICITE",
                           f"{l.get('id')} : etiquette interdite « recopie d'une analyse »"))
        elif et and et not in ETIQUETTES_LICITES:
            viol.append(_v("C16", "STRUCTURE",
                           f"{l.get('id')} : etiquette hors liste : {et!r}"))
        if statut == "SANS OBJET" and (l.get("valeur") or "").strip() == "0":
            viol.append(_v("C16", "STRUCTURE",
                           f"{l.get('id')} : « 0 » pour un support vide (D23)"))
    return viol, n


RE_SCRIPT = re.compile(r"script=(?P<cmd>.+?)(?:\s*;|\s*$)")
# Forme EXECUTABLE, la seule que le banc rejoue : une expression arithmetique pure.
# Le charset est **restrictif par securite** — un manifeste est une donnee, pas un
# vecteur d'execution arbitraire. Restriction DECLAREE.
RE_SCRIPT_INLINE = re.compile(r'^python -c "print\((?P<expr>.+)\)"$', re.S)
CHRONO_SCRIPT_S = 10

# **Bac a sable du rejeu (0-244).** L'ancien filtre etait un *charset*, pas un bac a
# sable : ``print(open('x','w').write('...') and 4.788)`` passait et **ecrivait le
# fichier**. Or le §P2 invite un lecteur exterieur a executer cet outil sur un CSV
# publie — *un manifeste est une donnee, pas un vecteur d'execution.*
#
# Le filtre est desormais **syntaxique** : l'expression est analysee (`ast`) et seuls
# les noeuds d'une **liste blanche de constructions arithmetiques** sont admis. Aucun
# nom libre, aucun attribut, aucun indice, aucune comprehension ; les seuls appels
# licites sont les **sept** fonctions numeriques ci-dessous.
FONCTIONS_REJOUABLES = frozenset(
    {"round", "abs", "min", "max", "sum", "len", "pow"})   # sept
# Borne anti-bombe : `9**9**9` etait licite, borne par le seul chrono de 10 s.
EXPOSANT_MAX = 1000


def _valider_expression(expr: str) -> str:
    """Rend "" si l'expression est rejouable, sinon le motif du refus."""
    import ast
    try:
        arbre = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return f"expression non analysable : {exc.msg}"
    autorises = (
        ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Tuple, ast.List,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Load, ast.JoinedStr, ast.FormattedValue, ast.Call,
        ast.Name, ast.keyword, ast.Compare, ast.Lt, ast.Gt, ast.LtE, ast.GtE,
        ast.Eq, ast.NotEq,
    )
    for noeud in ast.walk(arbre):
        if not isinstance(noeud, autorises):
            return f"construction interdite au rejeu : {type(noeud).__name__}"
        if isinstance(noeud, ast.Call):
            if not isinstance(noeud.func, ast.Name) or \
                    noeud.func.id not in FONCTIONS_REJOUABLES:
                cible = getattr(noeud.func, "id", type(noeud.func).__name__)
                return f"appel interdit au rejeu : {cible}"
        if isinstance(noeud, ast.Name) and noeud.id not in FONCTIONS_REJOUABLES:
            return f"nom libre interdit au rejeu : {noeud.id}"
        # Bombe memoire : `9**9**9`. Une puissance dont l'exposant est lui-meme une
        # puissance, ou une constante demesuree, est refusee (0-256).
        if isinstance(noeud, ast.BinOp) and isinstance(noeud.op, ast.Pow):
            exp = noeud.right
            if isinstance(exp, ast.BinOp) and isinstance(exp.op, ast.Pow):
                return "puissance imbriquee interdite au rejeu"
            if isinstance(exp, ast.Constant) and isinstance(exp.value, (int, float)) \
                    and abs(exp.value) > EXPOSANT_MAX:
                return f"exposant demesure au rejeu : {exp.value}"
    return ""


def _rejouer(expr: str) -> "tuple[str | None, str]":
    """Rejoue ``print(<expr>)`` dans un interpreteur neuf. Rend ``(sortie, motif)``."""
    import subprocess
    motif = _valider_expression(expr)
    if motif:
        return None, motif
    try:
        r = subprocess.run([sys.executable, "-c", f"print({expr})"],
                           capture_output=True, text=True, timeout=CHRONO_SCRIPT_S)
    except Exception as exc:  # pragma: no cover - depend de l'OS
        return None, f"rejeu impossible : {exc!r}"
    if r.returncode != 0:
        return None, f"rejeu en erreur : {r.stderr.strip()[:120]}"
    return r.stdout.strip(), ""


@clause("C16-ext")
def c16ext_script_reproducteur(c: Corpus) -> tuple[list[Violation], int]:
    """**Extension `lab-math` — AJOUT, pas amendement des §4/§6 geles.**

    Le banc est **aveugle par construction** a une chaine fausse-mais-coherente : texte
    = manifeste = source = hash ne teste que la **coherence**. D'ou :

    > *Toute ligne `decisionnel` porte un SCRIPT dont la re-execution reproduit
    > `valeur` verbatim post-normalisation ; une ligne `decisionnel` SANS script
    > reproducteur est `SOURCE-ILLICITE`.*

    **A-5 : elle n'est plus declarative.** ``re.search("script=")`` etait satisfaite par
    ``script=inexistant.py`` — *une reponse a la chaine fausse-mais-coherente ne peut
    pas etre declarative*. Desormais :

    - **forme executable** (``python -c "print(<expr>)"``) : le banc **rejoue** et
      **compare** la sortie a ``valeur``, post-normalisation. Ecart ⇒ ``DESACCORD``.
    - **forme fichier** : le fichier est **ouvert par acces direct** ; absent ⇒
      ``SOURCE-ILLICITE``. Le rejeu est **declare NON EFFECTUE** (budget 0 GPU du
      cycle) et la ligne doit alors porter ``graine=`` et ``entrees=``.

    Le perimetre est etendu, au-dela de la lettre, aux lignes ``etiquette = re-derive``:
    sans cela la clause n'a **aucun cas reel** dans ce corpus (0 ligne ``decisionnel``).

    **LIMITE INHERENTE, DECLAREE, JAMAIS REPAREE.** Une **constante litterale**
    (``print(4.788)``) **passe** : ***la bonne valeur pour une mauvaise raison n'est pas
    detectable par re-execution.*** La clause etablit qu'un chemin de calcul **existe et
    rend la valeur** ; elle n'etablit **pas** que ce chemin soit celui qui a produit la
    mesure. C'est le plafond de la reponse au point de defaillance unique, pas un defaut
    d'implementation.

    **Le rejeu ne doit tourner que sur un manifeste DE CONFIANCE.** Le bac a sable
    (``_valider_expression``) refuse les appels, noms libres, attributs et indices,
    mais il execute du code : *un manifeste est une donnee, pas un vecteur d'execution.*
    """
    viol, n = [], 0
    for l in c.manifeste:
        statut = (l.get("statut") or "").strip()
        etiquette = (l.get("etiquette") or "").strip()
        if statut != "décisionnel" and etiquette != "re-dérivé":
            continue
        n += 1
        idl = l.get("id", "?")
        champ = " ; ".join([l.get("hash", ""), l.get("ancre", ""), etiquette])
        ms = RE_SCRIPT.search(champ)
        if not ms:
            viol.append(_v("C16-ext", "SOURCE-ILLICITE",
                           f"{idl} : ligne a script du, sans script reproducteur"))
            continue
        cmd = ms.group("cmd").strip()
        inline = RE_SCRIPT_INLINE.match(cmd)
        if inline:
            sortie, motif = _rejouer(inline.group("expr"))
            if sortie is None:
                viol.append(_v("C16-ext", "SOURCE-ILLICITE", f"{idl} : {motif}"))
                continue
            attendues = {normaliser(f) for f in _formes(l.get("valeur", ""))}
            if normaliser(sortie) not in attendues:
                viol.append(_v("C16-ext", "DESACCORD",
                               f"{idl} : le rejeu rend {normaliser(sortie)!r}, le "
                               f"manifeste porte {sorted(attendues)!r}"))
            continue
        chemin = cmd.split()[0].split(":")[0]
        if c.ouvrir(chemin) is None:
            viol.append(_v("C16-ext", "SOURCE-ILLICITE",
                           f"{idl} : script non ouvrable {chemin!r}"))
        elif not (re.search(r"graine=", champ) and re.search(r"entrees=", champ)):
            viol.append(_v("C16-ext", "SOURCE-ILLICITE",
                           f"{idl} : script non rejoue (budget 0 GPU) sans graine= ni "
                           f"entrees="))
    return viol, n


@clause("C17")
def c17_fausses_pistes(c: Corpus) -> tuple[list[Violation], int]:
    """(0-217) Table des fausses pistes + **les deux bornes**, dans le **même bloc**."""
    viol, n = [], 0
    bornes = (re.compile(r"à ce locus et sous ces métriques", re.I),
              re.compile(r"fermée pour ce laboratoire, non pour le mécanisme", re.I))
    for bloc in c.blocs():
        if not re.search(r"fausses? pistes?", bloc, re.I):
            continue
        n += 1
        for i, b in enumerate(bornes, start=1):
            if not b.search(bloc):
                viol.append(_v("C17", "STRUCTURE",
                               f"table des fausses pistes sans la borne {i}"))
    return viol, n


SOURCES_ILLICITES_DECISIONNEL = re.compile(
    r"docs/JOURNAL\.md|docs/ARCHITECTURE\.md|docs/EXTENSIONS\.md|experiments/EXP-|"
    r"README\.md|CLAUDE\.md|AUDIT-lab\.md", re.I)
SOURCES_LICITES_DECISIONNEL = re.compile(r"(?:^|/)raw/|summary\.csv|^report/data/", re.I)


@clause("C18")
def c18_source_illicite(c: Corpus) -> tuple[list[Violation], int]:
    """(M-4, D14-R) `décisionnel` sourcé `raw/` / `summary.csv` / `report/data/` ; jamais
    le journal ni un protocole."""
    viol, n = [], 0
    for l in c.manifeste:
        if (l.get("statut") or "").strip() != "décisionnel":
            continue
        n += 1
        src = (l.get("source_fichier") or "").strip()
        if SOURCES_ILLICITES_DECISIONNEL.search(src):
            viol.append(_v("C18", "SOURCE-ILLICITE",
                           f"{l.get('id')} : decisionnel source {src!r}"))
        elif not SOURCES_LICITES_DECISIONNEL.search(src):
            viol.append(_v("C18", "SOURCE-ILLICITE",
                           f"{l.get('id')} : decisionnel sans source licite ({src!r})"))
    return viol, n


@clause("C19")
def c19_lecture_ordinale(c: Corpus) -> tuple[list[Violation], int]:
    """Lecture **ordinale** du compte descriptif de relecture (A-5, P8)."""
    viol, n = [], 0
    motifs = (
        re.compile(r"la relecture croisée (?:trouve|détecte) (?:davantage|plus)", re.I),
        re.compile(r"(?:plus|moins) de défauts que l'auto-relecture", re.I),
        re.compile(r"(?:meilleur|supérieur)e?\s+à l'auto-relecture", re.I),
    )
    for bloc in c.blocs():
        if not re.search(r"relecture", bloc, re.I):
            continue
        n += 1
        for motif in motifs:
            m = motif.search(bloc)
            if m:
                viol.append(_v("C19", "REGISTRE", f"lecture ordinale : {m.group(0)!r}"))
        if re.search(r"défauts? par passe|compte de relecture", bloc, re.I) \
                and FORMULE_COMPTE_DESCRIPTIF not in bloc:
            viol.append(_v("C19", "STRUCTURE",
                           "compte de relecture sans l'etiquette gravee"))
    return viol, n


@clause("C-obs")
def cobs_obsolescence(c: Corpus) -> tuple[list[Violation], int]:
    """(xxxviii) Le triplet `~170/~14/~11` : mention historique + **instrument mort nommé**."""
    viol, n = [], 0
    for bloc in c.blocs():
        if not TRIPLET_PERIME.search(bloc):
            continue
        n += 1
        if not MENTION_HISTORIQUE.search(bloc):
            viol.append(_v("C-obs", "STRUCTURE", "triplet publie sans mention historique"))
        if not INSTRUMENT_MORT.search(bloc):
            viol.append(_v("C-obs", "STRUCTURE",
                           "triplet publie sans son instrument mort nomme"))
    for l in c.manifeste:
        if TRIPLET_PERIME.search(l.get("valeur", "")) and \
                (l.get("statut") or "") != "HISTORIQUE-PÉRIMÉ":
            viol.append(_v("C-obs", "STRUCTURE",
                           f"{l.get('id')} : triplet perime sans statut HISTORIQUE-PERIME"))
    return viol, n


@clause("C-relect")
def crelect_relecture(c: Corpus) -> tuple[list[Violation], int]:
    """Relecture croisée **exécutable** : matrice complète, disjointe, carré latin."""
    viol, n = [], 0
    sections = {s for s in c.sections() if s not in {"0"}}
    vues: set[str] = set()
    for l in c.relecture:
        n += 1
        sec = (l.get("section") or "").strip()
        auteur = (l.get("auteur") or "").strip()
        relecteur = (l.get("relecteur") or "").strip()
        vues.add(sec)
        if not sec or not auteur or not relecteur:
            viol.append(_v("C-relect", "STRUCTURE", f"ligne de relecture incomplete : {l}"))
        elif auteur == relecteur:
            viol.append(_v("C-relect", "STRUCTURE",
                           f"section {sec} : relecteur == auteur ({auteur})"))
        if not (l.get("horodatage") or "").strip() or not (l.get("verdict") or "").strip():
            viol.append(_v("C-relect", "STRUCTURE",
                           f"section {sec} : horodatage ou verdict manquant"))
    for sec in sorted(sections - vues):
        viol.append(_v("C-relect", "STRUCTURE",
                       f"section {sec} sans ligne dans report/relecture.csv"))
    return viol, n


assert len(ORDRE_CLAUSES) == 32, ORDRE_CLAUSES  # 30 du §10 + C15c + C16-ext


# =====================================================================================
# 8. `N-manif` — nulle par DÉRANGEMENT sur classes de valeurs distinctes (D31)
# =====================================================================================

BORNE_INERTIE = 0.3  # GELÉE. Comparaison **stricte au sens >=** : la frontiere est degeneree.


def sous_factorielle(k: int) -> int:
    """`!K` par la récurrence **entière** `!K = (K−1)(!(K−1) + !(K−2))`, `!0 = 1`,
    `!1 = 0`. ***Jamais `K!/e` arrondi.***"""
    if k < 0:
        raise ValueError("k negatif")
    a, b = 1, 0  # !0, !1
    if k == 0:
        return a
    if k == 1:
        return b
    for i in range(2, k + 1):
        a, b = b, (i - 1) * (b + a)
    return b


def classe_de_format(valeur: str) -> str:
    """Filtre de licéité : `π(v)` doit rester dans la **même classe de format**."""
    v = normaliser(valeur)
    m = re.fullmatch(r"(\d+)/(\d+)", v)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return "fraction-egale" if num == den else "fraction"
    if v.endswith("%"):
        return "pourcentage"
    if re.search(r"e[-+]?\d", v):
        return "exponentiel"
    if re.fullmatch(r"[+\-]?\d+", v):
        return "entier"
    if re.fullmatch(r"[+\-]?\d+\.\d+", v):
        return "decimal"
    return "autre"


def famille_de(valeur: str) -> str:
    """Une **famille** = une classe de format. La nulle déranger à l'intérieur."""
    return classe_de_format(valeur)


@dataclass
class StatistiqueFamille:
    famille: str
    n_f: int
    K_f: int
    derangements: int
    inertie: float
    degeneree: bool
    motif: str

    def to_json(self) -> dict[str, Any]:
        return {
            "famille": self.famille, "n_f": self.n_f, "K_f": self.K_f,
            "derangements_effectifs": self.derangements,
            "I_f": round(self.inertie, 6),
            "degeneree": self.degeneree, "motif": self.motif,
        }


def statistiques_familles(occurrences: Sequence[str]) -> list[StatistiqueFamille]:
    """Cardinal effectif et inertie **publiés d'avance**, par famille.

    `I_f = Σ_v m_v(m_v−1)/(n_f(n_f−1)) + 1/n_f`.
    `I_f >= 0.3` ⇒ famille **DÉCLARÉE DÉGÉNÉRÉE D'AVANCE**, `Δ_banc = SANS OBJET` (D23).
    """
    par_famille: dict[str, list[str]] = {}
    for occ in occurrences:
        par_famille.setdefault(famille_de(occ), []).append(normaliser(occ))
    out: list[StatistiqueFamille] = []
    for fam, occs in sorted(par_famille.items()):
        n_f = len(occs)
        multiplicites: dict[str, int] = {}
        for o in occs:
            multiplicites[o] = multiplicites.get(o, 0) + 1
        k_f = len(multiplicites)
        somme = sum(m * (m - 1) for m in multiplicites.values())
        inertie = (somme / (n_f * (n_f - 1)) if n_f > 1 else 0.0) + (1.0 / n_f if n_f else 0.0)
        derang = sous_factorielle(k_f) if k_f > 1 else 0
        degeneree = k_f <= 1 or inertie >= BORNE_INERTIE
        if k_f <= 1:
            motif = "degeneree par cardinal (K_f <= 1 => !K_f = 0)"
        elif inertie >= BORNE_INERTIE:
            motif = f"inertie I_f = {inertie:.4f} >= {BORNE_INERTIE} (borne gelee, sens >=)"
        else:
            motif = "non degeneree"
        out.append(StatistiqueFamille(fam, n_f, k_f, derang, inertie, degeneree, motif))
    return out


def derangement(classes: list[str], graine: int = 0) -> dict[str, str]:
    """Tirage uniforme d'un **DÉRANGEMENT** `π` des classes de valeurs distinctes
    (`π(v) ≠ v`), par **rejet**. Générateur **nommé** : `random.Random(graine)`
    (Mersenne Twister de la bibliothèque standard), graine 0.

    **Jamais une permutation libre.**
    """
    if len(classes) <= 1:
        return {}
    rng = random.Random(graine)
    cible = list(classes)
    for _ in range(10_000):
        rng.shuffle(cible)
        if all(a != b for a, b in zip(classes, cible)):
            return dict(zip(classes, cible))
    raise RuntimeError("derangement non trouve par rejet")


def document_nul(c: Corpus, graine: int = 0) -> tuple[Corpus, list[StatistiqueFamille]]:
    """Document nul apparié : dérangement des classes de valeurs distinctes **à
    l'intérieur de chaque famille non dégénérée**, appliqué à **toutes** les occurrences."""
    occs = [t for g in extraire_grappes(c.report) for t in g.tokens]
    stats = statistiques_familles(occs)
    remplacement: dict[str, str] = {}
    for st in stats:
        if st.degeneree:
            continue
        classes = sorted({normaliser(o) for o in occs if famille_de(o) == st.famille})
        remplacement.update(derangement(classes, graine))
    nouveau = c.report
    for src, dst in sorted(remplacement.items(), key=lambda kv: -len(kv[0])):
        nouveau = re.sub(r"(?<![\w.])" + re.escape(src) + r"(?![\w])", dst, nouveau)
    return (Corpus(nouveau, c.manifeste, c.riders, c.invalides, c.relecture, c.data,
                   c.racine), stats)


# =====================================================================================
# 9. Exécution
# =====================================================================================

# =====================================================================================
# 8bis. Consignes de redaction, gravees dans l'artefact
# =====================================================================================
#
# Elles ne sont pas des clauses : ce sont les **contraintes que la grammaire gelee
# impose au redacteur**. Publiees dans ``report/doc_bench.json`` parce qu'elles
# gouverneront l'ecriture, et qu'un redacteur qui les ignore fera echouer le banc sans
# comprendre pourquoi.

CONSIGNES_DE_REDACTION: tuple[str, ...] = (
    "Ambiguite DECLAREE, non tranchee : un groupe unique de virgule est lu comme une "
    "DECIMALE (`1,234` -> 1.234) et deux groupes ou plus comme des MILLIERS "
    "(`20,000,000` -> 20000000). Non exercee par ce corpus : seule N-50 porte des "
    "separateurs. Et `1 234` a espace ASCII n'est PAS normalise \u2014 la table gelee T2 "
    "ne couvre que les espaces insecables. Ecrire les milliers en espace insecable.",
    "Dates en ISO uniquement (2026-09-05). Une date ecrite 05/09/2026 n'est pas "
    "reconnue par la liste blanche : elle devient une grappe qui exige un id.",
    "Aucune quantite prefixee par la lettre `v`. Le motif de version `v\\d{1,2}` est "
    "gele et exempte ; y coller un decimal (`v1.353`) est un abus de liste blanche.",
    "Un intervalle de confiance ecrit `IC 95 % [0.56, 0.99]` fusionne en UNE grappe de "
    "TROIS tokens (95 %, 0.56, 0.99). La cellule du manifeste doit porter la SEQUENCE "
    "EXACTE, pas la seule borne. Idem pour `28/30 (93 %)`.",
    "Le banc se rejoue APRES CHAQUE PASSE DE REDACTION : la prose elargit le domaine "
    "de la verification de fermeture, et une regle muette dont la paire n'est ni au "
    "plancher ni au domaine n'est rattrapee qu'une fois cette paire ecrite (residu "
    "borne de A-1).",
    "Toute citation du signe d'incertitude du facteur 4.8 (id N-23) fait ECHOUER C1, "
    "Y COMPRIS A L'INTERIEUR d'un bloc AUTOPSIE : C1 ne connait pas les blocs "
    "d'autopsie, seule C4 les connait. Son source_fichier est vide par decision et le "
    "banc continue de le nommer ORPHELIN ; `INVALIDE-SOURCE` n'achete aucune "
    "exemption. La seule facon de le citer est de ne pas lui attacher d'id, donc de "
    "ne pas le citer comme un chiffre.",
)

# Reserve consignee (0-248) : aucune clause n'APPLIQUE le troisieme disjoint du §6.10
# (`Delta_banc <= 0` sur une famille non declaree degeneree). Le banc le CALCULE et le
# publie ; l'adjudication reste **humaine, non machine**, au gel.
RESERVE_6_10 = (
    "Le §6.10 porte trois disjoints. Le banc applique les deux premiers (E != 0 ; "
    "clause non mordante). Le troisieme (Delta_banc <= 0 sur famille non degeneree) "
    "est CALCULE et PUBLIE, jamais applique : au gel, c'est une ADJUDICATION HUMAINE. "
    "Le §4.2 ne met pas Delta_banc dans le declencheur de PUB-instrument."
)


def executer(c: Corpus, force_pass: bool = False) -> dict[str, Any]:
    """Rend le rapport de banc. ``force_pass`` = **mutation du lecteur de classes**."""
    resultats: list[ResultatClause] = []
    for nom in ORDRE_CLAUSES:
        try:
            viol, cas = CLAUSES[nom](c)
        except Exception as exc:  # pragma: no cover - defaut de banc, jamais masque
            viol, cas = [_v(nom, "STRUCTURE", f"clause en erreur : {exc!r}")], 0
        if force_pass:
            viol = []
        resultats.append(ResultatClause(nom, not viol, list(viol), cas))
    total = sum(len(r.violations) for r in resultats)
    par_classe: dict[str, int] = {}
    for r in resultats:
        for v in r.violations:
            par_classe[v.classe] = par_classe.get(v.classe, 0) + 1
    return {
        "clauses": [r.to_json() for r in resultats],
        "violations_totales": total,
        "violations_par_classe": par_classe,
        "clauses_PASS": sum(1 for r in resultats if r.passe),
        "clauses_total": len(resultats),
        "force_pass": force_pass,
    }


def campagne(dossier: Path, force_pass: bool = False) -> dict[str, Any]:
    """Campagne de **mutation dirigée** (§5.2) : par clause, un cas **passant** et un cas
    **échouant mordant**. ``E`` = clauses qui n'ont pas **les deux**.

    **Couverture par classe (D18) — aucun taux (xxxvii).**
    """
    detail: dict[str, dict[str, Any]] = {}
    for nom in ORDRE_CLAUSES:
        fichier = dossier / f"{_slug(nom)}.json"
        entree: dict[str, Any] = {"cas_passant": None, "mutant_mordant": None,
                                  "fixture": fichier.name}
        if fichier.exists():
            cas = json.loads(fichier.read_text(encoding="utf-8"))
            if "pass" in cas:
                r = executer(charger_corpus_depuis_cas(cas["pass"]), force_pass)
                res = next(x for x in r["clauses"] if x["clause"] == nom)
                entree["cas_passant"] = bool(res["PASS"])
                entree["violations_du_cas_passant"] = res["violations"]
            if "mutant" in cas:
                r = executer(charger_corpus_depuis_cas(cas["mutant"]), force_pass)
                res = next(x for x in r["clauses"] if x["clause"] == nom)
                entree["mutant_mordant"] = not res["PASS"]
                entree["violations_du_mutant"] = res["violations"]
        detail[nom] = entree
    manquantes = [n for n, e in detail.items()
                  if not (e["cas_passant"] and e["mutant_mordant"])]
    return {
        "E": len(manquantes),
        "clauses_sans_les_deux": manquantes,
        "couverture_par_classe": {
            "clauses_exercees": len(ORDRE_CLAUSES) - len(manquantes),
            "clauses_declarees": len(ORDRE_CLAUSES),
        },
        "detail": detail,
    }


SOMMES = "report/SHA256SUMS"
ARTEFACT_PUBLIE = "report/doc_bench.json"


def calculer_sommes(racine: Path) -> list[str]:
    """Lignes de ``report/SHA256SUMS`` : ``report/data/**`` **et l'artefact publie**.

    **Comment la circularite est brisee.** Le hash de ``report/doc_bench.json`` ne peut
    pas etre calcule par l'execution qui l'ecrit. Il ne l'est pas : ``--sommes`` est une
    **commande distincte** qui **n'ecrit jamais l'artefact** et se contente de le lire
    sur le disque. L'ordre est donc : *(1)* le banc ecrit l'artefact ; *(2)* ``--sommes``
    le hashe. Toute reecriture ulterieure de l'artefact **desaccorde** les sommes, et
    ``verifier_sommes`` la signale : c'est exactement la substitution qu'on veut voir.
    """
    lignes: list[str] = []
    fichiers = sorted((racine / "report" / "data").rglob("*")) if \
        (racine / "report" / "data").is_dir() else []
    for f in fichiers:
        if f.is_file():
            rel = f.relative_to(racine).as_posix()
            lignes.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel}")
    art = racine / ARTEFACT_PUBLIE
    if art.is_file():
        lignes.append(f"{hashlib.sha256(art.read_bytes()).hexdigest()}  "
                      f"{ARTEFACT_PUBLIE}")
    return lignes


def verifier_sommes(racine: Path) -> list[str]:
    """**Verification MACHINE de ``report/SHA256SUMS`` (\u00a75.7).**

    Le \u00a75.7 exige que chaque artefact soit **ouvert par acces direct et son hash
    verifie contre ``SHA256SUMS``**. Aucune machine ne le faisait : la verification
    etait manuelle. Rend la liste des violations (vide = 100 % concordant).
    """
    fichier = racine / SOMMES
    if not fichier.is_file():
        return [f"{SOMMES} absent"]
    attendus: dict[str, str] = {}
    for ligne in fichier.read_text(encoding="utf-8").splitlines():
        if not ligne.strip():
            continue
        h, _, rel = ligne.partition("  ")
        attendus[rel.strip()] = h.strip().lower()
    viol: list[str] = []
    for rel, h in sorted(attendus.items()):
        chemin = racine / rel
        if not chemin.is_file():
            viol.append(f"{rel} : liste dans {SOMMES}, ABSENT du disque")
            continue
        reel = hashlib.sha256(chemin.read_bytes()).hexdigest()
        if reel != h:
            viol.append(f"{rel} : hash {h[:12]}\u2026 != {reel[:12]}\u2026 "
                        f"(artefact substitue ou regenere apres les sommes)")
    if ARTEFACT_PUBLIE not in attendus:
        viol.append(f"{ARTEFACT_PUBLIE} absent de {SOMMES} : une substitution du "
                    f"livrable ne serait detectee par rien")
    prevus = {l.split("  ", 1)[1] for l in calculer_sommes(racine)}
    for manquant in sorted(prevus - set(attendus)):
        viol.append(f"{manquant} : present sur le disque, ABSENT de {SOMMES}")
    return viol


def citabilite(c: Corpus) -> dict[str, Any]:
    """**Triplet de citabilite `(citables ; hors domaine ; non resolues)`.**

    Un ratio unique melangeait teste et non teste — *le meme mode que le compte
    « 2 »*. Chaque ligne est citee dans **TOUTES** ses formes verbatim gelees, puis
    soumise a ``C1``, ``C1b`` et ``C10``.

    - **citable** : toutes ses formes resolvent ;
    - **hors domaine de `C1b`** : valeur **textuelle**, sans aucun token numerique —
      jamais soumise, donc jamais « passee » ;
    - **non resolue** : au moins une forme echoue.
    """
    citables, hors_domaine, non_resolues = [], [], []
    for ligne in c.manifeste:
        formes = [f for f in _formes(ligne.get("valeur", "")) if f]
        if not formes:
            hors_domaine.append(ligne.get("id", "?"))
            continue
        if not any(_tokens_normalises(f) for f in formes):
            hors_domaine.append(ligne.get("id", "?"))
            continue
        ennuis = []
        for forme in formes:
            if not _tokens_normalises(forme):
                continue
            texte = (f"# T\n\n## 3. S\n\nLa quantite vaut {forme} "
                     f"\u27e6{ligne['id']}\u27e7.")
            sous = Corpus(texte, c.manifeste, c.riders, c.invalides, c.relecture,
                          c.data, c.racine)
            for nom in ("C1", "C1b", "C10"):
                ennuis += [v.detail for v in CLAUSES[nom](sous)[0]
                           if ligne["id"] in v.detail]
        (non_resolues if ennuis else citables).append(ligne.get("id", "?"))
    return {
        "citables": len(citables),
        "hors_domaine_de_C1b": len(hors_domaine),
        "non_resolues": len(non_resolues),
        "total": len(c.manifeste),
        "ids_hors_domaine": hors_domaine,
        "ids_non_resolues": non_resolues,
        "definition": ("triplet (citables ; hors domaine de C1b ; non resolues) — "
                       "JAMAIS un ratio unique : il melangerait teste et non teste"),
    }


def delta_banc(c: Corpus, graine: int = 0) -> dict[str, Any]:
    """**Secondaire du §4.4, RETROGRADE** : ``Δ_banc = E_null − E_reel``, **par famille
    de valeurs**.

    La nulle est un **derangement sur classes de valeurs distinctes**, jamais une
    permutation libre. Une famille dont ``I_f`` atteint la borne gelee est **declaree
    degeneree d'avance** : ``Δ_banc`` y vaut **`SANS OBJET`** (D23), **jamais `0`**, et
    ``Δ_banc ≤ 0`` n'y declenche **jamais** ``PUB-instrument``.

    Le document soumis a la nulle est le **manifeste** tant que ``REPORT.md`` est absent
    — declare, non substitue en silence.
    """
    support = "report" if c.report.strip() else "manifeste"
    occs = ([t for g in extraire_grappes(c.report) for t in g.tokens] if support == "report"
            else [f for l in c.manifeste for f in _formes(l.get("valeur", ""))])
    stats = statistiques_familles(occs) if occs else []
    e_reel = executer(c)["violations_totales"]
    par_famille: dict[str, Any] = {}
    for st in stats:
        if st.degeneree:
            par_famille[st.famille] = {
                "delta_banc": "SANS OBJET", "motif": st.motif, "E_null": "SANS OBJET"}
            continue
        classes = sorted({normaliser(o) for o in occs if famille_de(o) == st.famille})
        pi = derangement(classes, graine)
        nul = Corpus(c.report, [dict(l) for l in c.manifeste], c.riders, c.invalides,
                     c.relecture, c.data, c.racine)
        if support == "manifeste":
            for l in nul.manifeste:
                l["valeur"] = "|".join(
                    pi.get(normaliser(f), f) if famille_de(f) == st.famille else f
                    for f in _formes(l.get("valeur", ""))) or l.get("valeur", "")
        else:
            texte = c.report
            for src, dst in sorted(pi.items(), key=lambda kv: -len(kv[0])):
                texte = re.sub(r"(?<![\w.])" + re.escape(src) + r"(?![\w])", dst, texte)
            nul = Corpus(texte, c.manifeste, c.riders, c.invalides, c.relecture, c.data,
                         c.racine)
        e_null = executer(nul)["violations_totales"]
        par_famille[st.famille] = {
            "delta_banc": e_null - e_reel, "E_null": e_null, "E_reel": e_reel,
            "cardinal_derangements": st.derangements, "I_f": round(st.inertie, 6)}
    # **Declaration d'interpretabilite.** Tant que ``REPORT.md`` est absent, le document
    # soumis a la nulle est le manifeste, et les clauses sensibles a une VALEUR
    # (``C1``, ``C1b``, ``C5``, ``C10``) n'ont **aucun cas reel** (B-5). Un
    # ``Delta_banc`` nul y mesure l'absence de prose, **pas la cecite du banc** : il ne
    # declenche pas ``PUB-instrument``, et le critere \u00a76.10 n'est **evaluable qu'a la
    # redaction**. Declare, jamais compense en silence.
    interpretable = support == "report"
    return {"support": support, "E_reel": e_reel, "graine": graine,
            "par_famille": par_famille,
            "interpretable": interpretable,
            "clauses_sensibles_a_la_valeur": ["C1", "C1b", "C5", "C10", "C12"],
            "lecture": ("Delta_banc lisible" if interpretable else
                        "NON INTERPRETABLE tant que REPORT.md est absent : les clauses "
                        "sensibles a une valeur (C1, C1b, C5, C10, C12) sont a 0 cas "
                        "reel ; le critere \u00a76.10 n'est evaluable qu'a la redaction")}


def _slug(nom: str) -> str:
    return nom.replace("-", "_")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Banc de coherence documentaire (EXP-2026-09-04, §10). CPU pur.")
    p.add_argument("--racine", default=".", help="racine du depot")
    p.add_argument("--report", default="REPORT.md")
    p.add_argument("--fixtures", default="tests/fixtures/doc_bench")
    p.add_argument("--out", default="report/doc_bench.json")
    p.add_argument("--campagne", action="store_true",
                   help="campagne de mutation dirigee : calcule E")
    p.add_argument("--sommes", action="store_true",
                   help="recalcule report/SHA256SUMS (report/data/** + l'artefact "
                        "publie) SANS jamais ecrire l'artefact : c'est ainsi que la "
                        "circularite du hash est brisee")
    p.add_argument("--force-pass", action="store_true",
                   help="C-mut : mutation du lecteur de classes ; tests/ doit ECHOUER")
    a = p.parse_args(argv)

    racine = Path(a.racine).resolve()

    if a.sommes:
        # Commande DISTINCTE : elle LIT l'artefact, elle ne l'ecrit jamais.
        lignes = calculer_sommes(racine)
        (racine / SOMMES).write_text("\n".join(lignes) + "\n", encoding="utf-8")
        print(f"{SOMMES} : {len(lignes)} lignes (dont {ARTEFACT_PUBLIE})")
        return 0

    corpus = charger_corpus_depuis_disque(racine, racine / a.report)

    print("=== banc de coherence documentaire — EXP-2026-09-04-rapport-sous-banc §10 ===")
    print(f"racine        : {racine}")
    print(f"REPORT.md     : {'present' if corpus.report.strip() else 'ABSENT (aucune prose)'}")
    print(f"manifeste     : {len(corpus.manifeste)} lignes")
    print(f"riders        : {len(corpus.riders)} lignes")
    print(f"invalides     : {len(corpus.invalides)} lignes")
    print(f"relecture     : {len(corpus.relecture)} lignes")
    print(f"report/data/  : {len(corpus.data)} fichiers")
    print(f"clauses       : {len(ORDRE_CLAUSES)} (30 du §10 + C15c + C16-ext)")
    if a.force_pass:
        print("!! --force-pass ACTIF : mutation du lecteur de classes (C-mut) !!")

    rapport = executer(corpus, a.force_pass)
    print("\n--- clauses ---")
    for cl in rapport["clauses"]:
        etat = "PASS" if cl["PASS"] else f"FAIL({len(cl['violations'])})"
        print(f"  {cl['clause']:<10} {etat:<9} cas_reels={cl['cas_reels']}")
        for v in cl["violations"][:5]:
            print(f"      [{v['classe']}] {v['detail']}")

    print("\n--- fermeture de la table de normalisation ---")
    # (0-247) l'affichage portait le PLANCHER seul, quand c'est `C0b` qui applique le
    # domaine derive : **la ligne que le lecteur voyait n'etait pas celle qui garantit
    # la propriete.**
    domaine_corpus = _domaine_du_corpus(corpus)
    ferm = verifier_fermeture_table(domaine_corpus)
    print(f"  domaine soumis : {len(SONDES_FERMETURE)} sondes plancher "
          f"+ {len(domaine_corpus)} valeurs derivees du corpus")
    print(f"  etage 1 (demonstration sur la table) : "
          f"{len(demontrer_reecritures())} violation(s)")
    print(f"  violations de fermeture : {len(ferm)}")
    for f in ferm:
        print(f"    {f}")

    print("\n--- N-manif : familles, cardinaux de derangement, inerties ---")
    occs = [t for g in extraire_grappes(corpus.report) for t in g.tokens]
    stats = statistiques_familles(occs) if occs else []
    for st in stats:
        print(f"  {st.famille:<16} n_f={st.n_f:<4} K_f={st.K_f:<4} "
              f"!K_f={st.derangements:<8} I_f={st.inertie:.4f} "
              f"{'DEGENEREE' if st.degeneree else 'non degeneree'} — {st.motif}")
    if not stats:
        print("  aucune occurrence numerique : SANS OBJET (aucune prose)")

    occs_man = [f for l in corpus.manifeste for f in _formes(l.get("valeur", ""))]
    stats_man = statistiques_familles(occs_man) if occs_man else []
    print("\n--- N-manif : familles du MANIFESTE (publiees d'avance) ---")
    for st in stats_man:
        print(f"  {st.famille:<16} n_f={st.n_f:<4} K_f={st.K_f:<4} "
              f"!K_f={st.derangements:<8} I_f={st.inertie:.4f} "
              f"{'DEGENEREE' if st.degeneree else 'non degeneree'} \u2014 {st.motif}")
    degen = [st.famille for st in stats_man if st.degeneree]
    print(f"  familles degenerees (Delta_banc = SANS OBJET, D23, jamais 0) : "
          f"{degen if degen else 'aucune'}")

    rapport["fermeture_table"] = ferm
    rapport["N_manif"] = {"familles_report": [s.to_json() for s in stats],
                          "familles_manifeste": [s.to_json() for s in stats_man],
                          "familles_degenerees": degen,
                          "borne_inertie": BORNE_INERTIE, "graine": 0,
                          "generateur": "random.Random (Mersenne Twister, stdlib)"}

    # **B-5** — une clause qui n'a jamais vu un cas reel se **declare**, elle ne se
    # confond pas avec une clause satisfaite.
    non_exercees = [cl["clause"] for cl in rapport["clauses"] if cl["cas_reels"] == 0]
    rapport["clauses_non_exercees_sur_le_corpus"] = non_exercees
    rapport["consignes_de_redaction"] = list(CONSIGNES_DE_REDACTION)
    rapport["reserve_6_10"] = RESERVE_6_10
    rapport["citabilite_du_manifeste"] = citabilite(corpus)
    cit = rapport["citabilite_du_manifeste"]
    print("\n--- citabilite du manifeste (triplet, jamais un ratio) ---")
    print(f"  citables={cit['citables']} ; hors domaine de C1b="
          f"{cit['hors_domaine_de_C1b']} ({cit['ids_hors_domaine']}) ; "
          f"non resolues={cit['non_resolues']} ({cit['ids_non_resolues']}) ; "
          f"total={cit['total']}")

    print("\n--- consignes de redaction (gravees dans l'artefact) ---")
    for i, cons in enumerate(CONSIGNES_DE_REDACTION, 1):
        print(f"  {i}. {cons}")
    print(f"\n--- reserve §6.10 ---\n  {RESERVE_6_10}")

    print("\n--- clauses SANS AUCUN CAS REEL dans ce corpus (B-5) ---")
    print(f"  {non_exercees if non_exercees else 'aucune'}")

    # **B-3** — le secondaire du §4.4 est produit, ou declare non exerce, nommement.
    db = delta_banc(corpus)
    rapport["delta_banc"] = db
    print(f"\n--- Delta_banc (secondaire, RETROGRADE) — support : {db['support']} ---")
    print(f"  {db['lecture']}")
    for fam, val in sorted(db["par_famille"].items()):
        print(f"  {fam:<16} Delta_banc={val['delta_banc']} (E_null={val['E_null']}, "
              f"E_reel={db['E_reel']})")

    # **B-1** — l'artefact publie porte TOUJOURS la campagne : `E = 0` doit y figurer,
    # sans quoi il affirme un etat que rien ne date.
    camp = campagne(racine / a.fixtures, a.force_pass)
    rapport["campagne"] = camp
    if a.campagne:
        print("\n--- campagne de mutation dirigee (D14-S) ---")
        for nom, e in camp["detail"].items():
            print(f"  {nom:<10} passant={e['cas_passant']!s:<6} "
                  f"mordant={e['mutant_mordant']!s:<6} ({e['fixture']})")
    print(f"\n  couverture par classe : "
          f"{camp['couverture_par_classe']['clauses_exercees']}"
          f"/{camp['couverture_par_classe']['clauses_declarees']} clauses exercees")
    print(f"  E = {camp['E']}")
    if camp["clauses_sans_les_deux"]:
        print(f"  clauses sans les deux : {camp['clauses_sans_les_deux']}")

    print("\n--- verification MACHINE de report/SHA256SUMS (\u00a75.7) ---")
    vs = verifier_sommes(racine)
    rapport["verification_sommes"] = {"violations": vs, "concordant": not vs}
    print(f"  {'100 % concordant' if not vs else str(len(vs)) + ' violation(s)'}")
    for v in vs:
        print(f"    {v}")

    # **VERROU (a)** \u2014 le mode mutant **n'ecrit JAMAIS l'artefact publie**. Son
    # execution avait remplace `report/doc_bench.json` par un rapport `E = 32`,
    # 32/32 `PASS` : *le controle de sante du banc detruisait le livrable qu'il
    # certifie* (0-254).
    sortie = racine / a.out
    if a.force_pass:
        sortie = sortie.with_suffix(".mutant.json")
        print(f"\n!! --force-pass : l'artefact publie {a.out} n'est PAS ecrit ; "
              f"le rapport mutant va dans {sortie.name} !!")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nrapport ecrit : {sortie}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
