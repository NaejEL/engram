# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests du banc de cohérence documentaire (`tools/doc_bench.py`).

CPU pur, sans torch, sans transformers, sans téléchargement HF.

Trois familles :

1. **Mutation dirigée (D14-S, §5.2)** — par clause, un cas **passant** et un cas
   **échouant mordant**. `E = 0` est la condition de `PUB-net` (§4.2).
2. **Table de normalisation** — les sept règles, et la **propriété de fermeture**
   vérifiée mécaniquement (ce qui protège de 0-214).
3. **`N-manif` (D31)** — sous-factorielle par récurrence entière, inertie `I_f`,
   dérangement (jamais une permutation libre), filtre de licéité.

Plus le contrôle **`C-mut`** (D35 (i)) : *banc forcé à `PASS` ⇒ ces tests doivent
**ÉCHOUER***. C'est la seule méthode qui ait trouvé le défaut d'Étape A.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from tools import doc_bench as db  # noqa: E402

FIXTURES = RACINE / "tests" / "fixtures" / "doc_bench"


# =====================================================================================
# 1. Mutation dirigée — un cas passant ET un mutant mordant par clause
# =====================================================================================

def _fixture(nom: str) -> dict:
    p = FIXTURES / f"{nom.replace('-', '_')}.json"
    assert p.exists(), f"fixture manquante pour {nom} : {p}"
    return json.loads(p.read_text(encoding="utf-8"))


def _resultat(cas: dict, nom: str, force_pass: bool = False) -> dict:
    rapport = db.executer(db.charger_corpus_depuis_cas(cas), force_pass)
    return next(c for c in rapport["clauses"] if c["clause"] == nom)


def test_denominateur_de_couverture_nomme():
    """**Dénominateur nommé (arbitrage PI)** : 30 clauses documentaires du §10 (GELÉ),
    `C-mut` en **contrôle du banc** (§5.3, hors document), plus deux **ajouts** qui
    n'amendent rien — `C16-ext` (`lab-math`) et `C15c` (mutabilité de la table)."""
    assert len(db.ORDRE_CLAUSES) == 32
    assert "C16-ext" in db.ORDRE_CLAUSES
    assert "C15c" in db.ORDRE_CLAUSES
    assert "C-mut" not in db.ORDRE_CLAUSES, "C-mut est un controle, pas une clause"
    for attendue in ("C0", "C0b", "C1", "C1b", "C2", "C2b", "C2c", "C3", "C3b", "C4",
                     "C5", "C5b", "C5c", "C6", "C7", "C8", "C9", "C10", "C11", "C12",
                     "C13", "C14", "C15", "C15b", "C16", "C17", "C18", "C19",
                     "C-obs", "C-relect"):
        assert attendue in db.ORDRE_CLAUSES, attendue


@pytest.mark.parametrize("nom", db.ORDRE_CLAUSES)
def test_cas_passant(nom: str):
    res = _resultat(_fixture(nom)["pass"], nom)
    assert res["PASS"], f"{nom} : le cas passant echoue — {res['violations']}"


@pytest.mark.parametrize("nom", db.ORDRE_CLAUSES)
def test_mutant_mordant(nom: str):
    """Le mutant doit **mordre sur sa propre clause**, pas sur une voisine."""
    res = _resultat(_fixture(nom)["mutant"], nom)
    assert not res["PASS"], f"{nom} : le mutant ne mord pas"
    assert res["violations"], f"{nom} : mutant sans violation detaillee"


def test_E_vaut_zero():
    """`E ≠ 0` ⇒ classe `PUB-instrument` : *le banc est aveugle, aucun verdict de
    publiabilité n'est lisible* (§4.2)."""
    camp = db.campagne(FIXTURES)
    assert camp["E"] == 0, camp["clauses_sans_les_deux"]
    assert camp["couverture_par_classe"]["clauses_exercees"] == len(db.ORDRE_CLAUSES)


def test_couverture_publiee_par_classe_sans_taux():
    """(xxxvii) — le rapport de campagne publie un **compte de clauses**, jamais un taux."""
    camp = db.campagne(FIXTURES)
    couv = camp["couverture_par_classe"]
    assert set(couv) == {"clauses_exercees", "clauses_declarees"}
    assert all(isinstance(v, int) for v in couv.values())


# =====================================================================================
# 2. C-mut — mutation du lecteur de classes (D35 (i))
# =====================================================================================

def test_c_mut_banc_force_a_pass_tue_tous_les_mutants():
    """`doc_bench` forcé à rendre toujours `PASS` ⇒ **plus aucun mutant ne mord** ⇒ la
    famille `test_mutant_mordant` échoue en bloc. Le contrôle vérifie que la suite est
    bien *capable* d'échouer : une suite qui survivrait à cette mutation ne testerait
    rien."""
    camp = db.campagne(FIXTURES, force_pass=True)
    assert camp["E"] == len(db.ORDRE_CLAUSES), (
        "le banc force a PASS laisse encore mordre des mutants : "
        "la mutation du lecteur de classes n'est pas exercee")
    assert sorted(camp["clauses_sans_les_deux"]) == sorted(db.ORDRE_CLAUSES)


@pytest.mark.parametrize("nom", db.ORDRE_CLAUSES)
def test_c_mut_chaque_assertion_de_mordant_tomberait(nom: str):
    """Chaque assertion de `test_mutant_mordant` **tomberait** sous `--force-pass`."""
    res = _resultat(_fixture(nom)["mutant"], nom, force_pass=True)
    assert res["PASS"], f"{nom} : --force-pass ne neutralise pas la clause"


# =====================================================================================
# 3. Table de normalisation — sept règles, et la fermeture
# =====================================================================================

def test_table_a_exactement_sept_regles():
    assert [r.nom for r in db.TABLE_NORMALISATION] == \
        ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]


def test_chaque_regle_declare_ses_reecritures():
    """C'est la colonne `reecritures` qui rend la fermeture **démontrable** : elle est
    ce qu'inspecte l'étage 1, sans énumérer aucune valeur."""
    for regle in db.TABLE_NORMALISATION:
        assert hasattr(regle, "reecritures")
        if regle.nom != "T7":
            assert regle.reecritures, regle.nom


@pytest.mark.parametrize("brut,attendu", [
    ("−0.0352", "-0.0352"),                       # T1
    ("1 401", "1401"),                            # T2
    ("+/-0.378", "±0.378"),                       # T3
    ("2,437", "2.437"),                                # T4
    ("5.12E−06", "5.12e-6"),                      # T5
    ("5.12×10^−6", "5.12e-6"),               # T5
    ("5.12e⁻⁰⁶", "5.12e-6"),            # T5
    ("82,3 %", "82.3%"),                          # T4 + T6
    ("+0.1513", "+0.1513"),                            # T7 : le + est CONSERVE
])
def test_les_sept_regles(brut: str, attendu: str):
    assert db.normaliser(brut) == attendu


def test_fermeture_verifiee_mecaniquement():
    """Chaque règle est une bijection au niveau caractère **préservant la valeur réelle**
    ⇒ `normaliser(a) == normaliser(b)` implique `val(a) == val(b)`."""
    assert db.verifier_fermeture_table() == []
    assert db.demontrer_reecritures() == []


# =====================================================================================
# 3bis. Les mutants du VERIFIER (verdict CHANGES_REQUESTED du 2026-09-05)
# =====================================================================================
#
# *« Après correction : le banc doit mordre sur les mutants DU VERIFIER, pas seulement
# sur les tiens. »* Les cinq trous A-1 … A-5 deviennent ici des cas de test.

def _table_mutee(fn, reecritures):
    """Ajoute une 8ᵉ règle à la table gelée, le temps d'un test."""
    orig = db.TABLE_NORMALISATION
    db.TABLE_NORMALISATION = orig + (db.Regle("T8", "mutant", fn, reecritures),)
    return orig


def test_A1_huitieme_regle_declaree_est_refusee_sans_aucune_sonde():
    """**A-1** — *« ce n'est pas une preuve sur le domaine de la table : c'est un sondage
    sur 21 sondes codées en dur »*. L'étage 1 **démontre** : une règle déclarant
    `0.0156 → 0.031` — le mutant nommé par `lab-math` — est refusée **sans qu'aucune
    valeur ne soit énumérée**."""
    orig = _table_mutee(lambda t: t.replace("0.0156", "0.031"), (("0.0156", "0.031"),))
    try:
        viol = db.demontrer_reecritures()
        assert viol and "NON preservante" in viol[0]
        assert db.verifier_fermeture_table() != []
    finally:
        db.TABLE_NORMALISATION = orig


def test_A1_huitieme_regle_MUETTE_est_rattrapee_par_le_domaine():
    """Variante sournoise : la règle **ne déclare pas** sa réécriture. L'étage 2 la
    rattrape parce que `0.031` et `0.0156` sont au **plancher gelé du domaine** — le
    couple bilatéral / unilatéral est un désaccord décisionnel **réel** du corpus."""
    orig = _table_mutee(lambda t: t.replace("0.0156", "0.031"), ())
    try:
        assert db.demontrer_reecritures() == []
        viol = db.verifier_fermeture_table()
        assert any("absorbe un desaccord" in v for v in viol), viol
    finally:
        db.TABLE_NORMALISATION = orig


def test_A1_le_domaine_est_derive_du_corpus_et_non_code_en_dur():
    corpus = db.charger_corpus_depuis_cas(_fixture("C0b")["pass"])
    domaine = db._domaine_du_corpus(corpus)
    assert "0.031" in domaine and "0.0156" in domaine


def test_A2_l_espace_ascii_n_est_pas_un_separateur_de_milliers():
    """**A-2** — `SEPARATEURS_MILLIERS` avalait `U+0020` : `normaliser('3 5') == '35'`,
    et la propriété de fermeture était **fausse sur le domaine réel**."""
    assert " " not in db.ESPACES_INSECABLES
    assert db.normaliser("3 5") != db.normaliser("35")
    assert db.normaliser("3 5") == "3 5"


@pytest.mark.parametrize("forme", [
    "20,000,000",              # affichage de la source, verbatim
    "20\u202f000\u202f000",  # affichage francais, espace fine insecable
    "20\u00a0000\u00a0000",  # affichage francais, espace insecable
])
def test_A2_le_separateur_anglais_de_milliers_est_normalise(forme: str):
    """`T4` mangeait le s\u00e9parateur anglais : `normaliser('20,000,000') = '20.000.000'`
    \u21d2 **l'extrait verbatim de l'archive Qwen n'\u00e9tait pas citable**. Les trois
    affichages licites convergent."""
    assert db.normaliser(forme) == "20000000"


def test_A2_deux_formes_d_une_cellule_doivent_normaliser_a_l_identique():
    """Deux formes d'une **m\u00eame cellule** sont deux **affichages** d'une m\u00eame valeur ;
    deux quantit\u00e9s distinctes prennent **deux `id`** et un `paire_id`."""
    cas = json.loads(json.dumps(_fixture("C0b")["pass"]))

    def _mettre(v):
        for l in cas["manifeste"]:
            if l["id"] == "M-BILAT":
                l["valeur"] = v
        return _resultat(cas, "C0b")["PASS"]

    assert _mettre("20,000,000|20\u202f000\u202f000")
    assert not _mettre("20,000,000|20 000 000")   # espace ASCII : PAS le meme token


def test_A2_un_groupe_unique_de_virgule_reste_une_decimale():
    """Ambiguïté **déclarée** : `2,437` (un seul groupe) est décimal ; il faut **deux**
    groupes pour lire un séparateur de milliers."""
    assert db.normaliser("2,437") == "2.437"
    assert db.normaliser("82,3 %") == "82.3%"


@pytest.mark.parametrize("texte", ["le gain E+1.353", "la valeur x2.437",
                                   "le ratio p0.68"])
def test_A3_changer_la_lettre_ne_sort_plus_du_circuit(texte: str):
    """**A-3** — *« changer la lettre suffisait pour qu'un décimal non signé sorte
    entièrement du circuit de provenance »*."""
    c = db.charger_corpus_depuis_cas({"report": texte})
    assert db.CLAUSES["C0"](c)[0], f"C0 muet sur {texte!r}"
    assert db.CLAUSES["C1"](c)[0], f"C1 muet sur {texte!r}"


@pytest.mark.parametrize("texte", ["sur GPT-2 et SmolLM2-360M et Qwen2.5-1.5B",
                                   "la decision D14 du 2026-09-04",
                                   "le defaut 0-197 au §3"])
def test_A3_la_liste_enumeree_laisse_passer_les_vrais_identifiants(texte: str):
    c = db.charger_corpus_depuis_cas({"report": texte})
    assert not db.CLAUSES["C0"](c)[0], texte
    assert not db.CLAUSES["C1"](c)[0], texte


@pytest.mark.parametrize("phrase,entree", [
    ("le modèle retrouve l’unité", "xiii"),
    ("l’industrie confirme notre lecture", "xxxi"),
    ("il n’y a pas d’effet", "D28"),
])
def test_A4_l_apostrophe_typographique_n_eteint_plus_l_interdit(phrase: str,
                                                                entree: str):
    """**A-4** — *« un seul copier-coller d'éditeur dans `REPORT.md` éteint
    l'interdit-titre sans que `E` bouge »*. La substitution est **longueur-préservante**,
    donc les positions des grappes restent intactes."""
    c = db.charger_corpus_depuis_cas({"report": phrase})
    viol = db.CLAUSES["C3"](c)[0]
    assert any(entree in v.detail for v in viol), [v.detail for v in viol]


def test_A4_la_normalisation_typographique_preserve_les_positions():
    avant = "l’unité vaut +2.437 ⟦M⟧"
    apres = db.normaliser_typographie(avant)
    assert len(apres) == len(avant)
    assert apres.index("+2.437") == avant.index("+2.437")


def test_A5_script_inexistant_ne_satisfait_plus_c16_ext():
    """**A-5** — `re.search("script=")` était satisfaite par `script=inexistant.py`. *Une
    réponse à la chaîne fausse-mais-cohérente ne peut pas être déclarative.*"""
    res = _resultat(_fixture("C16-ext")["mutant"], "C16-ext")
    assert not res["PASS"]
    assert any(v["classe"] == "SOURCE-ILLICITE" for v in res["violations"])


def test_A5_le_script_est_REJOUE_et_sa_sortie_COMPAREE():
    """La forme exécutable est **ré-exécutée** et sa sortie comparée verbatim
    post-normalisation. Une valeur fausse-mais-cohérente est attrapée **ici**, et
    nulle part ailleurs."""
    cas = json.loads(json.dumps(_fixture("C16-ext")["pass"]))
    assert _resultat(cas, "C16-ext")["PASS"]
    for l in cas["manifeste"]:
        if l["id"] == "M-D":
            l["valeur"] = "4.789"      # chaine coherente, valeur FAUSSE
    res = _resultat(cas, "C16-ext")
    assert not res["PASS"]
    assert any(v["classe"] == "DESACCORD" for v in res["violations"])


def test_A5_perimetre_etendu_declare():
    """Périmètre étendu aux lignes `etiquette = re-dérivé` : sans cela la clause n'a
    **aucun cas réel** dans ce corpus (0 ligne `décisionnel`)."""
    src = (RACINE / "tools" / "doc_bench.py").read_text(encoding="utf-8")
    assert 'etiquette != "re-dérivé"' in src


def test_hors_table_la_precision_et_la_lateralite():
    """HORS table, explicitement : zéros terminaux et précision (domaine de `C10`) ; la
    **latéralité** (`0.031` / `0.0156` : deux quantités, deux `id`)."""
    assert db.normaliser("−0.0023") != db.normaliser("−0.00")
    assert db.normaliser("0.031") != db.normaliser("0.0156")
    assert db.normaliser("2.4370") != db.normaliser("2.437")


def test_t4_ne_casse_pas_une_enumeration_car_elle_s_applique_au_token():
    """T4 s'applique **au TOKEN**, jamais à la ligne : « 3, 5 » reste deux nombres."""
    grappes = db.extraire_grappes("La liste porte 3, 5 items.")
    assert [t for g in grappes for t in g.tokens] == ["3", "5"]


def test_c0b_mutant_absorbe_un_desaccord_decisionnel():
    """Mutant `C0b` gravé : l'équivalence `0.031 ≡ 0.0156` — un désaccord décisionnel
    **réel** du corpus (bilatéral / unilatéral) — doit faire **FAIL**."""
    res = _resultat(_fixture("C0b")["mutant"], "C0b")
    assert not res["PASS"]


# =====================================================================================
# 4. `N-manif` (D31) — statistique complète, gravée AVANT le gel
# =====================================================================================

@pytest.mark.parametrize("k,attendu", [(0, 1), (1, 0), (2, 1), (3, 2), (4, 9),
                                       (5, 44), (6, 265), (7, 1854)])
def test_sous_factorielle_par_recurrence_entiere(k: int, attendu: int):
    """`!K = (K−1)(!(K−1) + !(K−2))`, `!0 = 1`, `!1 = 0`. ***Jamais `K!/e` arrondi.***"""
    assert db.sous_factorielle(k) == attendu


def test_sous_factorielle_n_est_pas_k_factoriel_sur_e_arrondi():
    """***Jamais `K!/e` arrondi*** : la formule d'arrondi **se trompe** en `K = 0`
    (`round(1/e) = 0` contre `!0 = 1`) et n'est pas une définition entière."""
    import math
    assert db.sous_factorielle(0) == 1
    assert round(math.factorial(0) / math.e) == 0
    assert isinstance(db.sous_factorielle(30), int)
    assert db.sous_factorielle(30) == 29 * (db.sous_factorielle(29)
                                            + db.sous_factorielle(28))


def test_famille_n_sur_n_est_declaree_degeneree_par_inertie():
    """Vérification de `lab-math`, reproduite : famille « n/n » = `12/12` ×4, `24/24`,
    `3/3`, `30/30` ⇒ `n_f = 7`, `Σ m_v(m_v−1) = 12`, `I_f = 12/42 + 1/7 = 0.4286 > 0.3`
    ⇒ **DÉGÉNÉRÉE**."""
    occ = ["12/12", "12/12", "12/12", "12/12", "24/24", "3/3", "30/30"]
    (st,) = db.statistiques_familles(occ)
    assert st.n_f == 7
    assert st.K_f == 4
    assert st.derangements == 9          # !4, jamais 24/e arrondi
    assert st.inertie == pytest.approx(12 / 42 + 1 / 7, abs=1e-9)
    assert st.inertie == pytest.approx(0.428571, abs=1e-6)
    assert st.degeneree is True
    assert "0.3" in st.motif


def test_borne_gelee_comparaison_stricte_au_sens_superieur_ou_egal():
    """Le cas frontière `I_f == 0.3` se déclare **dégénéré** (conservateur)."""
    assert db.BORNE_INERTIE == 0.3
    src = (RACINE / "tools" / "doc_bench.py").read_text(encoding="utf-8")
    assert "inertie >= BORNE_INERTIE" in src


def test_famille_degeneree_par_cardinal():
    """`K_f ≤ 1 ⇒ !K_f = 0` : famille dégénérée **par cardinal**."""
    (st,) = db.statistiques_familles(["12/12", "12/12", "12/12"])
    assert st.K_f == 1
    assert st.derangements == 0
    assert st.degeneree is True
    assert "cardinal" in st.motif


def test_delta_banc_est_sans_objet_sur_une_famille_degeneree():
    """`Δ_banc = SANS OBJET` (D23) sur une famille dégénérée, **jamais `0`** — et
    `Δ_banc ≤ 0` n'y déclenche **jamais** `PUB-instrument` : le document nul ne dérange
    aucune de ses classes."""
    corpus = db.charger_corpus_depuis_cas(
        {"report": "Les comptes valent 12/12 et 24/24 et 3/3 et 30/30 et 12/12 et "
                   "12/12 et 12/12."})
    nul, stats = db.document_nul(corpus, graine=0)
    assert all(s.degeneree for s in stats)
    assert nul.report == corpus.report          # aucune occurrence derangee


def test_derangement_jamais_une_permutation_libre():
    classes = ["0.10", "0.20", "0.30", "0.40", "0.50"]
    pi = db.derangement(classes, graine=0)
    assert set(pi) == set(classes)
    assert set(pi.values()) == set(classes)
    assert all(pi[v] != v for v in classes), "point fixe : ce n'est pas un derangement"


def test_derangement_deterministe_a_la_graine_zero():
    classes = ["1.0", "2.0", "3.0", "4.0"]
    assert db.derangement(classes, 0) == db.derangement(classes, 0)


def test_filtre_de_liceite_le_format_est_preserve():
    """`π(v)` reste dans la **même classe de format** (un `n/n` reste `n/n`, `num ≤ den`)
    — et le cardinal se publie **APRÈS** filtre."""
    assert db.classe_de_format("12/12") == "fraction-egale"
    assert db.classe_de_format("28/30") == "fraction"
    assert db.classe_de_format("12/12") != db.classe_de_format("28/30")
    assert db.classe_de_format("+5.12e-06") == "exponentiel"
    assert db.classe_de_format("82.3%") == "pourcentage"
    assert db.classe_de_format("-0.0352") == "decimal"
    assert db.classe_de_format("24") == "entier"
    occ = ["12/12", "24/24", "28/30", "12/37", "0.10", "0.20", "0.30"]
    familles = {s.famille for s in db.statistiques_familles(occ)}
    assert familles == {"fraction-egale", "fraction", "decimal"}


def test_generateur_de_la_nulle_est_nomme():
    corpus = db.charger_corpus_depuis_cas({"report": "x"})
    rapport = db.executer(corpus)
    assert rapport is not None
    src = (RACINE / "tools" / "doc_bench.py").read_text(encoding="utf-8")
    assert "random.Random" in src and "Mersenne Twister" in src


# =====================================================================================
# 5. Propriétés structurelles du banc
# =====================================================================================

def test_le_banc_est_cpu_pur():
    """CPU pur : **sans torch ni HF**, aucun forward, `M` jamais instanciée."""
    src = (RACINE / "tools" / "doc_bench.py").read_text(encoding="utf-8")
    for interdit in ("import torch", "import transformers", "import numpy",
                     "from torch", "from transformers", "AutoModel"):
        assert interdit not in src, interdit


def test_spdx_present():
    src = (RACINE / "tools" / "doc_bench.py").read_text(encoding="utf-8")
    assert src.startswith("# SPDX-License-Identifier: AGPL-3.0-or-later")


def test_partition_des_statuts_est_exhaustive():
    assert db.STATUTS_LICITES == {
        "décisionnel", "descriptif", "rectifié", "SANS OBJET",
        "INVALIDE-SOURCE", "HISTORIQUE-PÉRIMÉ"}


def test_etiquette_recopie_d_une_analyse_est_interdite():
    assert db.ETIQUETTE_INTERDITE == "recopié d'une analyse"
    assert db.ETIQUETTE_INTERDITE not in db.ETIQUETTES_LICITES


def test_les_deux_etiquettes_ajoutees_sont_declarees():
    """0-222 — §3.1 n'offrait aucune valeur licite pour une ligne `descriptif`
    rétrogradée sous M-4/P3 (43 lignes sur 49 la portaient vide). Valeurs **proposées et
    déclarées**, jamais inventées en silence."""
    assert "re-lu du journal, non re-mesuré" in db.ETIQUETTES_LICITES
    assert "externe-archivé-dégradé" in db.ETIQUETTES_LICITES


@pytest.mark.parametrize("phrase", [
    "la table est figée à l'inférence",
    "la table serait figée après entraînement",
    "elle est entraînée puis figée",
    "c'est une table en lecture seule",
])
def test_c15c_la_mutabilite_affirmee_mord(phrase: str):
    """Formulation **INTERDITE** : la source ne se prononce pas."""
    cas = json.loads(json.dumps(_fixture("C15c")["pass"]))
    cas["report"] += "\n\nPar ailleurs, " + phrase + ".\n"
    assert not _resultat(cas, "C15c")["PASS"]


def test_c15c_exige_la_cause_de_l_ouverture():
    """*« … et elle reste ouverte parce que RIEN N'EST DIT, non parce que quelque chose
    serait établi. »* — la **cause** fait partie de la clause : une ouverture attribuée à
    un résultat, et non au silence de la source, est la même faute retournée."""
    cas = json.loads(json.dumps(_fixture("C15c")["pass"]))
    cas["report"] = cas["report"].replace(
        "reste ouverte parce que RIEN N'EST DIT, non parce que quelque chose serait "
        "établi", "reste ouverte au vu de nos mesures")
    res = _resultat(cas, "C15c")
    assert not res["PASS"]
    assert any("cause" in v["detail"] for v in res["violations"])


def test_la_section_6_ne_sort_pas_avec_l_archive_degradee():
    """`C15b` satisfaite (0-231 fermé) : chaque chiffre de la section 6 résout vers
    `report/data/`, artefact **ouvert et hashé**."""
    res = _resultat(_fixture("C15b")["pass"], "C15b")
    assert res["PASS"]


def test_c1b_detecte_la_transcription_pas_la_propagation():
    """**Correction `lab-math`, portée aux tests** : `C1b` détecte les **fautes de
    transcription**, **PAS** le mode 0-139 dans sa forme propagée — texte, manifeste et
    source d'accord sur une valeur fausse. ***Cohérence ≠ correction. Le banc ne vérifie
    pas les chiffres : il vérifie leur RÉSOLUTION.***"""
    cas = _fixture("C1b")["pass"]
    faux = json.loads(json.dumps(cas))
    # une valeur fausse, propagee a l'identique dans le texte ET le manifeste
    faux["report"] = faux["report"].replace("+2.437", "+9.999")
    for l in faux["manifeste"]:
        if l["id"] == "M-EXC":
            l["valeur"] = "+9.999"
    res = _resultat(faux, "C1b")
    assert res["PASS"], (
        "C1b passerait sur une chaine fausse-mais-coherente : c'est sa portee reelle, "
        "et c'est ce que C16-ext couvre")


def test_c16_ext_attrape_ce_que_c1b_ne_voit_pas():
    """L'extension `C16-ext` : une ligne `décisionnel` **sans script reproducteur** est
    `SOURCE-ILLICITE`. *Le précédent `cos_type ≈ −0.002` n'aurait été attrapé que par
    cette voie.*"""
    res = _resultat(_fixture("C16-ext")["mutant"], "C16-ext")
    assert not res["PASS"]
    assert any(v["classe"] == "SOURCE-ILLICITE" for v in res["violations"])


def test_adresse_non_ouvrable_est_un_orphelin_pas_une_imprecision():
    """(xxxvi, N-5) *« Un chemin que personne ne peut ouvrir n'est pas une adresse,
    c'est un défaut. »*"""
    cas = json.loads(json.dumps(_fixture("C1")["pass"]))
    cas["data"] = {}
    res = _resultat(cas, "C1")
    assert not res["PASS"]
    assert any(v["classe"] == "ORPHELIN" and "non ouvrable" in v["detail"]
               for v in res["violations"])


@pytest.mark.parametrize("neff,licite", [
    ("24", True),
    ("[19,8 ; 27,6] (méthode de quantile : bootstrap de grappes)", True),
    ("NON ÉTABLI (run unique, variance non estimée — Q-02/Q-09)", True),
    ("SANS OBJET (dérivée : quadrature Q-M9)", True),
    ("SANS OBJET", False),
    ("NON ÉTABLI", False),
    ("environ 12", False),
])
def test_c2c_les_quatre_formes_licites_de_n_eff(neff: str, licite: bool):
    """`N_eff` : entier · intervalle (méthode publiée) · `NON ÉTABLI (cause)` ·
    `SANS OBJET (dérivée : ⟨nom⟩)`. Rien d'autre."""
    cas = json.loads(json.dumps(_fixture("C2c")["pass"]))
    for l in cas["manifeste"]:
        if l["id"] == "M-QUAD":
            l["N_eff"] = neff
    assert _resultat(cas, "C2c")["PASS"] is licite


# --- les livrables CSV de ce tour ----------------------------------------------------

def test_les_csv_livres_portent_les_colonnes_du_paragraphe_3_1():
    import csv
    with (RACINE / "report" / "manifest.csv").open(encoding="utf-8-sig",
                                                   newline="") as fh:
        entetes = next(csv.reader(fh))
    assert entetes == list(db.COLONNES_MANIFESTE)


def test_invalides_csv_est_produit_par_enumeration():
    """§6.3 — *la liste nominative est un livrable d'entrée, produite **par
    énumération**, jamais présumée.* L'énumération du journal en rend **trois** ; le
    protocole en annonce **cinq**. Le fichier porte ce que l'énumération a trouvé."""
    import csv
    with (RACINE / "report" / "invalides.csv").open(encoding="utf-8-sig",
                                                    newline="") as fh:
        lignes = list(csv.DictReader(fh))
    assert lignes, "aucun run invalide enumere"
    for l in lignes:
        assert l["run"] and l["date"] and l["ligne"] and l["clause"] and l["verdict"]


def test_riders_csv_porte_les_huit_riders():
    import csv
    with (RACINE / "report" / "riders.csv").open(encoding="utf-8-sig",
                                                 newline="") as fh:
        lignes = list(csv.DictReader(fh))
    assert [l["rider_id"] for l in lignes] == [f"RD-0{i}" for i in range(1, 9)]
    for l in lignes:
        assert l["texte"].strip()


def test_B1_le_banc_tourne_sur_les_ARTEFACTS_VIVANTS():
    """**B-1** — *« aucun des 156 tests n'exécute le banc sur les artefacts vivants ;
    c'est par là que la dérive est passée. »* Ce test **ouvre les CSV et
    `report/data/` réels** et fait tourner les 32 clauses dessus."""
    corpus = db.charger_corpus_depuis_disque(RACINE, RACINE / "REPORT.md")
    assert corpus.manifeste and corpus.riders and corpus.invalides
    assert corpus.data, "report/data/ vide : refus de permission ou archive absente"
    rapport = db.executer(corpus)
    echecs = {c["clause"]: c["violations"] for c in rapport["clauses"] if not c["PASS"]}
    assert not echecs, echecs


def test_B1_l_artefact_publie_porte_la_campagne_et_son_E():
    """`report/doc_bench.json` doit porter `campagne` — sans quoi il affirme un état que
    rien ne date."""
    art = json.loads((RACINE / "report" / "doc_bench.json").read_text(encoding="utf-8"))
    assert "campagne" in art, "l'artefact publie n'a pas de clef `campagne`"
    assert art["campagne"]["E"] == 0
    assert art["campagne"]["couverture_par_classe"]["clauses_declarees"] == \
        len(db.ORDRE_CLAUSES)
    assert "clauses_non_exercees_sur_le_corpus" in art
    assert "delta_banc" in art


def test_le_triplet_de_citabilite_du_manifeste():
    """**Non bloquant 3** \u2014 *le \u00ab 51/52 \u00bb melangeait teste et non teste.*

    Le triplet **(49 ; 2 ; 1)**, avec ses **trois definitions** et **jamais un ratio
    unique** \u2014 c'est le meme mode que le compte \u00ab 2 \u00bb.
    """
    corpus = db.charger_corpus_depuis_disque(RACINE, RACINE / "REPORT.md")
    cit = db.citabilite(corpus)
    assert cit["total"] == 52
    assert cit["citables"] == 49
    assert cit["hors_domaine_de_C1b"] == 2
    assert cit["non_resolues"] == 1
    assert sorted(cit["ids_hors_domaine"]) == ["N-09", "N-52"]
    assert cit["ids_non_resolues"] == ["N-23"]
    assert cit["citables"] + cit["hors_domaine_de_C1b"] + cit["non_resolues"] == \
        cit["total"]


def test_N23_reste_un_ORPHELIN_que_le_banc_continue_de_nommer():
    """**Non bloquant 4** \u2014 ``INVALIDE-SOURCE`` **n'achete aucune exemption**.

    *Ce n'est pas un orphelin rebaptise : c'est un orphelin que le banc continue de
    nommer orphelin.* Toute citation du signe d'incertitude du facteur 4.8 hors bloc
    ``AUTOPSIE`` fait **echouer `C1`**.
    """
    corpus = db.charger_corpus_depuis_disque(RACINE, RACINE / "REPORT.md")
    ligne = next(l for l in corpus.manifeste if l["id"] == "N-23")
    assert ligne["statut"] == "INVALIDE-SOURCE"
    assert not ligne["source_fichier"].strip()
    texte = "# T\n\n## 3. S\n\nL'incertitude vaut \u00b1 du facteur 4.8 \u27e6N-23\u27e7."
    c = db.Corpus(texte, corpus.manifeste, corpus.riders, corpus.invalides,
                  corpus.relecture, corpus.data, corpus.racine)
    viol = db.CLAUSES["C1"](c)[0]
    assert any(v.classe == "ORPHELIN" and "N-23" in v.detail for v in viol), viol


# =====================================================================================
# 3ter. Le compte des cas REELS — le mode que ce cycle poursuit, dans son artefact
# =====================================================================================

def test_PORTE_SANS_DONNEES_aucune_clause_ne_compte_sur_un_corpus_vide():
    """**La porte qui n'a besoin d'aucune donnee.** Quatrieme fois dans ce cycle que
    le defaut est trouve par elle.

    Le garde-fou precedent etait **un grep sur deux chaines litterales**
    (``max(n, 1)``, ``max(len(viol), 1)``) : il fermait **deux orthographes**, pas une
    **classe**. Quatre clauses comptaient encore \u2014 ``C3`` et ``C7`` comptaient les
    **entrees de leur table** au lieu des objets du corpus, ``C15`` et ``C15b``
    portaient un **plancher LITTERAL** (``return [], 1``) qu'aucun retrait de
    ``max(n, 1)`` ne pouvait voir.

    La forme correcte est celle-ci : **sur un corpus VIDE, toute clause a inspecte zero
    objet**. Une seule exemption, et elle est **motivee en clair** \u2014 ``C0b`` verifie la
    **table de normalisation elle-meme**, dont le plancher gele de sondes existe
    independamment de tout corpus : c'est precisement ce qui la rend demontrable sans
    echantillon (A-1).
    """
    vide = db.charger_corpus_depuis_cas({})
    comptes = {nom: db.CLAUSES[nom](vide)[1] for nom in db.ORDRE_CLAUSES}
    non_nuls = {n: v for n, v in comptes.items() if v != 0}
    assert set(non_nuls) == {"C0b"}, non_nuls
    assert non_nuls["C0b"] == len(db.SONDES_FERMETURE)


def test_les_VINGT_CINQ_clauses_sans_cas_reel_sont_publiees():
    """*C'est la liste qui doit gouverner la redaction.* La precedente en publiait
    **21** : il en manquait **quatre**, dont ``C3`` \u2014 le registre des formulations
    interdites \u2014 et ``C7`` \u2014 le plafond-en-succes. **Les deux clauses les plus
    decisionnelles pour la prose, donnees pour eprouvees alors qu'elles n'avaient rien
    vu.**"""
    corpus = db.charger_corpus_depuis_disque(RACINE, RACINE / "REPORT.md")
    rapport = db.executer(corpus)
    non = [c["clause"] for c in rapport["clauses"] if c["cas_reels"] == 0]
    assert non == ["C0", "C1", "C1b", "C2b", "C3", "C3b", "C5", "C5b", "C5c", "C6",
                   "C7", "C8", "C9", "C10", "C11", "C12", "C14", "C15", "C15b",
                   "C15c", "C17", "C18", "C19", "C-obs", "C-relect"], non
    assert len(non) == 25
    for ajoutee in ("C3", "C7", "C15", "C15b"):
        assert ajoutee in non, ajoutee
    assert "C18" in non          # ZERO ligne `decisionnel` au manifeste
    c16 = next(c for c in rapport["clauses"] if c["clause"] == "C16-ext")
    assert c16["cas_reels"] == 3


def test_cas_exerces_reste_un_plancher_declare_jamais_une_donnee():
    r = db.ResultatClause("C-test", True, [], 0)
    assert r.cas_reels == 0 and r.cas_exerces == 1
    assert "plancher" in db.ResultatClause.cas_exerces.__doc__.lower()


# =====================================================================================
# 3quater. Bac a sable du rejeu, residus bornes, consignes
# =====================================================================================

@pytest.mark.parametrize("expr,licite", [
    ("round(2.437/0.509, 3)", True),
    ("round(1 - 0.05**(1/100), 4)", True),
    ("f'+{(0.0551/0.0352-1)*100:.1f} %'", True),
    ("open('C:/tmp/x','w').write('h') and 4.788", False),
    ("open('C:/tmp/x','w')", False),
    ("__import__('os').system('echo')", False),
    ("(4.788).real", False),
    ("[c for c in '4.788']", False),
    ("eval('4.788')", False),
])
def test_le_rejeu_est_un_bac_a_sable_pas_un_charset(expr: str, licite: bool):
    """**Non bloquant 1 (0-244)** \u2014 l'ancien filtre laissait passer
    ``print(open('x','w').write('...') and 4.788)`` : **le fichier etait ecrit et la
    clause passait en silence**. *Un manifeste est une donnee, pas un vecteur
    d'execution.*"""
    assert (db._valider_expression(expr) == "") is licite, db._valider_expression(expr)


def test_la_limite_inherente_de_c16_ext_est_DECLAREE():
    """**Non bloquant 2** \u2014 une **constante litterale** passe : *la bonne valeur pour
    une mauvaise raison n'est pas detectable par re-execution.* A declarer, jamais a
    reparer."""
    assert db._valider_expression("4.788") == ""      # elle passe, et c'est le plafond
    doc = db.CLAUSES["C16-ext"].__doc__
    assert "constante litterale" in doc.lower().replace("\u00e9", "e")
    assert "manifeste DE CONFIANCE" in doc


def test_A1_residu_ii_une_reecriture_ne_peut_effacer_un_porteur_de_sens():
    """Le temoin ``_valeur_reference`` depouille lui aussi ``\u00b1`` et ``%`` : *il partage
    l'hypothese qu'il controle.* Le refus est donc explicite, hors temoin."""
    orig = db.TABLE_NORMALISATION
    db.TABLE_NORMALISATION = orig + (
        db.Regle("T8", "efface le signe", lambda t: t.replace("\u00b1", ""),
                 (("\u00b1", ""),)),)
    try:
        viol = db.demontrer_reecritures()
        assert any("porteur de signe" in v for v in viol), viol
    finally:
        db.TABLE_NORMALISATION = orig


def test_A1_residu_iii_une_regle_sans_reecritures_se_refuse_proprement():
    class RegleMuette:
        nom = "T8"
    orig = db.TABLE_NORMALISATION
    db.TABLE_NORMALISATION = orig + (RegleMuette(),)
    try:
        viol = db.demontrer_reecritures()
        assert any("sans colonne" in v for v in viol), viol
    finally:
        db.TABLE_NORMALISATION = orig


@pytest.mark.parametrize("texte,mord", [
    ("la version v1.2 du banc", False),
    ("la version v1 du banc", False),
    ("le gain v1.353", True),
    ("l'excès v2.437", True),
])
def test_A3_residu_le_prefixe_v_n_avale_plus_une_decimale(texte: str, mord: bool):
    """**Non bloquant 6** \u2014 *\u00ab changer la lettre suffisait \u00bb etait devenu \u00ab la lettre
    `v` suffit \u00bb.* Une version n'a jamais plus de deux chiffres par composante."""
    c = db.charger_corpus_depuis_cas({"report": texte})
    assert bool(db.CLAUSES["C0"](c)[0]) is mord, texte


def test_les_consignes_de_redaction_sont_gravees_dans_l_artefact():
    """**Non bloquant 8** \u2014 elles gouverneront l'ecriture ; un redacteur qui les
    ignore fera echouer le banc sans comprendre pourquoi."""
    art = json.loads((RACINE / "report" / "doc_bench.json").read_text(encoding="utf-8"))
    cons = " ".join(art["consignes_de_redaction"])
    assert len(art["consignes_de_redaction"]) == 6
    assert "Ambiguite DECLAREE" in cons
    assert "ISO" in cons and "`v`" in cons and "IC 95" in cons
    assert "APRES CHAQUE PASSE" in cons and "N-23" in cons


def test_la_reserve_6_10_est_consignee():
    """**Non bloquant 9** \u2014 aucune clause n'**applique** le troisieme disjoint :
    au gel, c'est une **adjudication humaine, non machine**."""
    art = json.loads((RACINE / "report" / "doc_bench.json").read_text(encoding="utf-8"))
    assert "ADJUDICATION HUMAINE" in art["reserve_6_10"]
    assert "PUB-instrument" in art["reserve_6_10"]


def test_le_triplet_est_publie_dans_l_artefact():
    art = json.loads((RACINE / "report" / "doc_bench.json").read_text(encoding="utf-8"))
    cit = art["citabilite_du_manifeste"]
    assert (cit["citables"], cit["hors_domaine_de_C1b"], cit["non_resolues"]) == \
        (49, 2, 1)
    assert "JAMAIS un ratio unique" in cit["definition"]


def test_B2_la_partition_des_etiquettes_est_disjointe_et_chiffree():
    """**B-2** — l'étiquette unique recouvrait **trois** situations ; 9 lignes sur 40 ne
    venaient pas du journal. *C'est le mode du compte « 2 » que ce protocole poursuit.*"""
    import collections
    import csv
    with (RACINE / "report" / "manifest.csv").open(encoding="utf-8-sig",
                                                   newline="") as fh:
        lignes = list(csv.DictReader(fh))
    compte = collections.Counter(l["etiquette"] for l in lignes)
    assert "" not in compte, "une etiquette vide subsiste"
    assert set(compte) <= db.ETIQUETTES_LICITES, set(compte) - db.ETIQUETTES_LICITES
    # `externe-archive`, valeur GELEE au §3.1, a un support VIDE : cardinal 0, publie.
    assert compte["externe-archivé"] == 0
    # la partition separe bien les sources
    assert compte["re-lu du journal, non re-mesuré"] > 0
    assert compte["re-lu de l'architecture, non re-mesuré"] > 0
    assert compte["re-lu d'un protocole, non re-mesuré"] > 0


def test_B3_delta_banc_est_produit_et_sa_lecture_declaree():
    """**B-3** — le secondaire du §4.4 était *implémenté à moitié* : la garde portait,
    aucun `E_null` n'était produit."""
    corpus = db.charger_corpus_depuis_disque(RACINE, RACINE / "REPORT.md")
    db_res = db.delta_banc(corpus)
    assert db_res["par_famille"], "aucun E_null produit"
    for fam, val in db_res["par_famille"].items():
        assert "delta_banc" in val and "E_null" in val
    degen = [f for f, v in db_res["par_famille"].items()
             if v["delta_banc"] == "SANS OBJET"]
    assert degen, "aucune famille degeneree : la garde D23 n'est pas exercee"
    # Δ_banc n'est PAS interpretable tant que la prose n'existe pas — declare.
    assert db_res["interpretable"] is False
    assert "REPORT.md est absent" in db_res["lecture"]


def test_B5_les_clauses_sans_cas_reel_sont_nommees():
    """**B-5** — une clause qui n'a jamais vu un cas réel se **déclare** ; elle ne se
    confond pas avec une clause satisfaite. C'est une donnée du rapport."""
    corpus = db.charger_corpus_depuis_disque(RACINE, RACINE / "REPORT.md")
    rapport = db.executer(corpus)
    # `cas_exerces` est un PLANCHER : le lire ici etait le bloquant lui-meme (0-243).
    non_exercees = {c["clause"] for c in rapport["clauses"] if c["cas_reels"] == 0}
    assert len(non_exercees) == 25, sorted(non_exercees)
    assert non_exercees <= set(db.ORDRE_CLAUSES)
    assert all(c["cas_exerces"] >= 1 for c in rapport["clauses"])


def test_les_exclusions_d_invalides_portent_leur_motif_DANS_le_fichier():
    """Réserve du Verifier : les deux exclusions ne portaient leur motif que dans le
    compte rendu. *Un lecteur doit voir pourquoi elles n'y sont pas.*"""
    import csv
    with (RACINE / "report" / "invalides.csv").open(encoding="utf-8-sig",
                                                    newline="") as fh:
        lignes = list(csv.DictReader(fh))
    inclus = [l for l in lignes if l["inclus"] == "oui"]
    exclus = [l for l in lignes if l["inclus"] == "non"]
    assert len(inclus) == 3, "l'enumeration rend TROIS runs invalides"
    assert len(exclus) == 2
    for l in exclus:
        assert l["motif_d_exclusion"].strip(), l["run"]


def test_les_trois_classes_de_violation_sont_exercees():
    """§3.3 — `ORPHELIN`, `DÉSACCORD`, `SOURCE-ILLICITE` : la campagne de mutants les
    exerce **toutes les trois**."""
    vues: set[str] = set()
    for nom in db.ORDRE_CLAUSES:
        rapport = db.executer(db.charger_corpus_depuis_cas(_fixture(nom)["mutant"]))
        for cl in rapport["clauses"]:
            for v in cl["violations"]:
                vues.add(v["classe"])
    assert {"ORPHELIN", "DESACCORD", "SOURCE-ILLICITE"} <= vues, vues


# =====================================================================================
# 6. Les deux verrous du livrable, et la verification machine des sommes
# =====================================================================================

def test_VERROU_A_le_mode_mutant_n_ecrit_jamais_l_artefact_publie():
    """*Le controle de sante du banc detruisait le livrable qu'il certifie* (0-254).

    Une execution de `C-mut` avait remplace `report/doc_bench.json` par un rapport
    `E = 32`, **32/32 `PASS`**. Le mode mutant ecrit desormais sous un autre nom.
    """
    src = (RACINE / "tools" / "doc_bench.py").read_text(encoding="utf-8")
    assert 'sortie = sortie.with_suffix(".mutant.json")' in src
    assert "if a.force_pass:" in src
    art = (RACINE / "report" / "doc_bench.json").read_text(encoding="utf-8")
    assert '"force_pass": false' in art.replace(" ", "").replace('"force_pass":false',
                                                                 '"force_pass": false')


def test_VERROU_B_l_artefact_publie_figure_dans_SHA256SUMS():
    """*Sans (b), (a) ne ferme que le chemin connu.* Le livrable entre aux sommes."""
    sommes = (RACINE / "report" / "SHA256SUMS").read_text(encoding="utf-8")
    assert "report/doc_bench.json" in sommes


def test_la_verification_des_sommes_est_MACHINE_et_non_manuelle():
    """**§5.7** exigeait l'ouverture par acces direct et la verification du hash ;
    **aucune machine ne le faisait**."""
    viol = db.verifier_sommes(RACINE)
    assert viol == [], viol
    lignes = [l for l in (RACINE / "report" / "SHA256SUMS")
              .read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lignes) == 55


def test_la_circularite_du_hash_est_brisee_par_une_commande_DISTINCTE():
    """Le hash de l'artefact ne peut pas etre calcule par l'execution qui l'ecrit.
    Il ne l'est pas : ``--sommes`` **lit** l'artefact et **ne l'ecrit jamais**."""
    doc = db.calculer_sommes.__doc__
    assert "circularite" in doc.lower().replace("\u00e9", "e")
    src = (RACINE / "tools" / "doc_bench.py").read_text(encoding="utf-8")
    bloc = src[src.index("if a.sommes:"): src.index("corpus = charger_corpus_depuis_disque")]
    assert "write_text" in bloc                      # elle ecrit les SOMMES
    assert "rapport" not in bloc                     # jamais l'artefact


def test_une_substitution_de_l_artefact_est_DETECTEE():
    """C'est la propriete que les deux verrous achetent."""
    import hashlib
    chemin = RACINE / "report" / "doc_bench.json"
    octets = chemin.read_bytes()
    try:
        chemin.write_bytes(octets + b"\n")
        viol = db.verifier_sommes(RACINE)
        assert any("doc_bench.json" in v for v in viol), viol
    finally:
        chemin.write_bytes(octets)
    assert db.verifier_sommes(RACINE) == []


@pytest.mark.parametrize("expr,licite", [
    ("9**9**9", False),
    ("2**5000", False),
    ("0.05**(1/100)", True),
    ("2**10", True),
])
def test_la_bombe_memoire_est_bornee(expr: str, licite: bool):
    """`9**9**9` etait licite, borne par le **seul chrono de 10 s**."""
    assert (db._valider_expression(expr) == "") is licite


def test_sept_fonctions_rejouables_et_pas_de_code_mort():
    assert len(db.FONCTIONS_REJOUABLES) == 7
    src = (RACINE / "tools" / "doc_bench.py").read_text(encoding="utf-8")
    assert "NOEUDS_REJOUABLES" not in src
    assert "les **sept** fonctions" in src


def test_la_consigne_5_est_alignee_sur_le_CODE():
    """La consigne etait **plus faible que le code** : elle disait *hors bloc
    `AUTOPSIE`*, quand `C1` rend `ORPHELIN` **y compris a l'interieur**. *Un redacteur
    conforme a la consigne obtenait un `C1` FAIL.*"""
    cons = [c for c in db.CONSIGNES_DE_REDACTION if "N-23" in c]
    assert len(cons) == 1
    assert "Y COMPRIS A L'INTERIEUR" in cons[0]
    corpus = db.charger_corpus_depuis_disque(RACINE, RACINE / "REPORT.md")
    texte = ("# T\n\n## 3. S\n\n<!-- AUTOPSIE -->\nL'incertitude vaut "
             "\u00b1 du facteur 4.8 \u27e6N-23\u27e7.\n<!-- /AUTOPSIE -->")
    c = db.Corpus(texte, corpus.manifeste, corpus.riders, corpus.invalides,
                  corpus.relecture, corpus.data, corpus.racine)
    assert any(v.classe == "ORPHELIN" for v in db.CLAUSES["C1"](c)[0])


def test_l_ambiguite_des_separateurs_est_DECLAREE():
    cons = " ".join(db.CONSIGNES_DE_REDACTION)
    assert "Ambiguite DECLAREE" in cons
    assert db.normaliser("1,234") == "1.234"
    assert db.normaliser("20,000,000") == "20000000"
    assert db.normaliser("1 234") == "1 234"       # espace ASCII : NON normalise


def test_relecture_csv_existe_avec_son_entete():
    """`C-relect` et le §6.11 mordront au gel ; le fichier existe, vide."""
    import csv
    chemin = RACINE / "report" / "relecture.csv"
    assert chemin.is_file(), "report/relecture.csv absent"
    with chemin.open(encoding="utf-8-sig", newline="") as fh:
        entetes = next(csv.reader(fh))
    assert entetes == ["section", "auteur", "relecteur", "horodatage", "defauts",
                       "verdict"]
