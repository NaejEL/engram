# Journal d'expériences

Une entrée par run notable. Toujours : date, commit, config (les écarts au défaut
suffisent), résultat chiffré, conclusion en une phrase. Les runs invalides (0 write,
divergence) se notent aussi — ce sont des données.

Modèle d'entrée :

```markdown
## 2026-08-20 — titre court
- **Commit** : abc1234
- **Config** : model=gpt2, layer=6, lam=0.5, eta=0.05, threshold=4.0 (défauts sinon)
- **Run** : `python eval/fact_injection.py --seeds 10`
- **Résultat** : Δlog-prob = +X.XX ± Y.YY nats (N=10), top-10 hits 4/10, writes/run ≈ 12
- **Conclusion** : ...
- **Suite** : ...
```

---

## 2026-08-20 — setup : piège torch CPU sur Windows

- **Résultat** : le premier `pip install` avec extra-index cu124 a silencieusement
  installé torch 2.13.0+cpu — l'index cu124 s'arrête à torch 2.6, donc pip a préféré
  le 2.13 CPU de PyPI. Corrigé en passant à l'index cu126 (torch 2.13.0+cu126, qui
  gagne la résolution PEP 440). Driver de la machine : 581.29 (OK pour CUDA 12.6/13.0).
- **Tests** : `pytest tests/ -q` → 9/9 sur CPU (delta rule : rappel, réécriture,
  bornage vs Hebb pur, cap de lecture, decay, élagage, reset, dtype fp16).
- **Note** : transformers installé en v5 (5.15.1) → `from_pretrained(..., dtype=)`
  et non plus `torch_dtype=` (déprécié). `cortex.py` écrit pour v5.
- **Conclusion** : toujours vérifier `torch.cuda.is_available()` après toute
  réinstallation ; le repli CPU est silencieux.

## 2026-08-20 — setup : téléchargement HF Hub, backend Xet cassé

- **Résultat** : `snapshot_download('gpt2')` échoue en anonyme sur cette machine —
  404 sur `xet-read-token` (nouveau backend de stockage Xet). Contournement qui
  marche : `HF_HUB_DISABLE_XET=1` (retour au HTTP classique). GPT-2 (poids
  safetensors + tokenizer, sans variantes TF/Flax) est en cache HF standard.
- **Conclusion** : préfixer `HF_HUB_DISABLE_XET=1` pour tout futur téléchargement de
  modèle (SmolLM2 en v1.2 notamment) ; une fois en cache, plus besoin.

## 2026-08-20 — X0 : premier run E1, baseline naïve

- **Commit** : (pré-initial)
- **Config** : défauts (model=gpt2, layer=6, lam=0.5, eta=0.05, decay=1e-3, thr=4.0)
- **Run** : `python eval/fact_injection.py` (10 secrets), GPU RTX 3060, ~30 s/run
- **Résultat** : Δlog-prob moyen **+0.219 ± 0.271 nats** (N=10), 9/10 positifs,
  0/10 top-10, rangs quasi immobiles (ex. 14431 → 13513), ~9 writes/injection.
  Seul négatif : tambourine (−0.40, le seul secret à rang bas au départ, 185).
- **Conclusion** : signal faible mais directionnel — l'information traverse bien le
  vidage du cache KV, mais l'effet est loin d'un rappel fonctionnel. Profil attendu
  pour la version naïve.
- **Suite** : balayage layer ∈ {3,6,9} × (λ,η) ∈ {(0.5,0.05),(1.0,0.1),(2.0,0.2)}
  pour situer où vit le signal, puis X1 (gyrus denté) + E3.

## 2026-08-20 — X0 : balayage layer × λ × η sur E1

- **Config** : défauts sauf balayés ; 10 secrets par config, GPU.
- **Résultat** (Δlog-prob moyen ± σ, nats) :

  | layer | λ=0.5 η=0.05 | λ=1.0 η=0.1 | λ=2.0 η=0.2 |
  | --- | --- | --- | --- |
  | 3 | +0.076 ± 0.24 | +0.333 ± 1.23 | +0.434 ± 2.47 |
  | 6 | +0.219 ± 0.27 | **+0.735 ± 0.89** | +0.870 ± 1.61 |
  | 9 | +0.125 ± 0.31 | +0.205 ± 1.09 | +0.171 ± 1.41 |

  Top-10 : 0/10 partout.
- **Conclusion** : (1) la couche 6 domine nettement — l'intuition « milieu du réseau »
  (décision D3) est confirmée empiriquement ; (2) monter λ/η achète de la moyenne au
  prix d'une explosion de la variance (σ passe de 0.27 à 1.6 sur la couche 6) : des
  secrets gagnent beaucoup, d'autres se font écraser — c'est le profil exact de
  l'**interférence entre associations**, précisément ce que X1 (gyrus denté) doit
  corriger. Le déclencheur de X1 est donc observé, en plus de son statut d'assurance.
- **Point de fonctionnement retenu** : layer=6, λ=1.0, η=0.1 (meilleur ratio
  moyenne/σ) — devient le défaut de `EngramConfig` et la référence du tableau des
  poids pour X1.
- **Suite** : X1 (projection gyrus denté) + éval E3, benchmarkées contre cette ligne.

## 2026-08-20 — X1 : projection gyrus denté — RETENUE

- **Config** : dg_dim=8192, dg_topk=64, layer=6 ; E1, 10 secrets, GPU.
- **Implémentation** : `φ_dg(h) = normalize(topk(G·h, 64))`, G aléatoire gelée seedée
  (décision D9), M rectangulaire 768×8192. Tests 13/13 dont interférence
  (clés à cos ≈ 0.95 : rappel DG > rappel dense).
- **Résultat** :

  | Config | E1 Δlogp |
  | --- | --- |
  | X0 dense, λ=1.0, η=0.1 (référence) | +0.735 ± 0.886 |
  | X1 dg, λ=1.0, η=0.1 | +0.540 ± 0.566 |
  | X1 dg, λ=1.0, η=0.1, topk=128 | +0.563 ± 0.556 |
  | X0 dense, λ=2.0, η=0.2 (rappel du balayage) | +0.870 ± 1.609 |
  | **X1 dg, λ=2.0, η=0.2** | **+1.361 ± 1.588** |
  | X1 dg, λ=3.0, η=0.3 | +1.337 ± 1.726 (plateau) |

  Top-10 : 0/10 partout. tambourine toujours négatif (seul secret à rang initial bas,
  185 : l'injection agressive pénalise les tokens déjà probables — indice pour E3).
- **Conclusion** : à hyperparamètres doux, DG *baisse* la moyenne (empreinte plus
  faible : 64 colonnes touchées par write) mais divise la variance par 1.6. La vraie
  valeur est la **marge de manœuvre** : l'interférence domptée permet λ=2/η=0.2, soit
  +56 % de moyenne à variance égale vs X0 au même régime, et +85 % vs la meilleure
  X0. λ=3 plafonne : le coude est trouvé. **Poids net de X1 : +0.63 nats.**
- **Verdict** : retenu. Nouveaux défauts : dg=8192/64, λ=2.0, η=0.2 — reproduit
  +1.361 exactement. **Sous réserve E3** : λ=2.0 est agressif, le dommage collatéral
  n'est pas encore mesuré (le cas tambourine suggère qu'il existe).
- **Suite** : écrire et lancer E3, remplir la colonne manquante du tableau des poids.

## 2026-08-20 — E3 : le dommage collatéral est réel — et DG le réduit

- **Config** : `eval/collateral.py` (nouveau) — texte neutre 344 tokens, contrôle M=0
  calculé une fois (déterministe), M chargée via protocole E1 pour chacun des 10 secrets.
- **Résultat** :

  | Config | E3 ΔNLL/token (neutre) |
  | --- | --- |
  | X0 (dense, λ=1.0, η=0.1) | **+0.1991 ± 0.0241** |
  | X1 (dg=8192/64, λ=2.0, η=0.2) | **+0.1354 ± 0.0147** |

  Seuil d'alarme (+0.05) : dépassé dans les deux cas, ~3× pour X1.
- **Conclusion** : (1) le mode d'échec prédit (« ça marche mais ça rend le modèle
  plus bête ailleurs ») est réel et chiffré — la lecture, active à chaque token,
  injecte du bruit récupéré par des clés sans rapport ; (2) résultat inattendu et
  précieux : **DG réduit aussi le dommage collatéral** (+0.135 à λ=2 contre +0.199 à
  λ=1 en dense) — les clés creuses limitent les récupérations parasites. X1 gagne
  sur E1 ET E3. (3) Le point λ=2 reste **au-dessus du seuil** : selon notre règle
  (README §« prouver ou tuer »), son gain E1 « ne compte pas » tant que E3 > 0.05.
- **Suite** : courbe de compromis (λ × max_read_norm) pour trouver le meilleur gain
  E1 sous contrainte E3 < 0.05. Flag `--cap` ajouté aux évals E1/E3.

## 2026-08-20 — X1b : compromis λ × cap — point conforme E3 trouvé

- **Config** : dg=8192/64, η=0.2, layer=6 ; balayage λ ∈ {1, 2} × cap ∈ {0.25, 0.1}.
- **Résultat** (E1 = Δlogp sur 10 secrets ; E3 = ΔNLL/token sur texte neutre) :

  | λ | cap | E1 | E3 |
  | --- | --- | --- | --- |
  | 2.0 | 0.25 | +0.735 ± 0.85 | +0.127 (> seuil) |
  | 2.0 | 0.1 | +0.336 ± 0.37 | +0.046 ✓ |
  | **1.0** | **0.25** | **+0.740 ± 0.80** | **+0.023 ✓** |
  | 1.0 | 0.1 | +0.318 ± 0.35 | +0.031 ✓ |

- **Conclusion** : le cap de lecture agit comme **gating doux gratuit** — les
  récupérations pertinentes (fortes) saturent le cap quel que soit λ, donc λ=2
  n'ajoute rien à E1 à cap égal ; les récupérations parasites (faibles) restent sous
  le cap et sont amplifiées proportionnellement à λ, d'où l'E3 ×5.6 de λ=2. Autrement
  dit : λ contrôle le bruit, le cap contrôle le signal. Le point λ=1/cap=0.25 garde
  ~100 % du meilleur E1 précédent (+1.36 → +0.74 : on rend le gain brut de λ=2, mais
  ce gain était fictif puisque non conforme) pour un E3 à moitié du seuil.
- **Verdict** : nouveaux défauts λ=1.0, max_read_norm=0.25 (η=0.2, dg inchangés).
  Reproduit : E1 +0.740 ± 0.799, 13/13 tests. Premier point de fonctionnement
  **conforme aux trois contraintes déclarées** du PoC.
- **Suite** : E2 (dérive de domaine) jamais lancée — dernière colonne vide du tableau
  des poids ; puis E1b (paraphrase) pour tester le rappel par indice partiel.

## 2026-08-20 — E1b : le rappel généralise — déclencheur X2 non observé

- **Config** : défauts (dg=8192/64, λ=1.0, η=0.2, cap=0.25) ; E1 étendu à 4 questions
  (1 exacte + 3 paraphrases), 10 secrets.
- **Résultat** (Δlog-prob moyen) :

  | Question | Δlogp |
  | --- | --- |
  | exact « The password is » | +0.740 ± 0.799 |
  | para2 « Remember, the password was » | **+1.006 ± 0.918** |
  | para1 « The secret word is » | +0.345 ± 0.347 |
  | para3 « Enter the password: » | +0.168 ± 0.505 |

  Généralisation (moy. paraphrases / exact) : **0.68**.
- **Conclusion** : pas de par-cœur — le rappel se dégrade gracieusement avec le
  recouvrement sémantique de l'indice (para2, qui partage « password »/« Remember »
  avec le fait injecté, fait mieux que l'exact ; para3 garde ~1/3 du signal).
  **Déclencheur X2 (CA3) non observé** : pas de lecture itérée nécessaire à ce stade.
  Bon présage pour E2 (qui exige un rappel sur états jamais vus). tambourine reste
  négatif partout — le pattern « token déjà probable pénalisé » se confirme.
- **Suite** : E2 lancée dans la foulée.

## 2026-08-20 — E2 : la dérive de domaine est réelle — interaction −0.055

- **Config** : défauts (dg=8192/64, λ=1.0, η=0.2, cap=0.25) ; RFC 9293 (TCP), 6000
  tokens, chunks de 512 (cache KV vidé entre chunks), gating normal (thr=4.0).
- **Résultat** :

  | Condition | NLL 1ʳᵉ moitié | NLL 2ᵉ moitié | Δ |
  | --- | --- | --- | --- |
  | sans M | 2.3289 | 2.3236 | −0.0053 |
  | avec M | 2.3778 | 2.3174 | −0.0605 |

  **Interaction : −0.0551 nats/token** (≈ −5.4 % de perplexité attribuable à M).
  writes = 1401/6000 tokens (~23 % de taux d'écriture) — run valide.
- **Décomposition honnête** : (1) sans M, Δ ≈ 0 → le contrôle est propre, aucune
  fuite d'adaptation entre chunks ; (2) avec M, la 1ʳᵉ moitié PAIE (+0.049 vs
  contrôle — la taxe collatérale pendant que M se remplit, cohérente avec E3) puis la
  2ᵉ moitié passe SOUS le contrôle (2.3174 < 2.3236, −0.006 absolu). L'essentiel de
  l'interaction vient du recouvrement de ce coût initial ; le gain absolu final est
  petit mais net : **M finit par rapporter plus qu'elle ne coûte, dans le domaine**.
  La prédiction de POSSIBLE_APPROACH (« 2-3 % de perplexité, à trancher ») est
  dépassée : 5.4 % d'interaction, et le signe absolu est bon.
- **Bilan d'étape — les trois questions falsifiables du PoC ont leur réponse**, au
  point de fonctionnement conforme (X1b), sur GPT-2 124M :
  E1 +0.740 (rappel post-vidage KV réel) · E1b ratio 0.68 (association, pas
  par-cœur) · E2 −0.055 (adaptation en ligne réelle) · E3 +0.023 ✓ (sous le seuil).
- **Suite** : ablations de fond (delta rule vs Hebb pur ; gating vs toujours-écrire ;
  dg on/off sur E2), puis SmolLM2-360M (v1.2). Le cas « tambourine » (token déjà
  probable pénalisé) reste la meilleure piste d'amélioration du mécanisme de lecture.

## 2026-08-20 — Ablations v1.1 : le gating est le héros silencieux

- **Config** : défauts X1b partout, un flag inversé par run (`--hebbian`, `--thr 0`,
  `--dg-dim 0` — flags CLI ajoutés à cet effet).
- **Résultat** :

  | Ablation | Métrique | Référence X1b | Ablatée |
  | --- | --- | --- | --- |
  | A1 Hebb pur (E1) | Δlogp exact | +0.740 ± 0.80 | +0.693 ± 0.80 |
  | A2 toujours-écrire (E2) | interaction | −0.0551 (1401 writes) | **−0.0045** (5988 writes) |
  | A3 clés denses (E2) | interaction | −0.0551 | −0.0352 |

- **Conclusion** :
  - **A2, le résultat majeur : le gating par surprise porte ~92 % de l'effet E2.**
    Écrire chaque token noie M d'associations redondantes ET accélère le decay
    (appliqué par write : ×4.3 d'érosion des traces utiles). D6 validée in situ —
    le « j'apprends quand je suis surpris » n'est pas une économie, c'est la
    condition du fonctionnement.
  - A3 : DG apporte +57 % d'interaction sur E2 (−0.035 → −0.055). X1 est maintenant
    gagnant sur les trois évals (E1 marge λ/η, E3 −32 % de dommage, E2 +57 %).
  - A1 : Hebb ≈ delta sur E1 (+0.693 vs +0.740) — attendu a posteriori : une
    injection unique ne sollicite pas le terme correctif (rien à réécrire, pas
    d'accumulation). Le juge pertinent est un long stream → A1b (E2 Hebb) lancée.
- **Suite** : A1b, puis commit v1.1.

## 2026-08-21 — A1b + contrôles : Hebb pur, le challenger inattendu

- **Config** : défauts X1b, E2 (6000 tokens) et E3 ; runs `--hebbian` et `--eta 0.4`.
- **Résultat** :

  | Règle | E2 interaction | E2 NLL 2ᵉ moitié (abs.) | E3 |
  | --- | --- | --- | --- |
  | delta η=0.2 (défaut) | −0.0551 | 2.3174 | +0.0228 ✓ |
  | delta η=0.4 (contrôle C1) | −0.0488 | 2.3062 | — |
  | **Hebb pur η=0.2 (A1b)** | **−0.0948** | **2.3034** | +0.0331 ✓ (C2) |

  (contrôle sans M : 2ᵉ moitié 2.3236 ; E1 Hebb ≈ delta : +0.693 vs +0.740)
- **Conclusion** :
  - Hebb pur bat nettement la delta rule sur E2 et ce n'est **pas** un effet de pas
    d'apprentissage (delta η=0.4 régresse : −0.0488). Mécanisme : sur un stream
    homogène, le terme correctif éteint progressivement les writes répétés
    (l'erreur v−M·k → 0) — c'est sa fonction (bornage, réécriture) et son coût
    (adaptation amortie). Hebb renforce à pleine force, paie plus cher en début de
    document (2.4035 vs 2.3778) et en E3 (+45 %), mais reste conforme partout.
  - **D5 nuancée, pas renversée** : la delta rule reste le défaut pour ses vertus
    prouvées hors E2 — bornage (test unitaire : Hebb croît linéairement, son seul
    garde-fou réel est le cap de lecture + decay) et capacité de réécriture (E1
    « le mot de passe a changé », non testée par E2 où rien ne se contredit).
    Hebb est consigné comme **régime de forte adaptation** légitime.
- **Question ouverte** (déclencheur du prochain arbitrage) : E2 long horizon
  (≥ 20k tokens) — si Hebb sature/diverge quand ‖M‖ atteint son équilibre η/δ,
  la delta rule gagne au fond ; s'il tient, un mode hybride (Hebb + decay fort,
  ou delta avec plancher d'erreur) mérite une entrée X6 dans EXTENSIONS.
- **Suite** : commit v1.1 (flags d'ablation + résultats).

## 2026-08-21 — E2n narratif : l'adaptation est réelle, l'avantage Hebb était du n-gramme

- **Config** : défauts X1b ; *Pride and Prejudice* (Gutenberg, en-tête/pied retirés →
  `data/pnp_narrative.txt`), 6000 tokens, chunks 512 ; delta puis `--hebbian`.
- **Résultat** (réf. RFC 9293 entre parenthèses) :

  | Règle | Interaction narratif | Interaction RFC | Rétention |
  | --- | --- | --- | --- |
  | delta η=0.2 | **−0.0304** (writes 2218) | −0.0551 | ~55 % |
  | Hebb pur | −0.0335 (writes 2250) | −0.0948 | ~35 % |

- **Conclusion** — les deux questions d'AFTER_v1_THOUGHTS ont leur réponse :
  1. **Pas un pur cache de n-grammes** : l'effet survit sur de la fiction
     (−0.030 nats/token, ~55 % du niveau RFC). Une partie du gain RFC était bien de
     l'exploitation de redondance technique, mais le cœur est de l'adaptation.
  2. **L'avantage Hebb s'évapore hors régime redondant** : −0.0335 vs −0.0304 pour
     delta — égalité (runs uniques). Son +72 % sur la RFC venait précisément du
     renforcement à pleine force des motifs répétés. **Arbitrage D5 quasi clos** :
     delta ≈ Hebb sur texte général, et delta garde bornage + réécriture → la delta
     rule reste le défaut, avec confiance cette fois. L'E2 long horizon passe de
     « nécessaire » à « optionnel ».
- **Note** : la 2ᵉ moitié de P&P est intrinsèquement plus dure (contrôle +0.46) —
  c'est le contrôle qui absorbe ça, l'interaction reste propre. Taux d'écriture 37 %
  (vs 23 % RFC) : le narratif surprend plus souvent au seuil 4.0.
- **Suite** : I1 (prédicteur d'échec par similarité de clés), puis v1.2 SmolLM2-360M.

## 2026-08-21 — I1 : la jauge de saturation — le mur n'est pas la capacité

- **Config** : défauts X1b ; instrumentation `track_keys` (buffer circulaire de clés,
  cos max par write) ; E1 en trois régimes : mono-fait, multi partagé (`--multi`,
  10 faits même gabarit dans une seule M), multi varié (`--multi --varied`,
  10 faits à contextes distincts).
- **Résultat** :

  | Régime | keysim (étendue) | E1 Δlogp | Lecture |
  | --- | --- | --- | --- |
  | mono-fait (référence) | 0.21–0.26 | +0.740 ± 0.80 | sain |
  | multi, gabarit commun | 0.68–0.87 | **+0.025 ± 0.80** | rappel détruit |
  | multi, contextes variés | 0.26–0.52 | **+0.770 ± 0.72** | 10 faits sans coût |

- **Conclusion** :
  1. **La capacité de M n'est pas la contrainte** : 10 faits simultanés se rappellent
     aussi bien qu'un seul dès que leurs indices diffèrent. Le mur du régime
     « gabarit commun » est une collision d'indices — cohérent avec la théorie DG
     (la séparation de patterns ne peut rien pour des entrées identiques).
  2. **keysim est une jauge de saturation calibrée** (0.23/0.45/0.78 →
     +0.74/+0.77/+0.03, falaise au-delà de ~0.5) — c'est un détecteur de RÉGIME.
     L'hypothèse AFTER_v1 (corrélation négative par fait) n'est PAS soutenue
     dans les trois régimes — corr intra-régime +0.2 à +0.5, **n.s. à N=10**
     (requalifié à l'audit du 2026-08-21 : « falsifiée » était trop fort pour
     cette puissance) : le prédicteur utile est le niveau absolu, pas le
     gradient intra-régime. Débouché : gating d'écriture
     (alerter/refuser quand cos max > ~0.6) — candidat X6.
  3. Pattern prior confirmé en gradué : corr(Δlogp, logp a priori) = −0.50 en
     mono-fait — M aide l'improbable, pénalise le déjà-probable (tambourine négatif
     dans TOUS les régimes). Piste mécanistique pour le verrou top-10.
  4. Pas d'effet de récence net en régime collision (corr −0.05) : les faits en
     conflit ne se résolvent pas par « le dernier gagne » — ils se superposent en
     bruit (les clés ne sont pas assez identiques pour la réécriture propre de la
     delta rule).
- **Bug corrigé au passage** : les similarités étaient collectées après le
  `reset_memory()` du contrôle (toujours vides) — capture déplacée juste après
  l'injection.
- **Suite** : v1.2 — SmolLM2-360M, avec la jauge keysim en main.

## 2026-08-21 — v1.2 : protocole pré-enregistré (avant toute mesure)

- **Modèle** : HuggingFaceTB/SmolLM2-360M (base) — 32 couches, d=960, entraîné
  ~11T tokens (vs 12 couches, d=768, ~10B pour GPT-2 124M).
- **Protocole** : (1) balayage couche ∈ {8, 16, 24} sur E1, autres hyperparamètres
  aux défauts X1b ; (2) à la meilleure couche : E1 complet (+multi varié), E3, E2 RFC.
- **Questions posées avant mesure** :
  Q1. Le signal E1 survit-il au changement d'échelle ? (« artefact d'échelle ou
      début de quelque chose » — AFTER_v1)
  Q2. La règle « couche du milieu » (D3, confirmée sur GPT-2) se transpose-t-elle ?
  Q3. Un modèle mieux entraîné a des NLL plus basses partout → moins de writes au
      seuil 4.0, et le pattern « M aide l'improbable » (corr −0.50) prédit un
      bénéfice PLUS FAIBLE. Pari honnête : Δlogp E1 positif mais < +0.740.
  Q4. La jauge keysim garde-t-elle sa falaise (~0.5) dans un espace latent différent ?
- **Résultat** (mesuré les 2026-08-21, détail ci-dessous) :

  | Mesure (SmolLM2-360M) | Valeur | Réf. GPT-2 |
  | --- | --- | --- |
  | E1 balayage couche : 8 / 16 / 24 | +0.554 / **+0.852** / **−0.375** | couche 6/12 : +0.740 |
  | E1-multi varié (couche 16, cap 0.25) | +0.817 ± **0.214** | +0.770 ± 0.716 |
  | E3 (cap 0.25 → 0.15 → 0.1) | +0.079 / +0.067 / **+0.042 ✓** | +0.023 ✓ (cap 0.25) |
  | E1 au point conforme (cap 0.1) | **+0.494 ± 0.245** | +0.740 (cap 0.25) |
  | E2 RFC (cap 0.25) | −0.0500 (writes 1140) | −0.0551 |

- **Réponses aux questions pré-enregistrées** :
  Q1. **Pas un artefact d'échelle** — le signal survit et se renforce (+0.852 mono,
      +0.817 multi avec σ divisé par 3). Le rappel multi-faits est plus NET à 360M.
  Q2. **La règle du milieu se transpose exactement** (16/32 comme 6/12) — et
      l'injection tardive devient activement nocive (couche 24 : −0.375, là où
      GPT-2 couche 9 était juste plus faible).
  Q3. **Pari TENU au point conforme** *(reformulé à l'audit du 2026-08-21 — la
      version initiale disait « pari perdu dans le bon sens »)* : le pari
      pré-enregistré (« Δlogp positif mais < +0.740 ») se juge au point conforme
      E3, où E1 = **+0.494 < +0.740 ✓**. Le +0.852 qui l'aurait « perdu » est
      mesuré sans σ, à cap 0.25 NON conforme (E3 +0.079) — pas comparable. La
      frontière E3 s'est déplacée — les distributions plus pointues du modèle
      rendent la même injection plus coûteuse (E3 ×3.5 à cap égal). Nouveau
      point conforme : **cap=0.1** → E1 +0.494, E3 +0.042. La calibration λ/cap
      est PAR-MODÈLE, le mécanisme et la méthode de calibration se transposent
      tels quels.
  Q4. La corrélation keysim est négative à la couche 16 (−0.34 à −0.40) — le
      signe prédit par AFTER_v1 — mais **n.s. à N=10** (p ≈ 0.25 ; requalifié à
      l'audit du 2026-08-21) : une indication à re-tester en puissance, pas une
      émergence établie.
- **Conclusion v1.2** : le PoC tient sur deux modèles d'échelles différentes
  (124M/12 couches/10B tokens vs 360M/32 couches/11T tokens) — corrélations
  keysim par fait : voir Q4, n.s. à N=10. Ce qui est universel :
  delta rule + DG + gating + couche du milieu. Ce qui est par-modèle : le point
  (λ, cap) sur la frontière E1/E3.
- **Complément (E2 au point conforme, cap 0.1)** : interaction −0.0170, coût
  initial quasi nul (+0.019 en 1ʳᵉ moitié, parité absolue en fin de document).
  Le cap serré échange du gain d'adaptation contre l'innocuité — cohérent avec
  « λ contrôle le bruit, le cap contrôle le signal ».
- **Suite** : candidats — X6 (gating d'écriture par keysim), piste tambourine,
  ou consolidation v2 (replay génératif depuis M → LoRA).

## 2026-08-21 — X7 : protocole pré-enregistré (avant toute mesure)

- **Hypothèse** : l'injection de lecture aplatit la distribution de sortie — un seul
  mécanisme derrière tambourine, corr −0.50 et taxe E3 (voir EXTENSIONS §X7).
- **Campagne** (`eval/flattening.py`, GPT-2 d'abord) : courbe Δlogp vs prior sur
  ~24 cibles (échantillonnage top-heavy, rangs 1–500 sur-représentés), projection
  logit-lens de r sur W_U[cible] (fraction signal vs bruit), entropie avec/sans M,
  per-token sur les 10 secrets multi-tokens.
- **Prédictions posées avant mesure** :
  P1. Δlogp décroît avec le prior et **croise zéro** quelque part dans les rangs
      ~50–1000 (la loi corr −0.50 extrapolée).
  P2. La composante de r alignée sur la cible est petite devant sa norme (le bruit
      domine), et c'est le bruit qui coûte quand le prior est haut — l'entropie
      avec M monte sur les prompts où le modèle était confiant.
  P3. Sur les secrets multi-tokens, la pénalité liée au prior porte sur le token de
      tête ; les tokens suivants (conditionnés au premier) sont épargnés.
- **Résultat** (GPT-2, 20 cibles rangs 3–35000 + 10 multi-tokens) :
  - **P1 à moitié falsifiée, en mieux** : le gradient existe (corr Δlogp/prior =
    −0.66) mais il n'y a PAS de régime de pénalité — même les cibles de rang 3–10
    restent ≥ 0 (le gain tend vers zéro, il ne devient pas négatif). La loi est
    « M cesse d'aider le probable », pas « M combat le probable ». tambourine
    (Δ −0.41) est en-dessous de ce que la courbe prédit : outlier non expliqué par
    le prior seul.
  - **P2 confirmée avec un twist majeur** : ΔH = **+0.141 nats partout** (aplatissement
    uniforme, indépendant de la cible) ET cos(r, W_U[cible]) = **−0.01 en moyenne**
    contre 0.136 de base aléatoire — la lecture ne contient AUCUNE composante
    dirigée vers la cible dans le chemin direct. Le rappel fonctionne donc
    entièrement par **recomputation indirecte** : r déplace l'état à la couche 6 et
    les couches suivantes re-routent le calcul. « M contient ~0 % de signal
    directionnel et ~100 % de déplacement d'état » — l'aplatissement est le coût
    fixe de ce déplacement.
  - **P3 falsifiée sur le point intéressant** : le gain se concentre sur la tête
    (+0.567 vs +0.172, la suite étant souvent saturée à ~0), mais la pénalité de
    tambourine porte sur TOUTE la séquence (tête −0.414, suite −0.407).
- **Conséquence pour le gate X7 — feu vert renforcé** : l'injection coûte un
  aplatissement FIXE (+0.14 nats d'entropie) mais ne rapporte quasi rien quand le
  cortex est confiant (Δ→0 aux rangs bas). Gater la lecture par l'incertitude
  devrait donc conserver ~tout le bénéfice et éliminer ~tout le coût E3 — le
  mécanisme à deux facteurs (incertitude × keysim) a sa cible chiffrée.
- **Question ouverte** : le verrou top-10 n'est pas une pénalité active mais
  l'absence de composante directionnelle — un déplacement d'état diffus déplace les
  rangs de 20k → 15k mais ne peut pas hisser un token au sommet. Piste v2+ : valeurs
  stockées dans l'espace d'unembedding (v = u_token) ou tête de lecture apprise.

## 2026-08-21 — X9 : pas de falaise à 80 faits — d² est un problème théorique

- **Config** : défauts X1b, GPT-2 ; pool de 80 gabarits combinatoires distincts
  (`eval/pool.py`, 16 possesseurs × 20 entités × 5 verbes) + 80 secrets ;
  `eval/capacity.py`, N ∈ {5, 10, 20, 40, 80} dans la même M.
- **Résultat** :

  | N | Δlogp | positifs | keysim max (méd) |
  | --- | --- | --- | --- |
  | 5 | +0.781 ± 0.64 | 4/5 | 0.987 |
  | 10 | +0.763 ± 0.45 | 9/10 | 0.994 |
  | 20 | +0.742 ± 0.48 | 18/20 | 1.000 |
  | 40 | +0.702 ± 0.61 | 35/40 | 1.000 |
  | 80 | **+0.710 ± 0.61** | 71/80 | 1.000 |

  Seuil de falaise (50 % de N=5 = +0.391) : jamais approché. corr(Δ, keysim max)
  poolée : −0.04.
- **Conclusion** :
  1. **Pas de falaise jusqu'à 80 faits** (91 % de rétention, 720 writes dans M).
     Le critère pré-enregistré tranche : d² est un problème THÉORIQUE à cette
     échelle — les factorisations de M restent écartées, et la capacité ne bloque
     pas un cortex plus gros (décision « scaling » débloquée côté mémoire).
  2. **Le prédicteur par fait est mort, avec diagnostic** : keysim max sature à
     ~1.0 partout — les gabarits partagent des mots-outils (« is called », « The »)
     dont les clés collisionnent à cos ≈ 1 entre faits… sans nuire au rappel. Les
     collisions sur transitions génériques sont INOFFENSIVES (la delta rule
     converge sur clés identiques, elle n'interfère pas) ; la statistique max
     mesure « un token s'est-il répété », pas « l'indice discriminant
     collisionne-t-il ». La jauge de RÉGIME d'I1 (keysim moyen) reste valide.
  3. Mise en garde X8 : un gate keysim-max risque de s'ouvrir sur du texte
     générique pour la même raison. Alternative en poche si E3 le confirme : la
     force de récupération ‖M·φ(h)‖ comme facteur de pertinence.
- **Suite** : banc d'essai X8 (4 modes × E1/E3/E1c) — en cours.

## 2026-08-21 — X8 banc d'essai : keysim seul gagne, entropie anomalie, E1c révèle une limite

- **Config** : GPT-2, défauts X1b (cap 0.25), 4 modes de gate ; `eval/read_gate.py`.
- **Résultat** :

  | mode | E1 Δexact | E3 (seuil 0.05) | E1c ΔMarseille |
  | --- | --- | --- | --- |
  | none (cap seul) | +0.740 ± 0.80 | +0.0228 ✓ | −0.378 |
  | entropy | +0.736 ± 0.81 | **+0.0997 ✗** | −0.378 |
  | **keysim** | **+0.741 ± 0.81** | **−0.0076 ✓** | −0.374 |
  | two_factor | +0.738 ± 0.80 | +0.1005 ✗ | −0.378 |

- **Conclusion** :
  1. **keysim seul remporte le banc** : E1 intact, E3 ÉLIMINÉ (devient négatif).
     La crainte X9 (saturation sur mots-outils) ne se matérialise pas : les états
     d'un texte neutre ne matchent pas les clés d'un autre domaine — la saturation
     X9 n'existait qu'entre gabarits similaires du même régime.
  2. **Anomalie entropie, non résolue** : E3 *pire* qu'aucun gate (+0.0997 vs
     +0.0228) alors qu'un gain multiplicatif g ≤ 1 devrait réduire l'injection —
     contre-intuitif, à investiguer avant tout usage du facteur entropie. Défaut
     structurel relevé au passage : le gate lit l'entropie DÉCALÉE d'un pas (celle
     de la prédiction du token courant — la distribution en cours de production
     n'existe pas encore au moment de la lecture, œuf-et-poule du forward). Le
     two_factor hérite des deux problèmes via le soft-OR.
  3. **E1c échoue dans TOUS les modes** (ΔMarseille ≈ −0.38, ΔParis ≈ −0.51 : M
     aplatit les deux, ne hisse rien). Ce n'est pas un échec du gate mais du
     MÉCANISME : sans composante directionnelle (X7), M ne peut pas renverser un
     prior confiant. La correction de fait sous confiance erronée est hors de
     portée de la v1 — rejoint la piste « rappel directionnel » (v2+).
- **Verdict provisoire** : keysim = candidat par défaut, sous réserve du test
  « régime agressif rouvert » (gate + λ2/cap 0.5 — en cours) et d'un E2.
- **Suite** : ajouts roadmap intégrés le même jour (vision engram-par-projet →
  README ; E4 conventions ; LoRA v2 basse priorité actée par le verdict X9).

## 2026-08-21 — X8 validation : le gate rouvre le régime agressif — sauf E2 SmolLM2

- **Config** : gate keysim + λ=2.0/cap=0.5 (nouveaux défauts), GPT-2 et SmolLM2.
- **Résultat** :

  | Mesure | GPT-2 | SmolLM2 (layer 16) |
  | --- | --- | --- |
  | E1 | **+1.353 ± 1.58** (vs +0.740 ancien conforme) | **+0.755 ± 0.94** (vs +0.494) |
  | E3 | **−0.014 ✓** | **+0.003 ✓** |
  | E2 RFC | interaction +0.002, mais **absolu −0.055 sur les DEUX moitiés** | interaction +0.018, **absolu +0.03/+0.05 : ÉCHEC** |

- **Conclusions** :
  1. **Cibles E1/E3 dépassées sur les deux modèles, sans recalibration** : le gate
     élimine le dommage inter-domaines (E3) et rend le régime agressif au signal
     (+83 % GPT-2, +53 % SmolLM2 vs anciens points conformes).
  2. **Subtilité de métrique E2 (GPT-2)** : l'interaction tombe à ~0 parce que M
     aide DÈS la première moitié (−0.055 absolu partout) — plus de coût
     d'échauffement, donc plus de différentiel. La métrique honnête devient le
     bénéfice absolu ; l'interaction était conçue pour une mémoire qui paie avant
     de gagner.
  3. **Échec E2 SmolLM2 au régime agressif** : en domaine, le gate keysim est
     ouvert en permanence (les clés viennent du document lui-même) — il ne protège
     que HORS domaine. SmolLM2 subit donc λ=2/cap=0.5 non modulé en intra-domaine,
     que ses distributions pointues encaissent mal. C'est la signature du **second
     facteur manquant** (moduler par la confiance intra-domaine — le rôle prévu du
     facteur entropie, disqualifié pour anomalie). Le two_factor reste motivé,
     suspendu à la résolution de l'anomalie entropie.
  4. Coût du gate mesuré au passage : ratio paraphrases 0.68 → 0.38 (la
     sélectivité se paie en généralisation ; bouton `gate_keysim_mid`).
- **Verdict X8** : gate keysim RETENU (défauts : gate=keysim, λ=2, cap=0.5) —
  2 cibles sur 3 dépassées ; la 3ᵉ (E2 SmolLM2) documentée en échec au régime
  agressif, balayage caps intermédiaires en cours pour situer le point E2
  par-modèle. E4 est débloquée (X8 tient sur E1/E3).
- **Complément (caps intermédiaires E2 SmolLM2, gate actif)** : λ2/cap0.25 →
  interaction +0.001, absolu ≈ parité ; λ1/cap0.25 → interaction −0.002, absolu ≈
  parité. Sur SmolLM2, AUCUN point testé ne donne de gain E2 absolu net — le
  meilleur régime atteint la parité. Lecture : la RFC est largement DANS la
  compétence d'un modèle entraîné sur 11T tokens (NLL 1.88 vs 2.33 pour GPT-2) —
  la loi du prior encore : M aide où le cortex est faible. **E2-sur-RFC sature à
  mesure que les modèles s'améliorent** ; pour un cortex fort, le test pertinent
  du contexte diffus est un texte réellement étranger à son entraînement — c'est
  précisément E4 (conventions d'un projet spécifique). Convergence propre : la
  prochaine mesure utile est E4, doublement motivée.
- **Suite** : E4 ; anomalie entropie à investiguer (préalable au two_factor).

## 2026-08-21 — E4 : critère non atteint à cette échelle — le juge est en cause

- **Config** : défauts X8 ; `eval/conventions.py` — CLAUDE.md d'engram (~2.8k
  tokens) streamé en chunks, gating normal ; 10 paires minimales ; contrôles :
  autre-projet (spécificité), E3, mode force (sensibilité).
- **Résultat** :

  | Mesure | GPT-2 | SmolLM2 |
  | --- | --- | --- |
  | discrimination M vierge | **−0.135** (inversée !) | **−0.131** (inversée !) |
  | gain avec M conventions | +0.020 (6/10) | +0.012 (6/10) |
  | gain contrôle autre-projet | +0.004 ✓ | −0.013 ✓ |
  | E3 | −0.049 ✓ | +0.008 ✓ |
  | mode force (GPT-2) | gain +0.028 mais contrôle +0.036 : **spécificité PERDUE** | — |

- **Verdict selon les critères pré-enregistrés** : gain non significatif
  (+0.01/+0.02 pour σ inter-paires ~0.12, N=10) → **le critère de succès n'est
  pas atteint à cette échelle, et on le note** — comme le protocole l'exigeait.
- **Mais l'échec a une structure qui désigne le coupable** :
  1. **Le juge est cassé avant l'expérience** : les DEUX cortex trouvent les
     phrases violantes PLUS probables que les conformes (baseline −0.13) — un
     modèle qui ne peut pas juger la conformité ne peut pas montrer un gain de
     jugement, quelle que soit M. GPT-2 et SmolLM2 sont trop faibles en français
     technique. Le juge naturel de E4 est un cortex multilingue — Qwen2.5-1.5B,
     déjà sur la feuille de route.
  2. **Par paires, M fait exactement ce qu'on sait d'elle** : les paires
     quasi-verbatim du document (« clear_context vide le cache KV… » +0.095,
     « reset_memory remet M à zéro… » +0.178, cortex/hippocampe +0.148) gagnent
     nettement et spécifiquement ; les paires normatives abstraites (« tout
     hyperparamètre passe par EngramConfig ») perdent. Mémoire associative
     diffuse, pas inférence normative — cohérent avec 0.38–0.68.
  3. **Découverte collatérale (mode force, GPT-2)** : en écriture forcée, le
     contrôle autre-projet gagne PLUS que le document cible — le gating par
     surprise n'est pas une économie, c'est ce qui préserve la SPÉCIFICITÉ de la
     mémoire (écho direct de l'ablation A2).
- **Affinements avant de rejouer E4 sur un juge valide** : (a) scorer la NLL au
  token décisif de chaque paire (fp32/fp16…) plutôt qu'en moyenne de phrase —
  le signal est dilué ; (b) cortex Qwen2.5-1.5B (d=1536, ~3.1 Go fp16, tient sur
  la 3060). La vision par-projet n'est ni validée ni tuée : elle est EN ATTENTE
  D'UN JUGE.
- **Suite** : E4b (token décisif) + cortex Qwen = le même run, prochaine étape
  naturelle.

## 2026-08-21 — E4s : le jugement de conformité résiste à la mémoire (un seul mur)

- **Config** : `eval/conventions_simple.py` — conventions ARBITRAIRES d'un projet
  fictif (Zephyr) en anglais simple, scoring au token décisif, requêtes
  reformulées ; contrôle signé : document Boreas aux conventions OPPOSÉES ;
  8 variantes (2 modèles × gating/force × gate on/off × docs avec/sans négation).
- **Résultat (gain de discrimination vs M vierge, critère : > 0)** :

  | Variante | GPT-2 | SmolLM2 |
  | --- | --- | --- |
  | gating (base) | −0.076 (5/10) | −0.391 (2/10) |
  | force | −0.355 | −0.694 |
  | force + gate none | — | −0.807, **E3 +0.57 !** |
  | docs positifs (sans négation) | −0.068 | −0.386 |

  Boreas (contrôle signé) : TOUJOURS plus bas que Zephyr (différentiel +0.07 à
  +0.13, 8-10/10 paires ↓) — la trace spécifique existe, correctement signée.
- **Conclusions** :
  1. **Critère échoué dans les 8 variantes** — et cette fois le juge est hors de
     cause (baseline = pur biais de prior, anglais simple, token décisif). C'est
     un négatif de MÉCANISME : la teinte diffuse de M ne se traduit pas en
     préférence token-niveau pour la convention, même dans le cadre le plus
     favorable. Ce qui existe : une trace spécifique faible (~+0.05 par direction,
     le différentiel Zephyr/Boreas), noyée sous une dérive non spécifique.
  2. **Hypothèse « aveuglement à la négation » : RÉFUTÉE** — les docs sans
     négation ne changent rien. (L'argument théorique reste vrai — une mémoire
     associative encode la co-occurrence, pas l'affirmation — mais il n'explique
     PAS ce résultat.)
  3. **La dérive non spécifique n'est pas expliquée** : candidat naturel = la
     traction vers le marginal (X7 : perturbation d'état → distribution moins
     conditionnelle → tokens fréquents favorisés, or nos tokens conformes sont
     souvent les rares) — mais le pattern par-paire est brouillé, on le note
     comme candidat, pas comme conclusion.
  4. **Découverte collatérale majeure** : sans gate, E3 explose à +0.57 nats/token
     au régime λ2/cap0.5 — le gate keysim ne « réduit » pas la taxe, il retient
     un torrent. Sa valeur est bien plus grande que mesurée à cap 0.25.
  5. **Synthèse — un seul mur, plusieurs visages** : échec E1c (renverser un
     prior confiant), verrou top-10, échec E4/E4s (préférence token-niveau) ont
     la même racine, mesurée en X7 : **la lecture n'a aucune composante
     directionnelle** (cos(r, W_U) ≈ 0). M déplace l'état, ne désigne pas de
     token. Tout usage exigeant une préférence token-précise passera par le
     rappel directionnel (v2 : valeurs en espace d'unembedding ou tête de
     lecture apprise) — désormais LE chantier prioritaire de v2, devant la
     consolidation.
- **Statut vision par-projet** : sévèrement dégradée à cette échelle — E4-dur
  attend encore Qwen (juge multilingue) mais E4s retire l'excuse du juge. Ce qui
  survit : l'association quasi-verbatim (E4 §2) et le régime E1/E2 (rappel et
  adaptation), pas le jugement de conformité.
- **Suite** : Qwen (téléchargement en reprise) pour E4-dur — dernier test avant
  verdict final ; puis arbitrage v2 (directionnel d'abord).

## 2026-08-21 — Campagne Qwen2.5-1.5B : l'amorçage s'amplifie avec le cortex

- **Config** : Qwen2.5-1.5B (d=1536, 28 couches → layer 14), défauts X8 ; fumée E1
  (2 secrets), E4s, E4-dur. Le cortex avale l'architecture Qwen2 sans modification.
- **Résultat** :
  1. **E1 (fumée, N=2) : +4.18 ± 1.76 nats** (swordfish +5.42, rang 14770 → 2317
     — rang ÷6, le plus grand mouvement jamais mesuré). *(Requalifié à l'audit
     du 2026-08-21 — la version initiale promouvait une « série qui
     s'amplifie ».)* La « série » +0.74 → +0.85 → +4.18 mélange trois régimes
     (les deux premiers points = anciens points conformes SANS gate ; le
     troisième = défauts X8, régime où GPT-2 fait +1.353) et repose sur 2
     secrets : c'est une fumée prometteuse, pas une loi d'échelle. L'hypothèse
     « l'amorçage s'amplifie avec la qualité du cortex » reste OUVERTE — à
     trancher par un E1 à 10 secrets, même régime, mêmes couches relatives, sur
     les trois modèles (Q-08 de l'audit).
  2. **E4s : signes conformes, non significatifs** *(retitré à l'audit du
     2026-08-21 — « première double dissociation » était trop fort :
     +0.018/−0.010 à N=10 sans σ rapporté, et deux signes conformes = 1 chance
     sur 4 sous le hasard)* — baseline +0.076 (sain, comme conçu), Zephyr +0.018
     (positif, une première), Boreas −0.010 (négatif comme prédit), E3 ≈ 0. Une
     préférence token-niveau sur un juge fort reste À ÉTABLIR — les signes sont
     du bon côté des deux contrôles, rien de plus. Writes = 33 (stockage mince,
     seuil 4.0 sur de l'anglais simple — piste : seuil adaptatif à la NLL moyenne
     du cortex).
  3. **E4-dur : échec même sur Qwen** (gain −0.011, 3/10) — ET baseline encore
     inversé (−0.148) : même un juge multilingue préfère nos phrases « violantes ».
     Diagnostic : l'instrument est cassé — la NLL moyenne de phrase mesure la
     FLUIDITÉ, pas la conformité (nos violantes sont du français plus naturel).
     **E4-dur est retiré au profit d'E4s** (token décisif, baseline ≈ 0 par
     construction) comme instrument de la famille conventions.
- **Note technique** : keysim d'écriture ≈ 0.14 sur Qwen (vs ~0.23 GPT-2) — les
  statistiques de similarité dépendent de d ; `gate_keysim_mid=0.6` est calibré
  sur GPT-2 et mérite un re-balayage par modèle (E1 fonctionne, donc le gate
  s'ouvre sur les requêtes pertinentes, mais le point d'opération est à vérifier).
- **Conclusion** : le canal d'amorçage SEMBLE valoir cher sur un bon cortex (E1,
  fumée N=2 — Q-08 pour le confirmer), le canal de préférence token-niveau reste
  le mur (E4s +0.018, n.s.) — la hiérarchie des chantiers ne change pas : V2-D
  (canal de sortie directionnel) d'abord.

## 2026-08-21 — Diagnostics : théorie à quatre observations, non-monotonie des gates

- **Diagnostic 1 — anomalie entropie** (`eval/gate_anomaly.py`, 2×2 : mode ×
  lecture-pendant-injection, GPT-2, régime λ2/cap0.5) :

  | mode | inject read on | inject read off | ‖M‖ |
  | --- | --- | --- | --- |
  | none | +0.1354 | +0.1354 | 65.06 |
  | entropy | +0.2520 | +0.2520 | 65.06 |

  (Contrôle de cohérence : none à λ2/cap0.5 reproduit le +0.1354 de X1 à
  l'identique.)
  - **Hypothèse « boucle lecture→écriture » : RÉFUTÉE** — M est identique au
    centième près dans les 4 conditions, l'anomalie persiste à M égale. (Notre E3
    mesure déjà writes gelés ; le canal injection est nul aussi.)
  - **Résolution — la fonction de dommage est NON MONOTONE en g** : trois points
    la dessinent — g≈0.02 (keysim/neutre) → E3 ≈ 0 ; g≈1 (none) → +0.135 ;
    g intermédiaire/mixte (entropy) → +0.252. Scaler un vecteur de lecture quasi
    constant (voir diag. 2) ne donne pas une injection « plus douce » mais un
    déplacement d'état DIFFÉRENT, que les couches avales gèrent plus mal.
  - **Règle de design consignée : les gates de lecture doivent être quasi
    BINAIRES.** keysim l'est de fait (tau 0.05 → g ≈ 0 ou 1) — c'est
    rétrospectivement une des raisons de sa victoire au banc X8 ; l'entropie
    (tau 0.5) produisait exactement la zone toxique intermédiaire. Réhabiliter le
    facteur entropie = le durcir (seuil franc), pas le retirer. two_factor à
    rejouer en version binaire.
- **Diagnostic 2 — traction vers le marginal** (`eval/marginal_pull.py`, GPT-2,
  120 positions de texte neutre, M chargée d'un fait) :
  - **corr(W_U·r, log-fréquence unigramme) = +0.484 ± 0.002** — la lecture
    projette systématiquement vers les tokens FRÉQUENTS, avec une direction quasi
    CONSTANTE d'une position à l'autre (σ = 0.002 !). corr(W_U·r, logits
    courants) = +0.6. Norme de r : 24.8 brut vs 0.48 sous gate keysim (×50
    d'atténuation sur texte neutre — le mécanisme exact de la protection E3).
  - **La dérive non spécifique d'E4s est expliquée** : les tokens conformes
    étaient souvent les options rares ; la traction vers le fréquent les pénalise
    mécaniquement. **Quatrième observation ramenée à X7** (avec la loi du prior,
    la taxe E3, le verrou top-10) : ce n'est plus un bug, c'est une théorie —
    la lecture de M est un vecteur quasi fixe orienté « prior générique », et
    tout son bénéfice passe par la recomputation aval.
- **Suite** : X8.1 (gates binaires, two_factor durci) ; V2-D reste le chantier
  prioritaire — il attaque la racine que ces deux diagnostics viennent de
  confirmer.

## 2026-08-21 — X8.1 : protocole pré-enregistré (test CAUSAL de la loi 2)

- **Contexte** : la loi « les gates de lecture doivent être binaires » est
  observationnelle — entropie lisse (tau 0.5) perd, keysim raide (tau 0.05) gagne,
  mais les deux gates diffèrent AUSSI par leur signal. Intervention propre : même
  facteur entropie, même seuil (mid 2.0), seule la PENTE change (tau 0.5 → 0.02).
- **Instrumentation** : fraction des lectures dans la zone toxique g ∈ [0.05, 0.95]
  par config (collecteur `gate_log`) — relie l'intervention au mécanisme.
- **Prédictions posées avant mesure** (E3 GPT-2, régime λ2/cap0.5, M contrôlée ;
  repères : none = +0.135, entropy tau 0.5 = +0.252, keysim ≈ 0) :
  P1. entropy tau 0.02 → E3 ≤ +0.135 (l'excès +0.117 disparaît : il était
      entièrement dû aux g intermédiaires). La loi passe de corrélation à
      causalité si P1 tient.
  P2. La fraction toxique s'effondre avec tau 0.02 (>50 % → <10 %), et l'excès
      d'E3 est ordonné par cette fraction à travers les configs.
  P3. two_factor durci (tau_H 0.02) : E3 ≈ keysim (le OR avec un facteur binaire
      n'introduit plus de zone molle) ET E1 ≥ keysim seul (rien perdu).
- **Résultat** :

  | config | E3 | zone toxique | E1 exact |
  | --- | --- | --- | --- |
  | none | +0.1354 | — | |
  | entropy tau 0.5 | +0.2520 | 27.6 % | |
  | entropy tau 0.02 | **+0.2745** | **0.7 %** | |
  | keysim (réf) | −0.0136 | 0.0 % | +1.353 |
  | two_factor dur | +0.2738 | 0.7 % | +1.361 |

- **Verdict : P1 et P2 RÉFUTÉES par intervention propre** — la binarisation
  fonctionne mécaniquement (zone toxique 27.6 % → 0.7 %) et E3 EMPIRE. La théorie
  « g intermédiaire = poison » est morte. P3 réfutée côté E3 (E1 intact).
- **Nouvelle hypothèse, désignée par le tableau** : avec un gate dur, g ∈ {0, 1} —
  si le dommage était par-token, E3(dur) ≤ E3(none) ; il est au DOUBLE. Le dommage
  passe par le cache KV : un gate qui BASCULE token par token remplit le cache
  d'états incohérents (biaisés/non mélangés). Le poison n'est pas l'amplitude de
  g mais sa **volatilité temporelle**. keysim gagne parce que son signal est une
  propriété du TEXTE (stable sur un domaine), pas du token. Loi 2 réécrite en
  candidate : « gater sur des signaux LENTS ».
- **Test causal scellant, pré-enregistré** : gate forcé en créneaux, duty 50 %
  constant, seule la période varie — k ∈ {1, 4, 16, 64} tokens. Prédiction P4 :
  E3 décroît de façon monotone avec k (moins de bascules à duty égal) ; à grand k,
  E3 → ~50 % du dommage de none. Si P4 tient, l'incohérence temporelle est
  démontrée par intervention.

## 2026-08-21 — X8.1b + P5 : anomalie résolue — le dommage vit aux positions incertaines

- **X8.1b, créneaux forcés** (`eval/gate_cycle.py`, duty 50 %, période k variable,
  indépendant du contenu) : none +0.1354 ; k=1 → **+0.0393** ; k=4 → +0.0396 ;
  k=16 → +0.0800 ; k=64 → +0.0546. **P4 RÉFUTÉE** : aucune monotonie en k, et la
  bascule maximale (k=1) est la MEILLEURE — sous-proportionnelle au duty. La
  théorie « incohérence temporelle du cache KV » meurt à son tour.
- **Par élimination** : les créneaux sont indépendants du contenu ; le gate
  entropie ouvre précisément sur les tokens incertains. Hypothèse P5 : le dommage
  par lecture est concentré sur les positions FRAGILES.
- **P5, test par position** (343 positions de texte neutre, M chargée, gate none) :
  **corr(dommage, entropie baseline) = +0.394** ; partage à la médiane :
  positions confiantes **−0.099** (l'injection y aide légèrement !), incertaines
  **+0.364**. Confirmée sans ambiguïté.
- **L'anomalie entropie est entièrement résolue** — et chaque observation tombe
  en place : tau 0.5 (+0.252) = lectures pondérées vers l'incertain ; tau 0.02
  (+0.274, pire !) = lectures EXCLUSIVEMENT aux positions incertaines, le ciblage
  le plus pur des points fragiles ; créneaux (≤ proportionnel) = ciblage nul ;
  keysim (≈0) = lectures corrélées à la pertinence mémoire, décorrélées de la
  fragilité sur texte neutre.
- **Loi 2, forme finale (trois interventions : pente, période, ciblage)** :
  le dommage de lecture est concentré aux positions incertaines du cortex — un
  gate déclenché par l'incertitude est donc **adversarial par construction** (il
  lit exactement là où lire coûte). Gater la lecture côté MÉMOIRE (pertinence du
  match), jamais côté détresse du cortex. Le two_factor est enterré
  définitivement — et le cas « confiance erronée » qu'il visait relève de V2-D,
  pas d'un gate.
- **Note pour le papier** : l'arc entier (anomalie → 2 fausses théories réfutées
  par intervention → résolution par position) est le chapitre méthode rêvé — la
  loi finale (« ne lisez PAS quand le modèle hésite ») est contre-intuitive pour
  l'intuition RAG, mais c'est la prédiction du modèle cholinergique de Hasselmo
  (ACh haute en régime de nouveauté : encodage favorisé, rappel récurrent
  supprimé) — un ancrage bio, pas une curiosité (précision d'audit, 2026-08-21) ;
  chaque théorie morte est documentée avec son expérience tueuse.
- **Suite** : arc écriture (« Priming, not recall ») — v1 est close, X8.1 scellé.

## 2026-08-21 — Audit externe (/ai-lab-audit) + COR-02 : le 0.68 gagne son incertitude

- **Contexte** : audit en lecture seule, 5 lentilles + validation adversariale —
  39 findings bruts, 31 confirmés, **22 corrections (COR-01..22) toutes traitées
  ce jour** et 8 questions de recherche priorisées (6 retenues + 2 en réserve).
  Détail et statuts : `AUDIT-lab.md`. Aucune fuite dans le banc E1/E2/E3 — le
  contrôle D7 tient, le tableau des poids n'est pas invalidé.
- **Requalifications notables (éditées en place, marquées dans les entrées)** :
  la « série Qwen qui s'amplifie » redevient une fumée (N=2, trois régimes
  mélangés — Q-08 pour trancher) ; le pari v1.2 Q3 est TENU au point conforme ;
  E4s Qwen retitrée « signes conformes, non significatifs » ; corrélations
  keysim par fait : n.s. à N=10 ; la « loi 2 contre-intuitive » est la
  prédiction du modèle cholinergique de Hasselmo. Docs remis au 2026-08-21
  (défauts X8, X10 gelé, décision X6 écrite, D10 au tableau, V2-D prioritaire).
- **COR-02 — recalcul du ratio de généralisation** :
- **Config** : X1b exact (dg=8192/64, layer=6, λ=1.0, η=0.2, cap=0.25,
  read_gate=none) — rejoue E1b du 2026-08-20, 10 secrets × 4 questions.
- **Run** : script d'audit (bootstrap 10 000 tirages, resampling par secret).
- **Résultat** : run DÉTERMINISTE confirmé — exact +0.740 ± 0.799, ratio de
  moyennes **0.684**, reproduction au millième. **IC 95 % bootstrap du ratio de
  moyennes : [0.56, 0.99]** ; **par secret : médiane 0.59, IC 95 % [0.45,
  0.75]**. Deux pathologies de la statistique par-secret : marmalade (exact
  −0.011 ≈ 0 → ratio divergent) et tambourine (négatif partout → ratio +0.60
  trompeur) ; 8/10 secrets ont un ratio entre 0.44 et 1.16.
- **Conclusion** : la généralisation graduée est réelle (l'IC exclut largement
  zéro) mais « 0.68 » était un ratio de deux moyennes bruitées sans
  incertitude — le citer désormais **« ~0.6–0.7 (IC large, N=10) »**. Pour
  resserrer : plus de secrets (le pool X9 en a 80), pas plus de bootstrap.
- **Suite** : questions de l'audit dans l'ordre Q-01 → Q-03 → Q-02 → Q-04 →
  Q-06 → Q-05, chacune via `/lab-run` (AUDIT-lab.md).

## 2026-08-20 — v0 : squelette posé

- **Commit** : (initial)
- **Config** : n/a
- **Résultat** : structure du repo, hippocampe (delta rule + decay + prune + reset),
  hook cortex, boucle engine, deux évals, tests unitaires CPU. Rien n'a tourné sur GPU.
- **Conclusion** : prêt pour le premier run réel.
- **Suite** : `pytest tests/ -q`, puis `eval/fact_injection.py` avec les défauts, puis
  balayage de `layer_index` ∈ {3, 6, 9} et `lam` ∈ {0.1, 0.5, 1.0}.
