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

## 2026-08-20 — v0 : squelette posé

- **Commit** : (initial)
- **Config** : n/a
- **Résultat** : structure du repo, hippocampe (delta rule + decay + prune + reset),
  hook cortex, boucle engine, deux évals, tests unitaires CPU. Rien n'a tourné sur GPU.
- **Conclusion** : prêt pour le premier run réel.
- **Suite** : `pytest tests/ -q`, puis `eval/fact_injection.py` avec les défauts, puis
  balayage de `layer_index` ∈ {3, 6, 9} et `lam` ∈ {0.1, 0.5, 1.0}.
