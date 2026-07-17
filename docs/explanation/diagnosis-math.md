# PrefScope diagnosis — the math

What every number in the diagnosis pipeline means, and the formula behind it.
Borrows from WIMHF ("What's In My Human Feedback") and *Anatomy of Post-Training*
(the inside-vs-outside contrast and the predicted-vs-actual validation).

Notation: a **battle** is a prompt with two completions $A$, $B$ from models
$m_A, m_B$. $e_A, e_B \in \mathbb{R}^D$ are their L2-normalized response
embeddings. $y = P(A\ \text{preferred}) \in \{0, \tfrac12, 1\}$ is the label
(judge `y_judge` or human `human_pref`).

---

## 1. The difference lens (recap)

The SAE is trained on the **contrast vector** $e_A - e_B$ (WIMHF default). Its
encoder gives a sparse signed code

$$
z = f_{\text{SAE}}(e_A - e_B) \in \mathbb{R}^M ,\qquad
z_f > 0 \;\Leftrightarrow\; A\ \text{expresses concept } f\ \text{more than } B .
$$

The SAE is **non-linear** (BatchTopK threshold), so

$$
f_{\text{SAE}}(e_A - e_B)\ \neq\ -\,f_{\text{SAE}}(e_B - e_A).
$$

This is why orientation matters and why the bank (§3) projects both directions
explicitly rather than flipping a sign.

---

## 2. Per-model diagnosis: `net_direction`

To diagnose a target model $X$, orient every battle so $X$ is "self": for each
battle involving $X$ define the oriented code

$$
z^{(X)} = f_{\text{SAE}}\!\big(e_{\text{self}} - e_{\text{other}}\big),\qquad
z^{(X)}_f > 0 \;\Leftrightarrow\; X\ \text{over-expresses } f\ \text{vs its opponent.}
$$

Aggregate over $X$'s battles $\mathcal{B}_X$:

$$
\boxed{\ \text{net\_direction}_f(X) \;=\; \underbrace{P\big(z^{(X)}_f > 0\big)}_{\text{self\_more\_rate}} - \underbrace{P\big(z^{(X)}_f < 0\big)}_{\text{self\_less\_rate}}\ }
$$

Note $\operatorname{net\_direction}_f(X) = \mathbb{E}\big[\operatorname{sign}(z^{(X)}_f)\big]$,
since $\operatorname{sign}(z)\in\{-1,0,+1\}$. **Positive** = $X$ does more of
concept $f$ than its opponents; **negative** = a gap.

`outcome_assoc` crosses this with who won (the raw split, kept for back-compat):

$$
\text{outcome\_assoc}_f(X) = \mathbb{E}[\,w \mid z^{(X)}_f > 0\,] - \mathbb{E}[\,w \mid z^{(X)}_f < 0\,],
\qquad w = \text{outcome}\in\{0,\tfrac12,1\}.
$$

Arena has a strong **length bias** — longer answers win — so this raw split partly
measures verbosity. The helps-win signal is therefore length-controlled:
`outcome_assoc_lc` is the per-feature logistic **average marginal effect** (the
`win_relevance_logistic` $\Delta$win-rate, §5) fit on $X$'s battles with the
word-count gap $\ell^\Delta = \text{wc(self)} - \text{wc(other)}$ held fixed. When a
global win-relevance frame is passed in, its length-controlled `delta_win_rate` is
merged as `helps_win` (the global feature-reward weight); the within-model
`outcome_assoc_lc` is the secondary read. A `length_confound` column (correlation of
$\operatorname{sign}(z_f)$ with $\ell^\Delta$) surfaces features that are mostly
verbosity proxies.

Reading: `net_direction < 0` **and** `helps_win > 0` (or `outcome_assoc_lc > 0`) ⇒
$X$ *under*-expresses a concept that *helps* it win even after length control — a gap
worth closing.

---

## 3. The oriented-code bank (pool baseline)

`net_direction` alone is **relative to $X$'s opponents**, not to the field. Some
concepts are rare (or common) for everyone. To say what is *distinctive* about
$X$ we need a pool baseline, so we orient **every battle around each of its two
models**. For each battle ($N$ total) we emit two rows:

| orientation | code $z$ | self | other | win |
|---|---|---|---|---|
| `a` | $f_{\text{SAE}}(e_A - e_B)$ | $m_A$ | $m_B$ | $y$ |
| `b` | $f_{\text{SAE}}(e_B - e_A)$ | $m_B$ | $m_A$ | $1-y$ |

giving a bank $Z \in \mathbb{R}^{2N \times M}$ with a `self_model` tag. Then:

- inside $= \{$ rows with `self_model` $= X\}$ — these reproduce §2's $z^{(X)}$;
- outside $= \{$ rows with `self_model` $\neq X\}$ — the **pool**;
- the `orientation = a` rows reproduce the lens's natural $z_{\text{diff}}$ with
  `win` $= y$, which is exactly what win-relevance (§5) consumes.

---

## 4. Inside-vs-outside Welch contrast (`delta_vs_pool`)

For feature $f$, let $s = \operatorname{sign}(z_f)$. Compare the target's signed
presence against the pool's:

$$
\boxed{\ \Delta^{\text{pool}}_f(X) \;=\; \underbrace{\mathbb{E}\big[s \mid \text{self}=X\big]}_{\text{net\_direction}_f(X)} \;-\; \underbrace{\mathbb{E}\big[s \mid \text{self}\neq X\big]}_{\text{net\_direction\_pool}_f}\ }
$$

Because $\mathbb{E}[s] = \text{net\_direction}$, $\Delta^{\text{pool}}$ is literally
"how much more does $X$ over-express $f$ than the average model." Significance is a
**Welch two-sample $t$-test** (unequal variance) on the two sign-samples
$s_{\text{in}}, s_{\text{out}}$:

$$
t = \frac{\bar s_{\text{in}} - \bar s_{\text{out}}}
{\sqrt{\operatorname{Var}(s_{\text{in}})/n_{\text{in}} + \operatorname{Var}(s_{\text{out}})/n_{\text{out}}}},
\qquad
d = \frac{\bar s_{\text{in}} - \bar s_{\text{out}}}{\sqrt{(\operatorname{Var}(s_{\text{in}}) + \operatorname{Var}(s_{\text{out}}))/2}} ,
$$

with `welch_p` Bonferroni-corrected over the features tested
(`welch_p_bonferroni`). This turns "olmo3 has net_direction $-0.31$ on $f_0$" into
the *distinctiveness* claim "olmo3 under-expresses $f_0$ by $\Delta=-0.31$ vs the
pool, $d=-0.8$, $p<10^{-6}$". When a baseline is supplied, the diagnosis is sorted
by $\Delta^{\text{pool}}$ instead of raw `net_direction`.

---

## 5. Which features do humans reward? (`win_assoc`)

Model-independent, computed on the natural $z_{\text{diff}}$ and $y=P(A\ \text{pref})$
(this is the WIMHF reward question). With $\tilde y = 2y-1 \in\{-1,0,+1\}$:

$$
\text{win\_assoc}_f = \mathbb{E}[\,y \mid z_f > 0\,] - \mathbb{E}[\,y \mid z_f < 0\,],
\qquad
r_f = \operatorname{corr}\big(\operatorname{sign}(z_f),\,\tilde y\big)\ \text{over firing battles.}
$$

`win_assoc` $> 0$ ⇒ the side expressing $f$ more is the side humans prefer.
$r_f$ with Bonferroni $p$ gives the `significant` flag.

---

## 6. Predictive validation (`validate-diagnosis`)

The end-to-end check: a model that under-expresses human-rewarded features should
actually lose more. Define the **predicted advantage**

$$
\boxed{\ \hat s(m) \;=\; \sum_{f} \text{net\_direction}_f(m)\; \cdot\; w_f\ },
\qquad w_f = \text{delta\_win\_rate}_f\ \text{(significant features only)} .
$$

The default weight $w_f$ is the **length-controlled** `delta_win_rate` (the
logistic average marginal effect, §5 / `win_relevance_logistic`), not the raw
`win_assoc`: Arena's length bias means raw associations partly reflect verbosity, so
weighting by the length-controlled AME makes $\hat s$ length-controlled too. Compare
to the **actual win rate** $a(m) = \mathbb{E}[\,w \mid \text{self}=m\,]$ across all
models, and report

$$
R^2 = \operatorname{corr}\big(\hat s,\, a\big)^2 , \qquad \rho = \text{Spearman}\big(\hat s,\,a\big).
$$

This is **exploratory and associational across a small number of models** (n ≈ 10),
not a high-powered estimate. Read it through:

- a **bootstrap CI** (`*_r2_ci_lo/_hi`, `*_spearman_ci_lo/_hi`): resample the model
  rows with replacement (2000×) and take the 2.5/97.5 percentiles;
- a **permutation p-value** (`*_r2_perm_p`): shuffle the actual win rate across
  models (2000×) and count how often the permuted $R^2$ meets the observed one,
  $(\#\{R^2_{\text{perm}} \ge R^2_{\text{obs}}\}+1)/(B+1)$.

A single $R^2$ point estimate over ~10 points is fragile; the CI width and the null
tail are how to judge whether the association is real.

**Leave-one-model-out** (`--loo`) refits each $w_f$ on battles *not involving* $m$
(the bank's `orientation = a` rows), so $\hat s(m)$ never sees $m$'s own data. When
the bank carries a per-battle `length` (the word-count gap, persisted by
`build-bank`), the LOO refit is itself length-controlled (logistic AME) and
`loo_length_controlled` is `True`; on a legacy bank without `length` it falls back to
the raw association and reports `False`.

---

## 7. Command flow

```
build-lens --dump-embeddings EMB ...           # e_a/e_b/meta + z_diff + lens
build-bank --lens-dir LENS --from-embeddings EMB \
           --label human --corpus CORPUS --out BANK     # §3
win-relevance --lens-dir LENS --corpus CORPUS --out WR  # §5
diagnose --lens-dir LENS --model X --bank BANK ...      # §2 + §4 (delta_vs_pool)
validate-diagnosis --bank BANK --win-relevance WR --loo --out VAL   # §6
```

All of §3–§6 are pure NumPy/SciPy over cached code matrices — no GPU, no
re-embedding.
