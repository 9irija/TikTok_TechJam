# Project Story

_Devpost "Story" section — inspiration, what we learned, how we built it,
challenges we ran into. Markdown, with inline LaTeX for the math that
actually mattered while building this._

## Inspiration

We went into TikTok TechJam wanting two things at once: a real shot at a
hard, open-ended challenge, and a genuine excuse to get our hands dirty
with machine learning itself — not just calling an API, but actually
understanding how the frameworks differ, how a model's parameters move
during training, and why an ML engineer's day looks the way it does.

Challenge 2 turned out to be exactly that excuse. The premise is almost
recursive: build an agent that does what an ML research engineer does —
read the problem, inspect the data, engineer features, train and tune a
model, evaluate it, then reflect and revise — and have it run that loop
*on itself*, autonomously, against a real recommendation benchmark
(KuaiRand-Pure, 1.4M interactions from Kuaishou's short-video feed). To
build that agent well, we had to actually understand every stage of the
loop we were automating — you can't design a Diagnosis Engine that
recognizes *why* a model regressed without understanding backpropagation,
overfitting, and what a validation curve is supposed to look like when
it's healthy.

We also drew directly on three papers the problem statement pointed us
toward — **AIDE** (tree-structured search over ML solutions beats linear
iteration by roughly $4\times$ on MLE-Bench), **AI-Scientist-v2**
(separating the *decision* of what to try next from the LLM call that
writes the code), and **MLE-Bench** itself (the evaluation paradigm this
whole challenge is modeled on). Reading those before writing a line of
code changed the shape of what we built — we didn't want a chatbot that
edits a script in a loop; we wanted a system with real memory of its own
experiments.

## What it does

Given the KuaiRand-Pure benchmark and a fixed baseline, the agent
autonomously reproduces that baseline, then iterates — proposing,
training, evaluating, and diagnosing new candidates — until validation
score converges by the organizer's own rule ($\varepsilon = 0.002$,
$N = 3$: converged when the best score hasn't improved by more than
$\varepsilon$ over the last $N$ iterations). Every iteration's hypothesis,
code diff, metrics, and any error/recovery event gets logged automatically
and rendered into a clickable Research Map. The result that shipped —
`deepfm_mtl_v1`, a multi-task DeepFM — beats the official baseline by
**+0.0028 primary-metric delta on the hidden test set**, 3-seed verified,
and that edge holds (and even grows) on TikTok's own unbiased,
randomized-exposure log — real evidence it isn't an artifact of the
platform's normal serving bias.

## How we built it

We built this in layers, each one earning the right to add the next.

**Foundation first.** Before any modeling, we built the boring-but-critical
plumbing: an Evaluator Wrapper that calls the organizer's own scoring code
directly instead of reimplementing it (the single easiest way to silently
score yourself wrong), and a Convergence Detector that reads the
epsilon/N rule live from the organizer's own published file rather than
hardcoding it. Only once that harness self-checked correctly against a
known reference score did we trust anything it reported afterward.

**The models, hand-derived first.** Our first real model was a
Factorization Machine, ported faithfully from the official baseline:

$$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

— a global bias, a linear term per feature, and a learned low-rank
interaction term that lets sparse categorical features (`user_id`,
`video_id`, `author_id`, ...) cross with each other without an explosion
of parameters. We hand-derived and hand-coded its backward pass in plain
NumPy rather than reaching for a framework immediately — genuinely useful
for understanding what autograd is actually doing under the hood once we
did start using it. DeepFM extended this with a small MLP "deep" component
reading the same embeddings, still hand-rolled.

**PyTorch, added deliberately, not by default.** We kept the numpy-only
philosophy for as long as it made sense, and only reached for PyTorch when
autograd was a genuine win, not a shortcut — specifically, a five-headed
multi-task network (one main loss, four auxiliary engagement signals
sharing one embedding table) whose backward pass has five different
gradients merging back into a shared trunk. Hand-deriving that by hand
would have been real effort spent on the wrong problem. That one model —
`deepfm_mtl_v1` — turned out to be the actual project-best, and it landed
there on its very first attempt.

**A memory, not a log.** Rather than a flat list of "tried X, got Y," we
built a persistent, tree-structured Research Map (directly inspired by
AIDE) where every experiment is a node with a parent, an edge type
(`draft` / `improve` / `debug`), and a diagnosis. A Metric-Aware Diagnosis
Engine reads the *pattern* across both scored metrics — GAUC (per-user
ranking quality) and nDCG@5 (top-of-list precision, discounted by
position: $\text{DCG@5} = \sum_{i=1}^{5} \frac{2^{\text{rel}_i}-1}{\log_2(i+1)}$)
— not just whether the primary number went up, so "GAUC up, nDCG@5 down"
gets tagged a *ranking trade-off*, not lumped in with a clean win or a
clean loss. That distinction mattered more than once.

**Then we let an LLM drive.** With that memory and diagnosis layer in
place, we wired in Google Gemini as a genuine Research Strategist: it
reads the live Research Map — every prior hypothesis and result — and
proposes the next experiment on its own, validated against our model
registry before anything runs. On its first real round it independently
noticed an `overfitting_risk` flag our own hand-authored logic had missed
and proposed exactly the fix, which became a real, verified improvement.

**Then we searched, honestly, for more.** Once we had one real win, we
spent the rest of the project trying to beat it — and mostly failing, on
purpose testing everything we could: five variants of pairwise BPR loss
($-\ln\sigma(\hat{y}_{ui} - \hat{y}_{uj})$), LambdaRank (weighting each
pair by exactly how much fixing it would move nDCG@5), focal loss
($FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$, down-weighting examples
the model already gets confidently right), listwise softmax, a new
architecture (DCNv2), four different ensembling methods, checkpoint
averaging, and — the one genuinely non-standard idea — initializing
embeddings via LightGCN-style graph propagation over the user-video
interaction graph instead of random init. Almost all of it came back
negative or a tie. We reported every single one of those results exactly
as measured.

## Challenges we ran into

**The task definition itself was ambiguous, at first.** The problem
statement's own prose said one thing (`click` as the label, NDCG@10/
Recall@50); the Starter Kit's actual pinned code said another (`long_view`,
GAUC/nDCG@5). We had to decide — and defend — which one to trust
(the code, since it's what actually gets run) before writing a single
model. The organizers later updated the problem statement to match the
code directly, which was a genuinely good feeling to see confirmed.

**Windows fought us more than the ML did.** `multiprocessing` needs
`spawn` mode on Windows (no `fork`, no `SIGALRM`), which meant real
subprocess-isolation code for our Failure Recovery system, not a simple
`try/except`. The organizer's own submission validator crashes on a
default Windows console because of a Unicode checkmark in its success
message — a real, confusing "is my file broken?" moment the first time we
hit it, resolved by realizing the crash happened *after* every actual
check had already passed. LightGBM's native library outright refused to
load under a Windows security policy on our primary machine; a teammate
on macOS didn't have that problem at all.

**Silent bugs are the dangerous kind.** Twice, we found bugs that had been
silently wrong for a while without ever throwing an error: a report
generator whose import ordering meant two of its own sections had *never*
rendered correctly, and — more seriously — a "current best" headline that
had been silently stuck reporting an old result for days because nothing
cross-checked it against our own persistent memory. Neither crashed
anything, which is exactly what made them dangerous. We only found them by
actually reading the generated output critically instead of trusting that
"it ran without an error" meant "it's correct."

**Knowing when to stop looking for a bug and accept a real result.** After
roughly twenty further attempts beyond our first win all failed to beat
it, the hard part wasn't building the twenty-first — it was recognizing
that the *pattern itself* was the finding: this benchmark's learnable
signal, at this data volume, looks substantially extracted by a multi-task
training objective, and no amount of architecture or loss-function
tinkering was going to change that. Reporting a wall of negative results
honestly, instead of quietly going back for a twenty-second attempt and
hoping for a better story, was its own kind of discipline.

## Accomplishments that we're proud of

**A real, verified improvement, not a lucky seed.** +0.0028 primary-metric
delta over the official baseline on the hidden test set sounds small until
you look at the ceiling: the metrics don't span $[0,1]$ — a perfect
ranking only reaches primary $0.8645$ (27.1% of users have no positive
label at all, forcing their nDCG to 0 for any model), so our result is a
real, meaningful bite out of the actual attainable headroom, not a rounding
error. We didn't trust a single lucky run for it either — every number we
call "the result" is 3-seed verified, and we independently re-confirmed it
holds (and even grows) on TikTok's own unbiased, randomized-exposure log,
a genuinely different data distribution than the one we trained on.

**Genuine autonomy, not a rubber stamp.** Watching our LLM Research
Strategist read its own experiment history and independently flag an
overfitting pattern our own hand-authored logic had missed — then propose
exactly the right fix, which became a real, verified win — was the moment
this stopped feeling like a script and started feeling like a colleague.

**Catching real bugs before they mattered.** We found and fixed two
genuinely silent bugs (a report generator whose sections had never once
rendered correctly; a "current best" headline stuck on a stale result for
days) purely by reading our own output critically instead of trusting that
"no crash" meant "correct." Neither would have shown up in a casual demo.

**Robustness we actually tested, not just claimed.** Our Failure Recovery
system survived a genuine out-of-memory crash (a real ~293 TiB allocation
request, not a simulated one) and a real training timeout mid-run,
recovering both times without ever taking down the whole pipeline —
exactly the kind of thing that's easy to assert and easy to never verify.

**An honest, thorough search, even when it kept saying no.** Thirty-three
real, logged experiments; roughly two-thirds of them negative results, all
reported exactly as measured instead of quietly filed away. We're proud of
the discipline that took as much as we're proud of the one result that
worked.

## What we learned

Concretely, on the machine learning side: what a hand-derived backward
pass actually looks like for a bilinear interaction term, when autograd
genuinely earns its complexity cost versus when it's just convenience,
why pairwise ranking losses are theoretically appealing but empirically
harder to train stably than plain pointwise classification on a dataset
this size, and — the biggest surprise — that on a well-specified benchmark
like this one, the training *objective* (what you're optimizing for) can
matter far more than the model's *capacity* (how big it is). Every
architecture change we tried landed close to flat; the one lever that
worked was multi-task learning, a change to *what* the model was being
asked to predict, not how expressive it was allowed to be.

More broadly: a genuinely autonomous research loop is mostly an exercise
in restraint and record-keeping, not cleverness — the value was in never
letting a result go unrecorded, never trusting a number we hadn't
independently verified against the organizer's own pinned scoring code,
and being honest, every single time, about which parts of the process
were the agent's own reasoning and which were ours.

## What's next for tiktokers

**Turn the graph from a feature into an architecture.** Our one genuinely
non-standard experiment — initializing embeddings from a LightGCN-style
propagation over the user-video interaction graph — came back a clean
null, but only as a one-time *initialization*. We never tried making that
propagation a live part of the forward pass, recomputed against the
model's *current*, evolving embeddings every step instead of a frozen
snapshot from before training started. That's a real architecture change,
not a data-prep step, and it's the most promising untested idea we have.

**Let the LLM run longer, with a real budget to manage.** Our Research
Strategist has only ever run a handful of iterations at a time. With
budget-aware stopping already built, the natural next step is a long,
unattended run — hours, not minutes — to see whether it converges on
something none of us thought to try by hand.

**Generalize the Research Critic Gate.** Right now it catches one specific
repeated pattern (a pure-capacity hyperparameter change). A version that
recognizes "a fourth pairwise-loss variant" as the same *kind* of dead end
as "a fourth k-only change" would make the agent's own judgment sharper,
not just its search wider.

**Scale up, properly.** KuaiRand-1k and KuaiRand-27k are sitting right
there as bonus benchmarks. We deliberately didn't force a shortcut through
the Starter Kit's `_pure`-hardcoded loader just to claim the bonus points —
a real, config-driven, validated loader for the larger benchmarks is
worth building properly next, not rushed.

**Take the counterfactual angle further.** KuaiRand's randomized-exposure
log already let us confirm our result generalizes to unbiased data. The
same log is a genuine foundation for real off-policy evaluation and
debiased training — using the *unbiased* subset not just to check a model
trained on biased data, but to train one that corrects for that bias
directly.
