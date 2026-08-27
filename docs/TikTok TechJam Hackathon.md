# Problem statement

2. **Autonomous Machine Learning Research Agent for Recommender Systems**

In response to some queries, our engineers have provided updates to the problem statement to improve clarity and to support participants better.  
**Problem Statement last updated: 26 August 2026, 6:33PM**  
**Added downloadable kuairand-starter-kit.zip under '[Starter Kit](https://bytedance.larkoffice.com/wiki/DNtSwxgeciCS2nkiUefc5qqtnkf#R6QcdQ0G9oeZmQxHuEZm8LSxy8e)'**  
**Technical Workshop Webinar with Q\&A** will be held on **28 Aug, 2:00 to 2:45pm.**  
Click here to join the webinar\!

1. **Background**

### **Motivation**

Machine learning engineers (MLEs) spend much of their time on a single activity: **taking a dataset and a set of metrics, then iterating on a model again and again to push the score higher.** This work is inherently cyclic — every round repeats the same loop, shown in Figure 1\.  
![][image1]  
> **Figure 1\. The MLE iteration loop.** A closed cycle of five core stages, plus a reflection step that feeds the next round:

> 1. **Read the problem** — understand the given dataset and the target metrics.  
> 2. **Inspect data** — study data distribution through exploratory data analysis (EDA).  
> 3. **Engineer features** — build and select input features (see Appendix A.5).  
> 4. **Train \+ tune** — choose a model, set the loss function, and tune hyperparameters.  
> 5. **Evaluate** — read the metrics, check for overfitting, and consult the leaderboard.

> The result of the **evaluate** stage drives a **reflect \+ revise** step, which decides what to change and loops back into the next iteration — re-inspecting the data and adjusting the features. The cycle repeats until the score plateaus.  
Two of these stages — **engineer features** and **train \+ tune** — are carried out almost entirely in code: the engineer writes scripts to transform the data, define the model, and run training. In other words, each turn of the loop produces and modifies code. This is what makes the loop a natural target for automation: it is structured and repeatable, yet writing and revising that code is exactly the kind of task a code-generating LLM can take on.  
The loop is also repetitive and mechanical. It draws heavily on "engineering intuition," but many individual steps are well-structured and repeatedly exercised in practice — which is precisely why automating the whole cycle has become an active research direction.

### **Prior Work**

Over the past two years, a new line of work has set out to automate this loop: the **Autonomous ML Research Agent**, an LLM-driven agent that runs the cycle in Figure 1 on its own. It reads the problem, **writes the code** for each stage, trains and evaluates the model, reflects on the results, revises its approach, and finally produces a submission. Representative systems include:

* **MLE-Bench** \[1\] (OpenAI) — a benchmark of 75 Kaggle competitions, now a standard evaluation suite for such agents.  
* **AIDE** \[2\] (Weco AI) — a state-of-the-art agent that frames ML engineering as code optimization and explores the space of solutions via tree search.  
* **AI-Scientist-v2** \[3\] (Sakana AI) — an end-to-end agent for autonomous scientific and ML research, using agentic tree search to form hypotheses, run experiments, and write up results.

### **This Challenge**

This challenge asks participants to design an **autonomous ML research agent**. Given a public ML dataset and a set of metrics, the agent must **autonomously** run the full loop of Figure 1 — read the problem, engineer features, train and tune the model, evaluate, then reflect and iterate — to reach the highest possible score across the test sets. Writing the code for each stage is part of the agent's job, not something provided in advance.  
> **New to recommender systems?** All benchmarks in this challenge come from the recommendation domain (the KuaiRand family). If terms such as CTR, multi-task learning, NDCG, or Recall@K are unfamiliar, start with the **Appendix: A Primer on Recommender Systems** . At the end of this document — a concept map plus an annotated reading list designed to get you oriented in 1–2 hours.

2. **Problem Statement**

### **The Task**

Design and implement an Autonomous ML Research Agent. For each benchmark, the agent must autonomously:

1. **Reproduce the official baseline.** Stand up a working end-to-end pipeline and confirm it reaches the official baseline's reported validation score. (The official baseline is a fixed, organizer-provided reference — see *Benchmarks*. Any starter pipeline the agent builds for itself is an internal step, not the reference it is scored against.)  
2. **Iterate on the pipeline.** Autonomously draw on established methods from both industry and academia to improve each stage of the pipeline (see Figure 1), and apply those improvements in code. The agent develops using **only the training split and the public validation feedback** — it never has access to the hidden test set.  
3. **Improve over the baseline.** Through repeated iterations, drive the **validation** score above the official baseline. Improvement need not be strictly monotonic — as with real-world data, the trajectory may fluctuate — but the agent should show a clear, sustained ability to keep improving relative to the baseline. Final ranking is computed once, on the **hidden test set**, using the submission the agent designates as final.

### **Task Requirements**

1. **Runs end-to-end and aims to beat the baseline.** The agent must run the full pipeline on the required benchmark (KuaiRand-Pure) and reach a converged result; attempting the bonus benchmark (KuaiRand-1k & KuaiRand-27k) is optional. The target is a hidden-test score that exceeds the official baseline; the actual delta achieved — positive or negative — is what feeds into the Primary metric scoring (see Judging Criteria), so falling short of the baseline is scored continuously rather than treated as a disqualifying failure.  
2. **Iterates autonomously across the full stack.** The agent should improve the solution on its own, driven by its own evaluation of results. Improvements may target any part of the algorithmic stack — not just the model architecture, but every upstream and downstream module is fair game. The goal is to **minimize human intervention** — a fully autonomous run is the ideal, but a well-instrumented **semi-automated** pipeline that requires only a handful of interventions is an acceptable and realistic outcome; in practice, we measure how little human intervention a run requires (e.g. the number of manual interventions).  
3. **Robust operation.** The pipeline should run reliably with **minimal human intervention**. Robustness here is about how the agent *handles* difficulty, not how often it succeeds — we do not score it by failure count, since a capable agent may fail only on genuinely hard problems. What matters is that when a step fails (a code error, a timeout, an unexpected input), the agent can recover, retry, or route around it, and that long iterative runs neither crash, stall, nor diverge.  
3. **Constraints & Scope**

| Category | Constraints & Scope Details |
| :---- | :---- |
| In scope | Any open-source library or framework (PyTorch, RecBole, TorchRec, LightGBM, …) Any papers, public solutions, or pretrained weights Changes to any pipeline stage — not just the model |
| Out of scope | No external training data or pretrained weights trained on these benchmarks' test labels No hidden-test access during development (train \+ validation only) |
| Limits | **KuaiRand-Pure**: NDCG@10 / Recall@50, click \= positive (fixed) (*Required*); **KuaiRand-1k** & **KuaiRand-27k**: same task and metrics (*Bonus*) Hidden test scored once, on the final submission Compute budget: *TBD* |
| Allowed assumptions | Fixed `train / validation / hidden-test` split per dataset Official baseline, scores & evaluation script (incl. convergence rule) Example submission \+ output schema |

4.   
   **Available Resources & Data**

### **Starter Kit**

This content is only supported in a Feishu Docs  
To lower the barrier to entry — especially for participants new to recommender systems — the challenge provides a standard starting point. **Download: kuairand-starter-kit.zip** (above) — numpy only (no torch / pandas / scikit-learn); `python3 baseline.py --model fm` reproduces the official baseline in about 40 s on a single CPU core. It contains:

1. **Fixed data splits**: date-based, taken from the two standard logs (`log_standard_4_08_to_4_21_pure.csv` & `log_standard_4_22_to_5_08_pure.csv`). **train** \= date 20220408–20220421 (1,141,112 rows) / **validation** \= date 20220422–20220428 (124,909 rows) / **test** \= date 20220429–20220508 (170,588 rows). Teams develop on train \+ validation only; the hidden test set is scored once. Splitting by date rather than by row count avoids any tie-breaking ambiguity on equal timestamps.  
2. **Official baseline**: a fixed, organizer-provided reference pipeline shipped in the Starter Kit — a Factorization Machine (k=16, lr=0.001, 5 categorical fields), numpy only, about 40 s on CPU. Published **hidden-test** scores: GAUC **0.6610** / nDCG@5 **0.5282** / primary **0.5946** (mean over 5 seeds, std 0.0008). Validation: GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016. Reference rungs for harness self-check — random scoring: primary 0.4753; item popularity: primary 0.5715. Beating *this* baseline is what counts — not a baseline the team builds itself.  
3. **Evaluation script**: the exact scoring code (GAUC / nDCG@5) ships in the Starter Kit as `evaluate.py`. It is model-agnostic — it takes only `(user_ids, labels, scores)`, so any model can be scored with it. **Pinned conventions**: users with zero positives count as nDCG \= 0 and are included in the average; GAUC counts only users with 0 \< positives \< impressions, weighted by positive count; nDCG gain \= 2^rel − 1\. **Convergence rule: ε \= 0.002, N \= 3** — a run is converged when the validation primary score has not improved by more than ε over the last N consecutive iterations (ε ≈ 2.5σ of the baseline's 5-seed std of 0.0008). The absolute-delta aggregation is unchanged.  
4. **Submission format**: a CSV with the header `row_id,user_id,video_id,score`, one line per evaluation-split row. `row_id` is a 0-based, strictly increasing index into the split as produced by `data.load()`; `user_id` / `video_id` are redundant fields used only to verify alignment; `score` is any real number (only the relative order matters), and NaN / Inf are rejected. The `row_id` is required because `(user_id, video_id)` is **not unique** in the evaluation split — 3.06% of test rows are repeated pairs, up to 12 times — so it cannot serve as a key. Generate a runnable example with `python3 submit.py --make` and validate with `--check`, which rejects a wrong header, a row-count mismatch, `row_id` gaps, misalignment against the evaluation split, and non-numeric scores.  
5. **Run-log requirements**: each iteration should record its **hypothesis**, the **code diff**, the resulting **metrics**, and any **error / recovery events**. These logs are how judges assess **Autonomy** (scored under Impact & Relevance) and **Robustness** (scored under Technical Execution) — see Judging Criteria.  
6. **LLM coding agent**: you can use whatever you like, or use [Trae](https://www.trae.ai/pricing) from ByteDance, which provides "Limited offer: new user 7-day free trial".

### **Benchmarks**

**KuaiRand-Pure is required** and determines 100% of the primary score. **KuaiRand-1k and KuaiRand-27k are bonus datasets** — attempting them is optional and earns extra credit, but neither is required to complete the primary score.  
**Resource policy.** This is a hackathon, so external resources are open by default: use any open-source library (PyTorch, RecBole, TorchRec, LightGBM, …), read any papers, docs, or public solutions, and use pretrained model weights freely. The agent is expected to draw on whatever published methods it can find — that is what makes it a *research* agent.  
There is **one hard rule: no external training data.** Training must rely only on the KuaiRand datasets listed below — no augmenting, joining, or pre-training on any other dataset, and no pretrained model whose weights were trained on these benchmarks' test labels. This single rule is what keeps the hidden-test ranking fair; everything else is unrestricted.

| Dataset | Domain & Description | Metrics | Scale |
| :---- | :---- | :---- | :---- |
| **KuaiRand** (Kuaishou) Three released variants: **KuaiRand-Pure** is required, while **KuaiRand-1k** and **KuaiRand-27k** are bonus. | Short-video feed. 12 feedback signals (click / like / follow / comment / forward / long\_view / play\_time …) plus a randomized-exposure intervention that supports counterfactual evaluation. **Relevance label and K are fixed by the organizers** (see Starter Kit / TBD): the default task treats `click` as the positive relevance label and reports **NDCG@10 / Recall@50**. The exact label definition and K values are pinned in the Starter Kit so every team solves the same task. | NDCG@10 / Recall@50  | Pure: 1.4M interactions (27K users × 7.6K items). 1k: 11.7M. 27k: 322M. |

Links: KuaiRand — [https://kuairand.com](https://kuairand.com/)  
> KuaiRand's randomized-exposure data also enables off-policy / counterfactual evaluation (OPE).

5. **Deliverables**  
1. **Written Project Description (via Devpost)**  
* Provide a clear written description of your project that includes:  
  * How your solution addresses the problem statement  
  * Development tools used (e.g. VSCode, Colab, Jupyter)  
  * APIs used (e.g. OpenAI GPT-4o, Google Maps API)  
  * Libraries and frameworks used (e.g. Hugging Face Transformers, PyTorch, scikit-learn, pandas)  
  * Datasets and assets used (e.g. Google Local Reviews dataset, manually labelled data)  
2. **Public Code/GitHub Repository**  
* Submit a link to a public Code/GitHub repository containing:  
  * Well-structured, commented code covering all components of your solution  
  * A README file that includes:  
    * Project overview  
    * Setup and installation instructions  
    * Steps to reproduce your results  
    * A brief reflection on your solution's limitations and what you would improve given more time  
    * Team member contributions (if applicable, i.e. team participants, non-solo participants)  
3. **Run & Iteration Logs**  
* Submit the per-iteration log required in the Starter Kit (Run-log requirements), covering:  
  * Hypothesis for that iteration — what the agent intended to try and why  
  * The code diff applied  
  * The resulting metrics (NDCG@10 / Recall@50 for the KuaiRand benchmarks)  
  * Any error or recovery events encountered, and how the agent handled them  
* A short summary reporting the number of manual interventions during the run (used to assess autonomy per Task Requirement 2\)  
4. **Final Submission & Results Summary**  
* Submit your final model output/checkpoint for the required benchmark (KuaiRand-Pure), in the schema defined by the Starter Kit. If you also attempt the bonus benchmarks (KuaiRand-1k & KuaiRand-27k), submit their outputs as well for bonus scoring.  
* A results table reporting your validation-best score for the required benchmark's metrics (KuaiRand-Pure NDCG@10 / Recall@50), and its absolute delta over the official baseline (per the Evaluation section scoring formula); if you attempted the bonus benchmarks (KuaiRand-1k & KuaiRand-27k), include their NDCG@10 / Recall@50 results as well  
* Reported resource usage required to reach the converged result: total token consumption (input \+ output) from the agent's LLM calls, and total GPU time (GPU-hours) consumed during training and evaluation (used to score Feasibility & Practicality)  
6. **Judging Criteria**

| Judging Criteria | Weight |
| :---- | :---: |
| **Technical Execution** | **35%** |
| **Innovation & Problem Insight** | **20%** |
| **Impact & Relevance** | **20%** |
| **Feasibility & Practicality** | **15%** |
| **Presentation & Communication** Final Event Only | **10%** |

### **Technical Execution — Primary Metric & Robustness**

**Primary metric.** We score the **converged result**, not the peak and not the intermediate trajectory. A run is considered converged when **validation score has not improved by more than a small threshold ε over the last *N* consecutive iterations** (default: ε and *N* fixed by the organizers and published in the Starter Kit), *or* when the run hits the fixed compute/wall-clock budget — whichever comes first. The submission scored for ranking is the **validation-best checkpoint** at that point, evaluated **once on the hidden test set**. The agent develops only on train \+ validation; it never sees the hidden test set.

* **KuaiRand-Pure is the required benchmark** and determines 100% of the Primary metric score. **KuaiRand-1k and KuaiRand-27k are bonus benchmarks**: a strong result on either earns additional bonus points on top of the Primary metric score, but skipping them does not reduce the KuaiRand-Pure score.  
* Per-dataset metrics: **KuaiRand-Pure / KuaiRand-1k / KuaiRand-27k** → NDCG@10 / Recall@50. Within each dataset, the score is the **equal-weighted average of each metric's *absolute* improvement over the official baseline** on the hidden test set. For every metric *m*:

delta(m) \= score\_agent(m) − score\_baseline(m)

* score\_dataset \= mean over m of  delta(m)

**Robustness.** Not judged by whether the agent ever hits a failure, but by **how it handles one** — recovering, retrying, or routing around a failed step (a code error, a timeout, an unexpected input) so that long iterative runs neither crash, stall, nor diverge before hitting the compute/wall-clock budget.

### **Innovation & Problem Insight**

Judged on what the agent identified as worth trying and why — not on implementation.

* What the agent chose to target across the full algorithmic stack (features, model architecture, training strategy, evaluation loop, etc. — improvements are not limited to the model itself) and the reasoning behind that choice.  
* Originality in drawing on published methods, papers, or public solutions — rewarding agents that go beyond naive baseline tweaks.

### **Impact & Relevance — Autonomy**

**Autonomy.** How much of the improvement loop the agent drives on its own — proposing and testing changes based on its own evaluation of results, not just tuning the model architecture. Measured primarily by the **number of manual interventions** required to reach the converged result; fewer interventions score higher, with fully autonomous runs scoring highest. The fewer humans required, the more this reflects real acceleration of recommender-system R\&D.

### **Feasibility & Practicality — Resource Consumption**

How much it costs — in both LLM usage and GPU compute time — to reach the converged result.

* **Token consumption.** Total input \+ output tokens used by the agent's LLM calls across the run.  
* **GPU time.** Total GPU-hours consumed during training and evaluation to reach the converged result — captures the actual compute resources used in a way that wall-clock time alone cannot (e.g. running on more GPUs in parallel looks fast on the clock but is not necessarily cheaper).  
7. **References**

\[1\] J. S. Chan, N. Chowdhury, O. Jaffe, J. Aung, D. Sherburn, E. Mays, G. Starace, K. Liu, L. Maksin, T. Patwardhan, L. Weng, and A. Mądry, "MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering," OpenAI, 2024\. arXiv:2410.07095. [https://doi.org/10.48550/arXiv.2410.07095](https://doi.org/10.48550/arXiv.2410.07095)  
\[2\] Z. Jiang, D. Schmidt, D. Srikanth, D. Xu, I. Kaplan, D. Jacenko, and Y. Wu, "AIDE: AI-Driven Exploration in the Space of Code," 2025\. arXiv:2502.13138. [https://doi.org/10.48550/arXiv.2502.13138](https://doi.org/10.48550/arXiv.2502.13138)  
\[3\] Y. Yamada, R. T. Lange, C. Lu, S. Hu, C. Lu, J. Foerster, J. Clune, and D. Ha, "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search," 2025\. arXiv:2504.08066. [https://doi.org/10.48550/arXiv.2504.08066](https://doi.org/10.48550/arXiv.2504.08066)

8. **Appendix A. A Primer on Recommender Systems**

> This appendix gives participants without a recommender-systems background just enough to get started. It is a concept map plus an annotated reading list — not a textbook. Use it to understand the KuaiRand benchmarks and to know what to look up when you get stuck.

### **A.1 The Big Picture: The Recommendation Pipeline**

A modern industrial recommender does not score every item directly. It runs a funnel of stages, each narrowing the candidate set:  
Recall  →  Pre-ranking  →  Ranking  →  Re-ranking  
millions    thousands       hundreds     final list

* **Recall / Retrieval**: cheaply retrieve a few thousand candidates from millions.  
* **Pre-ranking**: a lightweight model trims the candidates further.  
* **Ranking**: a heavy, accurate model scores each candidate. **This challenge mostly lives here.**  
* **Reranking**: adjust the final ordering for diversity, business rules, and so on.

> For this competition you mainly need the **ranking** stage. The KuaiRand benchmarks are ranking/prediction tasks, not full end-to-end pipelines.  
This content is only supported in a Feishu Docs

### **A.2 Core Tasks: CTR and the Feedback Funnel**

Most industrial ranking is framed as predicting the probability of user feedback:

* **CTR (Click-Through Rate)** — `P(click | impression)`. The user saw the item; will they click?  
* **CVR (Conversion Rate)** — `P(conversion | click)`. The user clicked; will they convert (buy)? E-commerce background only; not a task in this challenge.  
* **The funnel**: `impression → click → deeper engagement` (in e-commerce, `→ conversion`). Because these stages are linked, two well-known problems arise:  
  * **Sample selection bias**: the post-click signal is only observed on *clicked* items, yet must be predicted for *all* impressions.  
  * **Data sparsity**: post-click signals such as `long_view` or `like` are far rarer than clicks.

> **KuaiRand** has no purchase label, so CVR itself is never scored here. But the same two problems reappear on its post-click signals (`long_view`, `like`, `follow` …), and ESMM-style multi-task modelling — see A.3 — is a legitimate approach to them.

### **A.3 Multi-Task & Multi-Feedback Learning**

Real users produce many signals (click, like, follow, comment, watch-time, and so on). Predicting them jointly — rather than training a separate model per signal — shares representations and tends to improve every task.

* Why it matters here: **KuaiRand** provides **12 feedback signals**, so a multi-task model can learn from several of them jointly even though only `click` is scored.  
* The key idea is to balance *shared* parameters (which transfer useful knowledge across tasks) against *task-specific* parameters (which prevent conflicting tasks from hurting one another — the "seesaw" problem).

### **A.4 Evaluation Metrics**

| Metric | Intuition | Used for |
| :---- | :---- | :---- |
| **AUC** | Probability that a random positive is ranked above a random negative. Threshold-free and robust to class imbalance. | CTR / CVR prediction in general (not scored in this challenge) |
| **NDCG** | Quality of a *ranked list*, rewarding relevant items near the top (with a position discount). | Ranking quality (KuaiRand) |
| **Recall** | Fraction of all relevant items that appear in the returned list. | Coverage (KuaiRand) |

> **Offline vs. online**: a higher offline metric does not always mean better real-world performance (because of distribution shift and feedback loops). This competition is evaluated offline, but it is worth knowing the gap exists.

### **A.5 Feature Engineering Basics**

* **ID features**: user ID, item ID, category ID — high-cardinality discrete features.  
* **Embedding**: map each discrete ID to a learnable dense vector. This is the foundation of all deep recommenders.  
* **Feature crossing**: combine features (e.g. user × category) to capture interactions. Models such as FM and DeepFM automate this.

### **A.6 Annotated Reading List**

\[Hints: If you find reading the following material challenging or find you have missing backgrounds, you can use ChatGPT / Claude / ... to explain it to you.\]  
The goal here is only to understand **how a recommender system is structured** — the recall → ranking → re-ranking pipeline — and where the ranking stage (which this challenge targets) sits within it. You do **not** need to read a whole course; the introductory overview is enough. **Read just one of the following:**

* Google, *Recommendation Systems* (Machine Learning Crash Course), the **Overview** section — `https://developers.google.com/machine-learning/recommendation` A short, official overview of the pipeline. Note: Google calls the ranking stage **"scoring"** — this is the same thing as **ranking**, and it is the part this challenge focuses on.  
* Wang Shusen, *Recommender Systems*, **Chapter 1 (Overview)** — `https://github.com/wangshusen/RecommenderSystem` The most beginner-friendly Chinese resource; the first chapter alone gives the full architecture.

![][image2]

# Outline

# **Project Brainstorming: \[Project Name — suggestion: "ReflectML" or "AutoRec Agent"\]**

**Team Members:** \[Fill in\] **Submission Deadline:** \[Fill in from hackathon rules\]

## **Context**

MLEs spend most of their time in one repeating loop: read the problem → inspect data → engineer features → train \+ tune → evaluate → reflect \+ revise → repeat. Two stages of that loop (feature engineering, train \+ tune) are almost pure code, which is why an LLM-driven agent can plausibly run the whole cycle autonomously. This challenge asks us to build exactly that agent, targeted at the KuaiRand-Pure recommendation benchmark (NDCG@10 / Recall@50, click \= positive label), with KuaiRand-1k/27k as optional bonus scale-ups.

The organizer-provided Factorization Machine baseline is fixed (hidden-test: GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946). Our agent has to reproduce that, then autonomously iterate past it — developing only on train \+ validation, never touching the hidden test set until the one final scored run.

It's worth being explicit about scope here: a production recommender runs a full funnel — Recall/Retrieval (millions of candidates) → Pre-ranking (thousands) → **Ranking** (hundreds, heavy accurate model) → Re-ranking (final list, diversity/business rules). This challenge lives entirely in the **Ranking** stage — we're not building retrieval or re-ranking, just the model that scores each candidate given (user, video) pairs. Keeping the agent's scope pinned to ranking (feature engineering \+ model \+ training/tuning against NDCG@10/Recall@50) avoids wasted effort building funnel stages nobody's scoring.

## **Problem Statement**

How might we build an ML research agent that autonomously runs the full read → inspect → engineer → train/tune → evaluate → reflect loop on KuaiRand-Pure, reliably beats the official baseline, and does so with minimal human intervention, bounded compute cost, and full traceability of what it tried and why?

## **Mission**

Build an autonomous ML research agent that:

* Reproduces the official FM baseline end-to-end before touching anything else  
* Iterates on every stage of the pipeline (not just model architecture) using published methods, papers, and public solutions  
* Detects its own convergence per the organizer's rule (ε \= 0.002, N \= 3\) and stops cleanly  
* Recovers from failures (code errors, timeouts, bad inputs) instead of crashing or stalling  
* Logs every iteration's hypothesis, code diff, metrics, and error/recovery events  
* Minimizes human intervention, token spend, and GPU-hours to reach its converged result

## **Target Stakeholders**

* **The Agent (autonomous system):** the thing doing the actual work — needs to plan, code, execute, evaluate, and self-correct without a human in the loop.  
* **Developer/Operator (us):** needs visibility into what the agent is doing, a safe way to intervene if it's stuck, and confidence it won't burn the compute budget on a bad idea.  
* **Judges/Evaluators:** need to see *why* the agent chose what it tried (Innovation & Problem Insight), *how autonomous* the run was (Impact & Relevance), and *what it cost* (Feasibility & Practicality) — not just the final score.

## **User Stories**

> Note: the PS was updated 26 Aug 2026 (starter kit link added; technical workshop webinar 28 Aug, 2:00–2:45pm) — worth having someone on the team attend that before locking Phase 1 assumptions.

### **Agent (Autonomous System)**

* As the agent, I want to run `baseline.py` first and confirm I hit the published baseline scores before writing any new code, so I know my harness is trustworthy.  
* As the agent, I want to profile the KuaiRand feature space (12 feedback signals, categorical fields) during EDA, so my feature-engineering hypotheses are grounded in the actual data, not generic recipes.  
* As the agent, I want to try multi-task formulations (e.g. ESMM/MMOE-style, using auxiliary signals like like/follow/long\_view) alongside the primary click task, so I can exploit KuaiRand's 12 signals instead of training on click alone.  
* As the agent, I want each iteration to log a hypothesis, a code diff, and resulting metrics *before* I decide on the next change, so my reasoning is auditable and I don't repeat failed strategies.  
* As the agent, when a training run throws an error or times out, I want to catch it, log the failure, and retry or route around it, so a single bad step doesn't kill the whole run.  
* As the agent, I want to check my validation primary score against the ε/N convergence rule after every iteration, so I stop at the right point instead of overfitting to validation or wasting compute.  
* As the agent, I want to validate my submission CSV against `submit.py --check` before finalizing, so I never get disqualified over a formatting or alignment error.  
* As the agent, I want to track my own cumulative token usage and GPU-hours per iteration, so I can flag when I'm approaching budget and prioritize higher-expected-value experiments.  
* As the agent, I want to store each experiment as a hypothesis → method → result → insight chain (not just a score in a list), so I can later ask "what areas of the solution space haven't I explored yet" instead of only "what's my last score."  
* As the agent, I want to diagnose *why* a metric moved — e.g. NDCG up but Recall down suggests a ranking problem, train↑/validation↓ suggests overfitting — so my next hypothesis targets the actual weakness instead of guessing.  
* As the agent, I want to generate several candidate experiments at once and score each by (expected gain × confidence × novelty ÷ compute cost), so I spend budget on the highest-value idea first instead of running whatever occurs to me next.  
* As the agent, I want to smoke-test a new idea on \~1% of the data for one epoch before committing to a full run, and only escalate to 10%, then 100%, if it looks promising, so I kill weak ideas cheaply instead of burning GPU-hours finding out at full scale.  
* As the agent, I want most experiments to go through a structured config (model, features, hyperparameters) rather than an LLM rewriting source code, so only genuinely novel ideas cost real tokens and a bad code-gen pass can't silently corrupt the pipeline.

### **Developer/Operator**

* As the developer, I want a live view of the agent's current stage (EDA / feature eng / training / evaluating / reflecting), so I know what it's doing without reading raw logs.  
* As the developer, I want to see the full iteration history (hypothesis → diff → metric delta) in one place, so I can assess autonomy and quality after the run without replaying it.  
* As the developer, I want an optional manual-intervention hook, so I can nudge the agent if it's stuck in a loop, but I want every intervention I make automatically counted and logged (it costs us on the Autonomy score).  
* As the developer, I want a dry-run / small-sample mode, so I can sanity-check a new agent strategy on a data subset before letting it loose on the full 1.14M-row train split.  
* As the developer, I want the bonus benchmarks (KuaiRand-1k/27k) to reuse the same pipeline with just a config change, so attempting them isn't a rewrite.

### **Judge/Evaluator**

* As a judge, I want to see *why* the agent tried a given technique (not just that it did), so I can score Innovation & Problem Insight fairly.  
* As a judge, I want a clear count of manual interventions and total resource usage, so I can score Autonomy and Feasibility without digging through raw logs.

## **Key Features / Requirements**

Ranked by tier, the same way we want the *agent itself* to rank experiments: **P0 \= foundation** (nothing works or scores without it), **P1 \= differentiators** (this is where Innovation, Autonomy, and Feasibility points are actually won), **P2 \= stretch** (build only if P0/P1 are solid and time remains). "Maps to" names the judging criterion each feature primarily earns.

### **P0 — Foundation (must exist before anything else matters)**

| Category | Feature Name | Description | Maps to | Effort | Owner | Status |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Core Loop | Orchestrator / Planner | Drives read → inspect → engineer → train/tune → evaluate → reflect; a fixed short list of predefined experiments is fine at this stage — no LLM reasoning yet. | Technical Execution | L |  |  |
| Modeling | Model Zoo (base) | Pre-built, not LLM-generated: FM (baseline) \+ DeepFM to start, so the agent selects/configures instead of writing models from scratch. | Technical Execution | M |  |  |
| Evaluation | Evaluator Wrapper | Wraps organizer's `evaluate.py` exactly (zero-positive → nDCG 0, GAUC weighting) — never reimplement it. | Technical Execution (correctness) | S |  |  |
| Evaluation | Convergence Detector | Implements ε \= 0.002 / N \= 3 automatically; flags the validation-best checkpoint. Getting this wrong silently changes what gets scored. | Technical Execution | S |  |  |
| Engineering | Structured Experiment Interface | A config schema (model, features, hyperparams) the agent fills in for \~80% of experiments; LLM only writes raw code for the genuinely novel \~20%. | Technical Execution \+ Feasibility | M |  |  |
| Observability | Structured Run Log | Auto-generates the required per-iteration log: hypothesis, code diff, metrics, error/recovery events. | Hard deliverable requirement | S |  |  |
| Robustness | Failure Recovery | Catches code errors, timeouts, bad inputs; retries/rolls back/routes around instead of crashing the run. | Robustness (Technical Execution) | M |  |  |
| Submission | Submission Validator | Wraps `submit.py --check` before finalizing; rejects bad headers/row-id gaps/misalignment automatically. | Avoids disqualification | S |  |  |

### **P1 — Differentiators (this is the "not just ChatGPT tuning a learning rate" story)**

| Category | Feature Name | Description | Maps to | Effort | Owner | Status |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Research Memory | Research Map / Experiment Tree | Each experiment is a node in a solution tree (not a flat list): a node \= a full pipeline config, edges \= a proposed modification (draft fresh / debug a failure / improve a working node — the three-mode split AIDE uses). Node metadata stores hypothesis → method → result → insight. | Innovation & Problem Insight | L |  | Directly modeled on AIDE's solution tree \+ AI-Scientist-v2's experiment-manager pattern — see "Grounding in Cited Prior Work" below. This is our answer to the Mission's "using published methods, papers, and public solutions." |
| Research Memory | Metric-Aware Diagnosis Engine | Reads the metric *pattern*, not just the number: NDCG↑/Recall↓ → ranking problem; train↑/val↓ → overfitting; both low → feature/model mismatch. Turns "score went down, try something else" into an actual diagnosis. | Innovation & Problem Insight | M |  |  |
| Planning | Best-First Node Selector | A dedicated selector role (separate from the LLM that writes code, per AI-Scientist-v2's experiment-manager split) greedily picks the most promising *node* to expand next, scored by (expected gain × confidence × novelty ÷ cost) — can branch off an older strong node, not just the most recent result. | Innovation \+ Feasibility | M |  | On MLE-Bench, AIDE's tree search won \~4× more medals than a linear (no-branching) agent — direct published evidence this beats a flat "portfolio, pick top one" design. |
| Efficiency | Multi-Fidelity Runner | Smoke-tests a new idea on \~1% data / 1 epoch, escalates to 10% then 100% only if promising; kills weak ideas before they burn full GPU time. Optionally runs 2–3 branches concurrently at the smoke-test tier (per AI-Scientist-v2's parallel-worker pattern) instead of testing one idea at a time. | Feasibility & Practicality | M |  | Biggest single lever on GPU-hours — could be worth more than any model-architecture choice. |
| Data & Features | Multi-Task Feature Exploitation | Uses KuaiRand's 12 feedback signals (like, follow, long\_view, forward…) as auxiliary tasks alongside click. | Innovation & Problem Insight | M |  | Cited directly in the primer (A.3) as a legitimate differentiator over naive tweaks. |
| Resource Management | Token/GPU Tracker \+ Budget-Aware Planner | Logs cumulative token/GPU usage per iteration; planner deprioritizes expensive experiments as budget depletes. | Feasibility & Practicality | S–M |  | Required deliverable numbers (token \+ GPU-hour totals) come straight out of this. |
| Evaluation & Convergence | Noise-Aware Convergence Check | Runs 2–3 seeds (or bootstrap-resamples validation) before accepting a result into the research map. Baseline's 5-seed std is 0.0008 vs ε \= 0.002 — only \~2.5σ margin, so a single noisy run can look like real improvement or false convergence. | Technical Execution \+ Innovation | S–M |  | Protects the research map itself — a false "insight" learned from noise poisons every downstream decision the agent makes off it. |
| Data & Features | Debiased Learning via Randomized-Exposure Subset | KuaiRand's randomized-exposure intervention subset is unbiased exposure data — usable for inverse-propensity-weighted training or counterfactual evaluation instead of just more rows. | Innovation & Problem Insight | M |  | Explicitly invited by the primer but not required — most teams will skim past it entirely. |
| Data & Features | Completion Rate & Rewatch Features | `completion_rate = play_time/duration`, `rewatch_flag = play_time > duration` — TikTok has publicly described both as unusually strong signals for its own product. | Innovation & Problem Insight | S |  | Cheapest, most defensible "grounded in TikTok's own disclosed methodology" claim on the list. |
| Data & Features | Fast-Skip Implicit Negative Signal | Near-instant skip (`play_time` below a small threshold) treated as an explicit negative label/auxiliary task, not just "no click" — matches TikTok's disclosed treatment of skips. | Innovation & Problem Insight | S |  |  |
| Data & Features | Creator-Level Aggregate Features | Rolling engagement-rate features per `author_id` across their other videos — creator quality is part of TikTok's real ranking story and `author_id` is already in KuaiRand. | Innovation & Problem Insight | M |  |  |
| Research Memory | Per-Segment Metric Diagnosis | Buckets users by activity decile (or items by popularity decile) and checks NDCG/Recall per bucket, not just in aggregate — an aggregate \+0.01 can hide "better for power users, worse for the long tail." | Innovation & Problem Insight | S–M |  | Cheap since labels already exist; extends the Metric-Aware Diagnosis Engine rather than replacing it. |

### **Grounding in Cited Prior Work**

The PS's Mission line — "iterates on the pipeline... using established methods from both industry and academia" — points at the three papers cited under Prior Work (MLE-Bench, AIDE, AI-Scientist-v2). Here's exactly what we're taking from each, so it's traceable rather than a vague namecheck:

* **AIDE** frames ML engineering as a *solution tree* search, not a flat iteration list: each node is a full pipeline script; a Solution Generator either drafts fresh, debugs a broken node using its traceback, or improves a working node; an Evaluator scores it; a Solution Selector greedily expands the most promising node next. We adopted this directly — it's why Research Map / Experiment Tree and Best-First Node Selector replaced our earlier flat "portfolio selector" draft. AIDE's own paper reports this tree structure winning \~4× more MLE-Bench medals than a linear (non-branching) agent, which is direct published evidence for the switch, not just aesthetic preference.  
* **AI-Scientist-v2** separates the *decision* of what to try next (a dedicated "experiment manager agent") from the LLM that actually *writes* the code for it, and runs best-first tree search with multiple parallel exploration workers rather than one strict serial thread. We already had this split (Orchestrator/Planner vs. Code-Gen), so this confirms rather than changes our design — but it does suggest running 2–3 branches concurrently under our Multi-Fidelity Runner (smoke-test several nodes at 1% in parallel, only escalate the winner) instead of always testing one idea at a time. Worth adding as a config knob if time allows.  
* **MLE-Bench** isn't a method to borrow features from — it's the standard *evaluation paradigm* this whole task is modeled on (fixed benchmark, agent iterates only on train/validation, scored once on held-out data). Its main value to us is as the source of AIDE's reported 4× number above, and as a sanity check that "tree search beats linear iteration" isn't just our opinion.

Everything else in the P1 table (multi-task feature exploitation, OPE/debiasing, TikTok-specific engineered features) comes from the Appendix primer and public knowledge of TikTok's disclosed ranking signals, not from these three papers — worth keeping that distinction clear in the written project description, since the deliverable explicitly asks what methods/papers were drawn on.

### **P2 — Stretch (build only once P0/P1 are solid)**

| Category | Feature Name | Description | Maps to | Effort | Owner | Status |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Research Memory | Research Critic Gate | A cheap pre-flight check (deterministic rules \+ one lightweight prompt) that can reject an expensive experiment before it runs, with a stated reason (cost, weak prior evidence, etc.). | Innovation polish \+ Feasibility | M |  | Don't burn two expensive LLM calls per experiment on this — deterministic checks first, cheap critic prompt second. |
| Modeling | Extended Model Zoo | Add DCNv2, Wide & Deep, LightGBM once FM/DeepFM are stable. | Technical Execution | M |  |  |
| Modeling | Hyperparameter Search | Lightweight autotuning (e.g. Optuna) wired into the config interface. | Technical Execution | S–M |  |  |
| Observability | Live Dashboard | Visualizes current stage \+ research-map coverage ("Feature Engineering 80%, Sequential Modelling 10%…") for demoing. | Presentation & Communication (final only) | S–M |  | Doesn't score during development, but this is what makes the final demo land. |
| Robustness | Checkpointing | Periodic state snapshots so a crash mid-run doesn't lose earlier progress. | Robustness | S |  |  |
| Bonus | Config-Driven Scale-Up | Same pipeline runs against KuaiRand-1k/27k via config change only. | Bonus points | S |  | Don't attempt before the required KuaiRand-Pure path is solid. |

## **Extra Features (Idea Bank)**

Not folded into the tiered table above — good ideas, but either lower-priority, more speculative, or better named as a limitation than half-built. Worth a scan if P0/P1 finish early.

* **Temporal Drift Check** — train is dated 4/08–4/21, validation 4/22–4/28; explicitly check whether feature distributions shift across that boundary and flag features that "worked" in train but drift in validation. A dataset-specific overfitting guard, not a generic one.  
* **Cheap Ensembling of Top Experiments** — right before finalizing, average/stack the top 2–3 distinct experiments' saved predictions. No retraining required, near-free, often a real NDCG bump — good ROI for Feasibility.  
* **Plateau Prediction** — fit a simple decay curve to the validation-score trajectory and estimate when further iterations are unlikely to help, instead of only reacting to the ε/N rule after the fact. Lets the agent bail early under budget pressure.  
* **Session/Sequential Context Features** — rolling features like previous video's engagement or position within a session. TikTok's feed is disclosedly session-aware; this extends the sequential-modelling direction already flagged as underexplored in the research map. If this gets built as a real sequence model rather than hand-rolled rolling features, it's the one place Hugging Face Transformers (a SASRec-style transformer over watch history) would plausibly fit — but that's speculative P2 scope, not a priority; simple rolling/aggregate features are the cheaper first pass.  
* **Cold-Start / Exploration Bonus** — sounds compelling but likely doesn't work here: evaluation is offline against fixed historical labels, so an agent can't get credit for "exploring" against a label set that already happened. Better named explicitly as a limitation/future-work item in the README than half-built.  
* **"Approximating TikTok's Value Model" framing** — not a feature, a pitch angle: describe the multi-task weighting as the team's approximation of TikTok's disclosed multi-objective ranking blend, rather than presenting it as generic multi-task learning. Costs nothing to build, but sharpens how Innovation & Problem Insight reads to judges who know the product.

## **Feature Rationale: What the Cited Papers Actually Say, and Why We're Building From Them**

The short version lives in the table above; this is the expanded version for anyone drafting the "how your solution addresses the problem statement" section of the written project description, where you'll want to cite these specifically.

### **AIDE: AI-Driven Exploration in the Space of Code (Weco AI, arXiv:2502.13138)**

**What it says.** AIDE frames ML engineering as a code-optimization problem and formulates the whole trial-and-error loop as a tree search over the solution space. A "solution" is a full script; the empty root is `s0`. Every discovered solution is stored as a node in a solution tree, with edges representing improvement steps. Three components do the work: a **Solution Generator** that proposes new candidates either by drafting from scratch or by modifying an existing node (fixing a bug, or introducing an improvement); an **Evaluator** that runs a candidate and scores it against the objective; and a **Solution Selector** that greedily picks the most promising node to expand next. The paper's framing is explicit that this strategically *reuses and refines* promising solutions rather than rewriting from scratch each round — trading compute for performance. On MLE-Bench (75 real Kaggle competitions), AIDE reportedly won roughly 4× more medals than "the best linear agent" — an agent that iterates on one script sequentially without branching.

**Which features this grounds.** Research Map / Experiment Tree (a node is a full pipeline config; edges are the modification type), Best-First Node Selector (greedy expansion of the most promising node, not just "rank a batch and run the top one"), and the Structured Experiment Interface (config-driven changes so most iterations really are cheap "modify an existing node" steps, not full rewrites).

**Why we should implement it.** This is the one piece of direct, published, quantified evidence in our reference list that a specific architectural choice (tree-structured, branching search) beats the naive alternative (linear, non-branching iteration) on the exact style of task this challenge is. Our earlier draft — a flat portfolio, scored independently, run one at a time — was structurally closer to the "linear agent" AIDE explicitly outperforms. Fixing that before Phase 4 (when the LLM reasoning gets wired in) is cheap now and expensive to retrofit later, since the research-memory schema and the planner's interface both depend on whether experiments are stored as a list or a tree.

### **The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search (Sakana AI, arXiv:2504.08066)**

**What it says.** An end-to-end agentic system that autonomously formulates hypotheses, designs and runs experiments, analyzes results, and (in their case) writes up manuscripts. Its key upgrade over v1 is dropping the requirement for a human-authored starting code template, and replacing v1's strictly linear hypothesis-testing routine with a **progressive agentic tree search** managed by a dedicated **experiment manager agent** — a role explicitly separated from the LLM call that writes the code for any given node. It implements best-first tree search with configurable parallel exploration workers: e.g. `num_workers=3, steps=21` explores up to 21 nodes total, expanding 3 of them concurrently at each step, rather than one strictly serial thread.

**Which features this grounds.** The existing Orchestrator/Planner vs. Code-Gen split in our P0 tier (this paper is confirming evidence that separation is a deliberate, published design choice, not just tidiness), the Best-First Node Selector (same best-first idea as AIDE, but this paper is explicit about keeping "decide what's next" and "write the code" as separate roles/calls), and the optional parallel-branch mode noted on the Multi-Fidelity Runner.

**Why we should implement it.** Two concrete, adoptable pieces beyond what AIDE alone gives us. First: don't let the same LLM call that's mid-way through generating code also be the one deciding the tree's strategic direction — a single noisy generation shouldn't get to both propose an idea and approve its own priority. Keeping those as separate calls/roles is cheap and reduces a specific failure mode. Second: running 2–3 branches concurrently at the cheap smoke-test tier of our Multi-Fidelity Runner (each is only 1% data / 1 epoch, so parallelizing is nearly free) surfaces a stronger candidate to escalate to full training than trying ideas one at a time within the same wall-clock budget — a genuine efficiency gain for Feasibility & Practicality, not architectural decoration.

### **MLE-Bench: Evaluating Machine Learning Agents on Machine Learning Engineering (OpenAI, arXiv:2410.07095)**

**What it says.** A benchmark of 75 real Kaggle competitions used as the standard evaluation suite for ML-engineering agents. It's not an architecture or method paper — it's the yardstick other agents (including AIDE) report their results against.

**Which features this grounds.** None directly — it isn't a source of design ideas.

**Why it still matters.** It's the evidentiary backbone behind the AIDE comparison above (the \~4× medal number is an MLE-Bench result), and it's a useful sanity check on our own harness: the challenge's structure here — fixed baseline, develop only on train+validation, score once on held-out test, log every iteration — is essentially MLE-Bench's evaluation paradigm applied to one recommender-systems task instead of 75 Kaggle competitions. Worth citing in the written project description as evidence the evaluation harness follows established practice, not something improvised for this hackathon.

### **What's explicitly *not* from these three papers**

The TikTok-specific engineered features (completion rate, rewatch, fast-skip, creator aggregates), the OPE/debiased-learning idea, and multi-task feature exploitation all come from the Appendix's recommender-systems primer and public knowledge of TikTok's disclosed ranking signals — a different, equally legitimate source of "published methods, papers, and public solutions," just not these three specific references. Worth keeping that distinction explicit in the Devpost write-up (it directly asks what methods/papers you drew on) — naming two separate, deliberate sources of technique (agent-architecture literature vs. recsys domain literature) reads as more careful than blending them into one vague "we used AI research" claim.

## **Tips — How We Win**

* **Most teams lose on plumbing, not ideas.** A large chunk of Technical Execution is just: does the convergence rule match the spec exactly, does the evaluator wrapper match the pinned conventions exactly, does the submission pass `--check` first try. These are boring, easy to get subtly wrong under time pressure, and a wrong implementation silently changes your score without you noticing. The team that wins has an airtight P0 layer, not just clever ideas on top of a shaky one — budget real time to check these against the spec line by line.  
* **Depth on 2–3 differentiators beats breadth on ten.** Trying to build the whole idea list is how teams end up with six half-working features instead of two that are genuinely sharp. Highest-ROI pair if forced to choose: **Research Map \+ Multi-Fidelity Runner** — strongest Innovation story (cheapest to demonstrate convincingly) and strongest Feasibility story (produces the token/GPU numbers you must report anyway).  
* **Autonomy is scored by counting, so make the count real, not decorative.** Don't build a "planner" that's actually a rubber stamp on a hardcoded decision tree — if the LLM isn't really making the call, you haven't earned that score. Conversely, don't hide genuine interventions — a transparent semi-autonomous run beats a run that vaguely claims full autonomy.  
* **The demo narrative is worth more than its official 10%.** Presentation is only formally scored at the final event, but it's what separates two teams with similar underlying numbers. Want a clean, replayable story: baseline reproduced → weakness diagnosed → candidates generated and ranked → weak ones rejected with stated reasons → winner smoke-tested cheaply, then scaled up → insight recorded → converged. Consider building the dashboard/log-replay earlier than its P2 slot suggests — it's cheap and it's your closing argument.  
* **Don't skip the "boring" deliverables.** README quality, a stated reflection on limitations, team contributions, a correctly-computed results table — these are graded and easy to half-ass at 2am. A technically strong project with a sloppy writeup loses to a modest project with a crisp one, because judges are reading many of these fast.

## **Technical Architecture**

* **Orchestration:** \[Agent framework choice — e.g. custom loop, LangGraph, CrewAI\]  
* **LLM:** \[Model choice for planning/code-gen/reflection — kept out of the loop until Phase 4, see roadmap\]  
* **Execution:** Sandboxed Python execution (numpy-only baseline; pandas for EDA/data handling, PyTorch/RecBole/TorchRec for modeling, scikit-learn for LightGBM's sklearn API \+ light preprocessing, all allowed once past baseline reproduction)  
* **Hugging Face Transformers:** not in scope — no text in this dataset, and the only plausible use (a SASRec-style transformer over session/watch history, see Extra Features) is speculative P2. Not a priority; only pull it in if that specific idea gets built.  
* **Data:** Fixed date-based splits per Starter Kit (train 1,141,112 rows / val 124,909 / hidden test 170,588)  
* **Evaluation:** Organizer's `evaluate.py`, wrapped, not reimplemented  
* **Research Memory:** Structured store (hypothesis → method → result → insight, linked) — the Research Map/Experiment Graph lives here, not just a flat `experiment_history.json`  
* **Experiment Interface:** Config schema (model / features / hyperparams / training strategy) the planner fills in; only genuinely novel ideas fall through to raw LLM code generation  
* **Multi-Fidelity Runner:** Staged execution (1% → 10% → 100% of train data) with early termination on weak trajectories  
* **Logging:** Structured per-iteration log (hypothesis / diff / metrics / errors) → feeds both the deliverable and the dashboard  
* **Compute:** \[CPU/GPU budget — TBD per organizer limits\]

## **Development Roadmap**

Sequenced deliberately so the *dumb but working* version exists before any LLM reasoning is added — a fully autonomous run built on a shaky harness is worse than a semi-automated one built on a solid one.

| Phase | Focus Area | Key Tasks |
| ----- | ----- | ----- |
| Phase 1 | Dumb Autonomous Loop | Baseline reproduction, evaluator wrapper, convergence detector; agent picks from a short predefined experiment list (no LLM reasoning yet) — prove `python run.py` works end-to-end |
| Phase 2 | Experiment Infrastructure | Structured experiment folders/config, central experiment log, submission validator wired in — this alone already satisfies most of the Run & Iteration Log deliverable |
| Phase 3 | Model Zoo \+ Tuning | Fill out FM → DeepFM → DCNv2/Wide\&Deep/LightGBM, hyperparameter search, multi-task feature exploitation |
| Phase 4 | LLM Research Reasoning | *Now* bring in the planner/strategist: feeds dataset summary \+ metric history \+ research map \+ budget remaining, outputs hypothesis \+ reasoning \+ expected effect \+ cost \+ priority. Cost-aware portfolio selection and metric-aware diagnosis plug in here. |
| Phase 5 | Multi-Fidelity \+ Early Termination | Staged 1%→10%→100% runner; kill weak experiments early, log GPU saved |
| Phase 6 | Recovery, Polish & Bonus | OOM/NaN/timeout/syntax-error handling, research-map dashboard, Research Critic gate if time remains, bonus benchmarks if time remains, README \+ reflection write-up |

## **Success Metrics**

* **Primary metric:** absolute delta over baseline on hidden-test NDCG@10 / Recall@50 (100% of Technical Execution's primary score)  
* **Robustness:** run completes without crash/stall/divergence across the full compute budget  
* **Autonomy:** number of manual interventions during the run (lower is better; log every one)  
* **Cost:** total input+output tokens and GPU-hours to reach the converged result  
* **Innovation:** the reflection log itself should show reasoning that goes beyond naive baseline tweaks  
* **Demo narrative:** the strongest pitch isn't "we built the smartest AI researcher" — it's a clean, traceable story: baseline reproduced → weakness diagnosed → candidate experiments generated and ranked → weak ones rejected on cost/gain grounds → promising one tested cheaply first, then scaled up → insight recorded → converged. That sequence alone demonstrates all five judging criteria at once.

## **Open Questions**

* Which orchestration framework are we standardizing on?  
* What's our actual GPU budget/access for training runs?  
* Do we attempt bonus benchmarks (1k/27k) at all, or spend that time hardening the required KuaiRand-Pure path?  
* Who owns the reflection/prompt-engineering for the "next hypothesis" step — that's likely the highest-leverage piece for the Innovation score.

Add ons:  
Worth explicitly protecting time for "just try to get a genuinely strong model" rather than assuming sophisticated tooling produces one automatically. 

# Features to ideate

I think the way to make yours genuinely different is to build it as an **experiment-driven research system with a strategic memory and an explicit understanding of why experiments succeeded or failed**.

## **1\. The architecture I would build**

Here is the high-level architecture:

                        ┌─────────────────────┐  
                         │   USER STARTS RUN   │  
                         │                     │  
                         │   python run.py     │  
                         └──────────┬──────────┘  
                                    │  
                                    ▼  
                  ┌────────────────────────────────┐  
                  │         ORCHESTRATOR           │  
                  │                                │  
                  │ Controls experiment lifecycle  │  
                  │ Budgets / convergence / retry  │  
                  └───────────────┬────────────────┘  
                                  │  
          ┌───────────────────────┼────────────────────────┐  
          │                       │                        │  
          ▼                       ▼                        ▼  
 ┌────────────────┐    ┌──────────────────┐    ┌──────────────────┐  
 │ DATA INTELLIGENCE│    │ RESEARCH MEMORY │    │ RESOURCE MANAGER │  
 │                 │    │                  │    │                  │  
 │ EDA             │    │ Experiments      │    │ Token budget     │  
 │ Feature stats   │    │ Hypotheses       │    │ GPU budget       │  
 │ Sparsity        │    │ Failures         │    │ Time budget      │  
 │ Temporal drift  │    │ Insights         │    │ Cost estimates   │  
 └────────┬────────┘    └─────────┬────────┘    └────────┬─────────┘  
          │                       │                       │  
          └───────────────────────┼───────────────────────┘  
                                  ▼  
                     ┌─────────────────────────┐  
                     │   RESEARCH STRATEGIST   │  
                     │                         │  
                     │ "What should we try?"   │  
                     │                         │  
                     │ Hypothesis generation   │  
                     │ Experiment selection    │  
                     │ Risk / reward reasoning │  
                     └────────────┬────────────┘  
                                  │  
                                  ▼  
                     ┌─────────────────────────┐  
                     │    EXPERIMENT PLANNER   │  
                     │                         │  
                     │ Converts hypothesis →   │  
                     │ executable experiment   │  
                     └────────────┬────────────┘  
                                  │  
                                  ▼  
          ┌────────────────────────────────────────────┐  
          │               CODE ENGINE                   │  
          │                                            │  
          │ Generate patch / config / feature changes  │  
          │ NOT rewrite entire project                 │  
          └─────────────────────┬──────────────────────┘  
                                │  
                                ▼  
                     ┌─────────────────────────┐  
                     │      VALIDATION GATE    │  
                     │                         │  
                     │ Syntax                  │  
                     │ Unit tests              │  
                     │ Smoke test              │  
                     │ Data validation         │  
                     └────────────┬────────────┘  
                                  │  
                    ┌─────────────┴──────────────┐  
                    │                            │  
                  PASS                          FAIL  
                    │                            │  
                    ▼                            ▼  
          ┌──────────────────┐         ┌──────────────────┐  
          │ EXPERIMENT RUNNER│         │ RECOVERY AGENT   │  
          │                  │         │                  │  
          │ Train            │◄────────│ Diagnose         │  
          │ Evaluate         │         │ Fix / retry      │  
          │ Save checkpoint  │         │ Fallback         │  
          └────────┬─────────┘         └──────────────────┘  
                   │  
                   ▼  
          ┌────────────────────┐  
          │ EVALUATION ANALYST │  
          │                    │  
          │ NDCG@10            │  
          │ Recall@50          │  
          │ Overfitting        │  
          │ Cost               │  
          │ Improvement        │  
          └─────────┬──────────┘  
                    │  
                    ▼  
          ┌────────────────────┐  
          │    REFLECTION      │  
          │                    │  
          │ Why did it work?   │  
          │ Why did it fail?   │  
          │ What next?         │  
          └─────────┬──────────┘  
                    │  
                    └───────────────↺  
                           LOOP

That is the **base architecture**.

But this alone won't necessarily make you different. A lot of teams could build something vaguely similar.

The interesting part is **how the Research Strategist decides what to do**.

---

# **2\. The thing I think can differentiate you: build an Experiment Intelligence System**

Most autonomous ML agents will probably do something like:

Experiment 1 → FM  
Experiment 2 → DeepFM  
Experiment 3 → Tune learning rate  
Experiment 4 → Add dropout  
Experiment 5 → Try DCN  
Experiment 6 → Tune embedding size

Basically:

> LLM, look at the last score and suggest something.

That is weak.

I would make your agent maintain an actual **research map**.

## **Instead of just remembering experiments, it learns relationships**

For example:

                       FEATURE INTERACTIONS  
                              │  
                    DeepFM ───┼── \+0.012  
                              │  
                         xDeepFM?  
                              │

                       TEMPORAL FEATURES  
                              │  
                    time\_bucket ─── \-0.004  
                              │  
                   sequential history?  
                              │

                       REGULARIZATION  
                              │  
                     dropout 0.2 ── \+0.002  
                     dropout 0.5 ── \-0.006

Now the agent isn't just storing:

> Experiment 4 scored 0.612.

It stores:

> **"Moderate regularisation appears beneficial. Excessive regularisation hurts. Temporal bucket features did not help this architecture, but temporal sequence modelling has not yet been tested."**

That is much closer to actual research.

---

# **3\. I would use a Research Knowledge Graph**

This could be one of your coolest differentiators.

Each experiment becomes connected to:

Hypothesis  
    │  
    ▼  
Method  
    │  
    ▼  
Change  
    │  
    ▼  
Observed Result  
    │  
    ▼  
Insight  
    │  
    ├──── Supports hypothesis  
    ├──── Rejects hypothesis  
    └──── Suggests next hypothesis

For example:

HYPOTHESIS:  
FM cannot model sufficiently complex interactions.

        │  
        ▼

METHOD:  
DeepFM

        │  
        ▼

RESULT:  
NDCG \+0.011  
Recall \+0.008

        │  
        ▼

INSIGHT:  
Higher-order feature interactions appear useful.

        │  
        ▼

NEXT POSSIBILITIES:  
├── xDeepFM  
├── DCNv2  
└── attention-based feature interactions

This creates a **directed experiment graph** instead of a dumb list of experiments.

Your agent can then ask:

> What areas of the solution space have we explored?

Feature Engineering      ████████░░ 80%  
Architecture             ██████░░░░ 60%  
Training Strategy        ██░░░░░░░░ 20%  
Sequential Modelling     █░░░░░░░░░ 10%  
Multi-task Learning      ░░░░░░░░░░ 0%

That means the agent can deliberately explore **underexplored but promising areas**.

This is much more interesting than random trial-and-error.

---

# **4\. Give the agent a portfolio of experiments, not one experiment at a time**

This is another thing I would do differently.

The agent generates, say:

Candidate experiments:

A. Add time features  
Expected gain: \+0.002  
Cost: Low  
Confidence: High

B. Switch DeepFM → DCNv2  
Expected gain: \+0.010  
Cost: Medium  
Confidence: Medium

C. Add sequential user history  
Expected gain: \+0.020  
Cost: High  
Confidence: Medium

D. Tune embedding dimensions  
Expected gain: \+0.004  
Cost: Low  
Confidence: High

Then your system selects using something like:

Experiment Value \=  
Expected Improvement  
× Confidence  
× Novelty  
÷ Compute Cost

So instead of the LLM going:

> Hmm, maybe try DeepFM.

Your agent explicitly reasons:

DCNv2:  
High potential but moderate cost.

Embedding tuning:  
Low potential but extremely cheap.

Sequential model:  
High potential but expensive.

Decision:  
Run embedding tuning first.  
If no improvement, allocate budget to DCNv2.

This makes your agent **resource-aware**, which directly addresses the judging criteria.

Remember: they're scoring:

* token usage  
* GPU hours  
* practicality

So being intelligent about *what not to run* could actually be as important as finding a good model.

---

# **5\. Multi-fidelity experimentation: this could be HUGE**

This is one of the biggest things I would add.

Don't run every experiment on the full 1.1M training rows for full training.

That wastes time.

Instead:

                   NEW EXPERIMENT  
                           │  
                           ▼  
                  ┌─────────────────┐  
                  │ Level 1: Smoke  │  
                  │ 1% data         │  
                  │ 1 epoch         │  
                  └────────┬────────┘  
                           │  
                      Promising?  
                     /          \\  
                   NO            YES  
                   │              │  
                   ▼              ▼  
                REJECT      Level 2: Quick  
                            10% / few epochs  
                                  │  
                             Promising?  
                             /        \\  
                           NO          YES  
                           │            │  
                           ▼            ▼  
                        REJECT     Level 3:  
                                   Full Training

This is **how real experimental systems save compute**.

Your agent can kill bad experiments early.

For example:

Experiment: DCNv2

Quick evaluation:  
Baseline: 0.6016  
Current projected performance: 0.596

Decision:  
Terminate experiment early.  
GPU saved: 85%.

This could massively improve your **Feasibility & Practicality** story.

---

# **6\. Build a "metric-aware diagnosis engine"**

Don't just give the LLM:

Score \= 0.612

Give it an actual diagnosis.

For example:

TRAIN SCORE: 0.82  
VALIDATION SCORE: 0.61

→ possible overfitting.

Or:

NDCG@10 ↑  
Recall@50 ↓

The agent can interpret:

> We are ranking the items we retrieve better, but retrieving fewer relevant items overall.

Then it chooses an appropriate direction.

Metric Pattern → Possible Direction

Low NDCG \+ Low Recall  
→ Representation/model capacity problem

High Recall \+ Low NDCG  
→ Ranking problem

High NDCG \+ Low Recall  
→ Candidate coverage problem

Train ↑↑ / Validation ↓  
→ Overfitting

Both low  
→ Feature/data/model mismatch

Now your system isn't just:

> Score went down, try something else.

It has a **diagnostic policy**.

This can combine deterministic rules with LLM reasoning.

---

# **7\. I would NOT let the LLM freely rewrite everything**

This is where many agent projects become messy.

Instead, create a controlled experiment interface.

For example:

Experiment(  
    model="deepfm",  
    embedding\_dim=32,  
    hidden\_layers=\[256, 128\],  
    dropout=0.2,  
    features=\[  
        "user\_id",  
        "video\_id",  
        "author\_id",  
        "time\_bucket"  
    \]  
)

Then the agent can modify:

model  
features  
hyperparameters  
training strategy  
loss  
sampling strategy

through a structured experiment specification.

Only when the agent proposes something genuinely new does it need to modify source code.

This gives you:

80% experiments → cheap structured config  
20% experiments → LLM-generated code patches

That means:

* fewer LLM tokens  
* fewer broken runs  
* better reproducibility  
* easier logging

And honestly, **this is a much stronger engineering design** than "let an LLM edit Python files endlessly."

---

# **8\. Your secret weapon could be an Experiment Compiler**

This is something I think could sound very cool in your final presentation.

The LLM doesn't directly create experiments.

Instead:

LLM Research Idea  
       ↓  
Structured Experiment Spec  
       ↓  
Experiment Compiler  
       ↓  
Valid Python Config / Code Patch  
       ↓  
Validation  
       ↓  
Execution

Example:

{  
  "hypothesis": "Higher-order feature interactions are under-modelled.",  
  "intervention": {  
    "type": "architecture",  
    "from": "deepfm",  
    "to": "dcnv2"  
  },  
  "budget": {  
    "max\_epochs": 10,  
    "early\_stopping": 3  
  }  
}

Your **Experiment Compiler** knows how to translate that into actual code.

That makes your architecture much more reliable.

---

# **9\. Add a Research Critic**

Before running an expensive experiment:

Researcher:  
"Let's add a transformer."

        ↓

Critic:  
"Rejected."

Reason:  
\- Dataset contains limited sequential context.  
\- Previous temporal features showed weak benefit.  
\- Estimated compute cost is 5×.  
\- Expected improvement confidence is low.

This is basically an internal debate, but I wouldn't necessarily use two expensive LLM calls.

You can have:

Research proposal  
        ↓  
Deterministic checks  
\+  
Cheap critic prompt  
        ↓  
Approve / Reject / Simplify

The key is that every experiment must justify:

Why?  
Expected improvement?  
Cost?  
Risk?  
What previous evidence supports this?  
---

# **10\. A strong overall system could look like this**

┌────────────────────────────────────────────────────────────┐  
│                    AUTONOMOUS RESEARCH AGENT               │  
└────────────────────────────────────────────────────────────┘

                         ┌──────────────┐  
                         │ DATA PROFILER│  
                         └──────┬───────┘  
                                │  
                                ▼  
                         ┌──────────────┐  
                         │ RESEARCH MAP │  
                         │              │  
                         │ What we know │  
                         │ What failed  │  
                         │ Unexplored   │  
                         └──────┬───────┘  
                                │  
                                ▼  
                      ┌────────────────────┐  
                      │ RESEARCH STRATEGIST│  
                      │                    │  
                      │ Generate candidates│  
                      └─────────┬──────────┘  
                                │  
                   ┌────────────┼─────────────┐  
                   ▼            ▼             ▼  
                Idea A        Idea B         Idea C  
                   │            │             │  
                   └────────────┼─────────────┘  
                                ▼  
                      ┌────────────────────┐  
                      │ EXPERIMENT SELECTOR│  
                      │                    │  
                      │ Gain × Confidence  │  
                      │ × Novelty / Cost   │  
                      └─────────┬──────────┘  
                                │  
                                ▼  
                      ┌────────────────────┐  
                      │ EXPERIMENT COMPILER│  
                      └─────────┬──────────┘  
                                │  
                                ▼  
                       ┌──────────────────┐  
                       │ VALIDATION GATE  │  
                       └────────┬─────────┘  
                                │  
                                ▼  
                    ┌────────────────────────┐  
                    │ MULTI-FIDELITY RUNNER  │  
                    │                        │  
                    │ 1% → 10% → 100%        │  
                    └───────────┬────────────┘  
                                │  
                                ▼  
                      ┌────────────────────┐  
                      │ METRIC DIAGNOSIS   │  
                      └─────────┬──────────┘  
                                │  
                                ▼  
                      ┌────────────────────┐  
                      │ REFLECTION ENGINE  │  
                      │                    │  
                      │ Learn causal-ish   │  
                      │ experiment insight │  
                      └─────────┬──────────┘  
                                │  
                                ▼  
                         UPDATE RESEARCH MAP  
                                │  
                                └──────↺

## **Phase 1 — Get a dumb autonomous loop working**

Goal:

Baseline  
→ run  
→ evaluate  
→ choose from predefined experiments  
→ run next experiment

No fancy LLM yet.

We prove:

python run\_agent.py

works end-to-end.

---

## **Phase 2 — Build the experiment infrastructure**

Create:

experiments/  
    experiment\_001/  
        config.json  
        hypothesis.md  
        results.json  
        logs.txt

And a central database:

experiment\_history.json

Every experiment automatically records:

* hypothesis  
* change  
* metrics  
* runtime  
* GPU usage  
* token usage  
* errors  
* recovery actions

At this point you already satisfy a lot of the deliverable requirements.

---

## **Phase 3 — Add baseline model zoo**

Don't make the LLM generate every model.

Implement a few strong candidates:

FM  
DeepFM  
DCNv2  
Wide & Deep  
LightGBM

Then your agent chooses among them and tunes them.

Later, it can generate new implementations if necessary.

This gives you a stable base.

---

## **Phase 4 — Add LLM research reasoning**

Only now.

The LLM receives:

Dataset summary  
\+  
Metric history  
\+  
Experiment insights  
\+  
Available models  
\+  
Compute remaining  
\+  
Research map

And outputs:

{  
  "hypothesis": "...",  
  "experiment": "...",  
  "reasoning": "...",  
  "expected\_metric\_effect": {  
    "ndcg": "...",  
    "recall": "..."  
  },  
  "estimated\_cost": "...",  
  "priority": 0.87  
}  
---

## **Phase 5 — Add multi-fidelity \+ early termination**

This is when it starts becoming genuinely efficient.

Cheap experiment  
      ↓  
Looks bad? → Kill

Looks promising?  
      ↓  
More resources  
      ↓  
Still promising?  
      ↓  
Full training  
---

## **Phase 6 — Add recovery**

Handle:

OOM  
NaN  
Timeout  
Syntax Error  
Bad model output

automatically.

---

# **What would make your project genuinely stand out?**

If we execute this properly, your pitch becomes:

> **Most autonomous ML agents treat experimentation as a sequence of isolated code-generation tasks. Our agent treats experimentation as a resource-constrained research process.**

Then explain:

### **1\. 🧠 Research Map**

The agent accumulates **structured scientific memory** rather than raw chat history.

### **2\. 🎯 Metric-Aware Planning**

It diagnoses whether the weakness is:

* ranking  
* recall  
* overfitting  
* underfitting

and chooses experiments accordingly.

### **3\. 💰 Cost-Aware Experiment Selection**

The agent explicitly optimises:

Expected Improvement  
        vs  
GPU Cost  
        vs  
LLM Token Cost

### **4\. ⚡ Multi-Fidelity Evaluation**

Bad experiments are killed early.

### **5\. 🔄 Autonomous Recovery**

Failures become part of the research process instead of ending the run.

### **6\. 🧬 Experiment Graph**

The agent remembers **relationships between hypotheses and outcomes**, preventing repeated dead ends and identifying unexplored directions.

---

## **If I were trying to win with this**

My strategy would be:

**Don't try to claim:**

> "We built the smartest AI researcher ever."

Instead, build something that is **extremely demonstrable**.

At the final presentation, you should be able to show:

START  
  │  
  ▼  
Agent reproduced FM baseline: 0.6016  
  │  
  ▼  
Agent analysed metric weaknesses  
  │  
  ▼  
Generated 5 candidate research directions  
  │  
  ▼  
Rejected 3 due to poor gain/cost ratio  
  │  
  ▼  
Tested DeepFM  
  │  
  ▼  
Improved \+0.011  
  │  
  ▼  
Tested temporal features  
  │  
  ▼  
Performance dropped  
  │  
  ▼  
Agent recorded insight and avoided similar experiments  
  │  
  ▼  
Tested DCNv2 using multi-fidelity evaluation  
  │  
  ▼  
Improved again  
  │  
  ▼  
Converged automatically

That would be **such a strong story** because you're demonstrating all five things the judges care about:

* **Technical execution:** final score  
* **Innovation:** research graph \+ intelligent experiment selection  
* **Impact:** minimal human intervention  
* **Practicality:** controlled token/GPU usage  
* **Presentation:** an extremely clear visual narrative

### **My recommendation for our next step**

We should **not start coding models yet**.

First, let's design the actual project architecture properly into modules and define the **MVP → competitive → standout** versions.

Then when you get the starter kit, we plug the real dataset/evaluator into that architecture instead of rebuilding everything later.

And yes — I genuinely think there is a way to make this project **more than "ChatGPT runs DeepFM and tunes the learning rate."** The strongest angle is building an agent that learns from its *experiment history* and treats **compute, failure, and exploration itself as things to optimise**. That is where I think your project can become distinctive.

# Written Project Description

1. **Written Project Description (via Devpost)**  
* Provide a clear written description of your project that includes:  
  * How your solution addresses the problem statement  
  * Development tools used (e.g. VSCode, Colab, Jupyter)  
  * APIs used (e.g. OpenAI GPT-4o, Google Maps API)  
  * Libraries and frameworks used (e.g. Hugging Face Transformers, PyTorch, scikit-learn, pandas)  
  * Datasets and assets used (e.g. Google Local Reviews dataset, manually labelled data)

# Run & Iteration logs

**What logs are required** — this is a real deliverable (Deliverables section, item 3 in the PS), separate from the checkpoint idea above. For **every iteration** your agent runs, it needs to record:

* **Hypothesis** — what it intended to try and why  
* **Code diff** — the actual change applied  
* **Resulting metrics** — NDCG@10 / Recall@50 for that iteration  
* **Error/recovery events** — anything that broke and how it was handled

Plus a short summary at the end reporting the **total number of manual (human) interventions** during the run — that's the number judges use to score Autonomy.

# Links

[https://bytedance.larkoffice.com/wiki/DNtSwxgeciCS2nkiUefc5qqtnkf](https://bytedance.larkoffice.com/wiki/DNtSwxgeciCS2nkiUefc5qqtnkf) 

[https://github.com/9irija/TikTok\_TechJam](https://github.com/9irija/TikTok_TechJam) 

[https://kuairand.com/](https://kuairand.com/)

[https://tiktoktechjam2026.devpost.com/](https://tiktoktechjam2026.devpost.com/)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>