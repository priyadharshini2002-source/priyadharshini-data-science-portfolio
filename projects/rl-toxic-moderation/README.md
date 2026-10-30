# Learning When to Act — RL for Adaptive Toxic Content Moderation

**MAT340 Reinforcement Learning & Decision Making · MSc Data Science, SASTRA Deemed to be University**
Author: Priyadharshini Sivasankar

A case study comparing **SARSA, Q-Learning, Expected SARSA and contextual-bandit policies** with different exploration strategies (greedy, ε-greedy, decaying ε, UCB, Thompson sampling). Each agent decides whether to **Approve, Escalate to human review, or Remove** a comment. The decision uses a toxicity classifier's score, a limited human-review budget and the comment's position in the moderation shift.

## Setup
- **Data:** Jigsaw Toxic Comment dataset (159,571 comments) split with no leakage: 60% trains the classifier, and 40% is held out for the RL environment.
- **Classifier:** TF-IDF + Logistic Regression (held-out AUC 0.972, PR-AUC 0.873), plus a deliberately degraded variant used to test robustness.
- **MDP:** 200 states (toxicity score × remaining review budget × shift position), 3 actions. Rewards: +1 for a correct decision, −5 for a missed toxic comment, −3 for a false removal, and a cost for each escalation.
- **Evaluation:** the main experiment includes concept drift and is run over multiple seeds (results are mean ± 95% CI). It also includes a sensitivity analysis on the cost of a missed comment and a test of how classifier quality affects results.

## Key results
| Policy | Eval reward |
|---|---|
| **Q-Learning · UCB** (best learned) | **62.0 ± 1.2** |
| Expected SARSA · Decaying ε | 60.1 ± 1.6 |
| Bandit · ε-greedy | 59.4 ± 3.4 |
| Two-threshold rule 0.3 / 0.7 (best fixed) | 57.3 ± 0.9 |
| Single threshold 0.5 | 45.3 |

- The best learned policy beat the best hand-tuned rule by about 8% in reward, and missed fewer toxic comments.
- The exploration strategy mattered more than the learning rule: the reward spread was 15.7 across exploration strategies versus 5.9 across algorithms.
- On-policy SARSA was more conservative near the review-budget limit than off-policy Q-Learning, and it adapted better during drift.
- A weaker classifier lowered achievable reward for every policy (Q-Learning 73.4 → 64.2). Moderation quality therefore depends on the classifier as much as on the decision policy.

## Interactive demo
The notebook ends with a 4-tab **Gradio** app where you can moderate a single comment, simulate a moderation shift, compare agents, and inspect learned policy maps.

## Files
- `RL_Toxic_Moderation_Case_Study.ipynb`: full Colab notebook (run all, then upload Jigsaw `train.csv`)
- `RL_Case_Study_Report.docx`: end-semester case-study report

**Tech:** Python · NumPy · pandas · scikit-learn · Matplotlib · Gradio · Google Colab

Companion project: [Toxic Comment Detection — Sequence Models vs Transformers](../toxic-comment-transformers/)
