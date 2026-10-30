# Patient Sentiment from Drug Reviews — BiLSTM vs DistilBERT vs BERT-base

**MSc Data Science, SASTRA Deemed to be University** · Author: Priyadharshini Sivasankar

Binary sentiment classification (Negative vs Positive) of patient drug reviews from **Drugs.com** (UCI / Kaggle).
Ratings 1–5 are labelled Negative and ratings 6–10 Positive. The 215,063 raw reviews became 128,108 after cleaning and de-duplication, and were split 80/10/10 with stratification and no text overlap between splits. Class weights handle the 70/30 imbalance.

| Model | Test Accuracy | Macro-F1 | Weighted-F1 | Params (M) | Train time (min) |
|---|---|---|---|---|---|
| **BERT-base** | **0.901** | **0.882** | **0.901** | 109.5 | 57.4 |
| DistilBERT | 0.900 | 0.880 | 0.900 | 67.0 | 28.9 |
| BiLSTM | 0.869 | 0.846 | 0.870 | 6.3 | 7.3 |

## Key findings
- Both transformers beat the BiLSTM by about 3 points of accuracy. The largest gain is on the minority **Negative** class (F1 0.785 for BiLSTM vs 0.83 for the transformers).
- **DistilBERT matches BERT-base** (0.880 vs 0.882 macro-F1) with about 40% fewer parameters and half the training time, so it is the practical choice for deployment.
- BERT-base had the smallest train–validation gap, so it overfit the least.
- Error analysis shows the remaining mistakes are mostly mixed reviews, where the drug worked but caused side effects, and borderline ratings (around 5–6).

## Files
- `Drug_Review_Sentiment_BiLSTM_DistilBERT_BERT.ipynb`: full Colab notebook (data download, cleaning, three models, comparison, error analysis)

**Tech:** Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas · Matplotlib · Google Colab (T4 GPU)
