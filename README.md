# 🧠 Predicting Tesla Stock Reactions to Elon Musk's Tweets

This project explores the short-term relationship between **Elon Musk’s tweets** and **Tesla’s stock price movements**.  
The goal was to test whether tweet sentiment and relevance can predict **abnormal price changes** in the hours following a tweet.

---

## 🚀 Project Overview

- **Objective:** Analyze how the sentiment and relevance of Elon Musk’s tweets affect Tesla’s hourly stock performance.  
- **Approach:** Combine NLP-based text analysis with financial time series data to model “abnormal” stock reactions.  
- **Core Idea:** A tweet may cause a sudden acceleration or drop in price movement.  
  We define such reactions as *abnormal increases* and try to predict them from textual features.

---

## 🧩 Data Sources

| Source | Description |
|--------|--------------|
| **Tweets dataset** | Elon Musk’s tweets (up to June 2023), including text, date, and metadata. |
| **Stock data (yfinance)** | Tesla hourly stock prices (2 years back from latest tweet). |
| **Libraries used** | `sentence-transformers`, `TextBlob`, `pandas`, `yfinance`, `scikit-learn`, `matplotlib` |

---

## 🧠 Methodology

1. **Text Preprocessing**
   - Removed URLs and special characters  
   - Computed sentiment using `TextBlob`  
   - Encoded semantic similarity with `SentenceTransformer (all-MiniLM-L6-v2)`  

2. **Relevance Scoring**
   - Matched tweets against curated Tesla-related keyword lists  
   - Computed a weighted “Tesla relevance” and “Financial relevance” score  

3. **Financial Alignment**
   - Merged tweet timestamps with Tesla hourly stock prices (`merge_asof`)  
   - Calculated slopes, percent changes, and 1-hour post-tweet deltas  

4. **Label Definition**
   ```python
   threshold = 3
   merged_df['absolute_change'] = abs(merged_df['percent_change_after'] - merged_df['percent_change_before'])
   merged_df['is_abnormal_increase'] = (merged_df['absolute_change'] > threshold).astype(int)
   ```
   - Tweets causing >3% change in price movement were labeled as “abnormal”.

5. **Modeling**
   - Logistic Regression (baseline & class-balanced)  
   - Features: `sentiment_score`, `final_relevance`  
   - Evaluation via Accuracy, Precision, Recall, F1  

---

## 📈 Results

| Metric | Baseline | Balanced |
|--------|-----------|-----------|
| Accuracy | 0.95 | 0.60 |
| Recall (Abnormal) | 0.00 | 0.26 |
| Precision (Abnormal) | 0.00 | 0.26 |

- Without class balancing, the model ignored rare abnormal cases (95% accuracy but 0 recall).  
- After balancing, recall improved — the model started detecting real abnormal reactions, though accuracy dropped.  
- Sentiment alone was not a strong predictor; relevance contributed slightly to separation.

---

## 🎨 Visualization

**Decision Boundary & Predictions**

![Decision Boundary](images/decision_boundary.png)

- Most points fall in the “non-abnormal” region.  
- Tweets with higher relevance and moderate sentiment show slightly higher abnormality likelihood.

---

## 💡 Key Insights

- Most tweets do **not** cause measurable stock movement — market remains efficient.  
- Tweets **highly relevant to Tesla operations** show stronger correlation with price volatility.  
- Sentiment polarity alone has limited predictive power — context and timing matter.  
- Future models could incorporate:
  - Tweet embeddings (BERT, finBERT)
  - Trading volume
  - Market sentiment indices

---

## 🧰 Tech Stack

`Python`, `pandas`, `scikit-learn`, `yfinance`, `sentence-transformers`, `TextBlob`, `matplotlib`

---

## 📂 Repository Structure

```
Tesla_Tweets_Stock_Prediction/
│
├── data/                        # (optional) tweets_with_analysis_scores_2_years.csv
├── notebooks/
│   └── Cleaning_tweets_data.ipynb
├── images/
│   └── decision_boundary.png
├── src/
│   └── model_utils.py
└── README.md
```

---

## 👤 Author

**Daniel Behar**  
🎓 B.Sc. Biotechnology Engineering, Braude College  
💻 B.Sc. Data Science & Engineering, Technion (expected Oct 2025)  
📍 QC & Analytical Development Associate @ Minovia Therapeutics  
🔗 [LinkedIn](https://linkedin.com/in/daniel-behar) | [GitHub](https://github.com/danielbehar)

---

## 📜 License

This project is for educational and research purposes only.  
Stock data © Yahoo Finance. Tweets © Twitter / X.
