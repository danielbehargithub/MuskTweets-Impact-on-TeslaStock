# 🧠 Predicting Tesla Stock Reactions to Elon Musk's Tweets

This project explores the short-term relationship between **Elon Musk’s tweets** and **Tesla’s stock price movements**.  
The goal was to test whether tweet sentiment and relevance can predict **abnormal price changes** in the hours following a tweet.

---

## 🚀 Project Overview

This project examines the short-term market impact of **Elon Musk’s tweets** on **Tesla’s stock price**.  
Prior research has shown that social media activity, particularly from influential figures like Elon Musk can affect investor sentiment and market behavior.  
Building upon previous findings by [Stokanović Šević et al. (2022)](https://www.atlantis-press.com/proceedings/iciitb-22/125984182),  
who analyzed the **semantic sentiment** of Musk’s tweets and demonstrated a relationship between their sentiment and Tesla’s market performance,  
and [Strauss & Smith (2019)](https://www.emerald.com/ccij/article-pdf/24/4/593/408731/ccij-09-2018-0091.pdf),  
who highlighted that such effects often occur **within minutes to hours**,  
this project focuses on **intra-day reactions** - exploring whether the sentiment and relevance of Musk’s tweets  
can predict **abnormal short-term movements** in Tesla’s stock price.

- **Objective:** Quantify the short-term (intra-day) impact of Elon Musk’s tweets on Tesla’s stock price by analyzing both **sentiment** and **topic relevance**.  
- **Approach:** Combine NLP-based sentiment and semantic similarity analysis with **hourly financial data** from Yahoo Finance to model abnormal price responses.  
- **Core Idea:** Tweets act as real-time market signals - we define **abnormal movements** as short-term price changes exceeding expected behavior,  
  and test whether tweet-level textual features can predict these reactions.

---

## 🧩 Data Sources

| Source | Description |
|--------|--------------|
| **Tweets dataset** | Historical tweets by Elon Musk (collected up to mid-2023), including text, timestamp, and metadata. Based on the [Elon Musk Tweets Dataset by G. Preda](https://www.kaggle.com/datasets/gpreda/elon-musk-tweets), publicly available on Kaggle. |
| **Stock data (Yahoo Finance)** | Tesla’s hourly stock prices covering a two-year window preceding the last tweet in the dataset, retrieved via [Yahoo Finance](https://finance.yahoo.com/quote/TSLA/) using the `yfinance` library. |
| **Libraries used** | `sentence-transformers`, `TextBlob`, `pandas`, `yfinance`, `matplotlib` |

> 📘 *Tweets were used strictly for analytical and educational purposes. Stock data and tweet timestamps were aligned to evaluate short-term (intra-day) reactions.*

---

## 🧠 Methodology

Both notebooks - [`Cleaning_tweets_data.ipynb`](https://github.com/danielbehargithub/MuskTweets-Impact-on-TeslaStock/blob/main/Cleaning_tweets_data.ipynb) and [`tweets_features.ipynb`](https://github.com/danielbehargithub/MuskTweets-Impact-on-TeslaStock/blob/main/tweets_features.ipynb) - together handle tweet cleaning and feature construction:

- Removed URLs and special characters  
- Built a **keyword relevance feature** using weighted term matching across Tesla, technology, and financial domains  
- Computed sentiment scores (1–10) using **TextBlob**  
- Encoded semantic similarity between each tweet and a Tesla reference paragraph using **SentenceTransformer (`all-MiniLM-L6-v2`)**

Finally, the **keyword relevance** and **semantic similarity** components were fused into a unified **final relevance score**.  
The combination used an *adaptive weighting scheme* - tweets containing richer Tesla-related vocabulary received higher weight on the semantic similarity component, while others were "punished" by there low vocabulary frequency.  
This allowed the model to balance explicit term presence with contextual meaning.  

The following examples illustrate how the adaptive weighting translates textual content into relevance scores:

- **“Tesla stock price is too high imo”**  
  → A historically impactful tweet that caused an immediate stock drop.  
  It includes direct Tesla and financial terms, yielding a **keyword score of 6**,  
  a **semantic similarity score of 8.6**, and a **final relevance of 8.2**.  
  High overall relevance reflects both explicit Tesla mention and strong contextual match.

- **“I should clarify: Tesla stock is obviously high based on past & present, but low if you believe in Tesla's future.”**  
  → Semantically rich and Tesla-focused - **final relevance ≈ 8.9**.

- **“Thanks for the longstanding faith in SpaceX...”**  
  → Although similar in tone, this tweet concerns **SpaceX**, not Tesla.  
  Only a minor overlap in clean energy terminology yields a **keyword score of 2**  
  and a **final relevance of ~2.5**, correctly identifying it as off-topic.

This demonstrates how the **adaptive weighting scheme** rewards tweets that are semantically aligned with Tesla-related language,  
while off-topic messages - even from the same author - receive substantially lower relevance scores.


Each tweet therefore receives two key features:  
- `sentiment_score` - representing tone and polarity  
- `final_relevance` - representing contextual relevance to Tesla

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
- After balancing, recall improved - the model started detecting real abnormal reactions, though accuracy dropped.  
- Sentiment alone was not a strong predictor; relevance contributed slightly to separation.

---

## 🎨 Visualization

**Decision Boundary & Predictions**

![Decision Boundary](images/decision_boundary.png)

- Most points fall in the “non-abnormal” region.  
- Tweets with higher relevance and moderate sentiment show slightly higher abnormality likelihood.

---

## 💡 Key Insights

- Most tweets do **not** cause measurable stock movement - market remains efficient.  
- Tweets **highly relevant to Tesla operations** show stronger correlation with price volatility.  
- Sentiment polarity alone has limited predictive power - context and timing matter.  
- Future models could incorporate:
  - Tweet embeddings (BERT, finBERT)
  - Trading volume
  - Market sentiment indices

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

## 📜 License

This project is for educational and research purposes only.  
Stock data © Yahoo Finance. Tweets © Twitter / X.
