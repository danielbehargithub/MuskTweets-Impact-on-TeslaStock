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

Each tweet therefore receives two key features:  
- `sentiment_score` - representing tone and polarity  
- `final_relevance` - representing contextual relevance to Tesla

## 🎯 Event Definition, Threshold Testing & Modeling

To study the market impact of individual tweets, each tweet was aligned with the **nearest Tesla stock price observations** before and after posting.  
The goal was to determine whether a tweet triggers an **abnormal short-term price reaction**, based solely on its textual properties (`sentiment_score` and `final_relevance`).

In this context, an **abnormal event** represents a tweet that causes a *meaningful, trade-worthy change* in Tesla’s stock trajectory —  
the kind of move that an investor could realistically act following the tweet.

### Defining “Abnormal” Reactions

Three definitions were tested, each reflecting a different type of short-term price behavior:

1. **Slope-based acceleration (`|slope_after| > 3 × |slope_before|`)**  
   → Measures abrupt changes in *price momentum* — how sharply the stock accelerates after a tweet.

2. **Absolute percent change ≥ 5%**  
   → Detects *large intraday swings* that would be considered major market reactions.

3. **Absolute percent change ≥ 3% (final definition)**  
   → Captures *moderate but significant short-term movements* that are still actionable for traders.

Tweets meeting the chosen condition (≥3%) were labeled as **abnormal**, others as **normal**.

---

### Modeling Approach

- **Model:** Logistic Regression (tested both standard and class-balanced versions)  
- **Features:** `sentiment_score`, `final_relevance`  
- **Metrics:** Accuracy, Precision, Recall, F1
  
- From an investment standpoint, **precision** is the key metric.
Every positive prediction represents a trade decision or capital exposure.
A low precision means the model generates false alarms — triggering trades that risk money
without a solid underlying signal.


---

### 📈 Experimental Results

| Definition | Abnormal Ratio | Accuracy | Precision (Abnormal=1) | Recall | Interpretation |
|-------------|----------------------|-----------|--------------------------|---------|----------------|
| Slope ratio `>3×` | <1% | 0.95 | 0.00 | 0.00 | Almost no positive samples detected |
| Δ% ≥ 5 | ~16% | 0.70 | **0.17** | 0.22 | Captures only extreme events |
| Δ% ≥ 3 *(final)* | ~27% | 0.60 | **0.26** | 0.26 | Best trade-off — first meaningful predictive signal |

Feature correlations confirmed that **sentiment** and **relevance** were nearly independent (`r ≈ 0.014`),  
with **relevance** showing some influence on price movement (`Coefficients ≈ 0.062`), where **sentiment** coefficients was nearlly zero.


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
