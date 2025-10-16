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

## 🎯 Event Definition & Modeling

To study the market impact of individual tweets, each tweet was aligned with the **nearest Tesla stock price observations** before and after posting.  

An **abnormal price reaction** was defined as a tweet followed by a **≥3% directional change** in Tesla’s short-term price movement.  
Tweets meeting this condition were labeled as **“abnormal”**, while others were labeled as **“normal”**.

The classification task was then framed as predicting whether a tweet would trigger an abnormal price reaction,
based solely on its textual properties (sentiment_score and final_relevance).

**Modeling Approach:**

- Logistic Regression (tested both standard and class-balanced versions)
- Features: sentiment_score, final_relevance
- Evaluation Metrics: Accuracy, Precision, Recall, F1

While the baseline model achieved high accuracy due to class imbalance, recall on abnormal events was limited -
highlighting the inherent challenge of detecting rare, high-impact tweets.

---

## 📊 Defining “Abnormal” Market Reactions

Before modeling, several alternative definitions were tested to formalize what counts as a **significant stock response** to a tweet.  
Each definition reflects a different *market behavior hypothesis*:

1. **Slope-based acceleration (`|slope_after| > 3 × |slope_before|`)**  
   - Focuses on *momentum shifts* - tweets that cause the price to accelerate sharply relative to its pre-tweet slope.  
   - Hypothesis: a strong reaction immediately after a tweet indicates that traders changed direction or intensity.  
   - Result: extremely few such cases, implying that true momentum flips are rare on short timeframes.

2. **Absolute percent change ≥ 5%**  
   - Looks for *large, visible price swings* within about an hour after the tweet.  
   - Hypothesis: only extreme events (e.g., “Tesla stock price is too high imo”) produce such reactions.  
   - Result: too strict - captured only a small set of dramatic movements.

3. **Absolute percent change ≥ 3%** *(final definition)*  
   - Captures *moderate but meaningful* intraday changes typical for a volatile stock like Tesla.  
   - Hypothesis: smaller shifts are still actionable for traders and reflect genuine sentiment-driven reactions.  
   - This threshold achieved a better balance between rarity and practical investability.

---

### 📈 Modeling Results

| Definition | Positive Label Ratio | Accuracy | Precision (Abnormal=1) | Recall | Interpretation |
|-------------|----------------------|-----------|--------------------------|---------|----------------|
| Slope ratio `>3×` | <1% | 0.95 | 0.00 | 0.00 | Almost no positive samples detected |
| Δ% ≥ 5 | ~16% | 0.70 | **0.17** | 0.22 | Captures only extreme events |
| Δ% ≥ 3 *(final)* | ~27% | 0.60 | **0.26** | 0.26 | Best trade-off - some predictive signal emerges |

Feature correlations showed that **sentiment** and **relevance** are nearly independent (`r ≈ 0.014`),  
with **relevance** having stronger model weight (`β ≈ 0.062`) and sentiment contributing minimally.

> From an investment standpoint, **precision** is the key metric -  
> when the model flags a tweet as “abnormal,” we care how often that signal is *truly actionable*.  
> While the baseline logistic regression achieves modest precision (~0.25),  
> it demonstrates that textual relevance alone carries some predictive power for short-term stock volatility.

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
