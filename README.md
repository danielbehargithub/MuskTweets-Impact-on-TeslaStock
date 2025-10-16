# 🧠 Predicting Tesla Stock Reactions to Elon Musk's Tweets

This project explores the short-term relationship between **Elon Musk’s tweets** and **Tesla’s stock price movements**.  
The goal was to test whether tweet sentiment and relevance can predict **abnormal price changes** in the hours following a tweet.

---

## 🚀 Project Overview

This project examines the short-term market impact of **Elon Musk’s tweets** on **Tesla’s stock price**.  
Prior research has shown that social media activity, particularly from influential figures like Elon Musk can affect investor sentiment and market behavior. Building upon previous findings by [Stokanović Šević et al. (2022)](https://www.atlantis-press.com/proceedings/iciitb-22/125984182), who analyzed the **semantic sentiment** of Musk’s tweets and demonstrated a relationship between their sentiment and Tesla’s market performance, and [Strauss & Smith (2019)](https://www.emerald.com/ccij/article-pdf/24/4/593/408731/ccij-09-2018-0091.pdf), who highlighted that such effects often occur **within minutes to hours**, this project focuses on **intra-day reactions** - exploring whether the sentiment and relevance of Musk’s tweets can predict **abnormal short-term movements** in Tesla’s stock price.

- **Objective:** Quantify the short-term (intra-day) impact of Elon Musk’s tweets on Tesla’s stock price by analyzing both **sentiment** and **topic relevance**.
- **Approach:** Combine NLP-based sentiment and semantic similarity analysis with **hourly financial data** from Yahoo Finance to model abnormal price responses.  
- **Core Idea:** Tweets act as real-time market signals - we define **abnormal movements** as short-term price changes exceeding expected behavior, and test whether tweet-level textual features can predict these reactions.

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
The combination used an *adaptive weighting scheme* - tweets containing richer Tesla-related vocabulary received higher weight on the semantic similarity component, while others were "punished" by thier low vocabulary frequency.  
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

## 🎯 Modeling

The modeling was mainly in this notebook [`logistic_regression_model.ipynb`](https://github.com/danielbehargithub/MuskTweets-Impact-on-TeslaStock/blob/main/logistic_regression_model.ipynb)

To study the market impact of individual tweets, each tweet was aligned with the **nearest Tesla stock price observations** before and after posting.
The goal was to determine whether a tweet triggers an **abnormal short-term price reaction**, based solely on its textual properties (`sentiment_score` and `final_relevance`).

In this context, an **abnormal event** represents a tweet that causes a *meaningful, trade-worthy change* in Tesla’s stock trajectory- the kind of move that an investor could realistically act following the tweet.

### Defining “Abnormal” Reactions


Three definitions were tested, each reflecting a different type of short-term price behavior:

1. **Slope-based acceleration (`|slope_after| > 3 × |slope_before|`)**  
   → Measures abrupt changes in *price momentum*- how sharply the stock accelerates after a tweet.

2. **Absolute percent change ≥ 5%**  
   → Detects *large intraday swings* that would be considered major market reactions.

3. **Absolute percent change ≥ 3% (final definition)**  
   → Captures *moderate but significant short-term movements* that are still actionable for traders.

Tweets meeting the chosen condition (≥3%) were labeled as **abnormal**, others as **normal**.

The relative price change Δ% was computed as  ((**Price_after** − **Price_before**) / **Price_before**) × 100

---

### Modeling Approach

- **Model:** Logistic Regression (tested both standard and class-balanced versions)  
- **Features:** `sentiment_score`, `final_relevance`  
- **Metrics:** Accuracy, Precision, Recall, F1
  
- From an investment standpoint, **precision** is the key metric.
Every positive prediction represents a trade decision or capital exposure.
A low precision means the model generates false alarms- triggering trades that risk money
without a solid underlying signal.


---

### 📈 Experimental Results

| Definition | Abnormal Ratio | Accuracy | Precision (Abnormal=1) | Recall | Interpretation |
|-------------|----------------------|-----------|--------------------------|---------|----------------|
| Slope ratio `>3×` | <1% | 0.95 | 0.00 | 0.00 | Almost no positive samples detected |
| Δ% ≥ 5 | ~16% | 0.70 | **0.17** | 0.22 | Captures only extreme events |
| Δ% ≥ 3 *(final)* | ~27% | 0.60 | **0.26** | 0.26 | Best trade-off- first meaningful predictive signal |

Feature correlations confirmed that **sentiment** and **relevance** were nearly independent (`r ≈ 0.014`),  
with **relevance** showing some influence on price movement (`Coefficients ≈ 0.062`), where **sentiment** coefficients were nearly zero.


---

## 🎨 Visualization

#### 1. Feature–Prediction Relationship
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/0e45b59b-c724-4364-8304-257845f8fed2" />

These scatterplots show how each feature relates to the model’s predicted probability of an **abnormal** price reaction.  
Each blue dot represents the model’s confidence, and red markers denote the true labels.  
While `sentiment_score` alone shows weak correlation, `final_relevance` displays some gradient-  
tweets with higher relevance tend to receive slightly higher predicted probabilities.

---

#### 2. Decision Boundary Visualization
<img width="1589" height="598" alt="image" src="https://github.com/user-attachments/assets/0dfc53d2-e456-4f3b-a233-27c702cafeee" />

Each point (tweet) receives a predicted **probability of being “abnormal”**,  
based on its position in the 2D feature space- `sentiment_score` on the x-axis and `final_relevance` on the y-axis.

- In the **left plot**, the background color represents the model’s confidence level:  
  yellow–red areas indicate *higher predicted probability* of an abnormal reaction, while green zones indicate *lower probability*.  
  The dashed line marks the 0.5 decision threshold- everything **above/right** of it would be classified as abnormal.  
  The nearly uniform yellow coloring shows that the model struggles to create distinct probability regions- most tweets receive similar confidence scores.

- In the **right plot**, that same decision rule is applied in a binary form:  
  Red = *predicted normal*, Blue = *predicted abnormal*. Pink dots represent tweets that were actually labeled as abnormal.  
  The dense overlap between normal and abnormal samples indicates that the current features provide limited discriminative power.
  This lack of clear separation suggests that the relationship between tweet content and stock reaction is more complex than what linear models and simple sentiment/relevance scores can capture.
---

## ⚠️ Limitations

- **Not every tweet–market link is causal.**  
  Musk’s influence extends beyond Tesla; tweets unrelated to the company can still shake the market through political or economic sentiment.

- **Feature reliability.**  
  Some clearly Tesla-related tweets were misclassified as low relevance, showing limits in the keyword and embedding logic.

- **Sentiment misinterpretation.**  
  `TextBlob` struggled with irony, sarcasm, and ambiguous tone- common in Musk’s writing style.

- **Market timing issues.**  
  Tweets posted **outside trading hours** skewed short-term reaction estimates.

- **Narrow feature space.**  
  Only `sentiment_score` and `final_relevance` were modeled, omitting engagement, trading volume, volatility, and other context signals.

- **Modeling simplicity.**  
  Logistic regression assumes linear separability; nonlinear models (e.g., gradient boosting, transformers) could capture deeper linguistic–financial dynamics.

---

## 🧭 Summary

Working on this project was both **challenging and fascinating**- not because of the final metrics, but because of the complexity behind them.  
Exploring how a single tweet from Elon Musk might move a trillion-dollar company’s stock turned out to be a deep lesson in **data realism**, **model design**, and **human language**.

The process highlighted how many subtle factors must be considered- from market timing and model selection, to NLP feature quality and contextual ambiguity.  
Even with simple tools, it was exciting to see how social and financial data intertwine, and how small modeling decisions can change an entire interpretation.

Despite these gaps, the project provides a solid starting point for exploring how text, time, and finance intersect- and a reminder that even imperfect models can reveal meaningful patterns.

---

## 📜 License

This project is for educational and research purposes only.  
Stock data © Yahoo Finance. Tweets © Twitter / X.

---

## 🎲 Bonus: Investment Toy Model

As a small experiment, an **investment-oriented toy analysis** was added- to imagine how a trader might use the model’s predictions in practice.

### 📈 Expected Return Map
Each model output can be interpreted as a “trade” on whether a tweet will cause an abnormal price move.  
Assuming an average **+3% gain** for a correct signal and a **−1% loss** for a false one, the following contour map visualizes the **expected return** as a function of model precision and recall:

Calculation of minimum value of precision to earn from this toy model:
Min Precision = 0.01 / (0.03 + 0.01) = 0.25
So we are slightly earning from this model

---

### ⚙️ Sharpe Ratio Snapshot
To estimate risk-adjusted performance, a basic **Sharpe ratio** was computed for the subset of tweets labeled as “abnormal”, assuming a 2% risk-free rate and using our toy return assumptions.
**Results:**
- Sharpe Ratio: -0.05
- Mean Return: -0.345
- Std Dev: 6.52

This negative Sharpe confirms that under naïve assumptions- the model’s predictive signals are **disadvantageous**.  
The exercise highlights the real-world importance of improving **precision**, as every false signal directly translates into unnecessary market exposure.

