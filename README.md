# 📈 S&P 500 – Stock Market Data Analysis and Predictive Insights

An end-to-end Data Analytics and Machine Learning project that analyses historical S&P 500 stock market data to identify company performance, market trends, financial risk and relationships between stocks.

The project uses Python, Exploratory Data Analysis (EDA), statistical analysis and machine learning techniques to extract meaningful insights from historical stock market data.

## 📌 Project Overview

The S&P 500 is one of the most widely followed stock market indices, representing major publicly traded companies in the United States.

This project analyses historical daily stock market data to understand how companies performed over time, compare their risk levels, investigate sector behaviour and explore the use of machine learning for predicting future stock returns.

The project also uses K-Means clustering to group companies based on their historical financial characteristics and performs hypothetical investment scenario analysis.

## 🎯 Project Objectives

- Analyse historical stock prices and trading volumes.
- Identify companies with the highest average closing prices.
- Identify companies with the highest average trading volumes.
- Calculate daily returns and cumulative stock performance.
- Identify top-performing companies based on historical returns.
- Evaluate stock risk using annualised volatility and maximum drawdown.
- Compare historical performance across different market sectors.
- Analyse correlations between selected stocks.
- Visualise monthly market return patterns.
- Group companies using K-Means clustering.
- Predict future stock returns using Linear Regression.
- Classify stock return direction using Logistic Regression.
- Simulate hypothetical investment outcomes using historical return scenarios.

## 📊 Dataset Description

The project uses two CSV datasets:

### 1. Historical Stock Market Dataset

**File:** `all_stocks_5yr.csv`

Contains daily historical stock market information.

| Column | Description |
|---|---|
| date | Trading date |
| open | Opening stock price |
| high | Highest stock price |
| low | Lowest stock price |
| close | Closing stock price |
| volume | Number of shares traded |
| Name | Stock ticker symbol |

**Dataset information:**
- Total records: 619,040
- Unique ticker symbols: 505
- Historical period: 8 February 2013 – 7 February 2018

### 2. S&P 500 Company Information Dataset

**File:** `sp500_companies.csv`

Contains company information used to associate stock ticker symbols with company names and sector classifications.

Important columns include:

- Symbol
- Security
- GICS_Sector
- GICS_Sub_Industry
- Headquarters
- Founded_Year
- Employees
- Website

The two datasets are merged using their ticker symbols.

**Note:** The company reference dataset contains 503 records, whereas the historical dataset contains 505 unique tickers. Some historical tickers do not have matching company or sector information.

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Interactive development environment |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Scikit-learn | Machine learning and model evaluation |

## 🔍 Project Workflow

### 1. Data Loading and Preprocessing
- Load both CSV datasets.
- Inspect dataset structure and statistical information.
- Standardise ticker symbols.
- Merge historical stock data with company information.
- Handle missing values and duplicate records.
- Convert trading dates into datetime format.

### 2. Exploratory Data Analysis

**Company Performance**
- Average closing price analysis.
- Average trading volume analysis.
- Daily stock returns.
- Cumulative returns.
- Top-performing companies.

**Financial Risk Analysis**
- Annualised volatility.
- Risk versus return comparison.
- Maximum drawdown.

**Market Analysis**
- Sector-wise performance.
- Stock return correlation.
- Monthly market return analysis.

### 3. Unsupervised Machine Learning

**K-Means Clustering**

Companies are grouped using:
- Average daily return.
- Daily return volatility.
- Average trading volume.

The model uses feature standardisation and the Elbow Method to examine the clustering structure.

### 4. Predictive Analytics

**Linear Regression – Stock Return Prediction**

Estimates stock returns over the next 21 trading days using historical financial features.

Input features:
- 5-day return.
- 21-day return.
- 21-day rolling volatility.
- 5-day trading-volume change.

**Logistic Regression – Stock Direction Classification**

Classifies whether the next 21-trading-day stock return is positive or non-positive.

**What-If Investment Scenario Analysis**

Uses historical return percentiles to simulate:
- Pessimistic scenario.
- Normal scenario.
- Optimistic scenario.

The hypothetical investment amount used is ₹10,000.

## 📈 Key Findings

Some of the major observations from the historical analysis include:

| Analysis | Finding |
|---|---|
| Highest average closing price | Priceline (PCLN) |
| Highest average trading volume | Bank of America |
| Highest historical total return | Nvidia (NVDA) |
| Highest annualised volatility | Chesapeake Energy (CHK) |
| Largest maximum drawdown | Chesapeake Energy (CHK) |
| Best-performing calculated sector | Information Technology |
| Lowest-performing calculated sector | Energy |

**K-Means Clustering:** The model grouped 505 companies into three clusters based on historical return, volatility and trading volume.

**Linear Regression:** The model achieved an R² score of approximately -0.0117 on the test dataset, indicating weak predictive performance.

**Logistic Regression:** The model achieved approximately 61.61% accuracy. However, the confusion matrix revealed a strong bias towards predicting positive returns, demonstrating the limitations of the current classification model.

These findings describe the historical dataset and should not be interpreted as current market conditions or investment recommendations.

## 📂 Repository Structure

```text
S&P-500-Stock-Market-Analysis/
│
├── README.md
├── requirements.txt
│
├── S&P_Predictions.ipynb
│
├── all_stocks_5yr.csv
├── sp500_companies.csv
│
└── Analysis_Results/
    ├── company_performance.csv
    ├── volatility_analysis.csv
    ├── drawdown_analysis.csv
    └── other generated analysis files
```

*The Analysis_Results folder is generated when the notebook is executed. The exact output files depend on the notebook execution.*

## ⚙️ Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### Step 2: Navigate to the Project Directory

```bash
cd YOUR_REPOSITORY
```

### Step 3: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

### Step 6: Run the Project

Open `S&P_Predictions.ipynb` and execute the notebook cells sequentially.

Ensure that both CSV datasets are available in the project directory or update the dataset paths in the notebook.

## 📌 Limitations

- The historical stock dataset covers 2013–2018 and does not represent current market data.
- The closing prices are not adjusted closing prices; dividends and certain corporate actions are not incorporated into the calculated returns.
- Some historical ticker symbols do not match the company reference dataset.
- Sector performance is calculated using equal-weighted averages rather than official market-capitalisation-weighted sector indices.
- The predictive models use a limited number of historical features.
- The current Linear Regression and Logistic Regression results demonstrate that the selected features and models require further improvement.
- Historical performance does not guarantee future results.

## 🚀 Future Improvements

- Integrate updated stock market data.
- Experiment with Random Forest and Gradient Boosting algorithms.
- Improve feature engineering using technical indicators such as RSI, MACD and moving averages.
- Apply time-series cross-validation.
- Improve classification using class weighting and threshold tuning.
- Incorporate macroeconomic indicators and company financial information.
- Extend the analysis to portfolio optimisation and risk assessment.
- Develop an interactive dashboard using Streamlit.

## 👨‍💻 Author

**Thanajeyan**

B.Tech – Computer Science and Engineering  
Lovely Professional University, Punjab

**Project Domain:** Data Science, Predictive Analytics and Machine Learning

---

*This project was developed for academic and educational purposes. The analysis and predictions are not financial advice or investment recommendations.*
