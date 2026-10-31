from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)

def load_data():
    stocks_path = ROOT / "all_stocks_5yr.csv"
    companies_path = ROOT / "sp500_companies.csv"
    if not stocks_path.exists() or not companies_path.exists():
        raise FileNotFoundError(
            "Place all_stocks_5yr.csv and sp500_companies.csv beside app.py."
        )

    stocks = pd.read_csv(stocks_path)
    companies = pd.read_csv(companies_path)

    company_info = companies.rename(columns={
        "Symbol": "Ticker", "Security": "Company_Name",
        "GICS_Sector": "Sector", "GICS_Sub_Industry": "Industry"
    })
    stocks["Ticker"] = stocks["Name"].astype(str).str.strip().str.upper()
    company_info["Ticker"] = company_info["Ticker"].astype(str).str.strip().str.upper()

    df = stocks.drop(columns=["Name"]).merge(
        company_info[["Ticker", "Company_Name", "Sector", "Industry"]],
        on="Ticker", how="left"
    )
    df["Company_Name"] = df["Company_Name"].fillna(df["Ticker"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(["date", "Ticker"]).dropna(
        subset=["date", "close", "Ticker"]
    )
    df = df.sort_values(["Ticker", "date"]).reset_index(drop=True)
    df["Daily_Return"] = df.groupby("Ticker")["close"].pct_change() * 100
    df["Cumulative_Return"] = df.groupby("Ticker")["Daily_Return"].transform(
        lambda x: ((1 + x.fillna(0) / 100).cumprod() - 1) * 100
    )
    df["Wealth_Index"] = df.groupby("Ticker")["Daily_Return"].transform(
        lambda x: (1 + x.fillna(0) / 100).cumprod()
    )
    df["Drawdown"] = df["Wealth_Index"] / df.groupby("Ticker")["Wealth_Index"].cummax() - 1
    return df

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/metadata")
def metadata():
    try:
        df = load_data()
        companies = (df[["Ticker", "Company_Name"]].drop_duplicates()
                     .sort_values("Company_Name").to_dict("records"))
        sectors = sorted(df["Sector"].dropna().unique().tolist())
        return jsonify({
            "companies": companies, "sectors": sectors,
            "min_date": df["date"].min().strftime("%Y-%m-%d"),
            "max_date": df["date"].max().strftime("%Y-%m-%d"),
            "rows": int(len(df)), "tickers": int(df["Ticker"].nunique())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/dashboard")
def dashboard():
    try:
        df = load_data()
        ticker = request.args.get("ticker", "AAPL")
        start = pd.to_datetime(request.args.get("start", df["date"].min()))
        end = pd.to_datetime(request.args.get("end", df["date"].max()))
        sectors = request.args.getlist("sector")
        data = df[(df.date >= start) & (df.date <= end)].copy()
        if sectors:
            data = data[data["Sector"].isin(sectors) | data["Sector"].isna()]
        company = data[data.Ticker == ticker].sort_values("date")
        if company.empty:
            return jsonify({"error": "No observations for this company and date range."}), 400

        first, last = float(company.close.iloc[0]), float(company.close.iloc[-1])
        metrics = {
            "avg_close": float(company.close.mean()),
            "max_close": float(company.close.max()),
            "min_close": float(company.close.min()),
            "period_return": float((last / first - 1) * 100) if first else None,
            "total_volume": float(company.volume.sum()),
            "company_name": str(company.Company_Name.iloc[0]),
            "ticker": ticker
        }
        price = company[["date", "close"]].copy()
        price["date"] = price.date.dt.strftime("%Y-%m-%d")
        price["moving_average"] = company.close.rolling(30, min_periods=1).mean().values

        monthly_volume = company.set_index("date")["volume"].resample("MS").sum()
        volume = [{"date": d.strftime("%Y-%m"), "volume": float(v)}
                  for d, v in monthly_volume.items()]

        returns = company.Daily_Return.dropna()
        hist_counts, bins = np.histogram(returns, bins=35) if len(returns) else ([], [])
        histogram = [{"range": f"{bins[i]:.2f} to {bins[i+1]:.2f}",
                      "count": int(hist_counts[i])} for i in range(len(hist_counts))]

        perf = data.groupby(["Ticker", "Company_Name"]).agg(
            first=("close", "first"), last=("close", "last"),
            avg_volume=("volume", "mean")
        ).reset_index()
        perf["total_return"] = (perf["last"] / perf["first"] - 1) * 100
        top = perf.nlargest(10, "total_return").sort_values("total_return")

        risk = data.groupby(["Ticker", "Company_Name"]).agg(
            daily_vol=("Daily_Return", "std"),
            total_return=("Cumulative_Return", "last"),
            max_drawdown=("Drawdown", "min")
        ).reset_index()
        risk["annual_vol"] = risk.daily_vol * np.sqrt(252)
        risk = risk.replace([np.inf, -np.inf], np.nan).dropna(subset=["annual_vol"])

        sector_daily = data.dropna(subset=["Sector"]).groupby(
            ["date", "Sector"]
        ).Daily_Return.mean().reset_index()
        sector_wide = sector_daily.pivot(index="date", columns="Sector", values="Daily_Return")
        sector_growth = ((1 + sector_wide.fillna(0) / 100).cumprod() - 1) * 100
        sector_lines = [{"date": d.strftime("%Y-%m-%d"), "sector": s, "return": float(v)}
                        for s in sector_growth.columns
                        for d, v in sector_growth[s].items() if pd.notna(v)]
        sector_close = data.dropna(subset=["Sector"]).groupby("Sector").close.mean().sort_values()
        sector_avg = [{"sector": s, "avg_close": float(v)} for s, v in sector_close.items()]

        corr_tickers = data.groupby("Ticker").volume.mean().nlargest(12).index.tolist()
        pivot = data.pivot_table(index="date", columns="Ticker", values="Daily_Return")
        corr = pivot.reindex(columns=corr_tickers).corr().fillna(0)
        corr_data = [{"x": x, "y": y, "value": float(corr.loc[y, x])}
                     for y in corr.index for x in corr.columns]

        market_daily = data.groupby("date").Daily_Return.mean()
        monthly = ((1 + market_daily / 100).resample("MS").prod() - 1) * 100
        monthly_data = [{"month": d.strftime("%Y-%m"), "return": float(v)}
                        for d, v in monthly.items() if pd.notna(v)]

        # K-Means on the same three company-level features used in the notebook.
        cluster = data.groupby("Ticker").agg(
            company=("Company_Name", "first"),
            avg_return=("Daily_Return", "mean"),
            volatility=("Daily_Return", "std"),
            avg_volume=("volume", "mean")
        ).dropna()
        cluster["log_volume"] = np.log1p(cluster.avg_volume)
        cluster_points = []
        if len(cluster) >= 3:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            features = cluster[["avg_return", "volatility", "log_volume"]]
            scaled = StandardScaler().fit_transform(features)
            labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)
            cluster["cluster"] = labels.astype(str)
            cluster_points = [{"ticker": idx, "company": r.company,
                               "avg_return": float(r.avg_return),
                               "volatility": float(r.volatility),
                               "cluster": str(r.cluster)}
                              for idx, r in cluster.iterrows()]

        # Predictive modelling: chronological holdout with 21-observation purge gap.
        work = df.sort_values(["Ticker", "date"]).copy()
        work["r1"] = work.groupby("Ticker").close.pct_change()
        work["r5"] = work.groupby("Ticker").close.pct_change(5)
        work["r21"] = work.groupby("Ticker").close.pct_change(21)
        work["vol21"] = work.groupby("Ticker").r1.transform(lambda x: x.rolling(21).std())
        work["volchg5"] = work.groupby("Ticker").volume.pct_change(5)
        work["target"] = work.groupby("Ticker").close.shift(-21) / work.close - 1
        features = ["r5", "r21", "vol21", "volchg5"]
        model = work[["date", "Ticker", "Company_Name", "target"] + features]
        model = model.replace([np.inf, -np.inf], np.nan).dropna()
        model_dates = sorted(model.date.unique())
        reg_metrics, cls_metrics, pred_rows = {}, {}, []
        if len(model_dates) > 60:
            split_i = int(len(model_dates) * .8)
            split_date = model_dates[split_i]
            train_end = model_dates[max(0, split_i - 21)]
            train = model[model.date < train_end]
            test = model[model.date >= split_date]
            if len(train) and len(test):
                from sklearn.linear_model import LinearRegression, LogisticRegression
                from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                    r2_score, accuracy_score, precision_score, recall_score, f1_score,
                    confusion_matrix)
                Xtr, Xte = train[features], test[features]
                ytr, yte = train.target, test.target
                reg = LinearRegression().fit(Xtr, ytr)
                yp = reg.predict(Xte)
                reg_metrics = {"mae": float(mean_absolute_error(yte, yp)),
                               "rmse": float(np.sqrt(mean_squared_error(yte, yp))),
                               "r2": float(r2_score(yte, yp)),
                               "split_date": pd.Timestamp(split_date).strftime("%Y-%m-%d")}
                ytr_cls = (ytr > 0).astype(int)
                yte_cls = (yte > 0).astype(int)
                clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr_cls)
                cp = clf.predict(Xte)
                cm = confusion_matrix(yte_cls, cp, labels=[0, 1])
                cls_metrics = {"accuracy": float(accuracy_score(yte_cls, cp)),
                               "precision": float(precision_score(yte_cls, cp, zero_division=0)),
                               "recall": float(recall_score(yte_cls, cp, zero_division=0)),
                               "f1": float(f1_score(yte_cls, cp, zero_division=0)),
                               "confusion": cm.tolist()}
                latest = work.sort_values("date").groupby("Ticker").tail(1).dropna(subset=features)
                latest = latest.copy()
                latest["predicted_return"] = reg.predict(latest[features]) * 100
                latest["probability_rise"] = clf.predict_proba(latest[features])[:, 1] * 100
                pred_rows = latest.sort_values("predicted_return", ascending=False).head(15)[
                    ["Ticker", "Company_Name", "date", "predicted_return", "probability_rise"]
                ].assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_dict("records")

        # Historical pooled 21-day return scenarios.
        target_values = model.target.dropna()
        scenarios = {}
        if len(target_values):
            for key, value in {
                "pessimistic": target_values.quantile(.10),
                "normal": target_values.median(),
                "optimistic": target_values.quantile(.90)
            }.items():
                scenarios[key] = float(value)

        return jsonify({
            "metrics": metrics,
            "price": price.to_dict("records"),
            "volume": volume,
            "histogram": histogram,
            "top_performers": top[["Ticker", "Company_Name", "total_return"]].to_dict("records"),
            "risk": risk[["Ticker", "Company_Name", "annual_vol", "total_return", "max_drawdown"]].to_dict("records"),
            "sector_lines": sector_lines, "sector_avg": sector_avg,
            "correlation": corr_data, "correlation_tickers": corr_tickers,
            "monthly": monthly_data, "clusters": cluster_points,
            "regression": reg_metrics, "classification": cls_metrics,
            "predictions": pred_rows, "scenarios": scenarios,
            "rows": int(len(data))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
