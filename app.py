from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import os

# strojno učenje
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# pandas
def clean_data(df):
    df = df.copy()

    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna()

    df["leto"] = pd.to_numeric(df["leto"], errors="coerce")
    df["kolicina"] = pd.to_numeric(df["kolicina"], errors="coerce")

    df = df[df["kolicina"] > 0]

    Q1 = df["kolicina"].quantile(0.25)
    Q3 = df["kolicina"].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df["kolicina"] >= Q1 - 1.5 * IQR) &
        (df["kolicina"] <= Q3 + 1.5 * IQR)
    ]

    df = df.sort_values(by=["obcina", "leto"])

    return df

# ----------------------------------------------------------
#  MONTE CARLO SIMULACIJA
# ----------------------------------------------------------
def monte_carlo_for_obcina(df, row_index, n_future=10, n_sim=5000):
    """
    df        : pandas DataFrame z Excel datoteko (header=None)
    row_index : indeks vrstice izbrane občine
    n_future  : število prihodnjih let
    n_sim     : število simulacij
    """

    # letnice (prva vrstica, stolpci B-G)
    years = df.iloc[0, 1:7].astype(int).values

    # podatki iz izbrane občine
    values = df.iloc[row_index, 1:7].astype(float).values
    values = values[values > 0]

    # logaritmične stopnje rasti
    growth_rates = np.diff(np.log(values))
    last_value = values[-1]

    sims = np.zeros((n_sim, n_future))
    for i in range(n_sim):
        rates = np.random.choice(growth_rates, n_future, replace=True)
        sims[i] = last_value * np.exp(np.cumsum(rates))

    return {
        "years": list(range(years[-1] + 1, years[-1] + 1 + n_future)),
        "mean": sims.mean(axis=0).round(1).tolist(),
        "low": np.percentile(sims, 5, axis=0).round(1).tolist(),
        "high": np.percentile(sims, 95, axis=0).round(1).tolist()
    }


# ----------------------------------------------------------
#  STROJNO UČENJE – POLINOMSKA REGRESIJA
# ----------------------------------------------------------
def ml_forecast(years, values, n_future=10, degree=2):
    X = np.array(years).reshape(-1, 1)
    y = np.array(values)

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    future_years = np.array(
        range(years[-1] + 1, years[-1] + 1 + n_future)
    ).reshape(-1, 1)

    future_poly = poly.transform(future_years)
    forecast = model.predict(future_poly)

    return forecast.round(1).tolist()


# ----------------------------------------------------------
#  FLASK APLIKACIJA
# ----------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "data", "podatki-odpadki.xlsx")


# ----------------------------------------------------------
#  ROUTES
# ----------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/tabela")
def tabela():
    df = pd.read_excel(EXCEL_PATH)

    columns = df.columns.tolist()
    data = df.to_dict(orient="records")

    return render_template(
        "tabela.html",
        columns=columns,
        data=data
    )


@app.route("/simulacija")
def simulacija():
    df = pd.read_excel(EXCEL_PATH, header=None)
    obcine = df.iloc[1:, 0].astype(str).str.strip().tolist()
    return render_template("simulacija.html", obcine=obcine)


@app.route("/simulacija/data")
def simulacija_data():
    obcina = request.args.get("obcina")
    n_future = request.args.get("n_future", default=10, type=int)

    df = pd.read_excel(EXCEL_PATH, header=None)
    obcine = df.iloc[1:, 0].astype(str).str.strip().tolist()

    if obcina not in obcine:
        return jsonify({"error": "Občina ne obstaja"}), 400

    idx = obcine.index(obcina) + 1  # +1 zaradi letnic

    # Monte Carlo
    result = monte_carlo_for_obcina(df, idx, n_future=n_future)

    # zgodovinski podatki
    years_hist = df.iloc[0, 1:7].astype(int).tolist()
    hist_values = df.iloc[idx, 1:7].astype(float).tolist()

    # strojno učenje
    ml_prediction = ml_forecast(
        years_hist,
        hist_values,
        n_future=n_future,
        degree=2
    )

    result["hist_years"] = years_hist
    result["hist_values"] = hist_values
    result["ml"] = ml_prediction

    return jsonify(result)


# ----------------------------------------------------------
#  ZAGON APLIKACIJE
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
