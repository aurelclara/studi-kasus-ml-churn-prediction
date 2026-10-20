"""
Studi Kasus: Prediksi & Segmentasi Churn Pelanggan Telco
Resume kuliah tamu "Introduction to Machine Learning" - Dr. Taniza Tajudin
(Informatics Practitioner Lecture Series 2026)

Alur mengikuti kerangka CRISP-DM / workflow ML dari materi:
  1. Business Understanding
  2. Data Understanding & Preparation  (missing values, anomaly detection, scaling, encoding)
  3. Modeling                          (supervised, unsupervised, semi-supervised, reinforcement)
  4. Evaluation                        (accuracy, precision, recall, F1, MAE, RMSE, R2, fairness)
  5. Deployment                        (simpan & pakai ulang model)

Data disimulasikan (seed tetap), jadi hasil selalu reproducible.
Skrip ini menulis results.json dan folder figures/, yang dibaca oleh generate_report.py.

Jalankan:
    python ml_analysis.py
    python generate_report.py
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # tanpa layar (aman untuk server / CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RANDOM_STATE = 42

NUM_FEATURES = ["age", "tenure_months", "num_addons", "support_calls", "monthly_charges", "total_charges"]
CAT_FEATURES = ["contract", "internet_service", "payment_method"]
# gender sengaja TIDAK dipakai sebagai fitur (lihat bagian fairness)

R = {}  # hasil terstruktur untuk generate_report.py


# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def tbl(df, index_name=""):
    """DataFrame -> dict sederhana agar mudah disimpan ke JSON."""
    d = df.rename_axis(index_name).reset_index()
    return {"columns": [str(c) for c in d.columns], "rows": d.to_numpy(dtype=object).tolist()}


def save_fig(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)


def build_preprocessor(num_cols, cat_cols):
    """Imputasi missing values + normalisasi (numerik) + one-hot encoding (kategorikal)."""
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


# --------------------------------------------------------------------------
# DATA UNDERSTANDING: simulasi data pelanggan
# --------------------------------------------------------------------------
def generate_customers(n=3000, seed=RANDOM_STATE):
    """Dataset sintetis pelanggan telco, sengaja dibuat 'kotor' seperti data nyata."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 71, n)
    gender = rng.choice(["Laki-laki", "Perempuan"], n)
    tenure = rng.integers(1, 73, n)  # lama berlangganan (bulan)
    contract = rng.choice(["Bulanan", "Tahunan", "Dua Tahun"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber", "Tanpa Internet"], n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(["Transfer", "E-Wallet", "Kartu Kredit"], n, p=[0.40, 0.40, 0.20])
    num_addons = rng.integers(0, 4, n)
    support_calls = rng.poisson(1.5, n)

    base = pd.Series(internet).map({"DSL": 30, "Fiber": 55, "Tanpa Internet": 15}).to_numpy()
    monthly = np.clip(base + 8 * num_addons + rng.normal(0, 4, n), 10, None)
    total = np.clip(monthly * tenure + rng.normal(0, 25, n), 0, None)

    # Injeksi anomali (misal: salah billing / aktivitas mencurigakan), 1,5% data
    is_anomaly = np.zeros(n, dtype=int)
    anomaly_idx = rng.choice(n, int(n * 0.015), replace=False)
    is_anomaly[anomaly_idx] = 1
    support_calls[anomaly_idx] = rng.integers(20, 40, len(anomaly_idx))
    monthly[anomaly_idx] = rng.uniform(200, 350, len(anomaly_idx))
    total[anomaly_idx] = monthly[anomaly_idx] * tenure[anomaly_idx]

    # Label churn dari model logistik (gender TIDAK berpengaruh)
    logit = (
        -0.6
        + 1.3 * (contract == "Bulanan")
        - 1.0 * (contract == "Dua Tahun")
        - 0.035 * tenure
        + 0.02 * (monthly - 45)
        + 0.30 * np.minimum(support_calls, 6)
        + 0.5 * (internet == "Fiber")
        - 0.01 * (age - 40)
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
            "gender": gender,
            "age": age,
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "num_addons": num_addons,
            "support_calls": support_calls.astype(float),
            "monthly_charges": monthly.round(2),
            "total_charges": total.round(2),
            "is_anomaly_injected": is_anomaly,  # hanya untuk evaluasi, bukan fitur
            "churn": churn,
        }
    )
    # Injeksi missing values
    for col, frac in [("total_charges", 0.03), ("support_calls", 0.03), ("payment_method", 0.02)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan
    return df


def data_understanding(df):
    section("1. DATA UNDERSTANDING")
    missing = df.isna().sum()
    missing = missing[missing > 0].to_frame("jumlah_missing")
    print(f"Jumlah data   : {len(df)} pelanggan, {df.shape[1]} kolom")
    print(f"Proporsi churn: {df['churn'].mean():.1%}")
    print("Missing values per kolom:")
    print(missing.to_string())

    by_contract = df.groupby("contract")["churn"].mean().sort_values().to_frame("churn_rate")
    bins = pd.cut(df["tenure_months"], [0, 12, 24, 48, 72], labels=["1-12", "13-24", "25-48", "49-72"])
    by_tenure = df.groupby(bins, observed=True)["churn"].mean().to_frame("churn_rate")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    by_contract["churn_rate"].plot.bar(ax=axes[0], color="#3b6fb6", rot=0)
    axes[0].set_title("Tingkat churn per jenis kontrak")
    axes[0].set_ylabel("Proporsi churn")
    by_tenure["churn_rate"].plot.bar(ax=axes[1], color="#d9822b", rot=0)
    axes[1].set_title("Tingkat churn per lama berlangganan (bulan)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Proporsi churn")
    save_fig(fig, "01_eksplorasi_churn.png")

    R["eda"] = {
        "n_rows": len(df),
        "n_cols": int(df.shape[1]),
        "churn_rate": float(df["churn"].mean()),
        "missing": tbl(missing, "kolom"),
        "by_contract": tbl(by_contract, "kontrak"),
        "by_tenure": tbl(by_tenure, "lama_berlangganan_bulan"),
    }


# --------------------------------------------------------------------------
# DATA PREPARATION + ANOMALY DETECTION (Unsupervised)
# --------------------------------------------------------------------------
def remove_anomalies(df):
    section("2. DATA PREPARATION: ANOMALY DETECTION (Isolation Forest)")
    cols = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls"]
    X = SimpleImputer(strategy="median").fit_transform(df[cols])
    flagged = IsolationForest(contamination=0.02, random_state=RANDOM_STATE).fit_predict(X) == -1
    truth = df["is_anomaly_injected"].to_numpy() == 1
    tp = int((flagged & truth).sum())
    print(f"Baris ditandai anomali : {flagged.sum()}")
    print(f"Anomali sebenarnya     : {truth.sum()} (diketahui karena data simulasi)")
    print(f"Anomali tertangkap     : {tp} (recall {tp / truth.sum():.1%}, precision {tp / flagged.sum():.1%})")
    R["anomaly"] = {
        "flagged": int(flagged.sum()),
        "true": int(truth.sum()),
        "caught": tp,
        "recall": tp / int(truth.sum()),
        "precision": tp / int(flagged.sum()),
    }
    return df.loc[~flagged].reset_index(drop=True)


# --------------------------------------------------------------------------
# MODELING: SUPERVISED - KLASIFIKASI
# --------------------------------------------------------------------------
def run_classification(df):
    section("3. MODELING - SUPERVISED: KLASIFIKASI CHURN")
    X, y = df[NUM_FEATURES + CAT_FEATURES], df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Data latih: {len(X_train)} | Data uji: {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    results, fitted, preds = {}, {}, {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", build_preprocessor(NUM_FEATURES, CAT_FEATURES)), ("model", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        fitted[name], preds[name] = pipe, y_pred

    table = pd.DataFrame(results).T.round(3)
    best_name = table["f1"].idxmax()
    print(table.to_string())
    print(f"Model terbaik (F1): {best_name}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, models):
        ConfusionMatrixDisplay.from_predictions(
            y_test, preds[name], ax=ax, display_labels=["Bertahan", "Churn"], colorbar=False, cmap="Blues"
        )
        ax.set_title(name)
    save_fig(fig, "02_confusion_matrix.png")

    rf = fitted["Random Forest"]
    names = pd.Index(rf.named_steps["prep"].get_feature_names_out()).str.replace(r"^(num|cat)__", "", regex=True)
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp.sort_values().plot.barh(ax=ax, color="#3b6fb6")
    ax.set_title("10 fitur terpenting (Random Forest)")
    ax.set_xlabel("Importance")
    save_fig(fig, "03_feature_importance.png")
    print("Fitur terpenting:")
    print(imp.round(3).to_string())

    R["classification"] = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "table": tbl(table, "model"),
        "best": best_name,
        "importance": tbl(imp.round(3).to_frame("importance"), "fitur"),
    }
    return {
        "best_name": best_name, "best_pipe": fitted[best_name],
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "y_pred": preds[best_name],
    }


# --------------------------------------------------------------------------
# MODELING: SUPERVISED - REGRESI
# --------------------------------------------------------------------------
def run_regression(df):
    section("4. MODELING - SUPERVISED: REGRESI (tagihan bulanan)")
    num = ["age", "tenure_months", "num_addons", "support_calls"]
    cat = ["contract", "internet_service", "payment_method"]
    X, y = df[num + cat], df["monthly_charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    results, preds = {}, {}
    for name, reg in models.items():
        pipe = Pipeline([("prep", build_preprocessor(num, cat)), ("model", reg)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": r2_score(y_test, pred),
        }
        preds[name] = pred
    table = pd.DataFrame(results).T.round(3)
    print(table.to_string())

    best = table["MAE"].idxmin()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_test, preds[best], s=10, alpha=0.5, color="#3b6fb6")
    lim = [y_test.min(), y_test.max()]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Tagihan sebenarnya")
    ax.set_ylabel("Tagihan prediksi")
    ax.set_title(f"Aktual vs prediksi ({best})")
    save_fig(fig, "04_regresi_aktual_vs_prediksi.png")
    R["regression"] = {"table": tbl(table, "model"), "best": best}


# --------------------------------------------------------------------------
# MODELING: UNSUPERVISED - CLUSTERING + PCA
# --------------------------------------------------------------------------
def run_clustering(df):
    section("5. MODELING - UNSUPERVISED: SEGMENTASI (K-Means + PCA)")
    cols = ["age", "tenure_months", "monthly_charges", "support_calls", "num_addons"]
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[cols]))

    ks, inertia, sil = list(range(3, 7)), [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs, km.labels_))
    best_k = ks[int(np.argmax(sil))]
    k_table = pd.DataFrame({"inertia": np.round(inertia, 1), "silhouette": np.round(sil, 3)}, index=ks)
    print(k_table.to_string())
    print(f"k terpilih (silhouette tertinggi): {best_k}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(ks, inertia, "o-", color="#3b6fb6")
    axes[0].set_title("Elbow method")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(ks, sil, "o-", color="#d9822b")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("k")
    save_fig(fig, "05_pemilihan_k.png")

    labels = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit_predict(Xs)
    profile = df.assign(segmen=labels).groupby("segmen").agg(
        jumlah=("churn", "size"), umur=("age", "mean"), lama_langganan=("tenure_months", "mean"),
        tagihan=("monthly_charges", "mean"), telp_support=("support_calls", "mean"),
        layanan_tambahan=("num_addons", "mean"), churn_rate=("churn", "mean"),
    ).round(2)
    print(profile.to_string())

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pts = pca.fit_transform(Xs)
    var = float(pca.explained_variance_ratio_.sum())
    print(f"PCA: 2 komponen menjelaskan {var:.1%} variansi")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="viridis", s=10, alpha=0.7)
    ax.legend(*sc.legend_elements(), title="Segmen")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Segmen pelanggan pada ruang PCA")
    save_fig(fig, "06_segmen_pca.png")

    R["clustering"] = {
        "k_table": tbl(k_table, "k"), "best_k": best_k,
        "profile": tbl(profile, "segmen"), "pca_variance": var,
    }


# --------------------------------------------------------------------------
# MODELING: SEMI-SUPERVISED
# --------------------------------------------------------------------------
def run_semi_supervised(clf):
    section("6. MODELING - SEMI-SUPERVISED (self-training, 10% data berlabel)")
    y_train, y_test = clf["y_train"].to_numpy(), clf["y_test"].to_numpy()
    prep = build_preprocessor(NUM_FEATURES, CAT_FEATURES)
    Xtr, Xte = prep.fit_transform(clf["X_train"]), prep.transform(clf["X_test"])

    labeled, _ = train_test_split(np.arange(len(y_train)), train_size=0.10, stratify=y_train, random_state=RANDOM_STATE)
    y_partial = np.full(len(y_train), -1)  # -1 = tidak berlabel
    y_partial[labeled] = y_train[labeled]
    print(f"Data berlabel: {len(labeled)} dari {len(y_train)} data latih")

    def make():
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    models = {
        "Supervised (10% label saja)": make().fit(Xtr[labeled], y_train[labeled]),
        "Semi-supervised (self-training)": SelfTrainingClassifier(make(), threshold=0.8).fit(Xtr, y_partial),
        "Supervised (100% label, pembanding)": make().fit(Xtr, y_train),
    }
    rows = {}
    for name, m in models.items():
        pred = m.predict(Xte)
        rows[name] = {"accuracy": accuracy_score(y_test, pred), "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred)}
    table = pd.DataFrame(rows).T.round(3)
    print(table.to_string())
    R["semi"] = {"n_labeled": len(labeled), "n_train": len(y_train), "table": tbl(table, "skenario")}


# --------------------------------------------------------------------------
# MODELING: REINFORCEMENT LEARNING (epsilon-greedy bandit)
# --------------------------------------------------------------------------
def run_reinforcement_learning():
    section("7. MODELING - REINFORCEMENT LEARNING (epsilon-greedy bandit)")
    offers = ["Diskon 20%", "Upgrade kecepatan gratis", "Cashback 50rb"]
    true_rates = np.array([0.12, 0.28, 0.20])  # peluang berhasil, TIDAK diketahui agen
    n_customers = 5000

    def run(policy, epsilon=0.1):
        rng = np.random.default_rng(RANDOM_STATE)
        counts, values, rewards = np.zeros(3), np.zeros(3), np.zeros(n_customers)
        for t in range(n_customers):
            if policy == "fixed":
                a = 0
            elif policy == "random":
                a = int(rng.integers(3))
            elif counts.min() == 0:
                a = int(np.argmin(counts))  # coba tiap tawaran sekali dulu
            elif rng.random() < epsilon:
                a = int(rng.integers(3))  # eksplorasi
            else:
                a = int(np.argmax(values))  # eksploitasi
            r = float(rng.random() < true_rates[a])
            counts[a] += 1
            values[a] += (r - values[a]) / counts[a]  # rata-rata inkremental
            rewards[t] = r
        return rewards, counts, values

    rows, fig_data = [], {}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy, label in [("epsilon", "Epsilon-greedy (RL)"), ("random", "Acak"), ("fixed", "Selalu tawaran #1")]:
        rewards, counts, values = run(policy)
        ax.plot(np.cumsum(rewards), label=label)
        rows.append([label, int(rewards.sum()), float(rewards.mean()), counts.astype(int).tolist()])
        print(f"{label:22s}: {int(rewards.sum())} dipertahankan ({rewards.mean():.1%}), pilihan {counts.astype(int).tolist()}")
        if policy == "epsilon":
            fig_data["estimates"] = values.round(3).tolist()
    ax.set_xlabel("Jumlah pelanggan yang ditawari")
    ax.set_ylabel("Kumulatif pelanggan dipertahankan")
    ax.set_title("Reinforcement learning: memilih tawaran retensi")
    ax.legend()
    save_fig(fig, "08_rl_bandit.png")
    R["rl"] = {
        "offers": offers, "true_rates": true_rates.tolist(),
        "estimates": fig_data["estimates"], "rows": rows,
    }


# --------------------------------------------------------------------------
# EVALUATION: FAIRNESS (etika)
# --------------------------------------------------------------------------
def fairness_check(df, clf):
    section("8. EVALUATION - ETIKA: PENGECEKAN FAIRNESS")
    idx = clf["X_test"].index
    ev = pd.DataFrame(
        {
            "gender": df.loc[idx, "gender"].to_numpy(),
            "kelompok_usia": pd.cut(df.loc[idx, "age"], [0, 29, 49, 200], labels=["<30", "30-49", "50+"]).astype(str),
            "y_true": clf["y_test"].to_numpy(),
            "y_pred": clf["y_pred"],
        }
    )
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, attr in zip(axes, ["gender", "kelompok_usia"]):
        rows = []
        for g, sub in ev.groupby(attr):
            rows.append(
                {
                    attr: g, "n": len(sub), "churn_aktual": sub["y_true"].mean(),
                    "diprediksi_churn": sub["y_pred"].mean(),
                    "recall": recall_score(sub["y_true"], sub["y_pred"], zero_division=0),
                    "precision": precision_score(sub["y_true"], sub["y_pred"], zero_division=0),
                }
            )
        tab = pd.DataFrame(rows).set_index(attr).round(3)
        ratio = float(tab["diprediksi_churn"].min() / tab["diprediksi_churn"].max())
        print(f"\nKelompok: {attr}")
        print(tab.to_string())
        print(f"Disparate impact ratio: {ratio:.2f} (aturan 80%: >= 0.80)")
        out[attr] = {"table": tbl(tab, attr), "ratio": round(ratio, 3)}
        tab[["churn_aktual", "diprediksi_churn", "recall"]].plot.bar(ax=ax, rot=0)
        ax.set_title("Perbandingan antar kelompok: " + attr.replace("_", " "))
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
    save_fig(fig, "07_fairness.png")
    R["fairness"] = out


# --------------------------------------------------------------------------
# DEPLOYMENT
# --------------------------------------------------------------------------
def deploy(clf):
    section("9. DEPLOYMENT - simpan model & prediksi pelanggan baru")
    path = ROOT / "churn_model.joblib"
    joblib.dump(clf["best_pipe"], path)
    model = joblib.load(path)  # dimuat ulang seperti di lingkungan produksi
    new_customers = pd.DataFrame(
        [
            {"age": 27, "tenure_months": 3, "num_addons": 0, "support_calls": 5, "monthly_charges": 60.0,
             "total_charges": 180.0, "contract": "Bulanan", "internet_service": "Fiber", "payment_method": "E-Wallet"},
            {"age": 45, "tenure_months": 60, "num_addons": 2, "support_calls": 0, "monthly_charges": 46.0,
             "total_charges": 2760.0, "contract": "Dua Tahun", "internet_service": "DSL", "payment_method": "Transfer"},
        ]
    )
    proba = model.predict_proba(new_customers)[:, 1]
    demo = []
    for i, (row, p) in enumerate(zip(new_customers.to_dict("records"), proba), start=1):
        status = "Risiko churn tinggi" if p >= 0.5 else "Kemungkinan bertahan"
        print(f"Pelanggan {i}: probabilitas churn = {p:.1%} -> {status}")
        demo.append({"no": i, "profil": row, "probabilitas": float(p), "status": status})
    R["deploy"] = {"model": clf["best_name"], "demo": demo}


def main():
    df = generate_customers()
    data_understanding(df)
    df = remove_anomalies(df)
    clf = run_classification(df)
    run_regression(df)
    run_clustering(df)
    run_semi_supervised(clf)
    run_reinforcement_learning()
    fairness_check(df, clf)
    deploy(clf)

    (ROOT / "results.json").write_text(
        json.dumps(R, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    print("\nSelesai. Jalankan `python generate_report.py` untuk membuat report.html")


if __name__ == "__main__":
    main()
