"""
Membuat report.html (satu file, grafik di-embed base64) dari hasil ml_analysis.py.

Jalankan setelah ml_analysis.py:
    python ml_analysis.py
    python generate_report.py
"""
import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results.json"
FIG_DIR = ROOT / "figures"
OUT = ROOT / "report.html"

if not RESULTS.exists():
    raise SystemExit("results.json belum ada. Jalankan dulu: python ml_analysis.py")
R = json.loads(RESULTS.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Helper pembuat HTML
# --------------------------------------------------------------------------
def num(v, decimals=3, pct=False):
    """Format angka gaya Indonesia (koma desimal, titik ribuan)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return html.escape(str(v))
    if pct:
        return f"{v * 100:.1f}%".replace(".", ",")
    if isinstance(v, int):
        return f"{v:,}".replace(",", ".")
    s = f"{v:,.{decimals}f}"  # 1,234.5 -> 1.234,5
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def table(t, pct_cols=(), decimals=3, decimals_map=None, bold_first=None, caption=None):
    decimals_map = decimals_map or {}
    head = "".join("<th>" + html.escape(c.replace("_", " ")) + "</th>" for c in t["columns"])
    body = []
    for row in t["rows"]:
        strong = bold_first is not None and row[0] == bold_first
        cells = []
        for col, v in zip(t["columns"], row):
            cells.append(f"<td>{num(v, decimals_map.get(col, decimals), col in pct_cols)}</td>")
        attr = ' class="best"' if strong else ""
        body.append("<tr" + attr + ">" + "".join(cells) + "</tr>")
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return f'<div class="tablewrap"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def img(name, alt, width=None):
    data = base64.b64encode((FIG_DIR / name).read_bytes()).decode()
    style = f' style="max-width:{width}px;margin-left:auto;margin-right:auto"' if width else ""
    return f'<figure{style}><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def row_of(t, first_value):
    for row in t["rows"]:
        if row[0] == first_value:
            return dict(zip(t["columns"], row))
    raise KeyError(first_value)


def card(label, value, note=""):
    return f'<div class="kpi"><div class="kpi-v">{value}</div><div class="kpi-l">{html.escape(label)}</div><div class="kpi-n">{html.escape(note)}</div></div>'


# --------------------------------------------------------------------------
# Ambil angka penting
# --------------------------------------------------------------------------
eda, an = R["eda"], R["anomaly"]
cl, rg, cs, sm, rl, fr, dp = R["classification"], R["regression"], R["clustering"], R["semi"], R["rl"], R["fairness"], R["deploy"]

best_cls = row_of(cl["table"], cl["best"])
lr, rf = row_of(cl["table"], "Logistic Regression"), row_of(cl["table"], "Random Forest")
best_reg = row_of(rg["table"], rg["best"])
sil_vals = [r[2] for r in cs["k_table"]["rows"]]
sm_sup = row_of(sm["table"], "Supervised (10% label saja)")
sm_semi = row_of(sm["table"], "Semi-supervised (self-training)")
sm_full = row_of(sm["table"], "Supervised (100% label, pembanding)")
rl_rows = {r[0]: r for r in rl["rows"]}
age_tab = {r[0]: dict(zip(fr["kelompok_usia"]["table"]["columns"], r)) for r in fr["kelompok_usia"]["table"]["rows"]}
gen_tab = {r[0]: dict(zip(fr["gender"]["table"]["columns"], r)) for r in fr["gender"]["table"]["rows"]}
lowest_recall_age = min(age_tab.items(), key=lambda kv: kv[1]["recall"])

# --------------------------------------------------------------------------
# Isi laporan
# --------------------------------------------------------------------------
dataset_cols = {
    "columns": ["kolom", "keterangan"],
    "rows": [
        ["gender, age", "Demografi pelanggan (gender TIDAK dipakai sebagai fitur model)"],
        ["tenure_months", "Lama berlangganan (bulan)"],
        ["contract", "Jenis kontrak: Bulanan / Tahunan / Dua Tahun"],
        ["internet_service", "DSL / Fiber / Tanpa Internet"],
        ["payment_method", "Transfer / E-Wallet / Kartu Kredit"],
        ["num_addons, support_calls", "Jumlah layanan tambahan; jumlah telepon ke customer service"],
        ["monthly_charges, total_charges", "Tagihan bulanan dan total tagihan"],
        ["churn", "TARGET: 1 = berhenti berlangganan"],
        ["is_anomaly_injected", "Penanda anomali sisipan, hanya untuk evaluasi (bukan fitur)"],
    ],
}

crisp = {
    "columns": ["tahap CRISP-DM", "penerapan di proyek ini", "konsep dari materi"],
    "rows": [
        ["1. Business Understanding", "Mengurangi churn pelanggan telco; menentukan 7 pertanyaan analisis", "ML vs pemrograman tradisional"],
        ["2. Data Understanding & Preparation", "Eksplorasi, imputasi missing values, deteksi anomali, normalisasi, one-hot encoding", "Pra-pemrosesan data, unsupervised (anomaly detection)"],
        ["3. Modeling", "Klasifikasi, regresi, K-Means + PCA, self-training, epsilon-greedy bandit", "Supervised, unsupervised, semi-supervised, reinforcement learning"],
        ["4. Evaluation", "Accuracy, precision, recall, F1, ROC-AUC, MAE, RMSE, R², silhouette, fairness", "Evaluasi metrik, etika & fairness"],
        ["5. Deployment", "Model disimpan dengan joblib, dimuat ulang untuk prediksi pelanggan baru", "Deployment, pengawasan manusia"],
    ],
}

questions = [
    "Pelanggan mana yang berisiko churn? (klasifikasi)",
    "Berapa tagihan bulanan yang wajar untuk profil tertentu? (regresi)",
    "Segmen pelanggan apa saja yang ada? (clustering)",
    "Adakah data tidak wajar yang perlu dibuang sebelum modeling? (anomaly detection)",
    "Bisakah model tetap berguna dengan label yang sangat sedikit? (semi-supervised)",
    "Tawaran retensi mana yang paling efektif? (reinforcement learning)",
    "Apakah model adil untuk semua kelompok pelanggan? (etika)",
]

deploy_rows = {
    "columns": ["pelanggan", "profil singkat", "probabilitas_churn", "status"],
    "rows": [
        [
            f"Pelanggan {d['no']}",
            f"usia {d['profil']['age']}, {d['profil']['tenure_months']} bulan, kontrak {d['profil']['contract']}, "
            f"{d['profil']['support_calls']} telepon support",
            d["probabilitas"],
            d["status"],
        ]
        for d in dp["demo"]
    ],
}

rl_table = {
    "columns": ["strategi", "dipertahankan", "tingkat_keberhasilan", "jumlah_pemilihan_per_tawaran"],
    "rows": [[r[0], r[1], r[2], " / ".join(str(x) for x in r[3])] for r in rl["rows"]],
}
rl_est = {
    "columns": ["tawaran", "peluang_sebenarnya", "estimasi_agen"],
    "rows": [[o, t, e] for o, t, e in zip(rl["offers"], rl["true_rates"], rl["estimates"])],
}

content = f"""
<header class="hero">
  <p class="eyebrow">Informatics Practitioner Lecture Series 2026</p>
  <h1>Prediksi &amp; Segmentasi Churn Pelanggan Telco</h1>
  <p class="lead">Studi kasus implementasi materi <em>Introduction to Machine Learning</em> (Dr. Taniza Tajudin):
  jenis pembelajaran mesin, alur kerja ML, metrik evaluasi, dan pertimbangan etika dalam satu kasus bisnis.</p>
  <div class="kpis">
    {card("pelanggan (data simulasi)", num(eda["n_rows"]), "seed tetap, reproducible")}
    {card("proporsi churn", num(eda["churn_rate"], pct=True), "target klasifikasi")}
    {card("F1 model terbaik", num(best_cls["f1"]), cl["best"])}
    {card("R² regresi tagihan", num(best_reg["R2"]), rg["best"])}
    {card("rasio fairness (usia)", num(fr["kelompok_usia"]["ratio"], 2), "ambang aturan 80%: 0,80")}
  </div>
</header>

<nav class="toc" aria-label="Daftar isi">
  <a href="#crisp">Kerangka</a><a href="#bu">1. Business</a><a href="#du">2. Data</a><a href="#dp">3. Preparation</a>
  <a href="#model">4. Modeling</a><a href="#etika">5. Etika</a><a href="#deploy">6. Deployment</a><a href="#simpulan">7. Kesimpulan</a>
</nav>

<main>
<section id="crisp">
  <h2>Kerangka CRISP-DM</h2>
  <p>Proyek mengikuti alur kerja ML dari materi, dipetakan ke tahapan CRISP-DM.</p>
  {table(crisp)}
</section>

<section id="bu">
  <h2>1. Business Understanding</h2>
  <p>Perusahaan telco kehilangan pendapatan setiap pelanggan berhenti berlangganan, dan mencari pelanggan baru
  jauh lebih mahal daripada mempertahankan yang lama. Pertanyaan yang dijawab:</p>
  <ol>{"".join(f"<li>{html.escape(q)}</li>" for q in questions)}</ol>
  <p class="note">Berbeda dengan pemrograman tradisional yang menulis aturan <em>if-then</em> manual, di sini algoritma
  mempelajari pola churn langsung dari data.</p>
</section>

<section id="du">
  <h2>2. Data Understanding</h2>
  <p>Data <strong>disimulasikan</strong> ({num(eda["n_rows"])} pelanggan, {eda["n_cols"]} kolom) agar proyek berjalan tanpa
  unduhan apa pun dan "kebenaran dasarnya" diketahui (misalnya anomali yang sengaja disisipkan). Data dibuat kotor
  seperti data nyata. Kekurangannya: pola di dalamnya lebih sederhana daripada data sungguhan.</p>
  {table(dataset_cols)}
  <h3>Missing values</h3>
  {table(eda["missing"])}
  <h3>Eksplorasi churn</h3>
  {img("01_eksplorasi_churn.png", "Churn paling tinggi pada kontrak bulanan dan pelanggan baru")}
</section>

<section id="dp">
  <h2>3. Data Preparation</h2>
  <h3>Anomaly detection (Isolation Forest, unsupervised)</h3>
  <div class="kpis small">
    {card("baris ditandai", num(an["flagged"]))}
    {card("anomali sebenarnya", num(an["true"]))}
    {card("tertangkap", num(an["caught"]), f"recall {num(an['recall'], pct=True)}")}
    {card("precision", num(an["precision"], pct=True), f"{an['flagged'] - an['caught']} baris normal ikut terbuang")}
  </div>
  <p>Semua anomali sisipan tertangkap, tetapi sebagian baris normal ikut terbuang. Ini <em>trade-off</em> nyata dalam
  membersihkan data. Baris yang ditandai dibuang sebelum pelatihan.</p>
  <h3>Pipeline pra-pemrosesan</h3>
  <ul>
    <li>Fitur numerik: imputasi median, lalu normalisasi (<code>StandardScaler</code>).</li>
    <li>Fitur kategorikal: imputasi modus, lalu <code>OneHotEncoder</code>.</li>
    <li>Pembagian data 80% latih / 20% uji (stratified); seluruh pra-pemrosesan dipelajari hanya dari data latih
        (di dalam <code>Pipeline</code>) untuk mencegah <em>data leakage</em>.</li>
  </ul>
</section>

<section id="model">
  <h2>4. Modeling &amp; Evaluation</h2>

  <h3>4.1 Supervised: klasifikasi churn</h3>
  <p>Data latih {num(cl["n_train"])} · data uji {num(cl["n_test"])}. Baris tebal = model terbaik berdasarkan F1.</p>
  {table(cl["table"], bold_first=cl["best"])}
  {img("02_confusion_matrix.png", "Confusion matrix kedua model pada data uji")}
  {img("03_feature_importance.png", "10 fitur terpenting menurut Random Forest", 560)}
  <ul>
    <li>Kedua model hampir setara: selisih F1 hanya {num(abs(lr["f1"] - rf["f1"]))}, terlalu kecil untuk menyimpulkan satu
        lebih baik. Model kompleks tidak otomatis unggul pada pola data yang sederhana. {html.escape(cl["best"])} dipilih
        karena F1 dan ROC-AUC sedikit lebih tinggi dan lebih mudah dijelaskan.</li>
    <li>Fitur terpenting: lama berlangganan, tagihan, dan jenis kontrak, sesuai pola pada eksplorasi data.</li>
    <li>Untuk churn, <strong>recall</strong> lebih penting daripada accuracy: melewatkan pelanggan yang akan pergi lebih
        merugikan daripada menawarkan promo ke pelanggan yang sebenarnya bertahan.</li>
  </ul>

  <h3>4.2 Supervised: regresi tagihan bulanan</h3>
  {table(rg["table"], bold_first=rg["best"])}
  {img("04_regresi_aktual_vs_prediksi.png", f"Tagihan aktual vs prediksi ({rg['best']})", 460)}
  <p><code>total_charges</code> tidak dipakai sebagai fitur karena merupakan hasil kali tagihan bulanan dengan lama
  berlangganan (<em>data leakage</em>). Model linear menang karena hubungan di data simulasi memang hampir linear.</p>

  <h3>4.3 Unsupervised: segmentasi (K-Means + PCA)</h3>
  <p>Jumlah segmen dipilih dari silhouette tertinggi pada k = 3 sampai 6, hasilnya k = {cs["best_k"]}.</p>
  {table(cs["k_table"], decimals_map={"inertia": 1})}
  {img("05_pemilihan_k.png", "Elbow method dan silhouette score")}
  {table(cs["profile"], pct_cols=("churn_rate",), decimals_map={"umur": 1, "lama_langganan": 1, "tagihan": 2, "telp_support": 2, "layanan_tambahan": 2}, caption="Profil segmen (churn_rate tidak dipakai saat clustering)")}
  {img("06_segmen_pca.png", f"Segmen pada ruang PCA (2 komponen = {num(cs['pca_variance'], pct=True)} variansi)", 520)}
  <p class="warn"><strong>Catatan jujur:</strong> silhouette hanya sekitar {num(max(sil_vals), 2)} dan hampir sama untuk
  semua k, artinya struktur cluster <strong>lemah</strong>; segmen saling tumpang tindih pada plot PCA. Baca segmen
  sebagai pengelompokan kasar, bukan kelompok yang benar-benar terpisah.</p>

  <h3>4.4 Semi-supervised: self-training dengan {num(sm["n_labeled"])} dari {num(sm["n_train"])} label</h3>
  {table(sm["table"])}
  <p>Self-training <strong>tidak memberi keuntungan</strong> pada data ini (F1 {num(sm_semi["f1"])} vs {num(sm_sup["f1"])}
  untuk supervised dengan label yang sama). Menariknya, 10% label sudah menyamai 100% label (F1 {num(sm_full["f1"])}),
  menandakan pola data cukup sederhana. Selisih sekecil ini masih dalam kisaran variasi acak satu kali pembagian data uji.</p>

  <h3>4.5 Reinforcement learning: memilih tawaran retensi</h3>
  <p>Agen epsilon-greedy belajar dari reward (pelanggan bertahan = 1) tanpa mengetahui peluang keberhasilan tiap tawaran.</p>
  {table(rl_table, pct_cols=("tingkat_keberhasilan",))}
  {table(rl_est, decimals=2)}
  {img("08_rl_bandit.png", "Kumulatif pelanggan yang dipertahankan per strategi", 520)}
  <p>Agen menemukan "{html.escape(rl["offers"][1])}" sebagai tawaran terbaik dan memakainya pada
  {num(rl["rows"][0][3][1])} dari 5.000 pelanggan. Ini bentuk paling sederhana (<em>multi-armed bandit</em>, tanpa state),
  bukan RL penuh seperti mobil otonom.</p>
</section>

<section id="etika">
  <h2>5. Etika: pengecekan fairness</h2>
  <p>Model terbaik dievaluasi per kelompok pada data uji. Gender tidak dipakai sebagai fitur; usia dipakai.</p>
  <h3>Gender (rasio laju prediksi churn: {num(fr["gender"]["ratio"], 2)})</h3>
  {table(fr["gender"]["table"], pct_cols=("churn_aktual", "diprediksi_churn", "recall", "precision"))}
  <h3>Kelompok usia (rasio laju prediksi churn: {num(fr["kelompok_usia"]["ratio"], 2)})</h3>
  {table(fr["kelompok_usia"]["table"], pct_cols=("churn_aktual", "diprediksi_churn", "recall", "precision"))}
  {img("07_fairness.png", "Churn aktual, churn diprediksi, dan recall per kelompok")}
  <ul>
    <li><strong>Gender:</strong> rasio {num(fr["gender"]["ratio"], 2)} lolos aturan 80%. Selisih churn aktual
        ({num(gen_tab["Laki-laki"]["churn_aktual"], pct=True)} vs {num(gen_tab["Perempuan"]["churn_aktual"], pct=True)})
        muncul walau di data gender tidak mempengaruhi churn, jadi kemungkinan besar kebetulan sampel kecil. Angka
        berbeda antar kelompok belum tentu berarti bias.</li>
    <li><strong>Usia:</strong> rasio {num(fr["kelompok_usia"]["ratio"], 2)} di bawah ambang 0,80. Sebagian karena churn aktual
        kelompok 50+ memang lebih rendah, tetapi <strong>recall kelompok {html.escape(lowest_recall_age[0])} paling rendah
        ({num(lowest_recall_age[1]["recall"], pct=True)})</strong>: model lebih sering melewatkan pelanggan di kelompok itu
        yang akan churn.</li>
    <li>Menghapus atribut sensitif dari fitur belum menjamin keadilan karena fitur lain bisa menjadi <em>proxy</em>;
        pemantauan per kelompok tetap perlu setelah deployment.</li>
  </ul>
</section>

<section id="deploy">
  <h2>6. Deployment</h2>
  <p>Model <strong>{html.escape(dp["model"])}</strong> disimpan dengan <code>joblib</code>, dimuat ulang, lalu dipakai
  memprediksi dua pelanggan baru.</p>
  {table(deploy_rows, pct_cols=("probabilitas_churn",))}
  <p class="note">Prediksi hanya alat bantu. Keputusan akhir dan verifikasi kritis tetap membutuhkan pengawasan manusia.</p>
</section>

<section id="simpulan">
  <h2>7. Kesimpulan &amp; Keterbatasan</h2>
  <ol>
    <li>Workflow ML (data, pra-pemrosesan, split, pelatihan, evaluasi, deployment) berhasil diterapkan end-to-end.</li>
    <li>Pra-pemrosesan menentukan: tanpa penanganan missing values dan anomali, model tidak dapat dilatih dengan baik.</li>
    <li>Model sederhana bersaing dengan model kompleks; algoritma dipilih mengikuti masalah, bukan yang paling canggih.</li>
    <li>Hasil tidak selalu sesuai harapan (clustering lemah, semi-supervised tidak lebih baik) dan dilaporkan apa adanya.</li>
    <li>Fairness menunjukkan perbedaan kinerja antar kelompok usia, sehingga pengawasan manusia tetap krusial.</li>
  </ol>
  <h3>Keterbatasan</h3>
  <ul>
    <li>Data simulasi; kinerja pada data nyata akan berbeda.</li>
    <li>Satu kali pembagian train/test, tanpa cross-validation maupun tuning hyperparameter.</li>
    <li>Agen RL hanya bandit sederhana; cek fairness hanya dua atribut dan satu ukuran utama.</li>
  </ul>
</section>
</main>

<footer>
  <p>Kuliah tamu <em>Informatics Practitioner Lecture Series 2026</em> · Pemateri: Dr. Taniza Tajudin ·
  Materi: Introduction to Machine Learning (16 Oktober 2026)</p>
  <p>Laporan dibuat otomatis oleh <code>generate_report.py</code> dari <code>results.json</code>.</p>
</footer>
"""

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--text:#1d2430;--muted:#5b6575;--line:#e2e6ec;--accent:#2f5fa7;--accent-soft:#e8f0fb;--warn:#fff4e0;--warn-line:#e6b45a;--best:#eaf3ea}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1c2027;--text:#e6e9ee;--muted:#9aa4b2;--line:#2b313b;--accent:#7aa7e6;--accent-soft:#222c3b;--warn:#2d261a;--warn-line:#a07a2c;--best:#1f2c22}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.65 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.hero,.toc,main,footer{max-width:980px;margin:0 auto;padding-left:20px;padding-right:20px}
.hero{padding-top:48px;padding-bottom:12px}
.eyebrow{margin:0;color:var(--accent);font-weight:600;font-size:.85rem;letter-spacing:.06em;text-transform:uppercase}
h1{font-size:clamp(1.7rem,4vw,2.4rem);line-height:1.2;margin:.3rem 0 .6rem}
.lead{color:var(--muted);max-width:70ch;margin:0 0 1.2rem}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi-v{font-size:1.6rem;font-weight:700;line-height:1.2}
.kpi-l{font-size:.85rem;color:var(--muted)}
.kpi-n{font-size:.75rem;color:var(--muted);opacity:.85}
.kpis.small .kpi-v{font-size:1.3rem}
.toc{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:6px;padding-top:10px;padding-bottom:10px;background:var(--bg);border-bottom:1px solid var(--line)}
.toc a{color:var(--accent);text-decoration:none;font-size:.85rem;padding:4px 10px;border-radius:999px;background:var(--accent-soft)}
.toc a:hover{text-decoration:underline}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:6px 24px 20px;margin:20px 0;scroll-margin-top:60px}
h2{font-size:1.35rem;margin:1.2rem 0 .6rem}
h3{font-size:1.05rem;margin:1.6rem 0 .5rem;color:var(--accent)}
p,li{max-width:80ch}
code{background:var(--accent-soft);padding:1px 6px;border-radius:5px;font-size:.88em}
.tablewrap{overflow-x:auto;margin:10px 0 16px}
table{border-collapse:collapse;width:100%;font-size:.92rem}
caption{caption-side:top;text-align:left;color:var(--muted);font-size:.85rem;padding-bottom:6px}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:var(--accent-soft);font-weight:600}
tr.best td{background:var(--best);font-weight:700}
figure{margin:14px 0;text-align:center}
figure img{max-width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:10px;padding:6px}
figcaption{font-size:.82rem;color:var(--muted);margin-top:6px}
.note,.warn{border-left:4px solid var(--accent);background:var(--accent-soft);padding:10px 14px;border-radius:0 8px 8px 0}
.warn{border-left-color:var(--warn-line);background:var(--warn)}
footer{color:var(--muted);font-size:.85rem;padding-top:8px;padding-bottom:40px}
@media print{.toc{display:none}section{break-inside:avoid-page;border:none}body{background:#fff}}
"""

page = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studi Kasus ML: Prediksi dan Segmentasi Churn Pelanggan Telco</title>
<style>{CSS}</style>
</head>
<body>
{content}
</body>
</html>
"""

OUT.write_text(page, encoding="utf-8")
print(f"report.html dibuat ({OUT.stat().st_size / 1024:.0f} KB)")
