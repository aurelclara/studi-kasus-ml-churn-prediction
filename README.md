# Machine Learning: Prediksi & Segmentasi Churn Pelanggan Telco

Studi kasus implementasi materi **"Introduction to Machine Learning"** (Dr. Taniza Tajudin, *Informatics Practitioner Lecture Series 2026*).
Satu kasus bisnis, yaitu pelanggan layanan internet yang berhenti berlangganan (*churn*), dipakai untuk mempraktikkan
jenis-jenis pembelajaran mesin, alur kerja ML, metrik evaluasi, dan pertimbangan etika dari materi,
menggunakan `pandas`, `numpy`, dan `scikit-learn`.

🔗 Laporan HTML:
https://aurelclara.github.io/studi-kasus-ml-churn-prediction/

## Kerangka CRISP-DM

| Tahap | Penerapan | Konsep dari materi |
|---|---|---|
| **1. Business Understanding** | Mengurangi churn pelanggan telco; menentukan 7 pertanyaan analisis | ML vs pemrograman tradisional |
| **2. Data Understanding & Preparation** | Eksplorasi data, imputasi *missing values*, deteksi anomali (Isolation Forest), normalisasi, *one-hot encoding* | Pra-pemrosesan, unsupervised learning |
| **3. Modeling** | Klasifikasi churn, regresi tagihan, K-Means + PCA, self-training, epsilon-greedy bandit | Supervised, unsupervised, semi-supervised, reinforcement learning |
| **4. Evaluation** | Accuracy, precision, recall, F1, ROC-AUC, MAE, RMSE, R², silhouette, pengecekan *fairness* | Evaluasi metrik, etika & fairness |
| **5. Deployment** | Model disimpan dengan `joblib`, dimuat ulang, dipakai memprediksi pelanggan baru | Deployment, pengawasan manusia |

## Struktur Proyek

```
studi-kasus-ml-churn-prediction/
├── ml_analysis.py       # pipeline ML lengkap (simulasi data -> modeling -> evaluasi -> deployment)
├── generate_report.py   # membuat report.html dari hasil analisis
├── report.html          # laporan interaktif satu file (grafik sudah ter-embed)
├── requirements.txt     # dependensi Python
├── .gitignore
└── README.md
```

`ml_analysis.py` juga menghasilkan `results.json`, folder `figures/`, dan `churn_model.joblib`.
Ketiganya diabaikan Git karena bisa dibuat ulang kapan saja, dan seluruh grafik sudah ada di dalam `report.html`.

## Cara Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/aurelclara/studi-kasus-ml-churn-prediction.git
cd studi-kasus-ml-churn-prediction

# 2. Buat virtual environment (opsional)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependensi
pip install -r requirements.txt

# 4. Jalankan analisis
python ml_analysis.py

# 5. Buat laporan HTML (hasilnya: report.html, buka dengan browser)
python generate_report.py
```

## Output

`ml_analysis.py` mencetak sembilan bagian ke terminal:

1. **Data understanding**: ukuran data, proporsi churn, *missing values*.
2. **Data preparation**: deteksi anomali dengan Isolation Forest.
3. **Supervised, klasifikasi**: Logistic Regression vs Random Forest (accuracy, precision, recall, F1, ROC-AUC, feature importance).
4. **Supervised, regresi**: Linear Regression vs Random Forest Regressor (MAE, RMSE, R²).
5. **Unsupervised**: pemilihan k (elbow & silhouette), profil segmen K-Means, PCA.
6. **Semi-supervised**: self-training dengan 10% data berlabel.
7. **Reinforcement learning**: epsilon-greedy bandit memilih tawaran retensi.
8. **Evaluasi etika**: pengecekan *fairness* per gender dan kelompok usia.
9. **Deployment**: simpan model dan prediksi dua pelanggan baru.

### Ringkasan hasil

| Bagian | Hasil |
|---|---|
| Klasifikasi churn | Logistic Regression: F1 0,665, recall 0,725, ROC-AUC 0,791. Random Forest setara (F1 0,654) |
| Regresi tagihan | Linear Regression: MAE 3,006, R² 0,956 |
| Segmentasi | k = 3, silhouette ±0,18, artinya struktur cluster **lemah** |
| Semi-supervised | Self-training tidak lebih baik dari supervised biasa (F1 0,680 vs 0,682) |
| Reinforcement learning | Agen mempertahankan 28,4% pelanggan vs 20,7% (acak) dan 12,0% (selalu tawaran #1) |
| Fairness | Rasio gender 0,97 (lolos); rasio usia 0,74 (di bawah 0,80), recall kelompok 50+ paling rendah |

Analisis lengkap, tabel, dan grafik ada di **`report.html`**.

> GitHub hanya menampilkan kode HTML, bukan halamannya. Untuk melihat laporan: unduh `report.html` lalu buka di browser,
> atau aktifkan **GitHub Pages** (*Settings → Pages → Deploy from a branch → `main` / root*). Setelah aktif,
> laporan bisa dibuka di `https://aurelclara.github.io/studi-kasus-ml-churn-prediction/report.html`.

## Catatan

- Data **disimulasikan** (`np.random.default_rng(42)`): 3.000 pelanggan dengan ±3% *missing values* dan 1,5% anomali sisipan.
  Hasil selalu sama setiap dijalankan, tetapi polanya lebih sederhana daripada data nyata. Ganti dengan data perusahaan sungguhan
  (misalnya dataset *Telco Customer Churn*) untuk analisis yang sebenarnya.
- `gender` sengaja tidak dipakai sebagai fitur model; dipakai hanya untuk pengecekan *fairness*.
- `total_charges` tidak dipakai pada regresi tagihan karena menyebabkan *data leakage*.
- Evaluasi memakai satu kali pembagian train/test (80/20) tanpa *cross-validation* maupun tuning hyperparameter,
  sehingga selisih kecil antar model tidak boleh dianggap signifikan.
- Reinforcement learning di sini hanya *multi-armed bandit* sederhana, bukan RL penuh dengan state dan transisi.
- Model hanya alat bantu; keputusan akhir dan verifikasi kritis tetap membutuhkan pengawasan manusia.

## Lisensi
kuliah tamu Informatics Practitioner Lecture Series 2026
dengan pemateri Dr. Taniza Tajudin, materi Introduction to Machine Learning (16 Oktober 2026).
