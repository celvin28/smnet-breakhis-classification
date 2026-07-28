# smnet-breakhis-classification
Deep learning-based classification of breast cancer histopathological images (BreaKHis) using SMNet — ShuffleNetV2-0.5x with dual-branch token mixer

# SMNet: Klasifikasi Multi-Kelas Citra Histopatologi Kanker Payudara

Implementasi **SMNet** — arsitektur hybrid berbasis **ShuffleNetV2-0.5x** dengan **Dual-Branch Token Mixer** — untuk klasifikasi 8 subtipe histopatologi kanker payudara pada dataset **BreaKHis**, dievaluasi pada empat tingkat magnifikasi (40X, 100X, 200X, 400X).

Proyek ini merupakan bagian dari skripsi (tugas akhir) di bidang *deep learning* untuk citra medis, dengan fokus pada perbandingan arsitektur yang adil (*fair comparison*) antar model.

---

## 📋 Daftar Isi

- [Ringkasan Proyek](#-ringkasan-proyek)
- [Dataset](#-dataset)
- [Arsitektur SMNet](#-arsitektur-smnet)
- [Struktur Notebook](#-struktur-notebook)
- [Hasil Evaluasi](#-hasil-evaluasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Requirement](#-requirement)
- [Struktur Repo](#-struktur-repo)
- [Kredit](#-kredit)

---

## 🔬 Ringkasan Proyek

SMNet dirancang untuk menangkap fitur morfologis lokal (susunan sel, tekstur jaringan) sekaligus konteks spasial global pada citra histopatologi, dengan menempatkan cabang **Residual Token Mixer** (bergaya PoolFormer/MetaFormer) pada dua titik backbone — `stage3` dan `stage4` — bukan hanya di keluaran akhir backbone.

Setiap notebook pada repo ini melatih dan mengevaluasi SMNet secara independen untuk satu tingkat magnifikasi, dengan pipeline yang identik (split data, augmentasi, optimizer, callback) agar hasil antar magnifikasi dapat dibandingkan secara adil.

Pipeline mencakup:
- Preprocessing & augmentasi data (stratified split 70/15/15)
- Class weighting untuk menangani ketidakseimbangan kelas
- Training dengan Early Stopping, Model Checkpoint, dan ReduceLROnPlateau
- Evaluasi kuantitatif: Classification Report, Confusion Matrix, ROC & Precision-Recall Curve
- Explainable AI dengan Grad-CAM untuk interpretasi visual prediksi model

## 📊 Dataset

Proyek ini menggunakan dataset **[BreaKHis](https://www.kaggle.com/datasets/ambarish/breakhis)** (Breast Cancer Histopathological Database), yang terdiri dari citra histopatologi pada 4 tingkat magnifikasi (40X, 100X, 200X, 400X) dengan 8 subtipe kelas:

| Kategori | Subtipe |
|---|---|
| **Jinak (Benign)** | Adenosis, Fibroadenoma, Phyllodes Tumor, Tubular Adenoma |
| **Ganas (Malignant)** | Ductal Carcinoma, Lobular Carcinoma, Mucinous Carcinoma, Papillary Carcinoma |

> Dataset tidak disertakan dalam repo ini karena ukurannya besar. Unduh langsung dari sumber di atas dan sesuaikan path pada bagian konfigurasi di masing-masing notebook.

## 🧠 Arsitektur SMNet

```
Input (224x224)
     │
  Backbone: ShuffleNetV2-0.5x (stem → maxpool → stage2)
     │
     ├──▶ stage3 (14x14, 96ch)
     │        │
     │   Feature Enhancement (1x1 Conv + BN + GELU)
     │        │
     │   ┌────┴─────────────────┐
     │   branch A3: identity    branch B3: Residual Token Mixer (dim=96, depth=4)
     │   └────────┬─────────────┘
     │            │ Concat → 1x1 Conv Fusion
     │            │
     │        GAP + GMP  ──▶ kontribusi stage3 (192-dim)
     │
     └──▶ stage4 (7x7, 192ch)
              │
         Feature Enhancement (1x1 Conv + BN + GELU)
              │
         ┌────┴─────────────────┐
         branch A4: identity    branch B4: Residual Token Mixer (dim=192, depth=4)
         └────────┬─────────────┘
                  │ Concat → 1x1 Conv Fusion
                  │
              GAP + GMP  ──▶ kontribusi stage4 (384-dim)

  Concat(stage3, stage4) → 576-dim
              │
          Dropout → Fully Connected (576 → 8) → Softmax
```

**Poin kunci desain:**
- **Dual-branch token mixer** ditempatkan di dua titik backbone (stage3 & stage4) agar model menangkap konteks spasial pada level fitur menengah maupun fitur dalam.
- **Token mixer non-parametrik** (average pooling − identity), mengikuti definisi resmi PoolFormer, sehingga seluruh kapasitas belajar pada blok mixer berasal dari channel MLP.
- **Partial fine-tuning**: backbone dibekukan, lalu 30 tensor parameter terakhir dibuka untuk pelatihan 
## 📁 Struktur Notebook

Setiap notebook mengikuti struktur 11 bagian yang konsisten:

1. Setup & Import Library
2. Konfigurasi Proyek & Loading Dataset
3. Data Augmentasi & Generator
4. Class Weighting (Penanganan Data Imbalance)
5. Arsitektur Model — SMNet
6. Setup Callbacks (Training)
7. Grafik Akurasi & Loss
8. Evaluasi — Classification Report & Confusion Matrix
9. Evaluasi Lanjutan — ROC Curve & Precision-Recall Curve
10. Explainable AI — Grad-CAM
11. Download Semua Laporan Evaluasi

| Magnifikasi | Notebook |
|---|---|
| 40X | [`smnet_40x.ipynb`](smnet_40x.ipynb) |
| 100X | [`smnet_100x.ipynb`](smnet_100x.ipynb) |
| 200X | [`smnet_200x.ipynb`](smnet_200x.ipynb) |
| 400X | [`smnet_400x.ipynb`](smnet_400x.ipynb) |

## 📈 Hasil Evaluasi

Ringkasan performa SMNet pada test set untuk setiap tingkat magnifikasi:

| Magnifikasi | Test Accuracy | Macro F1-score | Weighted F1-score | Best Epoch |
|---|---|---|---|---|
| 40X  | 0.91 | 0.90 | 0.91 | 67 |
| 100X | 0.81 | 0.81 | 0.81 | 41 |
| 200X | 0.85 | 0.84 | 0.85 | 72 |
| 400X | 0.79 | 0.76 | 0.79 | 50 |

> Akurasi tertinggi diperoleh pada magnifikasi 40X. Detail lengkap (precision/recall per kelas, confusion matrix, kurva ROC/PR, dan Grad-CAM) tersedia di masing-masing notebook.

## ⚙️ Cara Menjalankan

Notebook ini awalnya dijalankan pada **Kaggle Notebooks** (path dataset menggunakan `/kaggle/input/...`). Untuk menjalankan ulang:

1. Buka notebook di Kaggle atau Google Colab
2. Unduh dataset [BreaKHis](https://www.kaggle.com/datasets/ambarish/breakhis) dan sesuaikan `DATASET_PATH` pada sel konfigurasi (Bagian 2.1)
3. Sesuaikan `TARGET_MAGNIFICATION` jika ingin mengganti tingkat magnifikasi
4. Jalankan seluruh sel secara berurutan

## 📦 Requirement

```
torch
torchvision
opencv-python
scikit-learn
matplotlib
seaborn
pandas
numpy
```

## 🗂️ Struktur Repo

```
smnet-breakhis-classification/
├── README.md
├── LICENSE
├── smnet_40x.ipynb
├── smnet_100x.ipynb
├── smnet_200x.ipynb
└── smnet_400x.ipynb
```


