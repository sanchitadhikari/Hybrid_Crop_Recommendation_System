## Hybrid Crop Recommendation System

<div align="center">

<p align="center">
  <span style="background-color:#3776AB;color:white;padding:6px 12px;border-radius:20px;margin:4px;display:inline-block;">Python 3.9+</span>
  <span style="background-color:#F7931E;color:white;padding:6px 12px;border-radius:20px;margin:4px;display:inline-block;">scikit-learn 1.4</span>
  <span style="background-color:#FF6600;color:white;padding:6px 12px;border-radius:20px;margin:4px;display:inline-block;">XGBoost 2.0</span>
  <span style="background-color:#02B875;color:white;padding:6px 12px;border-radius:20px;margin:4px;display:inline-block;">LightGBM 4.3</span>
  <span style="background-color:#FFCC00;color:black;padding:6px 12px;border-radius:20px;margin:4px;display:inline-block;">CatBoost 1.2</span>
  <span style="background-color:#009688;color:white;padding:6px 12px;border-radius:20px;margin:4px;display:inline-block;">FastAPI 0.111</span>
  <span style="background-color:#9B59B6;color:white;padding:6px 12px;border-radius:20px;margin:4px;display:inline-block;">SHAP Explainability</span>
</p>

<p align="center">
  <span style="background-color:green;color:white;padding:6px 14px;border-radius:20px;">MIT License</span>
</p>

**An intelligent crop recommendation engine powered by a hybrid ensemble of 8 machine learning models, SHAP explainability, and Monte Carlo uncertainty estimation.**

[Features](#-features) · [Architecture](#-ml-architecture) · [Results](#-model-results) · [Setup](#-setup) · [Usage](#-usage) · [Explainability](#-shap-explainability) · [API](#-api)

</div>

---

 Overview

The **Hybrid Crop Recommendation System** predicts the most suitable crop to cultivate based on soil nutrients and environmental conditions. It analyzes **7 key agricultural parameters** — Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH level, and rainfall — to deliver accurate, explainable, and confidence-rated crop recommendations.

Unlike traditional single-model approaches, this system uses a **stacked ensemble architecture** that combines multiple ML algorithms including gradient boosting models, SVMs, and deep learning (TabNet), with a meta-learner producing the final prediction.

```
Input: [N, P, K, Temperature, Humidity, pH, Rainfall]
                          ↓
         ┌────────────────────────────┐
         │   Hybrid Ensemble Engine   │
         │  RF + XGB + LGBM + CatBoost│
         │     + TabNet + SVM + KNN   │
         └────────────┬───────────────┘
                      ↓
         ┌────────────────────────────┐
         │   Meta-Learner (Stacking)  │
         └────────────┬───────────────┘
                      ↓
     Recommended Crop + Confidence + SHAP
```

---

##  Features

| Feature | Description |
|---------|-------------|
|  **8 ML Models** | Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN, Decision Tree, Stacking Ensemble |
| **Stacking Ensemble** | Meta-learner combining base model predictions for highest accuracy |
| **SHAP Explainability** | Global importance, beeswarm plots, per-crop heatmap, single-prediction waterfall |
| **Uncertainty Estimation** | Monte Carlo sampling via Random Forest tree disagreement |
| **Rich Visualizations** | 15+ chart types — violin plots, correlation heatmaps, learning curves, confusion matrices |
| **Google Colab Ready** | Full notebook with step-by-step execution |

---

## Dataset

The system is trained on the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) from Kaggle.

| Property | Value |
|----------|-------|
| Samples | 2,200 |
| Features | 7 (N, P, K, temperature, humidity, pH, rainfall) |
| Classes | 22 crops |
| Balance | Perfectly balanced — 100 samples per crop |
| Missing Values | None |

**Crops covered:** Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

### Input Features

| Feature | Unit | Range | Description |
|---------|------|-------|-------------|
| N | mg/kg | 0 – 140 | Nitrogen content in soil |
| P | mg/kg | 5 – 145 | Phosphorus content in soil |
| K | mg/kg | 5 – 205 | Potassium content in soil |
| temperature | °C | 8.8 – 43.7 | Average temperature |
| humidity | % | 14.3 – 99.9 | Relative humidity |
| ph | 0–14 | 3.5 – 9.9 | pH value of soil |
| rainfall | mm | 20.2 – 298.6 | Annual rainfall |

---

##  ML Architecture

### Base Models

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT FEATURES (7)                      │
└──────┬──────┬──────┬──────┬──────┬──────┬──────┬───────────┘
       │      │      │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼      ▼      ▼
      RF    XGB   LGBM  CatBoost TabNet  SVM    KNN
       │      │      │      │      │      │      │
       └──────┴──────┴──────┴──────┴──────┴──────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  Meta-Learner         │
               │  (Logistic Regression)│
               └───────────┬───────────┘
                           │
                           ▼
                   FINAL PREDICTION
               + Confidence + SHAP + Uncertainty
```

| Model | Type | Key Strength |
|-------|------|-------------|
| **Random Forest** | Bagging Ensemble | Robust, low variance, good for feature importance |
| **XGBoost** | Gradient Boosting | High accuracy, handles outliers well |
| **LightGBM** | Gradient Boosting | Fast training, memory efficient |
| **CatBoost** | Gradient Boosting | Handles categorical data, often best accuracy |
| **TabNet** | Deep Learning | Attention-based tabular learning |
| **SVM** | Support Vector Machine | Effective in high-dimensional space |
| **KNN** | Instance-based | Simple, no training phase |
| **Decision Tree** | Tree | Fully interpretable |
| **Stacking Ensemble** | Meta-learning | Combines all above models |

### Additional Components

- **SHAP (SHapley Additive exPlanations)** — assigns a contribution score to each input feature for every prediction
- **Monte Carlo Uncertainty** — runs predictions across all 300 RF trees; variance = uncertainty score

---

##  Model Results

> Results on 20% held-out test set (440 samples) with 5-Fold Cross-Validation

| Rank | Model | Test Accuracy | CV Accuracy |
|------|-------|:---:|:---:|
| 🥇 | **Stacking Ensemble** | **99.77%** | — |
| 🥈 | CatBoost | 99.55% | 99.43% ± 0.21% |
| 🥉 | Random Forest | 99.32% | 99.26% ± 0.18% |
| 4 | XGBoost | 99.09% | 99.01% ± 0.22% |
| 5 | LightGBM | 98.86% | 98.79% ± 0.25% |
| 6 | SVM | 97.95% | 97.84% ± 0.31% |
| 7 | KNN | 97.27% | 96.98% ± 0.44% |
| 8 | Decision Tree | 95.68% | 95.11% ± 0.72% |
| 9 | Naive Bayes | 88.64% | 88.41% ± 0.91% |

**Sample prediction output:**

| **7 ML Models** | Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN, Decision Tree |
| **Stacking Ensemble** | Meta-learner (RF + XGBoost + LightGBM + CatBoost + SVM) with Logistic Regression as meta-learner |
| **SHAP Explainability** | Global feature importance, beeswarm per crop, full heatmap (all crops × features), single-prediction waterfall |
| **Uncertainty Estimation** | Monte Carlo sampling via Random Forest tree disagreement — returns confidence score per prediction |
| **Rich Visualizations** | Histograms, box plots, violin plots, pairplot, Pearson & Spearman heatmaps, confusion matrices, F1 bar chart, learning curves, feature importance bars, SHAP scatter plots |
| **Prediction Function** | `predict_crop(N, P, K, temp, humidity, ph, rainfall)` — works with any trained model |
| **Google Colab Ready** | Upload CSV via `files.upload()`, runs end-to-end with step-by-step cells |

## SHAP Explainability

The system uses SHAP to make every prediction transparent and interpretable.

### Plots Generated

| Plot | What It Shows |
|------|--------------|
| **Global Bar Chart** | Overall feature importance across all test samples |
| **Beeswarm Plot** | Distribution and direction of SHAP values per feature per crop |
| **Heatmap (22 × 7)** | Which features matter most for each individual crop |
| **Waterfall Plot** | Why a specific single prediction was made |
| **Dependence Plot** | How a feature's impact changes with its value |

**Key SHAP Finding:** `humidity`, `rainfall`, and `K (Potassium)` are consistently the top 3 most influential features across most crop classes.

---

##  Project Structure

crop-recommendation-system/
│
├── ml/                           ← Machine learning modules
│   ├── preprocess.py             ← Label encoding, StandardScaler, train/test split
│   ├── train_models.py           ← Train RF, XGBoost, LightGBM, CatBoost, SVM, KNN, DT
│   ├── ensemble.py               ← Stacking ensemble (LR meta-learner)
│   ├── explainability.py         ← SHAP: global, beeswarm, heatmap, waterfall, scatter
│   └── uncertainty.py            ← Monte Carlo via RF tree disagreement
│
├── models/                       ← Saved model files (auto-generated)
│   ├── model_rf.pkl
│   ├── model_xgb.pkl
│   ├── model_lgbm.pkl
│   ├── model_catboost.pkl
│   ├── model_svm.pkl
│   ├── model_knn.pkl
│   ├── model_dt.pkl
│   ├── stacked_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── data/
│   └── crop_recommendation.csv   ← Dataset (download from Kaggle)
│
├── notebooks/
│   └── crop_recommendation.ipynb ← Full Colab notebook
│
├── requirements.txt
├── .gitignore
└── README.md



| Step | Content |
|------|---------|
| 1  | Install libraries — xgboost, lightgbm, catboost, shap |
| 2  | Import all packages — numpy, pandas, sklearn, matplotlib, seaborn, shap |
| 3  | Load dataset — upload CSV via Google Colab, shape & preview |
| 4  | EDA — crop distribution (bar + pie), feature histograms, box plots, Pearson & Spearman heatmaps, violin plots per crop, pairplot (8 crops sample) |
| 5  | Preprocessing — label encoding, StandardScaler, train/test split, before vs after scaling visualization |
| 6  | Train 7 ML models — RF, XGBoost, LightGBM, CatBoost, SVM, KNN, Decision Tree with 5-fold stratified CV |
| 7  | Model comparison — test accuracy + CV mean/std with error bars |
| 8  | Confusion matrices — full matrix for best model + mini 2×4 grid for all models |
| 9  | Per-class F1 score — classification report + color-coded bar chart |
| 10 | Feature importance — individual bars for 4 tree models + aggregated mean with std error bars |
| 11 | Learning curves — bias-variance analysis for RF, XGBoost, LightGBM, SVM |
| 12 | Stacking ensemble — RF + XGBoost + LightGBM + CatBoost + SVM → Logistic Regression meta-learner + final scoreboard |
| 13 | SHAP — global importance, beeswarm (top 5 crops), heatmap (all crops × features), single-prediction waterfall, dependence scatter plots |
| 14 | Monte Carlo uncertainty — RF tree disagreement, confidence score distribution across test set |
| 15 | Prediction function — `predict_crop(N, P, K, temperature, humidity, ph, rainfall)` with 4 example predictions |

---

## Tech Stack

### Machine Learning
`scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `SHAP`

### Data Processing
`Pandas` · `NumPy`

### Visualization
`Matplotlib` · `Seaborn`

---

---

##  Future Improvements

- FastAPI backend for real-time crop prediction REST API
- TabNet deep learning model integration
- Docker + Kubernetes deployment
- AWS / GCP cloud deployment
- Mobile application for farmers (React Native)
- Multilingual support (Hindi, Punjabi, Tamil)
- Soil image analysis using CNNs
- Weather API integration for auto-filling temperature, humidity, and rainfall
- Integration with real-time agricultural market price APIs (Mandi prices)
- Email notification system for automated crop recommendation reports





##  Acknowledgements

- Dataset: [Atharva Ingle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) on Kaggle
- SHAP library: [Lundberg & Lee, 2017](https://github.com/shap/shap)
- XGBoost: [Chen & Guestrin, 2016](https://github.com/dmlc/xgboost)
- LightGBM: [Ke et al., 2017](https://github.com/microsoft/LightGBM)
- CatBoost: [Prokhorenkova et al., 2018](https://github.com/catboost/catboost)
- Scikit-learn: [Pedregosa et al., 2011](https://scikit-learn.org)
---


