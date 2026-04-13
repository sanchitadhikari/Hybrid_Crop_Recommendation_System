## Hybrid Crop Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-02B875?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-FFCC00?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-9B59B6?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

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
```
=======================================================
  CROP RECOMMENDATION RESULT
=======================================================
  Input: N=90, P=42, K=43, Temp=20.8°C, Humidity=82%, pH=6.5, Rainfall=202.9mm

   RECOMMENDED CROP : RICE
  Model              : Stacking Ensemble
  Confidence         : 98.2%
  Uncertainty Score  : 0.0198

  Top 3 Candidates:
    rice             98.2%  ███████████████████
    jute              0.5%
    pigeonpeas        0.1%
=======================================================
```

---

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

```
crop-recommendation-system/
│
├── app/                          ← FastAPI backend
│   ├── main.py                   ← App entry point & routes
│   ├── predict_api.py            ← Prediction endpoint logic
│   ├── price_fallback.py         ← Market price fallback handler
│   └── crop_mapping.py           ← Crop name ↔ ID mappings
│
├── ml/                           ← Machine learning modules
│   ├── preprocess.py             ← Data cleaning, scaling, splitting
│   ├── train_models.py           ← Train RF, XGBoost, LightGBM, CatBoost
│   ├── tabnet_model.py           ← TabNet deep learning model
│   ├── ensemble.py               ← Stacking ensemble builder
│   ├── explainability.py         ← SHAP plots and analysis
│   └── uncertainty.py            ← Monte Carlo uncertainty estimation
│
├── models/                       ← Saved model files (auto-generated)
│   ├── model_rf.pkl
│   ├── model_xgb.pkl
│   ├── model_lgbm.pkl
│   ├── model_catboost.pkl
│   ├── model_tabnet.zip
│   ├── stacked_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── data/
│   └── crop_recommendation.csv   ← Dataset (download from Kaggle)
│
├── static/                       ← Static files for frontend
├── templates/                    ← HTML templates
├── notebooks/
│   └── Hybrid_Crop_Recommendation_System.ipynb  ← Full Colab notebook
│
├── requirements.txt
├── .env.example
└── README.md
```

---



| Step | Content |
|------|---------|
| 1 | Install libraries |
| 2 | Import all packages |
| 3 | Load dataset |
| 4 | EDA — distributions, box plots, correlations, violin plots, pairplot |
| 5 | Preprocessing — label encoding, feature scaling, train/test split |
| 6 | Train 8 ML models with cross-validation |
| 7 | Model comparison charts + CV error bars |
| 8 | Confusion matrices (full + mini grid for all models) |
| 9 | Per-class F1 score analysis |
| 10 | Feature importance (4 tree models + aggregated) |
| 11 | Learning curves (bias-variance analysis) |
| 12 | Stacking ensemble with final scoreboard |
| 13 | SHAP — 5 plot types |
| 14 | Monte Carlo uncertainty estimation |
| 15 | Interactive prediction function |
| 16 | Summary |

---

## Tech Stack

### Machine Learning
`scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `pytorch-tabnet` · `SHAP`

### Backend
`FastAPI` · `Uvicorn` · `Python-dotenv`

### Data Processing
`Pandas` · `NumPy`

### Visualization
`Matplotlib` · `Seaborn`

### Utilities
`ReportLab` · `Joblib`

---

##  Future Improvements

- Integration with real-time agricultural market price APIs (Mandi prices)
-  Docker + Kubernetes deployment
-  AWS / GCP cloud deployment
-  Mobile application for farmers (React Native)
-  Multilingual support (Hindi, Punjabi, Tamil)
-  Soil image analysis using CNNs
-  Weather API integration for auto-filling temperature/humidity/rainfall

---





##  Acknowledgements

- Dataset: [Atharva Ingle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) on Kaggle
- SHAP library: [Lundberg & Lee, 2017](https://github.com/shap/shap)
- TabNet: [Arik & Pfister, 2021](https://github.com/dreamquark-ai/tabnet)

---

<div align="center">
Made with  for smarter, data-driven agriculture
</div>
