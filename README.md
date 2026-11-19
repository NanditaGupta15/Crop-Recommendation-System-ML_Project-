# 🌾 Crop Recommendation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![Status](https://img.shields.io/badge/Status-Model%20complete%20%7C%20App%20in%20progress-yellow)

A machine learning project that recommends the most suitable crop to grow based on soil nutrients and local climate conditions. The project covers the full workflow: exploratory data analysis, preprocessing, training and comparing four classifiers, cross-validation, feature importance analysis, and a reusable prediction function with input-range checks. The final Random Forest model is saved for deployment.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Project Workflow](#project-workflow)
4. [Key Findings from EDA](#key-findings-from-eda)
5. [Models and Results](#models-and-results)
6. [Feature Importance](#feature-importance)
7. [Error Analysis](#error-analysis)
8. [Using the Saved Model](#using-the-saved-model)
9. [Repository Structure](#repository-structure)
10. [Installation and Usage](#installation-and-usage)
11. [Limitations](#limitations)
12. [Future Work](#future-work)
13. [Tech Stack](#tech-stack)

---

## Problem Statement

Choosing the right crop for a given piece of land depends on soil chemistry and weather conditions. This project builds a multi-class classifier that takes seven agronomic measurements and recommends one of 22 crops. Instead of returning a single answer only, the prediction function also returns the **top-3 candidate crops with confidence scores**, which suits a recommendation use case where more than one crop may be viable.

## Dataset

- **Source:** Crop Recommendation Dataset (Kaggle)
- **Size:** 2,200 rows × 8 columns
- **Target:** `label`, with 22 crop classes and exactly 100 samples per class (perfectly balanced)
- **Missing values:** none

| Feature | Description |
|---|---|
| `N` | Nitrogen content in soil |
| `P` | Phosphorus content in soil |
| `K` | Potassium content in soil |
| `temperature` | Temperature in °C |
| `humidity` | Relative humidity in % |
| `ph` | Soil pH value |
| `rainfall` | Rainfall in mm |

**Crops (22):** apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbean, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon.

## Project Workflow

1. **Data loading and inspection:** shape, data types, descriptive statistics, class distribution.
2. **Exploratory data analysis:** missing values, histograms, skewness, IQR-based outlier detection, correlation heatmap, pairplots, 2-D PCA projection.
3. **Preprocessing:** label encoding of the target, stratified 70/30 train–test split (`random_state=42`).
4. **Modeling:** KNN, Decision Tree (Gini and entropy), Random Forest, and SVM.
5. **Evaluation:** accuracy, macro F1, classification reports, confusion matrices, and 5-fold stratified cross-validation on identical folds for all models.
6. **Interpretation:** feature importance and analysis of the most confused crop pair.
7. **Deployment prep:** `recommend_crop()` (top-N predictions), `check_ranges()` (out-of-range input warnings), and the model and label encoder saved with `joblib`.

## Key Findings from EDA

- **Balanced classes:** all 22 crops have the same number of samples, so accuracy is a fair metric.
- **No missing values** in any column.
- **Outliers:** present in P, K, pH, temperature and rainfall (K and rainfall are the most skewed). They were **retained**, because they reflect natural variation in soil and climate rather than data-entry errors.
- **Correlation:** the only strong relationship is between **P and K (≈ 0.74)**. Most other feature pairs are weakly correlated.
- **Crop-specific requirements:** some crops have very distinct profiles. For example, mothbean grows in more acidic soil, orange and papaya in high-humidity conditions, and chickpea in low-humidity conditions.
- **Non-linear structure:** the pairplots show blocky, threshold-like class boundaries with partial overlap. This suggested that tree-based and other non-linear models would outperform linear ones.
- **PCA:** the 2-D projection shows natural clusters but also noticeable overlap between several crops, so two components are not enough to separate all classes.

## Models and Results

All models were trained on the same stratified 70% split (1,540 samples) and evaluated on the held-out 30% (660 samples). Cross-validation used the same 5 stratified folds for every model.

| Model | Train accuracy | Test accuracy |
|---|---|---|
| KNN (k=5, unscaled) | 98.70% | 97.88% |
| Decision Tree (depth 10, Gini) | 99.03% | 98.03% |
| Decision Tree (depth 10, entropy) | 99.68% | 98.79% |
| **Random Forest** (depth 10, entropy, `min_samples_leaf=5`) | **99.81%** | **99.55%** |

- **Random Forest** was the best model on both the test split and in cross-validation: **99.45% ± 0.23%** (5-fold CV), and it was the most stable across folds. It made only **3 errors out of 660** test samples.
- The notebook's **Model Comparison** section also evaluates a scaled KNN and an SVM (both with default hyperparameters) alongside the Decision Tree and Random Forest, reporting test accuracy, macro F1 and CV mean/std in one table.
- Tree-based models outperformed KNN and SVM, which is consistent with the threshold-like feature ranges seen in the EDA.
- KNN was scored across k = 1–20 and stays between roughly 97.1% and 98.3%, so it is not sensitive to k. k = 5 was kept instead of the test-set peak to avoid selecting a hyperparameter on the test data.
- A scaled KNN pipeline (`StandardScaler`) is used in the comparison, since distance-based models are sensitive to features with large ranges such as K and rainfall.
- The gap of about 1 point between train and test accuracy for the single decision trees shows mild overfitting, which the Random Forest reduces.

## Feature Importance

Random Forest feature importances:

| Feature | Importance |
|---|---|
| humidity | 0.23 |
| K | 0.20 |
| rainfall | 0.19 |
| N | 0.15 |
| P | 0.14 |
| temperature | 0.06 |
| pH | 0.03 |

Humidity, potassium and rainfall together account for about **62%** of the total importance, while temperature and pH matter least. A duplicate-row and "giveaway feature" check found no single feature with a suspiciously dominant importance, so the model is not relying on a leaked column.

## Error Analysis

The remaining confusion is concentrated in a few crops:

- **Rice ↔ Jute:** the main confusion. Two rice samples were predicted as jute by the Random Forest. KNN showed the same weakness (rice recall 0.77, jute precision 0.80), and the notebook compares their humidity and rainfall distributions to investigate it.
- **Blackgram:** one blackgram sample was predicted as maize by the Random Forest, and blackgram was also among the weaker classes for the entropy decision tree (recall 0.90).

## Using the Saved Model

The trained Random Forest and the label encoder are saved as `crop_model.joblib` and `label_encoder.joblib`. Note that the saved model was trained on the 70% training split.

```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load("crop_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall, top_n=3):
    """Return the top_n recommended crops with their confidence scores."""
    sample = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES
    )
    probs = model.predict_proba(sample)[0]
    top_idx = np.argsort(probs)[::-1][:top_n]
    crops = label_encoder.inverse_transform(model.classes_)
    return pd.DataFrame(
        {"Crop": crops[top_idx], "Confidence (%)": (probs[top_idx] * 100).round(1)}
    )

recommend_crop(N=90, P=42, K=43, temperature=21.0, humidity=82.0, ph=6.5, rainfall=203.0)
```

> **Note:** the "confidence" value is the Random Forest's vote share across trees. It is useful for ranking candidates but is not a calibrated probability.

The notebook also includes `check_ranges()`, which warns when an input value falls outside the range seen in the training data, since predictions for such inputs may be unreliable.

## Repository Structure

```
.
├── Crop_Recommendation_System.ipynb   # Full analysis, modeling and evaluation
├── Crop_recommendation.csv            # Dataset
├── crop_model.joblib                  # Saved Random Forest model
├── label_encoder.joblib               # Saved LabelEncoder (class index → crop name)
├── requirements.txt                   # Python dependencies
└── README.md
```

## Installation and Usage

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook Crop_Recommendation_System.ipynb
```

Run the cells from top to bottom. The notebook reads `Crop_recommendation.csv` from the same folder.

**`requirements.txt`**

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

> The saved `.joblib` files should be loaded with the same scikit-learn version they were created with. If you see a version warning, retrain the model from the notebook.

## Limitations

- The dataset is clean, balanced and highly separable, so the ~99.5% accuracy is likely **optimistic compared with real farm data**. The model has not been validated on real-world measurements.
- Each sample has exactly one correct crop, whereas in practice several crops can be suitable for the same conditions. This is why the top-3 output is more informative than a single prediction.
- SVM and KNN were used with default hyperparameters, and the Random Forest settings (depth 10, `min_samples_leaf=5`) were not tuned with a grid or randomized search.
- The model uses only seven inputs and ignores factors such as soil type, market price, irrigation availability and seasonality.
- Rice and jute remain the hardest pair to separate.

## Future Work

- [ ] Build a **Streamlit web app** where users enter the seven values and receive the top-3 crop recommendations, with out-of-range warnings.
- [ ] Wrap the model in a scikit-learn **Pipeline** so the feature order, model and label decoding are saved as one object.
- [ ] Tune hyperparameters for all models (grid or randomized search) for a fairer comparison.
- [ ] Refit the final model on the full dataset after cross-validation.
- [ ] Report **top-3 accuracy** alongside top-1 accuracy and calibrate the predicted probabilities.
- [ ] Test on external or real-world data.

## Tech Stack

- **Language:** Python
- **Data and visualization:** NumPy, pandas, Matplotlib, seaborn
- **Machine learning:** scikit-learn (KNN, Decision Tree, Random Forest, SVM, PCA, cross-validation)
- **Model persistence:** joblib
- **Environment:** Jupyter Notebook
