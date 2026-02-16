# Predictive Modeling for Click-Through Rate Optimization

## Overview

A machine learning solution to predict whether users will click on advertisements based on demographic and behavioral data. This project helps optimize ad targeting strategies by identifying high-probability clickers, reducing wasted ad spend and improving Return on Ad Spend (ROAS).

## Problem Statement

Online advertising campaigns often show ads to users with low engagement probability, leading to wasted ad spend and poor Click-Through Rates (CTR). This project develops a predictive model to estimate click probability, enabling more efficient ad targeting.

## Dataset

**File:** `advertising.csv`

**Features:**
- Daily Time Spent on Site (numeric)
- Age (numeric)
- Area Income (numeric)
- Daily Internet Usage (numeric)
- Ad Topic Line (categorical/text)
- City (categorical)
- Male (binary)
- Timestamp (datetime)
- **Target:** Clicked on Ad (binary: 1 = clicked, 0 = not clicked)

## Data Preprocessing

- **Missing Data:** Checked for null values; imputed or dropped as necessary
- **Outlier Handling:** Detected and clipped extreme values in Age, Daily Time Spent, and Daily Internet Usage
- **Feature Engineering:** Extracted Hour of Day and Day of Week from Timestamp; created ratio features
- **Categorical Encoding:** Binary encoding for Male; dropped high-cardinality features (Ad Topic Line, City)
- **Feature Scaling:** Standardized numerical features using StandardScaler
- **Data Split:** 80/20 train-test split with separate scaling on train set

## Model Development

**Algorithm:** Logistic Regression
- Chosen for interpretability and computational efficiency
- Binary classification output (likely to click vs unlikely to click)

**Pipeline:**
1. Data preprocessing (scaling, encoding, feature selection)
2. Logistic regression model training
3. Performance evaluation and visualization

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | ~92%  |
| Precision | ~90%  |
| Recall    | ~93%  |
| AUC       | ~0.95 |

The model effectively distinguishes between likely and unlikely clickers.

## Business Impact

- **Improved CTR:** Ads targeted to users with higher engagement probability
- **Reduced Ad Waste:** Lower spend on low-probability users
- **Better ROAS:** Stronger returns for advertising clients
- **Competitive Advantage:** Data-driven targeting enhances market position

## Project Structure

```
├── ctr_model.pkl              # Serialized pipeline (preprocessing + model)
├── custom_transformers.py     # Data transformation and outlier handling
├── train_model.py             # Model training and serialization script
├── evaluate_model.py          # Model evaluation and visualization script
├── advertising.csv            # Dataset
├── README.md                  # This file
└── visualizations/            # Generated plots (ROC, precision-recall, confusion matrix)
```

## Installation

### Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Setup

```bash
# Clone repository
git clone <repository-url>
cd ctr-prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train_model.py
```

This script:
- Loads and preprocesses the advertising dataset
- Trains the logistic regression model
- Saves the pipeline as `ctr_model.pkl`

### Evaluating the Model

```bash
python evaluate_model.py
```

This script:
- Loads the trained model
- Generates predictions on test data
- Displays metrics (accuracy, precision, recall, F1, AUC)
- Generates visualizations (ROC curve, confusion matrix, precision-recall curve)

## Deliverables

- **ctr_model.pkl** – Serialized preprocessing + logistic regression pipeline
- **custom_transformers.py** – Data transformation utilities
- **train_model.py** – Training script
- **evaluate_model.py** – Evaluation and visualization script
- **Visualizations** – ROC curve, precision-recall curve, confusion matrix

## Future Work

- Experiment with advanced models (Random Forest, XGBoost, Gradient Boosting)
- Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
- Deploy as REST API (Flask/FastAPI) for real-time predictions
- Incorporate additional features (device type, browsing history, user location)
- A/B testing for model performance in production

## Results and Insights

The logistic regression model achieves strong performance with ~92% accuracy and 0.95 AUC, making it reliable for identifying high-probability clickers. The model's interpretability allows stakeholders to understand feature importance and make data-driven decisions.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests with improvements.

## License

This project is open source and available under the MIT License.

## Contact

For questions or collaboration, feel free to reach out.
email : <vivek.rupapara.g@gmail.com>

---

**Note:** This project demonstrates end-to-end machine learning workflow including data preprocessing, model development, evaluation, and business impact analysis.
