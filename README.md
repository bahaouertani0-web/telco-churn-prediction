# Telco Customer Churn Prediction

Machine-learning project that predicts which telecom customers are likely to leave (churn), so that a company can target retention actions.

## Dataset

IBM **Telco Customer Churn** dataset: 7,043 customers, 21 columns (demographics, subscribed services, contract, billing, churn label). About 26.5 % of customers churn.

The data file is not included in this repository. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` (for example from Kaggle, "Telco Customer Churn") and place it in the `data/` folder.

## Approach

1. **Cleaning**: `TotalCharges` is stored as text and contains blank values; it is converted to numeric and missing values are filled with the median.
2. **Exploratory analysis**: distributions of numerical features, churn rate by contract type and internet service.
3. **Preprocessing**: stratified 80/20 train/test split, one-hot encoding, standard scaling (fitted on the training set only).
4. **Models**: Dummy baseline, Logistic Regression (balanced class weights), Random Forest (balanced class weights).
5. **Metric choice**: accuracy is misleading here (a model that always predicts "stay" gets 73 %), so the focus is on the **recall and F1-score of the churn class**.

## Results (test set, churn class = "Yes")

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Dummy (always "No") | 0.735 | 0.00 | 0.00 | 0.00 |
| Logistic Regression (balanced) | 0.739 | 0.51 | **0.78** | **0.61** |
| Random Forest (balanced) | 0.789 | 0.63 | 0.50 | 0.56 |

Logistic Regression is the preferred model: it detects 78 % of the customers who leave, at the cost of some false alarms.

## Project structure

```
telco-churn-prediction/
├── data/                  # put the CSV here (not tracked by git)
├── notebooks/
│   └── churn_prediction.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## How to run

```bash
git clone https://github.com/bahaouertani0-web/telco-churn-prediction.git
cd telco-churn-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook notebooks/churn_prediction.ipynb
```

## Possible improvements

- Hyperparameter tuning and cross-validation
- ROC-AUC and precision-recall curves
- Decision-threshold optimisation
- Interpretation of the model coefficients (which features drive churn)

## License

MIT, see [LICENSE](LICENSE).

Author: Baha Ouertani
