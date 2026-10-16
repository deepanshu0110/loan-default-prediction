# Loan Default Prediction
[![CI](https://github.com/deepanshu0110/loan-default-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/deepanshu0110/loan-default-prediction/actions)


![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=flat-square&logo=streamlit)
![Data](https://img.shields.io/badge/Data-synthetic-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

Predicts loan default probability from borrower financial data. Includes a Streamlit risk calculator with Approve / Review / Reject output.

**Data note:** the 10,000 loan records are synthetic. `loan_prediction.py` generates them with a fixed seed (42) from a rule that links default to credit score, debt-to-income, loan-to-income and prior defaults. The scores below show how well each model recovers that rule, not performance on real lending data.

---

## Business Problem

Lenders lose money on defaults and lose customers on false rejections. A risk scoring model helps credit teams make faster, consistent decisions — flagging high-risk applicants while approving creditworthy ones.

---

## Results

Test set: 2,000 records (stratified 80/20 split, seed 42).

| Model | Accuracy | ROC-AUC | Average precision |
|---|---|---|---|
| Logistic Regression | 83% | 0.913 | 0.888 |
| **Gradient Boosting** | **97%** | **0.998** | **0.997** |

Gradient Boosting's near-perfect score is expected on synthetic data: the model learns the exact rule that generated the labels. On real loan data, expect much lower scores.

**Default rate by risk factor** (overall default rate 41.3%):
- Credit score < 600 → 76.4%
- Loan-to-income > 0.5 → 60.3%
- Prior defaults > 0 → 64.2%

---

## Quickstart

```bash
git clone https://github.com/deepanshu0110/loan-default-prediction.git
cd loan-default-prediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # pins matplotlib 3.7; newer versions break the boxplot call
python loan_prediction.py   # generates data, trains, exports plots
streamlit run app.py        # launches risk calculator
```

---

## Streamlit App

Enter applicant details and get: default probability %, risk level (Low/Medium/High), decision (Approve/Review/Reject), and feature importance breakdown.

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

The tests runs the full training script, checks both ROC-AUC scores and the synthetic data, and starts the Streamlit app. GitHub Actions runs them on every push.

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit

---

## Roadmap

- [ ] SHAP explainability layer
- [ ] FastAPI REST endpoint
- [ ] Model monitoring dashboard

---

## Author

**Deepanshu Garg** — Freelance Data Scientist
- GitHub: [@deepanshu0110](https://github.com/deepanshu0110)
- Hire: [freelancer.com/u/deepanshu0110](https://www.freelancer.com/u/deepanshu0110)

MIT License