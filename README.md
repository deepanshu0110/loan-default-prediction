# Loan Default Prediction
[![CI](https://github.com/deepanshu0110/loan-default-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/deepanshu0110/loan-default-prediction/actions)


![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=flat-square&logo=streamlit)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.88-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

Predicts loan default probability using borrower financial data. Gradient Boosting achieves 85% accuracy and ROC-AUC 0.88. Includes a Streamlit risk calculator with Approve / Review / Reject output.

---

## Business Problem

Lenders lose money on defaults and lose customers on false rejections. A risk scoring model helps credit teams make faster, consistent decisions — flagging high-risk applicants while approving creditworthy ones.

---

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 78% | 0.82 |
| **Gradient Boosting** | **85%** | **0.88** |

**Top Risk Factors:**
- Credit score < 600 → 45% default rate
- Loan-to-income > 0.5 → 38% default rate
- Prior defaults > 0 → 52% default rate

---

## Quickstart

```bash
git clone https://github.com/deepanshu0110/loan-default-prediction.git
cd loan-default-prediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python loan_prediction.py   # generates data, trains, exports plots
streamlit run app.py        # launches risk calculator
```

---

## Streamlit App

Enter applicant details and get: default probability %, risk level (Low/Medium/High), decision (Approve/Review/Reject), and feature importance breakdown.

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