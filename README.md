# Credit Scoring Model
## Overview
This project predicts whether an individual is creditworthy using financial and 
demographic data. It was developed as part of the CodeAlpha Machine Learning 
Internship.
## Objective
To classify applicants as Creditworthy or Not Creditworthy based on features 
such as age, income, debt, credit score, loan amount, and payment history.
## Features Used- Age- Gender- Education- Income- Debt- Credit Score- Loan Amount- Loan Term- Number of Credit Cards- Payment History- Employment Status- Residence Type- Marital Status
## Algorithms Used- Random Forest Classifier
## Evaluation Metrics
- Accuracy- Precision- Recall- F1 Score- ROC-AUC Score
## Additional Features- Outlier Handling- Feature Importance Plot- ROC Curve- Streamlit Web Application
## Files Included- credit_model.py- app.py- credit_model.pkl- scaler.pkl
## How to Run
```bash
pip install -r requirements.txt
python credit_model.py
streamlit run app.py
