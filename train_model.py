import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

np.random.seed(42)
n_samples = 2500

# ---------------------------------------------------------
# STAGE 1 MODEL: First-Time Borrowers (Alternative Data)
# Features: avg_daily_income, volatility, service_rating, bill_punctuality
# ---------------------------------------------------------
avg_income_1 = np.random.uniform(200, 2500, n_samples)
volatility_1 = np.random.uniform(0.05, 0.8, n_samples)
ratings_1 = np.random.uniform(1.0, 5.0, n_samples)
punctuality_1 = np.random.uniform(30, 100, n_samples)

# Initial Safe Loan: ₹5,000 to ₹25,000
safe_loan_1 = (avg_income_1 * 12) * (1 - volatility_1 * 0.3) * (ratings_1 / 5.0) * (punctuality_1 / 100)
safe_loan_1 = np.clip(safe_loan_1, 5000, 25000)

X1 = pd.DataFrame({
    'avg_daily_income': avg_income_1,
    'income_volatility': volatility_1,
    'service_rating': ratings_1,
    'billing_punctuality': punctuality_1
})

model_stage1 = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
model_stage1.fit(X1, safe_loan_1)
joblib.dump(model_stage1, 'model_stage1.pkl')

# ---------------------------------------------------------
# STAGE 2 MODEL: Repeat Borrowers (NBFC Repayment Track Record)
# Features: previous_score_100, repayment_rate_pct, delayed_installments
# ---------------------------------------------------------
prev_score = np.random.uniform(40, 95, n_samples)
repayment_rate = np.random.uniform(50, 100, n_samples) # % of installments paid on time
delayed_count = np.random.randint(0, 6, n_samples)

# Enhanced Safe Loan for repeat borrowers: up to ₹60,000
safe_loan_2 = (prev_score * 300) + (repayment_rate * 300) - (delayed_count * 2500)
safe_loan_2 = np.clip(safe_loan_2, 8000, 60000)

X2 = pd.DataFrame({
    'prev_score': prev_score,
    'repayment_rate': repayment_rate,
    'delayed_count': delayed_count
})

model_stage2 = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
model_stage2.fit(X2, safe_loan_2)
joblib.dump(model_stage2, 'model_stage2.pkl')

print("✅ Both Stage 1 and Stage 2 ML Models trained and saved successfully!")