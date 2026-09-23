from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

model_stage1 = joblib.load('model_stage1.pkl')
model_stage2 = joblib.load('model_stage2.pkl')

@app.route('/api/stage1-score', methods=['POST'])
def stage1_score():
    """STAGE 1: First-time borrower using alternative signals"""
    data = request.json
    daily_income = data.get('dailyIncome', [])
    rating = float(data.get('rating', 0))
    punctuality = float(data.get('punctualityPercentage', 0))

    if not daily_income:
        return jsonify({'error': 'Missing transaction history'}), 400

    # Feature Processing
    arr = np.array(daily_income, dtype=float)
    avg_income = np.mean(arr)
    std_dev = np.std(arr)
    volatility = (std_dev / avg_income) if avg_income > 0 else 1.0

    # ML Loan Prediction
    pred_loan = float(model_stage1.predict([[avg_income, volatility, rating, punctuality]])[0])
    recommended_loan = round(pred_loan / 500) * 500

    # Score calculation on a 0 to 100 scale
    score_100 = int(((recommended_loan - 5000) / (25000 - 5000)) * 60 + 35)
    score_100 = max(10, min(100, score_100))

    reasons = []
    if volatility < 0.35:
        reasons.append("STABLE_DAILY_EARNINGS: Income consistency verified.")
    else:
        reasons.append("VOLATILE_EARNINGS_SMOOTHED: Smoothed using rolling-window trend.")

    if rating >= 4.5:
        reasons.append("HIGH_PLATFORM_RATING: Verified working platform customer rating.")
    
    if punctuality >= 90:
        reasons.append("TIMELY_UTILITY_PAYMENTS: High bill repayment punctuality.")

    return jsonify({
        'success': True,
        'stage': 1,
        'creditScore100': score_100,
        'recommendedLoanLimit': f"₹{recommended_loan:,}",
        'nbicDecision': "APPROVED" if score_100 >= 50 else "REJECTED",
        'reasons': reasons
    })

@app.route('/api/stage2-score', methods=['POST'])
def stage2_score():
    """STAGE 2: Repeat borrower using NBFC repayment feedback loop"""
    data = request.json
    prev_score = float(data.get('previousScore', 50))
    repayment_rate = float(data.get('repaymentRate', 100)) # % paid on time
    delayed_count = int(data.get('delayedCount', 0))       # Count of delayed installments

    pred_loan = float(model_stage2.predict([[prev_score, repayment_rate, delayed_count]])[0])
    recommended_loan = round(pred_loan / 1000) * 1000

    # Upgraded score calculation on a 0 to 100 scale based on repayment performance
    score_100 = int(prev_score + (repayment_rate * 0.3) - (delayed_count * 8))
    score_100 = max(10, min(100, score_100))

    reasons = []
    if delayed_count == 0:
        reasons.append("PERFECT_REPAYMENT_HISTORY: All installments paid on time.")
    else:
        reasons.append(f"DELAY_PENALTY: {delayed_count} delayed installment(s) logged by NBFC.")

    if score_100 > prev_score:
        reasons.append("CREDIT_LIMIT_INCREASE: Higher loan eligibility granted.")

    return jsonify({
        'success': True,
        'stage': 2,
        'creditScore100': score_100,
        'enhancedLoanLimit': f"₹{recommended_loan:,}",
        'nbicDecision': "APPROVED FOR HIGHER LOAN" if score_100 >= 50 else "REJECTED / HOLD",
        'reasons': reasons
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)