fraud_xgb_model.pkl is your REAL trained model - already in place, ready to use.

scaler_amount.pkl and scaler_time.pkl are MISSING ON PURPOSE.

Why: the scaler.pkl in your original repo has a bug - it was fit twice
on the same StandardScaler object (once on Amount, then again on Time),
and the second fit silently overwrote the first. The saved file only
remembers Time's statistics, so using it to scale Amount at inference
time produces wrong values (confirmed: its mean_/scale_ match Time's
range, not Amount's).

Your trained model itself is fine - it was trained on correctly-scaled
data before the bug occurred. Only the saved scaler.pkl is broken.

