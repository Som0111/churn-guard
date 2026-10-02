#!/usr/bin/env bash
# Smoke test for a running ChurnGuard API: wait until it is ready, then score a customer.
# Usage: scripts/smoke_test.sh [base_url]     (default http://localhost:8000)
set -euo pipefail

BASE="${1:-http://localhost:8000}"

echo "Waiting for ${BASE}/health/ready ..."
for _ in $(seq 1 60); do
    if curl -fsS "${BASE}/health/ready" > /dev/null 2>&1; then
        break
    fi
    sleep 2
done
curl -fsS "${BASE}/health/ready"
echo

echo "Scoring one customer ..."
RESPONSE=$(curl -fsS -X POST "${BASE}/predict" -H "Content-Type: application/json" -d '{
  "tenure": 1, "Contract": "Month-to-month", "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check", "TechSupport": "No", "OnlineSecurity": "No",
  "MonthlyCharges": 95.0, "TotalCharges": 95.0}')
echo "${RESPONSE}"

# Fail unless the response carries a probability and a decision.
echo "${RESPONSE}" | grep -q '"churn_probability"'
echo "${RESPONSE}" | grep -q '"will_churn"'
echo "Smoke test passed."
