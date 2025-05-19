# AI-Driven Quality Control System for Manufacturing

## 🚀 Objective
To build an intelligent, real-time quality control system for manufacturing industries using Artificial Intelligence. The system automates defect detection in products based on dimensional, weight, and temperature parameters using sensor data and machine learning.

---

## 🌟 Features

- ✅ Real-time defect detection using sensor data (width, weight, temperature)
- 📊 Live dashboard displaying model accuracy, responsiveness, and prediction speed
- 🛠 Integration-ready with existing production lines
- 🔐 Product-level traceability with timestamped logs
- 📩 Alert generation for defective products
- 🔄 Modular design for future scalability and maintainability

---

## 🛠 Technology Used

- Programming Language: Python
- ML Library: Scikit-learn (Random Forest)
- Data Handling: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Jupyter Notebook: For code execution and visualization
- Version Control: Git & GitHub
- Platform: Local system simulation (future-ready for Raspberry Pi integration)

---

## ⚙ How It Works

1. Data Collection from sensors measuring product parameters (width, weight, temperature).
2. Preprocessing: Clean and normalize the data.
3. Model Prediction: Random Forest model classifies products as "Defective" or "Non-Defective".
4. Visualization: Real-time dashboard updates with accuracy and predictions.
5. Logging & Alerts: Defects are logged with time and prediction confidence; alerts are generated for quality control staff.

---

## 📁 Data Collection

- Synthetic + Real Sensor Simulation
- Data was either simulated or sourced from lab sensors measuring:
  - Width
  - Weight
  - Temperature
- Labeled manually for model training (defective / non-defective)
-
