# Phase 5 Project: AI-Driven Quality Control in Manufacturing

# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Simulate Manufacturing Sensor Data
np.random.seed(42)
n_samples = 200
width = np.random.normal(loc=50, scale=0.9, size=n_samples)
weight = np.random.normal(loc=200, scale=5, size=n_samples)
temperature = np.random.normal(loc=70, scale=3, size=n_samples)

defective = (
    (abs(width - 50) > 1.0) |
    (abs(weight - 200) > 6.0) |
    (abs(temperature - 70) > 4.0)
).astype(int)

df = pd.DataFrame({
    'Width': width,
    'Weight': weight,
    'Temperature': temperature,
    'Defective': defective
})

# Step 3: Visualize Feature Distributions
sns.pairplot(df, hue='Defective')
plt.suptitle("Feature Distribution by Product Type", y=1.02)
plt.show()

# Step 4: Prepare Data for Training
X = df[['Width', 'Weight', 'Temperature']]
y = df['Defective']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 7: Feature Importance Analysis
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance in Predicting Defects")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Step 8: Predict on New Sample
sample = pd.DataFrame([[50.5, 198, 69]], columns=['Width', 'Weight', 'Temperature'])
prediction = model.predict(sample)
print("Sample Prediction:", "Defective" if prediction[0] == 1 else "Non-defective")

# Step 9: Interactive Visualizations with Plotly
# Pie Chart of Defective vs Non-defective
defect_counts = df['Defective'].value_counts()
labels = ['Non-defective', 'Defective']
fig_pie = go.Figure(data=[go.Pie(labels=labels, values=defect_counts, hole=.4)])
fig_pie.update_layout(title="Product Quality Distribution")
fig_pie.show()

# Gauge Chart for Quality Score
defect_rate = 100 * df['Defective'].mean()
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=100 - defect_rate,
    title={'text': "Quality Score (%)"},
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': "green"},
           'steps': [
               {'range': [0, 50], 'color': "red"},
               {'range': [50, 80], 'color': "yellow"},
               {'range': [80, 100], 'color': "lightgreen"}]
           }
))
fig_gauge.show()

# Step 10: Export Data to Excel
df['Prediction'] = model.predict(X)
df['Status'] = df['Prediction'].apply(lambda x: 'Defective' if x else 'Non-defective')
df.to_excel("quality_control_data.xlsx", index=False)
print("Excel file saved as 'quality_control_data.xlsx'")
