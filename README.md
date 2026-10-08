NEXUS-AI — Intelligent Warehouse Prediction & Optimization
A software-only AI/ML prototype using synthetic warehouse data. Includes regression, classification, anomaly detection, clustering-ready data exploration, and congestion-weighted grid path planning.

Run on Windows (PowerShell)
cd path\to\NEXUS_AI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
Open the localhost URL printed in the terminal (usually http://localhost:8501).

Included
Linear and multiple-feature regression, polynomial regression, decision tree and random forest regression
Logistic regression, decision tree, random forest, KNN and SVM classification
Isolation Forest anomaly detection with synthetic ground-truth evaluation
Interactive warehouse grid with A*, Dijkstra and congestion-weighted adaptive A*
Dataset preview and CSV download
Data and safety
The application uses generated synthetic data for demonstration, not real warehouse measurements. Results are educational and must not be treated as validated operational performance or used to control a real robot.
