# 🩸 GlucoShield AI – Early Diabetes Risk Analytics System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**GlucoShield AI** is an end-to-end medical decision-support platform designed to assess early diabetes risk using machine learning. Built on the PIMA Indians Diabetes Database, the application cleans physiological anomalies (such as zero-value blood pressure or glucose readings), scales inputs dynamically, and delivers real-time clinical predictions along with patient risk stratification.

---

## 🌟 Key Features

* **🩺 Individual Patient Assessment**: Interactive input sliders with clinical reference ranges and real-time risk probability calculation.
* **📁 Bulk Batch Screening**: Enables healthcare workers to upload a dataset (`.csv`) for fast multi-patient risk analysis.
* **⚡ Modern Preprocessing Pipeline**: Handles non-viable physiological zero values via targeted median imputation and standardizes metrics using `StandardScaler`.
* **📈 Transparent Metrics**: Integrated dashboard displaying algorithm details, model parameters, and key clinical risk indicators.

---

## 🏗️ Tech Stack

* **Language**: Python
* **Machine Learning**: Scikit-Learn (`RandomForestClassifier`, `StandardScaler`)
* **Data Processing**: Pandas, NumPy
* **Model Serialization**: Joblib
* **Web UI**: Streamlit

---

## 🚀 Quick Start & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/himanshusingh-45/diabetes-prediction-system.git](https://github.com/himanshusingh-45/diabetes-prediction-system.git)
cd diabetes-prediction-system
