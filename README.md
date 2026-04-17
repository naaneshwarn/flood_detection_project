# 🌊 FloodGuard India: Risk Forecasting System

An end-to-end Machine Learning project to predict flood risk levels across Indian states using historical IMD data (1967–2023).

## 🚀 Features
- **FastAPI Backend:** RESTful API for real-time risk inference.
- **Random Forest Model:** Classified into Low, Medium, High, and Extreme risk.
- **Streamlit UI:** Dynamic dashboard that changes theme based on risk severity.

## 🛠️ Tech Stack
- **Language:** Python 3.11
- **ML:** Scikit-Learn, Pandas, Numpy
- **API:** FastAPI, Uvicorn
- **Frontend:** Streamlit, Plotly

## 🏃 How to Run
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Train Model:** `python train_and_save.py`
3. **Start API:** `uvicorn main:app --reload`
4. **Launch UI:** `streamlit run front-end.py`
