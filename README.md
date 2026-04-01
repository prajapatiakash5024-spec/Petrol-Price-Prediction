# ⛽ PetrolIQ — Petrol Price Prediction using Machine Learning

A professional Streamlit web application that predicts global petrol prices using Linear Regression on a dataset of 181 countries (June 2022).

## 🚀 Live Demo
Project link (https://petrol-price-prediction-8wch3uwugdbsf5wmaowxde.streamlit.app/)

---

## 📋 Features

- **Overview Dashboard** — Key metrics, price distribution, and top expensive countries
- **Data Explorer** — Raw data table, correlation heatmap, GDP vs price scatter plot
- **Model & Metrics** — Actual vs Predicted chart, residuals, feature coefficients, R² score
- **Live Predictor** — Interactive sliders to input parameters and get real-time price predictions with a gauge chart

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| `Python` | Core language |
| `Streamlit` | Web application framework |
| `Scikit-Learn` | Linear Regression model |
| `Pandas / NumPy` | Data manipulation |
| `Plotly` | Interactive charts |
| `Seaborn / Matplotlib` | Heatmap visualization |
| `OpenPyXL` | Excel file reading |

---

## 📂 Project Structure

```
├── app.py                                              # Main Streamlit app
├── requirements.txt                                    # Python dependencies
├── Petrol_Dataset_June_23_2022_--_Version_2_csv.xlsx  # Dataset
└── README.md
```

---

## ⚙️ Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/petrol-price-predictor.git
cd petrol-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Select your repo → Set `app.py` as the main file
4. Click **Deploy** ✅

---

## 📊 Dataset

- **Source:** Global Petrol Prices (June 23, 2022)
- **Rows:** 181 countries
- **Target:** `Price Per Liter (USD)`
- **Features:** Daily oil consumption, world share, GDP per capita, yearly gallons per capita, etc.

---

## 🤖 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Linear Regression |
| Train/Test Split | 70% / 30% |
| R² Score | ~0.99+ |

---

## 👨‍💻 Author

Mini Project — Machine Learning  
*Petrol Price Prediction using Global Economic Indicators*
