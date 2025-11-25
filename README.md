# 🌧️ Bangladesh Rainfall Prediction System
A machine learning-based web application that predicts rainfall probability across 35 weather stations in Bangladesh using historical weather patterns and real-time data.

### 🎯 Overview
This project implements a rainfall prediction system specifically designed for Bangladesh weather patterns. The system uses a CatBoost classifier trained on historical weather data to predict rainfall probability with 87% accuracy. The web interface allows users to input current weather conditions and get instant rainfall predictions with actionable insights.

### ✨ Features
🌍 Multi-Station Support: Covers 35 weather stations across Bangladesh
🤖 ML-Powered: CatBoost model with 87.03% accuracy and 94.52% AUC
📱 Responsive UI: Bootstrap-based interface optimized for all devices
🔮 Smart Predictions: Rainfall probability with meaningful recommendations
⏰ Real-time Data: Integrates with OpenWeatherMap API for historical data
📊 Advanced Features: 28 engineered features including temporal, cyclic, and lag patterns
🎥 Demo

<img width="731" height="685" alt="image" src="https://github.com/user-attachments/assets/c5847016-7ef8-4291-8338-8fff0276c8ec" />


Live Demo: [Coming Soon]

### 🛠️ Installation
Prerequisites
Python 3.8+

OpenWeatherMap API key

### 🚀 Usage
- Select Weather Station: Choose from 35 available stations in Bangladesh
- Enter Date: Input the prediction date (defaults to current date)
- Input Weather Parameters:
- Sunshine hours
- Humidity percentage
- Temperature in Celsius
- Location coordinates (latitude/longitude)
- Get Prediction: Click "Predict Rainfall Probability" to see results
- Interpret Results: Receive probability percentage with actionable advice

### Prediction Interpretation
Probability Range	Message	Recommendation
80%+	High chance of rainfall	Strongly recommend carrying umbrella
60-79%	Moderate to high chance	Better to be prepared with rain protection
40-59%	Some possibility	Keep an eye on weather updates
20-39%	Low chance	Light showers possible
<20%	Very low chance	Enjoy the dry weather

### 📊 Model Performance 
The CatBoost classifier demonstrates excellent performance:
- Accuracy: 87.03%
- AUC Score: 94.52%
- Precision: 88% (No Rain), 84% (Rain)
- Recall: 92% (No Rain), 77% (Rain)

### Feature Engineering
- The model uses 28 engineered features:
- Basic Features: Station, Year, Month, Day, Sunshine, Humidity, Temperature
- Temporal Features: DayOfYear, WeekOfYear
- Cyclic Features: Month sin/cos, Day sin/cos
- Lag Features: 1,2,3-day lags for Rainfall, Temperature, Humidity
- Rolling Means: 3,7-day averages for key variables

### 🙏 Acknowledgments
Bangladesh Meteorological Department for weather data
OpenWeatherMap for API services
CatBoost for the powerful ML framework
Bootstrap for the responsive UI components

