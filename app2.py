from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os

app = Flask(__name__)

# Load your trained model
model = joblib.load('./bd_weather_output/best_cat.joblib')  # Make sure to save your model with this name

# OpenWeatherMap API configuration
API_KEY = "API"  # Replace with your actual API key
BASE_URL = "http://api.openweathermap.org/data/2.5/onecall"

def get_historical_weather(lat, lon):
    """Fetch historical weather data for lag features"""
    try:
        url = f"{BASE_URL}?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        
        # Extract daily data for past 3 days
        daily_data = data.get('daily', [])[:3]
        
        lag_features = {}
        for i, day_data in enumerate(daily_data, 1):
            lag_features[f'temp_lag{i}'] = day_data['temp']['day']
            lag_features[f'humidity_lag{i}'] = day_data['humidity']
            lag_features[f'rainfall_lag{i}'] = day_data.get('rain', 0)
            
        return lag_features
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return {}

def calculate_cyclic_features(month, day):
    """Calculate cyclic features for time patterns"""
    month_sin = np.sin(2 * np.pi * (month / 12.0))
    month_cos = np.cos(2 * np.pi * (month / 12.0))
    day_sin = np.sin(2 * np.pi * (day / 31.0))
    day_cos = np.cos(2 * np.pi * (day / 31.0))
    
    return month_sin, month_cos, day_sin, day_cos

def calculate_temporal_features(year, month, day):
    """Calculate temporal features"""
    date_obj = datetime(year, month, day)
    day_of_year = date_obj.timetuple().tm_yday
    week_of_year = date_obj.isocalendar()[1]
    
    return day_of_year, week_of_year

def get_crop_advisory(probability, temperature, humidity, month):
    """Generate crop-specific advisories based on weather conditions"""
    
    # Determine season based on month
    if month in [11, 12, 1, 2]:
        season = "Rabi"  # Winter season
        main_crops = ["Wheat", "Potato", "Mustard", "Pulses", "Vegetables"]
    elif month in [3, 4, 5]:
        season = "Kharif-1"  # Pre-monsoon
        main_crops = ["Aus Rice", "Jute", "Maize", "Mango", "Litchi"]
    else:  # 6,7,8,9,10
        season = "Kharif-2"  # Monsoon season
        main_crops = ["Aman Rice", "Jute", "Vegetables", "Banana"]
    
    advisories = []
    
    # High rainfall scenarios
    if probability >= 70:
        advisories.append(f"🌧️ <strong>High Rainfall Alert for {season} Season</strong>")
        
        if "Rice" in main_crops:
            if probability >= 80:
                advisories.append("🚫 <strong>Rice Crops:</strong> Risk of flooding and waterlogging. Ensure proper drainage in fields.")
            else:
                advisories.append("⚠️ <strong>Rice Crops:</strong> Monitor water levels. Young seedlings may need protection.")
        
        if "Wheat" in main_crops:
            advisories.append("🚫 <strong>Wheat:</strong> Heavy rain can cause fungal diseases. Apply fungicide if continuous rain persists.")
        
        if "Potato" in main_crops:
            advisories.append("🚫 <strong>Potato:</strong> Risk of late blight. Harvest mature potatoes before heavy rain.")
        
        if "Jute" in main_crops:
            advisories.append("⚠️ <strong>Jute:</strong> Harvested jute should be stored in covered areas to avoid damage.")
        
        if "Vegetables" in main_crops:
            advisories.append("🚫 <strong>Vegetables:</strong> Tomato, eggplant, and chili plants need protection from heavy rain. Use covers if possible.")
    
    # Moderate rainfall scenarios
    elif probability >= 40:
        advisories.append(f"🌦️ <strong>Moderate Rainfall Expected for {season} Season</strong>")
        
        if "Rice" in main_crops:
            advisories.append("✅ <strong>Rice:</strong> Beneficial for irrigation. Good for Aman rice transplantation.")
        
        if "Jute" in main_crops:
            advisories.append("✅ <strong>Jute:</strong> Moderate rain helps in retting process.")
        
        if "Vegetables" in main_crops:
            advisories.append("⚠️ <strong>Vegetables:</strong> Monitor for fungal diseases. Ensure proper drainage.")
    
    # Low rainfall scenarios
    else:
        advisories.append(f"☀️ <strong>Low Rainfall Expected for {season} Season</strong>")
        
        if probability <= 20:
            if "Rice" in main_crops:
                advisories.append("💧 <strong>Rice:</strong> Irrigation required. Monitor soil moisture levels.")
            
            if "Wheat" in main_crops:
                advisories.append("💧 <strong>Wheat:</strong> Critical irrigation needed during flowering stage.")
            
            if "Jute" in main_crops:
                advisories.append("💧 <strong>Jute:</strong> Additional irrigation needed for proper growth.")
        
        else:
            advisories.append("✅ <strong>Most crops:</strong> Favorable conditions. Continue regular farming activities.")
    
    # Temperature and humidity specific advisories
    if temperature > 35 and humidity > 80:
        advisories.append("🔥 <strong>Heat & Humidity:</strong> High risk of pest attacks. Monitor for rice hispa, stem borers.")
    
    if temperature < 15 and "Wheat" in main_crops:
        advisories.append("❄️ <strong>Cold Weather:</strong> Wheat crops may need protection from cold injury.")
    
    # Add general farming advice
    advisories.append(f"🌱 <strong>Current Main Crops:</strong> {', '.join(main_crops)}")
    
    return advisories

def get_rainfall_message(probability, temperature, humidity, month):
    """Generate meaningful message based on rainfall probability with crop advisories"""
    
    # General public message
    if probability >= 80:
        general_msg = "High chance of rainfall today. Strongly recommend carrying an umbrella."
    elif probability >= 60:
        general_msg = "Moderate to high chance of rainfall. Better to be prepared with rain protection."
    elif probability >= 40:
        general_msg = "Some possibility of rainfall. Keep an eye on the weather updates."
    elif probability >= 20:
        general_msg = "Low chance of rainfall. Light showers possible."
    else:
        general_msg = "Very low chance of rainfall. Enjoy the dry weather!"
    
    # Get crop advisories
    crop_advisories = get_crop_advisory(probability, temperature, humidity, month)
    
    return general_msg, crop_advisories

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        station = request.form['station']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        sunshine = float(request.form['sunshine'])
        humidity = float(request.form['humidity'])
        temperature = float(request.form['temperature'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        
        # Calculate features
        day_of_year, week_of_year = calculate_temporal_features(year, month, day)
        month_sin, month_cos, day_sin, day_cos = calculate_cyclic_features(month, day)
        
        # Get historical data for lag features
        historical_data = get_historical_weather(latitude, longitude)
        
        # Prepare feature dictionary with all required features
        features = {
            'Station': station,
            'Year': year,
            'Month': month,
            'Day': day,
            'Sunshine': sunshine,
            'Humidity': humidity,
            'Temperature': temperature,
            'DayOfYear': day_of_year,
            'WeekOfYear': week_of_year,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'day_sin': day_sin,
            'day_cos': day_cos,
        }
        
        # Add lag features (use 0 if historical data not available)
        for lag in [1, 2, 3]:
            features[f'Rainfall_lag{lag}'] = historical_data.get(f'rainfall_lag{lag}', 0)
            features[f'Temperature_lag{lag}'] = historical_data.get(f'temp_lag{lag}', temperature)
            features[f'Humidity_lag{lag}'] = historical_data.get(f'humidity_lag{lag}', humidity)
        
        # Calculate rolling means (simplified - using available lag data)
        rainfall_lags = [features[f'Rainfall_lag{i}'] for i in [1, 2, 3]]
        temp_lags = [features[f'Temperature_lag{i}'] for i in [1, 2, 3]]
        humidity_lags = [features[f'Humidity_lag{i}'] for i in [1, 2, 3]]
        
        features['Rainfall_rollmean_3'] = np.mean(rainfall_lags[:3]) if len(rainfall_lags) >= 3 else 0
        features['Rainfall_rollmean_7'] = np.mean(rainfall_lags) if len(rainfall_lags) >= 3 else 0
        features['Temperature_rollmean_3'] = np.mean(temp_lags[:3]) if len(temp_lags) >= 3 else temperature
        features['Temperature_rollmean_7'] = np.mean(temp_lags) if len(temp_lags) >= 3 else temperature
        features['Humidity_rollmean_3'] = np.mean(humidity_lags[:3]) if len(humidity_lags) >= 3 else humidity
        features['Humidity_rollmean_7'] = np.mean(humidity_lags) if len(humidity_lags) >= 3 else humidity
        
        # Convert to DataFrame for model prediction
        feature_df = pd.DataFrame([features])
        
        # Make prediction
        probability = model.predict_proba(feature_df)[0][1]  # Probability of rain
        rainfall_percentage = round(probability * 100, 2)
        
        # Get meaningful message and crop advisories
        general_msg, crop_advisories = get_rainfall_message(rainfall_percentage, temperature, humidity, month)
        
        return jsonify({
            'success': True,
            'rainfall_probability': rainfall_percentage,
            'message': general_msg,
            'crop_advisories': crop_advisories,
            'station': station,
            'temperature': temperature,
            'humidity': humidity
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':

    app.run(debug=True)
