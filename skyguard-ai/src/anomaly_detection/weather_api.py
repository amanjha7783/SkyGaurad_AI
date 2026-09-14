import os
import requests
from dotenv import load_dotenv

load_dotenv()

class WeatherAPIClient:
    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
    def get_current_weather(self, lat, lon):
        if not self.api_key:
            raise ValueError("WEATHER_API_KEY is missing from environment variables.")
            
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric" # Celsius, hPa, %
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "temperature_c": data["main"]["temp"],
                "pressure_hpa": data["main"]["pressure"],
                "relative_humidity_pct": data["main"]["humidity"],
                "timestamp": data["dt"]
            }
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch weather data: {e}")
