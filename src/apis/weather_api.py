# src/apis/weather_api.py
from __future__ import annotations
import os
import argparse
from typing import Literal
import requests
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool

from dotenv import load_dotenv
load_dotenv()

WEATHER_ENDPOINT = "https://api.openweathermap.org/data/2.5/weather"
DEFAULT_TIMEOUT = 10


# ---------------------------
# Pydantic models
# ---------------------------
class WeatherInput(BaseModel):
    city: str = Field(description="Nombre de ciudad (puede incluir país: 'Madrid,ES').")
    units: Literal["metric", "imperial", "standard"] = Field(
        default="metric", description="Sistema de unidades."
    )


class WeatherSummary(BaseModel):
    city: str
    country: str
    condition: str
    temperature: float
    feels_like: float
    humidity: int
    wind_speed: float
    units: str


# ---------------------------
# Core function 
# ---------------------------
def fetch_current_weather(
    city: str,
    units: Literal["metric", "imperial", "standard"] = "metric",
) -> WeatherSummary:
    """
    Devuelve el tiempo actual para una ciudad usando OpenWeatherMap.
    Requiere la variable de entorno OPENWEATHER_KEY.
    """
    api_key = os.getenv("OPENWEATHER_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENWEATHER_KEY en variables de entorno.")

    params = {"q": city, "appid": api_key, "units": units}
    try:
        resp = requests.get(WEATHER_ENDPOINT, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Error llamando a OpenWeatherMap: {e}") from e

    if data.get("cod") != 200:
        raise RuntimeError(f"OpenWeatherMap devolvió error: {data}")

    weather = (data.get("weather") or [{}])[0]
    main = data.get("main") or {}
    wind = data.get("wind") or {}
    sys_info = data.get("sys") or {}

    return WeatherSummary(
        city=data.get("name") or city,
        country=sys_info.get("country") or "",
        condition=weather.get("description") or "",
        temperature=float(main.get("temp") or 0.0),
        feels_like=float(main.get("feels_like") or 0.0),
        humidity=int(main.get("humidity") or 0),
        wind_speed=float(wind.get("speed") or 0.0),
        units=units,
    )


# ---------------------------
# Tool para LlamaIndex
# ---------------------------
def current_weather_tool() -> FunctionTool:
    # Asegura que los modelos están listos (evita errores de forward refs)
    WeatherInput.model_rebuild()
    WeatherSummary.model_rebuild()

    return FunctionTool.from_defaults(
        fn=fetch_current_weather,
        name="current_weather",
        description=(
            "Obtiene el tiempo actual para una ciudad (temperatura, sensación térmica, "
            "humedad, viento y condición)."
        ),
        fn_schema=WeatherInput,  
        return_direct=False,
    )


# ---------------------------
# CLI rápido
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--units", default="metric", choices=["metric", "imperial", "standard"])
    args = parser.parse_args()

    res = fetch_current_weather(args.city, args.units)
    # Nota: OpenWeather usa m/s para 'metric' y mph para 'imperial'
    speed_unit = {"metric": "m/s", "imperial": "mph", "standard": "m/s"}[res.units]
    print(
        f"{res.city}, {res.country} | {res.condition} | temp {res.temperature} "
        f"(sens {res.feels_like}) | hum {res.humidity}% | viento {res.wind_speed} {speed_unit}"
    )
