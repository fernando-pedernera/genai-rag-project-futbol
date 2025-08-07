import requests
import os
from datetime import datetime
import pytz
from langchain.schema import Document
from dotenv import load_dotenv

# Carga variables de entorno
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"x-apisports-key": API_KEY}

# Lista blanca de ligas relevantes
# ⚙️ Lista blanca de ligas relevantes (como tuplas)
ligas_relevantes = {
    ("Liga Profesional Argentina", "Argentina"),
    ("Copa Libertadores", None),
    ("Copa Sudamericana", None),
    ("Champions League", None),
    ("Premier League", "England"),
    ("La Liga", "Spain"),
    ("Serie A", "Italy"),
    ("Bundesliga", "Germany"),
    ("Ligue 1", "France"),
    ("MLS", "USA"),
    ("UEFA Euro", None),
    ("World Cup", None),
    ("Copa Argentina", "Argentina"),
    ("Brasileirão", "Brazil"),
    ("Copa Do Brasil", "Brazil"),
    ("Primera División", "Chile"),
    ("Primera A", "Colombia"),
    ("Liga MX", "Mexico"),
    ("Division Profesional - Apertura", "Paraguay"),
    ("Primera División", "Peru"),
    ("Primeira Liga", "Portugal"),
    ("Primera División - Clausura", "Uruguay"),
}

def obtener_partidos_argentina():
    """Obtiene partidos del día en Argentina filtrando solo ligas relevantes."""
    tz_arg = pytz.timezone('America/Argentina/Buenos_Aires')
    hoy = datetime.now(tz_arg).strftime("%Y-%m-%d")
    
    params = {
        "date": hoy,
        "timezone": "America/Argentina/Buenos_Aires"
    }
    
    try:
        response = requests.get(BASE_URL, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        partidos = data.get("response", [])
        
        # Filtra partidos con datos completos y ligas relevantes (por nombre y país)
        filtrados = [
            p for p in partidos
            if all(key in p for key in ["fixture", "teams", "league"])
            and (p["league"]["name"], p["league"].get("country")) in ligas_relevantes
        ]
        return filtrados

    except Exception as e:
        print(f"⚠️ Error al consultar API: {e}")
        return []

def formatear_partido(partido):
    """Formatea un partido: Equipos, Liga y Hora (ARG)."""
    fixture = partido["fixture"]
    liga = partido["league"]["name"]
    pais = partido["league"].get("country")
    hora_arg = datetime.strptime(
        fixture["date"], 
        "%Y-%m-%dT%H:%M:%S%z"
    ).astimezone(pytz.timezone('America/Argentina/Buenos_Aires')).strftime("%H:%M")
    
    liga_completa = f"{liga} ({pais})" if pais else liga
    return f"⚽ {partido['teams']['home']['name']} vs {partido['teams']['away']['name']} | {liga_completa} | {hora_arg} (ARG)"


def generar_documento():
    """Crea un Document de LangChain con partidos relevantes del día."""
    partidos = obtener_partidos_argentina()
    if not partidos:
        return Document(page_content="No hay partidos relevantes programados hoy.")
    
    contenido = "\n".join([formatear_partido(p) for p in partidos])
    return Document(page_content=contenido)

def cargar_chunks_eventos_deportivos():
    """Devuelve una lista con un documento LangChain."""
    return [generar_documento()]

