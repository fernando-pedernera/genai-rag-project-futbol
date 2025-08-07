# ⚽ Consulta de Partidos de Fútbol con GenAI + RAG

Aplicación conversacional para consultar partidos de fútbol utilizando Inteligencia Artificial Generativa (GenAI) y Retrieval-Augmented Generation (RAG).

## 🚀 Características principales

- Consulta en lenguaje natural sobre partidos de fútbol
- Soporte para 20+ ligas y competiciones internacionales
- Respuestas precisas con contexto actualizado
- Interfaz intuitiva desarrollada con Streamlit
- Arquitectura en contenedores Docker

## 🧠 Tecnologías utilizadas

| Categoría | Tecnologías |
| --- | --- |
| Backend | Python, FastAPI |
| IA  | LangChain, OpenAI GPT, RAG |
| Web Scraping | BeautifulSoup, Requests |
| Frontend | Streamlit |
| Infraestructura | Docker, Docker Compose |
| CI/CD | GitHub Actions |

## 🏗️ Estructura del proyecto

GENAI-RAG-PROJECT-FUTBOL/
├── devcontainer/
│ └── ... # Configuración para VSCode
├── app/
│ ├── init.py
│ ├── embeddings.py # Gestión de embeddings vectoriales
│ ├── ingestion.py # Pipeline de ingesta de datos
│ ├── main.py # Lógica principal
│ ├── rag_engine.py # Motor RAG personalizado
│ └── app.streamlit.py # Interfaz de usuario
├── .gitignore
├── docker-compose.yml # Orquestación de servicios
├── Dockerfile # Configuración del contenedor
├── requirements.txt # Dependencias Python
└── README.md # Documentación

## 🧪 ¿Cómo probar la aplicación?

### Ligas y competiciones soportadas

#### 🏆 Competiciones Internacionales

- **CONMEBOL**: Copa Libertadores, Copa Sudamericana
- **UEFA**: Champions League, UEFA Euro
- **FIFA**: World Cup

#### 🌍 Ligas Nacionales

| País | Competiciones |
| --- | --- |
| Argentina | Liga Profesional Argentina, Copa Argentina |
| Brasil | Brasileirão, Copa do Brasil |
| Chile | Primera División |
| Colombia | Primera A |
| España | La Liga |
| Francia | Ligue 1 |
| Alemania | Bundesliga |
| Italia | Serie A |
| México | Liga MX |
| Paraguay | División Profesional - Apertura |
| Perú | Primera División |
| Portugal | Primeira Liga |
| Uruguay | Primera División - Clausura |
| Inglaterra | Premier League |
| USA | MLS |

### Pasos para probar:

1. **Verificar partidos del día**  
  Consulta [Promiedos](https://www.promiedos.com.ar/) para equipos con partidos programados
  
2. **Realizar consultas**:
  
  ```plaintext
  ¿Qué equipos juegan hoy en la Premier League?
  ¿A qué hora juega Boca Juniors?
  ¿En qué torneo participa el Manchester City esta semana?
  ¿Juega hoy algún equipo de la Liga MX?
  ```
  

## 🚀 Instalación

**Opción 1: Docker (recomendado)**

```bash
docker-compose up --build
```

Accede a: http://localhost:8501

**Opción 2: Instalación local**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app/app.streamlit.py
```

## 🌐 [Demo en Streamlit](https://cgenai-rag-project-futbol-consulta-partidos-de-futbol.streamlit.app/) 

👉 Demo en Streamlit Cloud

## 🛠️ Roadmap

- MVP Funcional
- Integración con más fuentes de datos
- Sistema de caché para consultas frecuentes
- Soporte para múltiples idiomas

## 📄 Licencia

MIT License - Ver LICENSE para detalles.

Fernando Pedernera  
Data Engineer | Especialista en IA  
🔗 [LinkedIn](https://www.linkedin.com/in/fgpedernera/) 
📍 Córdoba, Argentina | 🚀 Proyecto personal en constante evolución



