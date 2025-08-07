import streamlit as st
from app.rag_engine import RAGEngine

# Instancia del motor RAG
engine = RAGEngine(embedding_device='cpu')

# Configuración de página
st.set_page_config(page_title="Fútbol RAG - Consulta", layout="wide")

# Estilo personalizado
st.markdown("""<style>
    html, body {
        background-color: #0d1117;
        color: #39ff14;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2 {
        text-align: center;
        color: #00ffff;
    }
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: #39ff14;
        border: 1px solid #39ff14;
    }
    .stButton > button {
        background-color: #39ff14;
        color: #000;
        font-weight: bold;
        border-radius: 8px;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #00cc66;
        color: white;
    }
    .response-box {
        background-color: #111;
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #00ffcc;
        color: #ffffff;
        margin-bottom: 2rem;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .liga-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
        margin-top: 2rem;
        padding: 0 2rem;
        justify-items: center;
    }
    .liga {
        background-color: #1a1a1a;
        border: 1px solid #39ff14;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
        color: #39ff14;
        box-shadow: 0 0 10px #39ff1490;
        transition: transform 0.2s ease;
        text-align: center;
        max-width: 160px;
        min-height: 120px;
    }
    .liga:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px #39ff14;
    }
    .liga img {
        width: 38px;
        height: 38px;
        border-radius: 4px;
        object-fit: contain;
    }
    footer {
        margin-top: 3rem;
        text-align: center;
        font-size: 0.85rem;
        color: #888;
    }
</style>""", unsafe_allow_html=True)

# Título
st.markdown("<h1>Consulta partidos de fútbol de hoy</h1>", unsafe_allow_html=True)

# Entrada del usuario
pregunta = st.text_input("✍️ Escribí tu pregunta", placeholder="¿Qué partidos hay esta tarde?")

if st.button("Preguntar"):
    if not pregunta.strip():
        st.warning("⚠️ Por favor, escribí una pregunta válida.")
    else:
        try:
            with st.spinner("💬 Buscando respuesta..."):
                resultado = engine.query(pregunta)
                respuesta = resultado["answer"] if isinstance(resultado, dict) else str(resultado)
                st.markdown("### 🤖 Respuesta", unsafe_allow_html=True)
                st.markdown(f"<div class='response-box'>{respuesta}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error: Tiempo de espera agotado o falló el motor de respuesta.\n\n**Detalle:** `{e}`")

# Sección de ligas destacadas
st.markdown("## 🏆 Ligas destacadas", unsafe_allow_html=True)

ligas = [
    ("Liga Profesional Argentina", "https://paladarnegro.net/escudoteca/ligas/ligas/img/argentina2.jpg"),
    ("Copa Libertadores", "https://paladarnegro.net/escudoteca/confederaciones/confederaciones/img/conmebol.jpg"),
    ("Copa Sudamericana", "https://logosenvector.com/logo/img/copa-sudamericana-conmebol-37620.png"),
    ("Champions League", "https://logosenvector.com/logo/img/uefa-champions-league-613.jpg"),
    ("Premier League", "https://logosenvector.com/logo/img/premier-league-37294.png"),
    ("La Liga", "https://paladarnegro.net/escudoteca/ligas/ligas/img/espana2.jpg"),
    ("Serie A", "https://paladarnegro.net/escudoteca/ligas/ligas/img/italia.jpg"),
    ("Bundesliga", "https://paladarnegro.net/escudoteca/ligas/ligas/img/alemania.jpg"),
    ("Ligue 1", "https://paladarnegro.net/escudoteca/ligas/ligas/img/francia.jpg"),
    ("MLS", "https://paladarnegro.net/escudoteca/ligas/ligas/img/estadosunidos.jpg"),
    ("UEFA Euro", "https://paladarnegro.net/escudoteca/confederaciones/confederaciones/img/uefa.jpg"),
    ("World Cup", "https://pbs.twimg.com/media/FwaCi8rWYAA9Vfy?format=jpg&name=small"),
    ("Copa Argentina", "https://seeklogo.com/images/C/copa-argentina-nuevo-logo-AF2D2FF2DA-seeklogo.com.png"),
    ("Brasileirão", "https://paladarnegro.net/escudoteca/ligas/ligas/img/brasil.jpg"),
    ("Copa do Brasil", "https://tse1.mm.bing.net/th/id/OIP.ULySJJ-MJIulqhqxPfGurQAAAA?pid=ImgDetMain"),
    ("Primera División (Chile)", "https://paladarnegro.net/escudoteca/ligas/ligas/img/chile.jpg"),
    ("Primera A (Colombia)", "https://paladarnegro.net/escudoteca/ligas/ligas/img/colombia.jpg"),
    ("Liga MX", "https://paladarnegro.net/escudoteca/ligas/ligas/img/mexico.jpg"),
    ("División Profesional (Paraguay)", "https://paladarnegro.net/escudoteca/ligas/ligas/img/paraguay.jpg"),
    ("Primera División (Perú)", "https://paladarnegro.net/escudoteca/ligas/ligas/img/peru.jpg"),
    ("Primeira Liga (Portugal)", "https://paladarnegro.net/escudoteca/ligas/ligas/img/portugal.jpg"),
    ("Primera División (Uruguay)", "https://paladarnegro.net/escudoteca/ligas/ligas/img/uruguay.jpg"),
]

cols = st.columns(6)
for idx, (nombre, logo) in enumerate(ligas):
    with cols[idx % 6]:
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 0.5rem;">
            <img src="{logo}" width="40" height="40" style="border-radius: 5px;" />
            <span style="color: #39ff14; font-weight: bold; text-align: center; font-size: 0.85rem;">{nombre}</span>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<footer>⚡ Datos actualizados automáticamente desde <strong>api-football</strong>.</footer>", unsafe_allow_html=True)
