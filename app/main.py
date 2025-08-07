from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from app.rag_engine import RAGEngine
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa la app
app = FastAPI(
    title="Fútbol RAG API",
    description="Consulta sobre partidos de fútbol usando un motor RAG (Retrieval-Augmented Generation) con embeddings y LLM.",
    version="1.0.0"
)

# Instancia global del motor RAG
rag_engine = RAGEngine()

# Modelo para la entrada del usuario
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask", tags=["Consultas"])
def ask_question(payload: QuestionRequest):
    try:
        result = rag_engine.query(payload.question)

        # Eliminar duplicados manteniendo el orden
        seen = set()
        unique_docs = []
        for doc in result["docs_used"]:
            content = doc["content"]  # Ahora accedemos como diccionario
            if content not in seen:
                seen.add(content)
                unique_docs.append(content)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "docs_used": unique_docs
        }
    except Exception as e:
        logger.error(f"[ERROR] Fallo al procesar pregunta: {e}")
        raise HTTPException(status_code=500, detail="Error al procesar la pregunta.")

@app.post("/reload", tags=["Admin"])
def reload_index():
    """
    Recarga el índice FAISS desde disco.
    Útil luego de actualizar los embeddings.
    """
    try:
        rag_engine.load_index()
        return {"message": "Índice FAISS recargado correctamente."}
    except Exception as e:
        logger.error(f"[ERROR] Error recargando índice: {e}")
        raise HTTPException(status_code=500, detail="Error al recargar el índice.")

@app.get("/", response_class=HTMLResponse, tags=["Interfaz Web"])
def home():
    """
    Interfaz HTML simple para enviar preguntas desde el navegador.
    """
    return """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Fútbol RAG API - Consulta</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #e9f5e9;
            color: #1e1e1e;
            max-width: 900px;
            margin: 40px auto;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 0 15px rgba(0,0,0,0.15);
            background-image: linear-gradient(135deg, #f3fff3, #d2f2d2);
        }
        h1 {
            text-align: center;
            color: #0d4f2f;
            font-size: 2.2em;
            margin-bottom: 10px;
        }
        h2 {
            margin-top: 40px;
            color: #0d4f2f;
            border-bottom: 2px solid #c4d6b0;
            padding-bottom: 5px;
        }
        .liga {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
        }
        .liga img {
            height: 24px;
            width: 24px;
        }
        form {
            margin-top: 20px;
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }
        label {
            font-weight: 600;
            font-size: 1.2em;
            min-width: 90px;
            color: #0d4f2f;
        }
        input[type="text"] {
            flex: 1;
            font-size: 1.1em;
            padding: 10px 14px;
            border: 2px solid #ccc;
            border-radius: 8px;
            transition: border-color 0.3s ease;
        }
        input[type="text"]:focus {
            border-color: #198754;
            outline: none;
        }
        button {
            background-color: #198754;
            border: none;
            color: white;
            padding: 10px 18px;
            font-size: 1.1em;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        button:hover {
            background-color: #146c43;
        }
        pre#answer {
            background: white;
            border: 1px solid #ddd;
            padding: 18px;
            border-radius: 10px;
            font-size: 1.05em;
            line-height: 1.5;
            white-space: pre-wrap;
            min-height: 150px;
            box-shadow: inset 0 0 6px #e0e0e0;
            color: #212529;
        }
        footer {
            margin-top: 40px;
            font-size: 0.9em;
            text-align: center;
            color: #555;
        }
    </style>
</head>
<body>
    <h1>⚽ Consulta partidos de fútbol de hoy</h1>

    <form id="queryForm" autocomplete="off">
        <label for="question">Pregunta:</label>
        <input type="text" id="question" name="question" placeholder="¿Qué partidos hay esta tarde?" required />
        <button type="submit">Enviar</button>
    </form>

    <h2>Respuesta:</h2>
    <pre id="answer">Aquí aparecerá la respuesta...</pre>

        <<div style="display: flex; flex-direction: column; gap: 20px; margin-top: 20px;">
        <!-- Primera sección -->
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
            <!-- Liga Profesional Argentina -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/argentina2.jpg" alt="Liga Profesional Argentina" width="32" height="32">
                <span>Liga Profesional Argentina</span>
            </div>

            <!-- Copa Libertadores -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/confederaciones/confederaciones/img/conmebol.jpg" alt="Copa Libertadores" width="32" height="32">
                <span>Copa Libertadores</span>
            </div>

            <!-- Copa Sudamericana -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://logosenvector.com/logo/img/copa-sudamericana-conmebol-37620.png" alt="Copa Sudamericana" width="32" height="32">
                <span>Copa Sudamericana</span>
            </div>

            <!-- Champions League -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://logosenvector.com/logo/img/uefa-champions-league-613.jpg" alt="Champions League" width="32" height="32">
                <span>Champions League</span>
            </div>

            <!-- Premier League -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://logosenvector.com/logo/img/premier-league-37294.png" alt="Premier League" width="32" height="32">
                <span>Premier League</span>
            </div>
        </div>

        <!-- Segunda sección -->
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
            <!-- La Liga -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/espana2.jpg" alt="La Liga" width="32" height="32">
                <span>La Liga</span>
            </div>

            <!-- Serie A -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/italia.jpg" alt="Serie A" width="32" height="32">
                <span>Serie A</span>
            </div>

            <!-- Bundesliga -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/alemania.jpg" alt="Bundesliga" width="32" height="32">
                <span>Bundesliga</span>
            </div>

            <!-- Ligue 1 -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/francia.jpg" alt="Ligue 1" width="32" height="32">
                <span>Ligue 1</span>
            </div>

            <!-- MLS -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/estadosunidos.jpg" alt="MLS" width="32" height="32">
                <span>MLS</span>
            </div>

            <!-- UEFA Euro -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/confederaciones/confederaciones/img/uefa.jpg" alt="UEFA Euro" width="32" height="32">
                <span>UEFA Euro</span>
            </div>

            <!-- World Cup -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://pbs.twimg.com/media/FwaCi8rWYAA9Vfy?format=jpg&name=small" alt="World Cup" width="32" height="32">
                <span>World Cup</span>
            </div>

            <!-- Copa Argentina -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://seeklogo.com/images/C/copa-argentina-nuevo-logo-AF2D2FF2DA-seeklogo.com.png" alt="Copa Argentina" width="32" height="32">
                <span>Copa Argentina</span>
            </div>
        </div>

        <!-- Tercera sección (Brasil y otros) -->
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
            <!-- Brasileirão -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/brasil.jpg" alt="Brasileirão" width="32" height="32">
                <span>Brasileirão</span>
            </div>

            <!-- Copa do Brasil -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://tse1.mm.bing.net/th/id/OIP.ULySJJ-MJIulqhqxPfGurQAAAA?r=0&rs=1&pid=ImgDetMain&o=7&rm=3" alt="Copa do Brasil" width="32" height="32">
                <span>Copa do Brasil</span>
            </div>

            <!-- Primera División (Chile) -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/chile.jpg" alt="Primera División Chile" width="32" height="32">
                <span>Primera División (Chile)</span>
            </div>

            <!-- Primera A (Colombia) -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/colombia.jpg" alt="Primera A Colombia" width="32" height="32">
                <span>Primera A (Colombia)</span>
            </div>

            <!-- Liga MX -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/mexico.jpg" alt="Liga MX" width="32" height="32">
                <span>Liga MX</span>
            </div>
        </div>

        <!-- Cuarta sección -->
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
            <!-- División Profesional (Paraguay) -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/paraguay.jpg" alt="División Profesional Paraguay" width="32" height="32">
                <span>División Profesional (Paraguay)</span>
            </div>

            <!-- Primera División (Peru) -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/peru.jpg" alt="Primera División Peru" width="32" height="32">
                <span>Primera División (Peru)</span>
            </div>

            <!-- Primeira Liga (Portugal) -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/portugal.jpg" alt="Primeira Liga Portugal" width="32" height="32">
                <span>Primeira Liga (Portugal)</span>
            </div>

            <!-- Primera División (Uruguay) -->
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://paladarnegro.net/escudoteca/ligas/ligas/img/uruguay.jpg" alt="Primera División Uruguay" width="32" height="32">
                <span>Primera División (Uruguay)</span>
            </div>
        </div>
    </div>

    <footer style="margin-top: 40px; font-size: 0.9em; text-align: center; color: #555;">
        Datos actualizados automáticamente desde <strong>api-football</strong>.
    </footer>

    <script>
        const form = document.getElementById('queryForm');
        const answer = document.getElementById('answer');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const question = document.getElementById('question').value.trim();
            if (!question) {
                answer.textContent = 'Por favor, escribe una pregunta.';
                return;
            }
            answer.textContent = 'Procesando...';

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                if (!response.ok) {
                    answer.textContent = 'Error en la consulta.';
                    return;
                }

                const data = await response.json();
                answer.textContent = data.answer || 'No se encontró respuesta.';
            } catch (err) {
                answer.textContent = 'Error al conectar con el servidor.';
            }
        });
    </script>
</body>
</html>
"""
