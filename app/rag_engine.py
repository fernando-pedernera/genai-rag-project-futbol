import os
import json
import requests
import threading
from datetime import datetime, date
from collections import OrderedDict
from dotenv import load_dotenv
from typing import List, Dict, Optional
from pydantic import BaseModel
from app.embeddings import EmbeddingGenerator

# Cargar variables de entorno
load_dotenv()

class RAGConfig(BaseModel):
    """Configuración del motor RAG"""
    index_path: Optional[str] = None
    max_results: int = 3
    cache_size: int = 100
    llm_model: str = "google/gemma-3n-e4b-it"
    llm_temperature: float = 0.2
    llm_timeout: int = 10

class RAGEngine:
    def __init__(self, config: Optional[RAGConfig] = None, embedding_device='cpu'):
        self.config = config or RAGConfig()
        self.embedding_generator =  EmbeddingGenerator(device=embedding_device)
        self.index_path = self.config.index_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "vector_store",
            "faiss_index"
        )
        self.index_metadata_path = os.path.join(os.path.dirname(self.index_path), "metadata.json")
        self.vector_store = None
        self.last_update_date = None
        self._query_cache: OrderedDict[str, dict] = OrderedDict()
        self._initialize_async()

    def _initialize_async(self):
        """Inicialización no bloqueante en segundo plano"""
        init_thread = threading.Thread(target=self._load_or_regenerate_index, daemon=True)
        init_thread.start()

    def _load_or_regenerate_index(self):
        """Carga o regenera el índice según necesidad"""
        try:
            if os.path.exists(self.index_path) and self._is_index_current():
                self._load_index()
            else:
                self._regenerate_index()
        except Exception as e:
            print(f"[ERROR] Error inicializando índice: {e}")

    def _is_index_current(self) -> bool:
        """Verifica si el índice es del día actual"""
        try:
            if not os.path.exists(self.index_metadata_path):
                return False
                
            with open(self.index_metadata_path, 'r') as f:
                metadata = json.load(f)
                self.last_update_date = datetime.strptime(metadata['last_update'], '%Y-%m-%d').date()
                return self.last_update_date == date.today()
        except Exception:
            return False

    def _load_index(self):
        """Carga optimizada del índice FAISS"""
        from time import time
        start = time()
        
        try:
            self.vector_store = self.embedding_generator.load_saved_index(self.index_path)
            print(f"[PERF] Índice cargado en {time()-start:.2f}s | Documentos: {self.vector_store.index.ntotal}")
        except Exception as e:
            print(f"[ERROR] Error cargando índice: {e}")
            self.vector_store = None

    def _regenerate_index(self):
        """Regeneración optimizada del índice"""
        from time import time
        start = time()
        
        try:
            self.vector_store = self.embedding_generator.generate_embeddings_from_api(self.index_path)
            
            metadata = {
                'last_update': date.today().isoformat(),
                'source': 'api-football',
                'documents': self.vector_store.index.ntotal if self.vector_store else 0
            }
            with open(self.index_metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            self.last_update_date = date.today()
            print(f"[PERF] Índice regenerado en {time()-start:.2f}s | Documentos: {metadata['documents']}")
        except Exception as e:
            print(f"[ERROR] Error regenerando índice: {e}")
            raise

    def _add_to_cache(self, query: str, result: dict):
        """Manejo optimizado de caché LRU"""
        if len(self._query_cache) >= self.config.cache_size:
            self._query_cache.popitem(last=False)
        self._query_cache[query.lower().strip()] = result

    def search_documents(self, query: str, k: Optional[int] = None) -> List[dict]:
        """Búsqueda semántica optimizada con caché"""
        if not self.vector_store:
            self._load_index()
            if not self.vector_store:
                return []

        k = k or self.config.max_results
        
        from time import time
        start = time()
        
        try:
            resultados = self.vector_store.similarity_search(query, k=k)
            print(f"[PERF] Búsqueda '{query[:20]}...' en {time()-start:.2f}s | Resultados: {len(resultados)}")
            return resultados
        except Exception as e:
            print(f"[ERROR] Búsqueda fallida: {e}")
            return []

    def generate_response(self, context: str, question: str) -> str:
        """Generación optimizada de respuestas con LLM"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY no configurada")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Fútbol RAG"
        }

# En generate_response(), ajustar el prompt:
        prompt = (
            "Eres un comentarista español de fútbol apasionado y carismático. Resume los partidos con emoción pero manteniendo la claridad:\n\n"
            f"CONTEXTO:\n{context}\n\n"
            "INSTRUCCIONES:\n"
            "1. Comienza con un saludo energético (ej: '¡Buena tarde, afición!')\n"
            "2. Agrupa los partidos por horarios para mejor fluidez\n"
            "3. Usa formato: '⚽ [EQUIPO_LOCAL] vs [EQUIPO_VISITANTE] - [LIGA] a las [HORA]'\n"
            "4. Incluye 1-2 adjetivos emocionales por partido importante (ej: 'clásico emocionante', 'duelo clave')\n"
            "5. Destaca 1-2 partidos estrella con una frase breve\n"
            "6. Termina con una despedida motivadora\n"
            "7. Mantén un tono alegre pero profesional\n"
            "8. Usa máximo 300 tokens en total\n"
            "\n"
            "EJEMPLO DE ESTILO:\n"
            "'¡Hola, amigos del fútbol! Hoy tenemos una jornada para no perderse...'\n"
            "'A las 17:00, el Atletico Grau recibe al Deportivo Garcilaso en un duelo peruano lleno de pasión'\n"
            "'Y no se pierdan el clásico brasileño entre Flamengo y Atlético-MG a las 19:00, ¡promete fuego!'\n"
        )

        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            "temperature": self.config.llm_temperature,
            "max_tokens": 500,
            "top_p": 0.9
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.llm_timeout
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.Timeout:
            return "Error: Tiempo de espera agotado al generar respuesta"
        except Exception as e:
            print(f"[ERROR] OpenRouter: {e}")
            return f"Error al generar respuesta: {str(e)}"

    def _clean_response(self, text: str) -> str:
        """Limpieza optimizada de la respuesta"""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        return "\n".join(unique_lines)

    def query(self, question: str, use_cache: bool = True) -> dict:
        """Pipeline completo optimizado"""
        # Verificar caché primero
        cache_key = question.lower().strip()
        if use_cache and cache_key in self._query_cache:
            return self._query_cache[cache_key]

        try:
            # Búsqueda semántica
            docs = self.search_documents(question)
            
            if not docs:
                result = {
                    "question": question,
                    "answer": "No encontré información relevante.",
                    "docs_used": [],
                    "cache_hit": False
                }
                self._add_to_cache(cache_key, result)
                return result

            # Generar contexto
            context = "\n".join(doc.page_content for doc in docs[:5])  # Limitar a 3 docs
            
            # Generar respuesta
            respuesta = self.generate_response(context, question)
            
            # Procesar resultado
            result = {
                "question": question,
                "answer": self._clean_response(respuesta),
                "docs_used": [{"content": doc.page_content[:200]+"..." if len(doc.page_content) > 200 else doc.page_content,
                              "metadata": doc.metadata} for doc in docs[:3]],
                "cache_hit": False
            }

            # Almacenar en caché
            if use_cache:
                self._add_to_cache(cache_key, result)
                result["cache_hit"] = True

            return result

        except Exception as e:
            print(f"[ERROR] Pipeline RAG: {e}")
            return {
                "question": question,
                "answer": f"Error procesando la pregunta: {str(e)}",
                "docs_used": [],
                "cache_hit": False
            }

# Ejemplo de uso
if __name__ == "__main__":
    # Configuración personalizada
    config = RAGConfig(
        max_results=5,
        cache_size=200,
        llm_model="anthropic/claude-3-haiku"  # Modelo más rápido
    )
    
    engine = RAGEngine(config)
    
    # Prueba interactiva
    while True:
        pregunta = input("\nIngrese pregunta (o 'salir'): ").strip()
        if pregunta.lower() == 'salir':
            break
            
        resultado = engine.query(pregunta)
        
        print("\n🔍 Resultado:")
        print(f"Pregunta: {resultado['question']}")
        print(f"\nRespuesta:\n{resultado['answer']}")
        
        if resultado['docs_used']:
            print("\n📄 Documentos usados:")
            for i, doc in enumerate(resultado['docs_used'], 1):
                print(f"{i}. {doc['content']}")
                print(f"   Metadata: {doc['metadata']}\n")
        
        print(f"\nℹ️ Cache: {'HIT' if resultado.get('cache_hit') else 'MISS'}")