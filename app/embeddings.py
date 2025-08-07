import os
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
from app.ingestion import cargar_chunks_eventos_deportivos
  # Importa tu función existente

load_dotenv()

class EmbeddingGenerator:
    def __init__(self, embedding_type="huggingface", device="cpu"):
        """
        :param embedding_type: Actualmente solo soporta "huggingface"
        :param device: dispositivo para cargar el modelo (cpu o cuda)
        """
        self.embedding_type = embedding_type
        self.device = device
        self.embeddings = self._load_embedding_model()

    def _load_embedding_model(self):
        """Carga el modelo de embeddings de HuggingFace"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': self.device},  # aquí usamos self.device
            encode_kwargs={'normalize_embeddings': False}
        )

    def generate_embeddings_from_api(self, save_path="vector_store/faiss_index"):
        """Genera embeddings desde los datos de la API de fútbol"""
        try:
            # Obtiene los chunks usando tu función existente
            chunks = cargar_chunks_eventos_deportivos()
            
            # Convierte a documentos LangChain
            documents = [
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        "source": "api-football",
                        "date": str(datetime.now()),
                        "content_type": "football-match"
                    }
                ) for chunk in chunks
            ]
            
            # Genera y guarda los embeddings
            vector_store = FAISS.from_documents(documents, self.embeddings)
            os.makedirs(save_path, exist_ok=True)
            
            # Guarda con seguridad
            vector_store.save_local(
                folder_path=save_path,
                index_name="index"
            )
            return vector_store

        except Exception as e:
            print(f"❌ Error generando embeddings: {str(e)}")
            raise

    def load_saved_index(self, path="vector_store/faiss_index"):
        """Carga un índice FAISS existente de forma segura"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directorio no encontrado: {path}")
            
        if not all(os.path.exists(os.path.join(path, f)) for f in ["index.faiss", "index.pkl"]):
            raise FileNotFoundError(f"Archivos FAISS incompletos en {path}")
            
        return FAISS.load_local(
            folder_path=path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,  # Necesario pero seguro en este contexto
            index_name="index"
        )

    def get_embedding_model(self):
        """Devuelve el modelo de embeddings para uso directo"""
        return self.embeddings