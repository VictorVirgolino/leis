import os
import psycopg2
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()

def get_db_connection():
    """Estabelece e retorna uma conexão com o banco de dados PostgreSQL."""
    try:
        conn = psycopg2.connect(
            # host=os.getenv("DB_HOST", "localhost"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ Erro ao conectar ao PostgreSQL: {e}")
        return None

def semantic_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Realiza uma busca semântica no banco de dados vetorial.
    Gera o embedding para a query e busca os documentos mais similares.
    """
    print(f"🔍 Realizando busca semântica para: '{query}'")
    
    conn = get_db_connection()
    if not conn:
        return []

    try:
        # 1. Gerar embedding para a query
        ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        embeddings_model = OllamaEmbeddings(model="nomic-embed-text", base_url=f"http://{ollama_host}:11434")
        query_embedding = embeddings_model.embed_query(query)
        print("  -> Embedding da query gerado.")

        # 2. Realizar a busca por similaridade de cosseno
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT paperless_id, title, preview_link, download_link, tags, 1 - (embedding <=> %s) AS similarity
                FROM documents
                ORDER BY similarity DESC
                LIMIT %s;
                """,
                (str(query_embedding), limit)
            )
            results = [dict(zip([desc[0] for desc in cur.description], row)) for row in cur.fetchall()]
            print(f"  -> {len(results)} documentos encontrados no banco de dados vetorial.")
            return results
    finally:
        if conn:
            conn.close()