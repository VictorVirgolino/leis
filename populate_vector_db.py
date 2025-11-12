import os
import psycopg2
import argparse
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from paperless_client import PaperlessClient
from vector_db import get_db_connection

load_dotenv()

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# Dimensão do vetor de embedding (768 para nomic-embed-text)
EMBEDDING_DIM = 768

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """Você é um especialista em análise e resumo de documentos. Sua tarefa é criar um resumo denso e detalhado de um documento, com até 2000 palavras.

INSTRUÇÕES:
1.  **Leia o conteúdo completo** do documento fornecido.
2.  **Extraia as informações mais importantes**:
    -   **Propósito Principal**: Qual o objetivo do documento? (e.g., instituir um código, regulamentar uma lei, etc.)
    -   **Entidades e Pessoas**: Quais órgãos, cargos, ou pessoas são mencionados? (e.g., Controladoria-Geral, Prefeito, Servidores).
    -   **Definições e Conceitos**: Quais termos são definidos? (e.g., o que é considerado "conduta ética", "vedações").
    -   **Regras e Procedimentos**: Liste os principais deveres, vedações, etapas, prazos e penalidades.
    -   **Datas e Números de Leis/Decretos**: Inclua todos os números de leis, decretos e datas relevantes.
3.  **Estruture o resumo** de forma clara, usando parágrafos para separar os tópicos.
4.  **Seja detalhado**: Não omita informações. O objetivo é criar um texto rico que possa ser usado para busca semântica.

CONTEÚDO DO DOCUMENTO:
---
{document_content}
---

RESUMO DETALHADO (ATÉ 2000 PALAVRAS):
"""
)

# ============================================================================
# FUNÇÕES DO BANCO DE DADOS
# ============================================================================

def setup_database(conn, reset=False):
    """Cria a extensão pgvector e a tabela de documentos, se não existirem."""
    with conn.cursor() as cur:
        if reset:
            print("  -> ⚠️  Opção --reset ativada. Apagando tabela 'documents' existente...")
            cur.execute("DROP TABLE IF EXISTS documents;")

        print("  -> Ativando extensão 'vector'...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        print("  -> Criando tabela 'documents'...")
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            paperless_id INTEGER UNIQUE NOT NULL,
            title TEXT,
            original_file_name TEXT,
            preview_link TEXT,
            download_link TEXT,
            summary TEXT,
            embedding VECTOR({EMBEDDING_DIM}),
            tags TEXT[]
        );
        """)
        conn.commit()
        print("✅ Setup do banco de dados concluído.")

def get_processed_ids(conn):
    """Obtém os IDs dos documentos do Paperless que já foram processados."""
    with conn.cursor() as cur:
        cur.execute("SELECT paperless_id FROM documents;")
        return {row[0] for row in cur.fetchall()}

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Popula o banco de dados vetorial com documentos do Paperless-NGX.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Apaga a tabela de documentos existente e recomeça o processo do zero."
    )
    args = parser.parse_args()

    print("🚀 Iniciando script para popular o banco de dados vetorial...")
    
    # 1. Inicializar clientes e modelos
    conn = get_db_connection()
    if not conn:
        return

    paperless_client = PaperlessClient()
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY não encontrada no arquivo .env. Verifique o seu arquivo .env.")

    # O modelo foi alterado para 1.5-flash para corresponder ao erro, mas a lógica é a mesma.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=gemini_api_key)
    
    ollama_host = os.getenv("OLLAMA_HOST", "localhost")
    embeddings_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=f"http://{ollama_host}:11434"
    )
    summarizer = SUMMARY_PROMPT | llm | StrOutputParser()

    # 2. Preparar o banco de dados
    setup_database(conn, reset=args.reset)

    # 3. Buscar documentos do Paperless
    print("\n🔄 Buscando todos os documentos do Paperless-NGX...")
    tag_map = paperless_client.get_all_tags()
    if tag_map:
        print(f"  -> {len(tag_map)} tags encontradas e mapeadas.")

    all_doc_ids = paperless_client.get_all_document_ids()
    processed_ids = get_processed_ids(conn)
    
    docs_to_process = [doc_id for doc_id in all_doc_ids if doc_id not in processed_ids]
    
    if not docs_to_process:
        print("✅ Nenhum documento novo para processar. O banco de dados está atualizado.")
        conn.close()
        return

    print(f"  -> {len(all_doc_ids)} documentos no total. {len(docs_to_process)} novos para processar.")

    # 4. Processar cada documento
    for i, doc_id in enumerate(docs_to_process, 1):
        print(f"\n--- Processando documento {i}/{len(docs_to_process)} (ID: {doc_id}) ---")
        
        try:
            # Obter metadados e conteúdo completo
            metadata = paperless_client.get_document_metadata(doc_id)
            if not metadata or not metadata.get("content"):
                print(f"  ⚠️ Conteúdo não encontrado para o documento ID {doc_id}. Pulando.")
                continue

            content = metadata["content"]
            title = metadata.get("title", "Sem Título")
            print(f"  📄 Título: {title}")

            # Obter nomes das tags
            tag_ids = metadata.get("tags", [])
            tag_names = [tag_map.get(tag_id, str(tag_id)) for tag_id in tag_ids]
            print(f"  🏷️ Tags: {tag_names if tag_names else 'Nenhuma'}")

            # Gerar resumo
            print("  -> Gerando resumo com IA...")
            summary = summarizer.invoke({"document_content": content})
            print(f"  -> Resumo gerado com {len(summary):,} caracteres.")

            # Gerar embedding
            print("  -> Gerando embedding para o resumo...")
            embedding = embeddings_model.embed_query(summary)
            print(f"  -> Embedding gerado (dimensão: {len(embedding)}).")

            # Preparar dados para inserção
            data_to_insert = (
                doc_id,
                title,
                metadata.get("original_file_name"),
                f"{paperless_client.api_url}/documents/{doc_id}/preview/",
                f"{paperless_client.api_url}/documents/{doc_id}/download/",
                summary,
                embedding,
                tag_names
            )

            # Inserir ou atualizar no banco de dados
            print("  -> Salvando no banco de dados...")
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (paperless_id, title, original_file_name, preview_link, download_link, summary, embedding, tags)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (paperless_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        original_file_name = EXCLUDED.original_file_name,
                        summary = EXCLUDED.summary,
                        embedding = EXCLUDED.embedding,
                        tags = EXCLUDED.tags;
                """, data_to_insert)
            conn.commit()
            print(f"  ✅ Documento ID {doc_id} salvo com sucesso!")

        except Exception as e:
            print(f"  ❌ Erro ao processar documento ID {doc_id}: {e}")
            conn.rollback() # Desfaz a transação em caso de erro

    print("\n🎉 Processo de população concluído!")
    conn.close()

if __name__ == "__main__":
    main()