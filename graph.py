import os
import json
from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from paperless_client import PaperlessClient
from vector_db import semantic_search

load_dotenv()


class GraphState(TypedDict):
    """Estado do grafo LangGraph para RAG com Paperless-NGX."""
    question: str
    search_query: str
    documents: List[dict]
    generation: str
    error: str


# Inicializa o cliente Paperless e LLM
paperless_client = PaperlessClient()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY não encontrada no arquivo .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    convert_system_message_to_human=True,
    temperature=0.3
)


# ============================================================================
# PROMPTS OTIMIZADOS
# ============================================================================

QUERY_TRANSFORM_PROMPT = ChatPromptTemplate.from_template(
    """Você é um especialista em otimização de buscas. Sua tarefa é extrair APENAS os 3-5 TERMOS MAIS IMPORTANTES de uma pergunta para busca em documentos.

REGRAS CRÍTICAS:
1. Extraia APENAS 3-5 termos essenciais (no máximo 5!)
2. Priorize: nomes próprios, conceitos-chave, termos técnicos
3. REMOVA: palavras comuns, artigos, preposições, verbos auxiliares
4. Se houver expressão técnica/nome próprio, mantenha completa (ex: "Tarifa Zero", "IPTU")
5. Prefira termos que REALMENTE aparecem em documentos formais

ESTRATÉGIA:
- Identifique o CONCEITO PRINCIPAL da pergunta
- Adicione 1-2 sinônimos OU termos de contexto direto
- PARE! Não adicione mais nada

EXEMPLOS:

Pergunta: "O que é a Tarifa Zero?"
Query: Tarifa Zero transporte gratuito
Explicação: "Tarifa Zero" é o termo principal, "transporte" e "gratuito" são contexto direto

Pergunta: "Quais são os procedimentos do regime jurídico das parcerias?"
Query: regime jurídico parcerias procedimentos
Explicação: 3 termos principais que definem o assunto

Pergunta: "Como funciona a cobrança de IPTU?"
Query: IPTU cobrança imposto predial
Explicação: IPTU é o termo principal, "cobrança" é a ação, "imposto predial" é contexto

Pergunta: "Quais os tributos de competência municipal?"
Query: tributos municipais competência
Explicação: Termos essenciais sem redundância

Pergunta: "O que diz a lei sobre licitações?"
Query: licitações lei processo licitatório
Explicação: Termos-chave do universo de licitações

AGORA É SUA VEZ:
Pergunta: "{question}"

Query otimizada (APENAS 3-5 termos essenciais, separados por espaço):"""
)


RAG_GENERATION_PROMPT = ChatPromptTemplate.from_template(
    """Você é um assistente jurídico especializado em legislação e documentos administrativos brasileiros.
Sua missão é fornecer respostas COMPLETAS, DETALHADAS e BEM FUNDAMENTADAS usando EXCLUSIVAMENTE os documentos fornecidos.

INSTRUÇÕES CRÍTICAS:

1. **LEIA TODO O CONTEÚDO DOS DOCUMENTOS** antes de responder
   - Não resuma demais - use todas as informações relevantes disponíveis
   - Extraia TODOS os detalhes pertinentes à pergunta

2. **ESTRUTURE A RESPOSTA DE FORMA COMPLETA:**
   - Comece com uma introdução clara do que foi encontrado
   - Desenvolva todos os pontos importantes com detalhes
   - Se houver artigos, incisos, parágrafos: CITE-OS NA ÍNTEGRA ou resuma seu conteúdo
   - Se houver listas, procedimentos, requisitos: ENUMERE TODOS
   - Se houver definições: TRANSCREVA ou paráfrase completamente

3. **CITAÇÕES OBRIGATÓRIAS:**
   - Cite a fonte após CADA informação importante: [Nome do Documento](link)
   - Use citações diretas quando o texto legal for relevante

4. **CITAÇÃO DA BASE TEXTUAL :**
   - Ao final de cada parágrafo ou seção que resume uma informação, inclua um bloco de citação com o trecho exato do documento que serviu de base.
   - Formate assim:
     > "Trecho literal do documento que comprova a informação..." [Nome do Documento](link)

5. **FORMATAÇÃO PARA MÁXIMA CLAREZA:**
   - Use **negrito** para termos-chave e títulos de seções
   - Use listas numeradas para procedimentos/etapas
   - Use bullet points para requisitos/características
   - Use blocos de citação (>) para transcrições literais importantes

6. **COMPLETUDE É ESSENCIAL:**
   - Se o documento tem 10 procedimentos, liste os 10
   - Se há requisitos, prazos, penalidades: inclua TODOS
   - Não diga apenas "o documento menciona X" - EXPLIQUE o que o documento diz sobre X

7. **SE NÃO HOUVER INFORMAÇÃO SUFICIENTE:**
   - Seja explícito: "Os documentos mencionam [X], mas não detalham [Y]"
   - Indique o que foi encontrado e o que está ausente

---
DOCUMENTOS PARA VISUALIZAÇÃO (PREVIEW):
{preview_links}
---

DOCUMENTOS DISPONÍVEIS (CONTEÚDO COMPLETO PARA SUA ANÁLISE):
{context}

---
PERGUNTA DO USUÁRIO:
{question}

RESPOSTA DETALHADA E COMPLETA (com todas as citações necessárias):"""
)


RELEVANCE_RANKING_PROMPT = ChatPromptTemplate.from_template(
    """Você é um especialista em análise de relevância. Sua tarefa é analisar uma lista de documentos e determinar quais são relevantes para responder a uma pergunta, e então rankeá-los.

PERGUNTA DO USUÁRIO:
{question}

DOCUMENTOS DISPONÍVEIS:
{context}

INSTRUÇÕES:
1. Leia a pergunta e o conteúdo de cada documento.
2. Determine quais documentos são **realmente úteis** para formular uma resposta completa.
3. Retorne uma lista Python com os **números** dos documentos relevantes, ordenados do **mais importante para o menos importante**.

REGRAS DE SAÍDA:
- Sua resposta deve ser APENAS a lista de números. Exemplo: `[3, 1]`
- Se o Documento 3 for o mais relevante e o Documento 1 for o segundo mais relevante, sua resposta será `[3, 1]`.
- Se nenhum documento for relevante, retorne uma lista vazia: `[]`

LISTA DE RELEVÂNCIA (APENAS NÚMEROS):"""
)


# ============================================================================
# NODOS DO GRAFO
# ============================================================================


def retrieve(state: GraphState) -> GraphState:
    """Recupera documentos relevantes do Paperless-NGX com estratégia multi-tentativa."""
    print("\n📚 RECUPERANDO DOCUMENTOS (BUSCA SEMÂNTICA)...")
    # Usa a pergunta original do usuário diretamente para a busca semântica
    search_query = state["question"]
    
    try:
        # 1. Realiza a busca semântica no banco de dados vetorial
        similar_docs = semantic_search(search_query, limit=7)

        if not similar_docs:
            print("   ❌ Nenhum documento encontrado na busca semântica.")
            return {"documents": []}

        # 2. Recupera o conteúdo completo de cada documento do Paperless
        print("\n   🔄 Buscando conteúdo completo dos documentos encontrados...")
        full_documents = []
        for doc_info in similar_docs:
            paperless_id = doc_info['paperless_id']
            full_doc = paperless_client.get_document_metadata(paperless_id)
            if full_doc:
                # Adiciona o score de similaridade e links do DB para o objeto completo
                full_doc['score'] = doc_info.get('similarity', 0)
                full_doc['link'] = doc_info.get('preview_link') # Garante o link correto
                full_documents.append(full_doc)
                print(f"      ✅ Conteúdo completo para '{full_doc['title']}' (ID: {paperless_id}) obtido.")
            else:
                print(f"      ⚠️ Falha ao obter conteúdo completo para o ID: {paperless_id}")

        return {"documents": full_documents}
        
    except Exception as e:
        print(f"   ❌ Erro na recuperação: {e}")
        return {"documents": [], "error": str(e)}


def check_relevance(state: GraphState) -> GraphState:
    """Verifica se os documentos encontrados são relevantes para a pergunta."""
    print("\n🎯 VERIFICANDO RELEVÂNCIA...")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return state
    
    # Prepara preview dos documentos
    context_preview = ""
    for i, doc in enumerate(documents, 1):  # Analisa até 5 documentos
        content = doc.get("highlights") or doc.get("content", "")
        if content:
            preview = content[:2000] if len(content) > 2000 else content
            context_preview += f"--- DOCUMENTO {i} ---\nTítulo: {doc['title']}\nConteúdo: {preview}...\n\n"
    
    try:
        relevance_ranker = RELEVANCE_RANKING_PROMPT | llm | StrOutputParser()
        ranking_str = relevance_ranker.invoke({
            "context": context_preview,
            "question": question
        }).strip()
        
        print(f"   📊 Análise de relevância retornou: {ranking_str}")
        
        # Tenta avaliar a string como uma lista Python
        try:
            ranked_indices = json.loads(ranking_str)
            if not isinstance(ranked_indices, list):
                raise ValueError("Não é uma lista")
        except (json.JSONDecodeError, ValueError):
            print("   ⚠️ Não foi possível decodificar o ranking, mantendo a ordem original.")
            return state

        if not ranked_indices:
            print("   🗑️ Nenhum documento considerado relevante. Descartando todos.")
            return {"documents": []}

        # Filtra e reordena os documentos com base no ranking
        ranked_docs = []
        for index in ranked_indices:
            if 1 <= index <= len(documents):
                ranked_docs.append(documents[index - 1])
        
        print(f"   ✅ Documentos filtrados e reordenados. {len(ranked_docs)} mantidos.")
        return {"documents": ranked_docs}
        
    except Exception as e:
        print(f"   ⚠️ Erro na verificação: {e}")
        return state  # Continua com os documentos em caso de erro


def generate(state: GraphState) -> GraphState:
    """Gera a resposta usando o LLM com base nos documentos."""
    print("\n✍️ GERANDO RESPOSTA...")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {
            "generation": "❌ **Não encontrei documentos relevantes no sistema para responder sua pergunta.**\n\n"
                         "💡 **Sugestões:**\n"
                         "- Tente usar termos diferentes ou mais específicos\n"
                         "- Verifique se há documentos sobre esse assunto no Paperless\n"
                         "- Reformule a pergunta de forma mais direta"
        }
    
    # Formata contexto estruturado com TODO O CONTEÚDO
    context_parts = []
    preview_links_parts = []
    for i, doc in enumerate(documents, 1):
        # Prioriza highlights, mas pega TODO o conteúdo disponível
        content = doc.get("content", "") or doc.get("highlights", "")
        
        if content:
            # SEM LIMITE - envia todo o conteúdo do documento
            score_info = f"Relevância: {doc.get('score', 0):.2%}" if doc.get('score') else ""
            
            context_parts.append(
                f"=== DOCUMENTO {i} ===\n"
                f"Título: {doc['title']}\n"
                f"Link: {doc['link']}\n"
                f"{score_info}\n"
                f"Conteúdo COMPLETO:\n{content}\n"
            )
            
            print(f"   📄 Doc {i}: {doc['title'][:50]}... ({len(content):,} caracteres)")
            
            preview_links_parts.append(f"- [{doc['title']}]({doc['link']})")
    
    context = "\n\n".join(context_parts)
    preview_links = "\n".join(preview_links_parts)

    print(f"\n   📊 ESTATÍSTICAS DO CONTEXTO:")
    print(f"   • Total de caracteres: {len(context):,}")
    print(f"   • Número de documentos: {len(documents)}")
    print(f"   • Tamanho médio por documento: {len(context)//len(documents):,} caracteres")
    print(f"   • Tokens estimados: ~{len(context)//4:,} tokens")
    
    # # DEBUG: Mostra preview mais longo do contexto
    # print("\n" + "=" * 70)
    # print("🔍 PREVIEW DO CONTEXTO ENVIADO PARA A LLM:")
    # print("=" * 70)
    
    # # Mostra primeiros 2000 caracteres de cada documento
    # for i, part in enumerate(context_parts, 1):
    #     lines = part.split('\n')
    #     preview_lines = []
    #     char_count = 0
        
    #     for line in lines:
    #         if char_count + len(line) > 2000:
    #             preview_lines.append(f"\n... [DOCUMENTO CONTINUA POR MAIS {len(part) - char_count:,} CARACTERES] ...")
    #             break
    #         preview_lines.append(line)
    #         char_count += len(line)
        
    #     print(f"\n--- DOCUMENTO {i} ---")
    #     print('\n'.join(preview_lines))
    
    # print("\n" + "=" * 70)
    # print(f"📝 PERGUNTA ENVIADA: {question}")
    # print("=" * 70 + "\n")
    
    try:
        rag_chain = RAG_GENERATION_PROMPT | llm | StrOutputParser()
        generation = rag_chain.invoke({
            "context": context,
            "question": question,
            "preview_links": preview_links
        })
        
        print("   ✅ Resposta gerada com sucesso")
        print(f"   📏 Tamanho da resposta: {len(generation)} caracteres")
        return {"generation": generation}
        
    except Exception as e:
        print(f"   ❌ Erro na geração: {e}")
        return {
            "generation": "Desculpe, ocorreu um erro ao processar os documentos.",
            "error": str(e)
        }


# ============================================================================
# CONSTRUÇÃO DO GRAFO
# ============================================================================

def build_rag_graph() -> StateGraph:
    """Constrói e compila o grafo RAG."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("generate", generate)
    
    # O ponto de entrada agora é a recuperação (retrieve)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "check_relevance")
    workflow.add_edge("check_relevance", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


app = build_rag_graph()


# ============================================================================
# TESTE COM NOVAS PERGUNTAS
# ============================================================================

# if __name__ == "__main__":
#     print("=" * 70)
#     print("🧪 TESTANDO WORKFLOW RAG COM PAPERLESS-NGX")
#     print("=" * 70)
    
#     # NOVAS PERGUNTAS DE TESTE
#     test_questions = [
#         "Explique o código de conduta para servidores?",
#         "Quais são os procedimentos do regime jurídico das parcerias?",
#     ]
    
#     for question in test_questions:
#         print(f"\n\n{'=' * 70}")
#         print(f"❓ PERGUNTA: {question}")
#         print("=" * 70)
        
#         try:
#             final_state = app.invoke({"question": question})
            
#             print("\n" + "=" * 70)
#             print("📝 RESPOSTA FINAL:")
#             print("=" * 70)
#             print(final_state.get("generation", "Nenhuma resposta gerada"))
            
#             documents = final_state.get("documents", [])
#             if documents:
#                 print("\n" + "=" * 70)
#                 print("📚 FONTES UTILIZADAS:")
#                 print("=" * 70)
#                 for i, doc in enumerate(documents, 1):
#                     score = doc.get('score', 0)
#                     print(f"{i}. [{doc['title']}]({doc['link']}) - Score: {score:.2%}")
            
#             if final_state.get("error"):
#                 print(f"\n⚠️ Aviso: {final_state['error']}")
                
#         except Exception as e:
#             print(f"\n❌ ERRO: {e}")
#             import traceback
#             traceback.print_exc()
        
#         print("\n" + "=" * 70 + "\n")