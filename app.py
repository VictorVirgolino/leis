import streamlit as st
from graph import app as langgraph_app, paperless_client
from datetime import datetime

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Assistente de Documentos Legais",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar a aparência
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
    }
    .source-card:hover {
        background-color: #e8eaf0;
        transition: background-color 0.3s;
    }
    .highlight-preview {
        font-size: 0.9em;
        color: #555;
        font-style: italic;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR COM INFORMAÇÕES
# ============================================================================

with st.sidebar:
    st.title("⚖️ Assistente Legal")
    st.markdown("---")
    
    st.markdown("### 📋 Sobre")
    st.markdown("""
    Este chatbot busca informações em documentos do **Paperless-NGX** 
    e fornece respostas baseadas nos **3 documentos mais relevantes**.
    
    ✨ **Powered by:**
    - 🤖 Google Gemini 2.5-Flash
    - 📚 LangGraph RAG
    - 📁 Paperless-NGX
    """)
    
    st.markdown("### 💡 Dicas de uso")
    st.markdown("""
    - ✅ Faça perguntas específicas
    - ✅ Use termos técnicos quando apropriado
    - ✅ Pergunte sobre artigos ou leis específicas
    - ✅ As fontes sempre serão citadas
    - ✅ Clique nos links para ver o documento completo
    """)
    
    st.markdown("---")
    
    # Botão para limpar histórico
    if st.button("🗑️ Limpar Conversa", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    # Estatísticas da sessão
    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        total_msgs = len(st.session_state.messages)
        user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        assistant_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        st.markdown("---")
        st.markdown("### 📊 Estatísticas da Sessão")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📝 Perguntas", user_msgs)
        with col2:
            st.metric("💬 Respostas", assistant_msgs - 1)  # -1 para excluir boas-vindas
        
        # Total de fontes consultadas
        total_sources = sum(
            len(m.get("sources", [])) 
            for m in st.session_state.messages 
            if m["role"] == "assistant"
        )
        st.metric("📚 Documentos consultados", total_sources)


# ============================================================================
# CABEÇALHO PRINCIPAL
# ============================================================================

st.title("⚖️ Assistente de Documentos Legais")
st.caption("🔍 Busca inteligente em documentos do Paperless-NGX com IA")

# Separador visual
st.markdown("---")


# ============================================================================
# GERENCIAMENTO DE ESTADO
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Mensagem de boas-vindas
    st.session_state.messages.append({
        "role": "assistant",
        "content": """👋 **Olá! Sou seu assistente de documentos legais.**

Posso ajudá-lo a encontrar informações em seus documentos do Paperless-NGX. 

**Como funciono:**
1. Você faz uma pergunta
2. Busco nos documentos mais relevantes
3. Apresento a resposta com as fontes citadas

**Exemplos de perguntas:**
- "Quais sâo príncipios de conduta dos servidores?"
- "O que diz sobre IPTU atrasado?"
- "Como funciona a cobrança de ISS?"
- "Quais são os códigos de conduta dos servidores públicos?"

📝 **Digite sua pergunta abaixo para começar!**""",
        "sources": []
    })


# ============================================================================
# EXIBIÇÃO DO HISTÓRICO
# ============================================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Exibe fontes se existirem
        if message.get("sources"):
            num_sources = len(message["sources"])
            with st.expander(f"⬇️ Baixar {num_sources} Documento{'s' if num_sources > 1 else ''} Utilizado{'s' if num_sources > 1 else ''}", expanded=False):
                for source in message["sources"]:
                    download_link = f"{source['api_url']}/documents/{source['id']}/download/"
                    st.markdown(f"""
                    <div class="source-card">
                        <a href="{download_link}" target="_blank" style="text-decoration: none;">
                            📄 Baixar: {source['title']}
                        </a>
                    </div>
                    """, unsafe_allow_html=True)


# ============================================================================
# PROCESSAMENTO DE NOVA MENSAGEM
# ============================================================================

if prompt := st.chat_input("💬 Digite sua pergunta sobre os documentos...", key="user_input"):
    # Valida entrada
    prompt_stripped = prompt.strip()
    if len(prompt_stripped) < 3:
        st.warning("⚠️ Por favor, faça uma pergunta mais específica (mínimo 3 caracteres).")
        st.stop()
    
    # Adiciona mensagem do usuário
    st.session_state.messages.append({
        "role": "user",
        "content": prompt_stripped
    })
    
    # Exibe mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt_stripped)
    
    # Processa com o LangGraph
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        sources_placeholder = st.empty()
        status_placeholder = st.empty()
        
        with st.spinner("🔍 Buscando nos documentos e preparando resposta..."):
            try:
                # Invoca o grafo LangGraph
                final_state = langgraph_app.invoke({"question": prompt_stripped})
                
                generation = final_state.get(
                    "generation",
                    "Desculpe, não foi possível gerar uma resposta."
                )
                documents = final_state.get("documents", [])
                error = final_state.get("error")
                
                # Exibe a resposta
                message_placeholder.markdown(generation)
                
                # Prepara dados das fontes para salvar
                sources_data = []
                for doc in documents:
                    sources_data.append({
                        "id": doc.get("id"),
                        "title": doc.get("title", "Sem título"),
                        "link": doc.get("link", "#"),
                        "highlights": doc.get("highlights", "Sem preview"),
                        "score": doc.get("score", 0),
                        "api_url": paperless_client.api_url
                    })
                
                # Salva no histórico
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": generation,
                    "sources": sources_data,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                # Exibe fontes
                if sources_data:
                    num_sources = len(sources_data)
                    with sources_placeholder.expander(f"⬇️ Baixar {num_sources} Documento{'s' if num_sources > 1 else ''} Utilizado{'s' if num_sources > 1 else ''}",
                        expanded=True
                    ):
                        for source in sources_data:
                            download_link = f"{source['api_url']}/documents/{source['id']}/download/"
                            
                            st.markdown(f"""
                            <div class="source-card">
                                <a href="{download_link}" target="_blank" style="text-decoration: none;">
                                    📄 Baixar: {source['title']}
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Feedback de sucesso
                    status_placeholder.success(
                        f"✅ Resposta gerada com sucesso baseada em {num_sources} documento{'s' if num_sources > 1 else ''}!",
                        icon="✅"
                    )
                else:
                    status_placeholder.info(
                        "ℹ️ Nenhum documento encontrado. Tente reformular sua pergunta.",
                        icon="ℹ️"
                    )
                
                # Exibe aviso se houver erro (mas não interrompe)
                if error:
                    st.warning(f"⚠️ Aviso: {error}")
                
            except Exception as e:
                error_msg = f"❌ Ocorreu um erro ao processar sua pergunta: {str(e)}"
                message_placeholder.error(error_msg)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
                
                # Log detalhado no terminal para debug
                import traceback
                print("\n" + "="*70)
                print("ERRO DETALHADO:")
                print("="*70)
                print(f"Pergunta: {prompt_stripped}")
                print(f"Erro: {str(e)}")
                traceback.print_exc()
                print("="*70 + "\n")
                
                # Sugestão amigável para o usuário
                st.info(
                    "💡 **Sugestão:** Tente reformular sua pergunta ou verifique se há documentos relevantes no Paperless.",
                    icon="💡"
                )


# ============================================================================
# RODAPÉ
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; padding: 1rem;'>
    <p>💡 <strong>Lembre-se:</strong> As respostas são baseadas exclusivamente nos documentos do Paperless-NGX</p>
    <p>⚠️ Sempre verifique as fontes citadas para confirmar as informações</p>
    <p style='margin-top: 1rem; font-size: 0.8em; color: #999;'>
        Desenvolvido com ❤️ usando Streamlit • LangGraph • Google Gemini
    </p>
</div>
""", unsafe_allow_html=True)