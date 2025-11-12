import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

class PaperlessClient:
    """Cliente para interação com a API do Paperless-NGX."""
    
    def __init__(self):
        self.api_url = os.getenv("PAPERLESS_API_URL")
        self.username = os.getenv("PAPERLESS_USERNAME")
        self.password = os.getenv("PAPERLESS_PASSWORD")
        self.base_url = os.getenv("PAPERLESS_BASE_URL", 
                                   self.api_url.replace("/api", "") if self.api_url else None)

        self._validate_credentials()
        
        self.auth = (self.username, self.password)
        self.headers = {"Content-Type": "application/json"}
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update(self.headers)

    def _validate_credentials(self):
        """Valida se todas as credenciais necessárias estão configuradas."""
        missing = []
        if not self.api_url:
            missing.append("PAPERLESS_API_URL")
        if not self.username:
            missing.append("PAPERLESS_USERNAME")
        if not self.password:
            missing.append("PAPERLESS_PASSWORD")
        
        if missing:
            raise ValueError(
                f"As seguintes variáveis de ambiente estão faltando: {', '.join(missing)}. "
                "Configure-as no arquivo .env"
            )

    def _make_request(self, method: str, endpoint: str, 
                     params: Optional[Dict] = None, 
                     data: Optional[Dict] = None) -> Optional[Dict]:
        """Executa uma requisição HTTP com tratamento de erros robusto."""
        url = f"{self.api_url}{endpoint}"
        
        try:
            response = self.session.request(
                method, url, params=params, json=data, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout ao acessar {url}")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"❌ Erro HTTP {response.status_code}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro na requisição: {e}")
            return None

    def search_documents(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Busca documentos no Paperless-NGX.
        
        Args:
            query: Termo de busca
            limit: Número máximo de documentos a retornar (padrão: 5)
            
        Returns:
            Lista de dicionários com informações dos documentos
        """
        endpoint = "/documents/"
        params = {
            "query": query,
            "page_size": limit,
            "ordering": "-score"  # Ordena por relevância
        }
        
        print(f"🔍 Buscando no Paperless: '{query}' (limite: {limit})")
        
        data = self._make_request("GET", endpoint, params=params)
        
        if not data or not data.get("results"):
            print(f"ℹ️ Nenhum documento encontrado para: '{query}'")
            return []
        
        documents = []
        for doc in data["results"]:
            doc_id = doc["id"]
            
            # Link direto para o preview do documento
            doc["link"] = f"{self.api_url}/documents/{doc_id}/preview/"
            
            # Processa highlights (trechos relevantes destacados)
            highlights_html = doc.get("__search_hit__", {}).get("highlights", "")
            highlights_cleaned = self._clean_highlights(highlights_html)
            doc["highlights"] = highlights_cleaned
            
            # Garante que o campo 'content' tenha o conteúdo dos highlights como fallback
            doc["content"] = doc.get("content", "")
            
            documents.append(doc)
        
        print(f"✅ Encontrados {len(documents)} documento(s)")
        return documents

    def _clean_highlights(self, highlights_html: str) -> str:
        """Remove HTML mantendo apenas o texto limpo."""
        if not highlights_html:
            return ""
        
        soup = BeautifulSoup(highlights_html, 'html.parser')
        # Preserva quebras de linha e remove espaços extras
        text = soup.get_text(separator="\n", strip=True)
        # Remove linhas vazias múltiplas
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)

    def get_all_document_ids(self) -> List[int]:
        """Busca e retorna os IDs de todos os documentos no Paperless, lidando com paginação."""
        all_ids = []
        endpoint = "/documents/"
        params = {"page_size": 100, "fields": "id"}  # Busca apenas IDs para eficiência
        
        while endpoint:
            data = self._make_request("GET", endpoint, params=params if endpoint == "/documents/" else None)
            if not data or not data.get("results"):
                break
            
            ids = [doc['id'] for doc in data['results']]
            all_ids.extend(ids)
            
            endpoint = data.get("next")
            if endpoint:
                # Remove a base da URL, pois _make_request já a adiciona
                endpoint = endpoint.replace(self.api_url, "")
                print(f"  -> Carregando próxima página... ({len(all_ids)} IDs encontrados)")

        return all_ids

    def get_all_tags(self) -> Dict[int, str]:
        """Busca e retorna um dicionário com todos os IDs e nomes de tags."""
        tag_map = {}
        endpoint = "/tags/"
        params = {"page_size": 100}

        print("🔄 Mapeando todas as tags do Paperless-NGX...")
        while endpoint:
            data = self._make_request("GET", endpoint, params=params if endpoint == "/tags/" else None)
            if not data or not data.get("results"):
                break

            for tag in data["results"]:
                tag_map[tag['id']] = tag['name']

            endpoint = data.get("next")
            if endpoint:
                endpoint = endpoint.replace(self.api_url, "")
                print(f"  -> Carregando próxima página de tags... ({len(tag_map)} tags mapeadas)")
        
        return tag_map

    def download_document_content(self, doc_id: int) -> Optional[bytes]:
        """
        Baixa o conteúdo binário de um documento.
        
        Args:
            doc_id: ID do documento no Paperless
            
        Returns:
            Bytes do arquivo ou None em caso de erro
        """
        endpoint = f"/documents/{doc_id}/download/"
        url = f"{self.api_url}{endpoint}"
        
        print(f"📥 Baixando documento ID: {doc_id}")
        
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            print(f"✅ Download concluído: {len(response.content)} bytes")
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro ao baixar documento {doc_id}: {e}")
            return None

    def get_document_metadata(self, doc_id: int) -> Optional[Dict]:
        """Obtém metadados completos de um documento."""
        endpoint = f"/documents/{doc_id}/"
        return self._make_request("GET", endpoint)


if __name__ == "__main__":
    try:
        client = PaperlessClient()
        query = "iptu"
        
        documents = client.search_documents(query, limit=3)
        
        if documents:
            print(f"\n📄 Resultados para '{query}':\n")
            for i, doc in enumerate(documents, 1):
                print(f"{i}. {doc['title']}")
                print(f"   🔗 {doc['link']}")
                print(f"   📝 Preview: {doc['content'][:100]}...")
                print()
        else:
            print(f"Nenhum documento encontrado para '{query}'")
            
    except ValueError as e:
        print(f"⚙️ Erro de configuração: {e}")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")