# Use uma imagem base oficial do Python
FROM python:3.13

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Desativa o buffer de saída do Python para que os prints apareçam em tempo real nos logs do Docker
ENV PYTHONUNBUFFERED=1

# Copia o arquivo de dependências para o diretório de trabalho
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação para o diretório de trabalho
COPY . .

# Expõe a porta que o Streamlit usa
EXPOSE 8501

# Comando para rodar a aplicação Streamlit
CMD ["streamlit", "run", "app.py"]