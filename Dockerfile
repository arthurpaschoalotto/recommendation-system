# Dockerfile
FROM python:3.11-slim

# Instala dependências
RUN pip install --upgrade pip

# Cria diretório da aplicação
WORKDIR /app

# Copia os arquivos
COPY . /app

# Instala as dependências
RUN pip install -r requirements.txt

# Expõe a porta usada pelo Streamlit
EXPOSE 8501

# Comando para rodar o Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
