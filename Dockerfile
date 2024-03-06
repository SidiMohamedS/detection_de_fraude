# Utilisez l'image Python officielle en tant qu'image de base
FROM python:3.12.2


# Définir le répertoire de travail dans le conteneur
WORKDIR /app


COPY requirements.txt /app
# Installez les dépendances pour Flask et Streamlit
RUN pip install -r requirements.txt

# Copiez les fichiers requis dans le conteneur
COPY app.py /app
COPY frontend.py /app

# Configurez l'environnement Flask
ENV FLASK_APP=app.py

# Exposez les ports nécessaires
EXPOSE 5000
EXPOSE 8001

# Commande pour démarrer les applications Flask et Streamlit
CMD ["sh", "-c", "flask run --host 0.0.0.0 --port 5000 & streamlit run --server.port 8001 frontend.py"]
