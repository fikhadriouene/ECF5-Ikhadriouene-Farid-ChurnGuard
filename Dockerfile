

# Stage 1 — Builder


FROM python:3.11-slim AS builder

WORKDIR /app

# Création du virtualenv
RUN python -m venv /opt/venv

# Utilisation du venv
ENV PATH="/opt/venv/bin:$PATH"

# Copie des dépendances
COPY requirements.txt .

# Installation des dépendances
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt



# Stage 2 — Runtime


FROM python:3.11-slim

WORKDIR /app

# Installation de curl pour le healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Variables d'environnement
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copie du virtualenv depuis le builder
COPY --from=builder /opt/venv /opt/venv

# Copie du code source
COPY . .

# Télécharger les données
RUN python scripts/download_data.py

# Création d'un utilisateur non-root
RUN useradd -m appuser

# Utilisation de l'utilisateur non-root
USER appuser

# Exposition du port FastAPI
EXPOSE 8000

# Healthcheck Docker
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Lancement de l'API FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

