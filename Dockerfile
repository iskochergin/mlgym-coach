# Простой одно-контейнерный запуск mlgym-coach: Streamlit-дашборд + раннер/агент.
# Песочница НЕ изолирована (агент пишет наш же baseline-код) — это осознанно для MVP.
FROM python:3.11-slim

WORKDIR /app

# venv в /app/.venv: дашборд запускает фоновый раннер через _REPO_ROOT/.venv/bin/python
# (dashboard/app.py), поэтому он обязан существовать со всеми зависимостями.
ENV VENV=/app/.venv
ENV PATH="$VENV/bin:$PATH"
RUN python -m venv "$VENV"

# Сначала зависимости — лучше кэшируется при изменениях кода.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Затем код проекта (.env и прочее исключены через .dockerignore).
COPY . .

EXPOSE 8501

# Streamlit слушает на 0.0.0.0, чтобы порт прокидывался наружу контейнера.
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
