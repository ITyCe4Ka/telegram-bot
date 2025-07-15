# Используйте официальный образ Python
FROM python:3.11-slim

# Установите рабочую директорию
WORKDIR /app

# Скопируйте файлы проекта врабочую директорию
COPY . /app

# Установите необходимые пакеты
RUN pip install --trusted-host pypi.python.org -r requirements.txt


# Запустите приложение
CMD ["python", "main.py"]
