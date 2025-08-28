# ML_REC_SYS
# Instacart Recommendation System API

🚀 Cистема рекомендаций для электронной коммерции на основе гибридного подхода (SVD + KNN)

## 📋 О системе

Система предоставляет персонализированные рекомендации товаров для пользователей на основе их истории покупок. Использует современные методы машинного обучения для анализа паттернов покупок и предсказания наиболее релевантных товаров.

### 🔧 Технологический стек

- **FastAPI** - высокопроизводительный веб-фреймворк
- **Pandas/Numpy** - обработка и анализ данных
- **Scikit-learn** - машинное обучение и метрики
- **Surprise** - алгоритмы рекомендательных систем
- **PyTorch** - глубокое обучение (опционально)
- **Uvicorn** - ASGI сервер

### 🎯 Ключевые особенности

- **Гибридная модель**: Комбинация SVD и KNN для максимальной точности
- **12 фичей рейтинга**: Многофакторная система оценки предпочтений
- **Автоматическое обнаружение данных**: Интеллектуальное определение структуры данных
- **Пакетная обработка**: Поддержка массовых запросов
- **Real-time обновления**: Возможность добавления новых данных без перезапуска

## 🚀 Быстрый старт

### Установка и запуск

```bash
# Клонирование репозитория
git clone <your-repo-url>
cd hybresys_insta
```
# Создание виртуального окружения
```
python -m venv instacart_env
source instacart_env/bin/activate  # Linux/Mac
```
# или
```
instacart_env\Scripts\activate     # Windows
```
# Установка зависимостей
```
pip install -r requirements.txt
```
# Запуск сервера
```
python main.py
```
Сервер будет доступен по адресу: http://localhost:8000

### 📊 Подготовка данных
Поместите файлы с данными в папку data/:

transactions.csv - основные данные о транзакциях

Формат: user_id, product_id, order_id, order_number, reordered, add_to_cart_order, order_dow, order_hour_of_day

Система автоматически определит структуру ваших данных!

📖 API Документация
После запуска откройте: http://localhost:8000/docs для интерактивной документации Swagger.

🔍 Основные endpoints
1. Получить рекомендации для пользователя
```http
GET /recommend/{user_id}?k=10
```
user_id - идентификатор пользователя

k - количество рекомендаций (по умолчанию: 10)

Пример ответа:

``` json
{
  "user_id": 1,
  "recommendations": [196, 12427, 10258, 25133, 47766],
  "count": 5
}
```
2. Пакетные рекомендации
``` http
POST /recommend-batch
```
Тело запроса:

``` json
{
  "user_ids": [1, 2, 3, 4, 5],
  "k": 5
}
```
3. Добавить новые транзакции (JSON)
```http
POST /add-transactions
```
Тело запроса:

```json
{
  "transactions": [
    {
      "user_id": 100001,
      "product_id": 50001,
      "order_id": 200001,
      "order_number": 1,
      "reordered": 0,
      "add_to_cart_order": 1,
      "order_dow": 3,
      "order_hour_of_day": 14
    }
  ]
}
```
4. Загрузить файл транзакций
```http
POST /upload-transactions
```
Загрузите CSV файл через форму

Поддерживается drag&drop в документации

5. Обновить информацию о товарах
```http
POST /update-products
```
Тело запроса:

```json
{
  "products": [
    {
      "product_id": 50001,
      "product_name": "Organic Bananas",
      "department": "produce",
      "aisle": "fresh fruits"
    }
  ]
}
```
6. Переобучить модель
```http
POST /retrain?full_retrain=true&optimize=true
```
full_retrain - полное переобучение (true/false)

optimize - оптимизация гиперпараметров (true/false)

🎮 Примеры использования
Через браузер
Откройте http://localhost:8000/docs

Выберите нужный endpoint

Нажмите "Try it out"

Заполните параметры

Нажмите "Execute"


## Через curl
### Получить рекомендации
```
curl -X GET "http://localhost:8000/recommend/1?k=5"
```
### Пакетные рекомендации
```bash
curl -X POST "http://localhost:8000/recommend-batch" \
  -H "Content-Type: application/json" \
  -d '{"user_ids": [1,2,3], "k": 3}'
```
### Проверить статус
```
curl -X GET "http://localhost:8000/health"
```
## Через Python
```python
import requests

# Получить рекомендации
response = requests.get("http://localhost:8000/recommend/1?k=5")
recommendations = response.json()["recommendations"]

# Добавить транзакции
new_transactions = {
    "transactions": [
        {
            "user_id": 100001,
            "product_id": 50001,
            "order_id": 200001,
            "order_number": 1,
            "reordered": 0,
            "add_to_cart_order": 1,
            "order_dow": 3,
            "order_hour_of_day": 14
        }
    ]
}
response = requests.post("http://localhost:8000/add-transactions", json=new_transactions)
```
🧠 Как работает система
📊 Анализируемые данные
Система анализирует 12 ключевых факторов:

История покупок - частота и регулярность покупок

Реордеры - повторные покупки товаров

Время покупок - утренние/вечерние паттерны

Популярность товаров - глобальные предпочтения

Позиция в корзине - порядок добавления товаров

Размер заказа - соотношение с средним чеком

Временные промежутки - время между заказами

Лояльность - постоянство покупок

Экспериментальность - разнообразие покупок

Дни недели - паттерны по дням

Время суток - предпочтения по времени

Сезонность - временные тенденции

🎯 Алгоритмы машинного обучения
1. SVD (Singular Value Decomposition)
Назначение: Выявление скрытых паттернов

Преимущества: Работа с разреженными матрицами

Оптимизация: Автоподбор оптимального количества факторов

2. K-Nearest Neighbors
Назначение: Поиск похожих товаров

Метрики: Косинусное расстояние, евклидово расстояние

Оптимизация: Динамический подбор количества соседей

3. Гибридный подход
Взвешивание: Оптимальное сочетание SVD и KNN

Адаптивность: Автоматическая настройка весов

Точность: Улучшенная точность предсказаний

⚡ Производительность
Обработка: до 100,000+ пользователей

Рекомендации: < 100ms на запрос

Пакетная обработка: параллельные вычисления

Обучение: инкрементальное обновление моделей

🔧 Конфигурация
Переменные окружения
Создайте файл .env:

```env
HOST=0.0.0.0
PORT=8000
WORKERS=4
LOG_LEVEL=INFO
DATA_PATH=./data
```
Настройка параметров
В коде можно настроить:

Веса фичей рейтинговой системы

Параметры алгоритмов ML

Логирование и мониторинг

Пути к данным

📈 Мониторинг и логи
Система предоставляет:

Журналирование: Подробные логи выполнения

Метрики: Статистика по пользователям и товарам

Health checks: Мониторинг состояния системы

Тайминги: Время выполнения операций

Просмотр логов:

```bash
tail -f app.log
```
🚀 Производственное развертывание
### Docker развертывание
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```
### Kubernetes развертывание
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommender-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: recommender
        image: your-repo/recommender:latest
        ports:
        - containerPort: 8000
```
🤝 Contributing
Форкните репозиторий

Создайте feature branch: `git checkout -b feature/amazing-feature`

Коммитте изменения: `git commit -m 'Add amazing feature'`

Запушьте ветку: `git push origin feature/amazing-feature`

Создайте Pull Request


🆘 Поддержка
Если у вас возникли вопросы:

Проверьте документацию API: http://localhost:8000/docs

Посмотрите логи приложения
