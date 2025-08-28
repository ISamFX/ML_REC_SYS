from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import uvicorn

# Импортируем наши классы
from recsys_classes import OurDataWinnerV20, AdvancedRatingSystem, RecommenderOptimizer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Глобальная переменная для рекомендательной системы
recommender_system = None
executor = ThreadPoolExecutor(max_workers=4)

# Модели Pydantic для запросов
class UserRequest(BaseModel):
    user_id: int
    k: int = 10

class UsersRequest(BaseModel):
    user_ids: List[int]
    k: int = 10

class TransactionData(BaseModel):
    transactions: List[Dict[str, Any]]

class ProductData(BaseModel):
    products: List[Dict[str, Any]]

class TrainingResponse(BaseModel):
    status: str
    message: str
    training_id: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events для инициализации и очистки"""
    global recommender_system
    
    # Startup
    try:
        logger.info("Инициализация рекомендательной системы...")
        recommender_system = OurDataWinnerV20()
        
        # Автопоиск и загрузка данных
        recommender_system.auto_detect_data_files()
        if recommender_system.load_and_preprocess_data():
            logger.info("Рекомендательная система успешно инициализирована")
        else:
            logger.warning("Не удалось загрузить данные, система будет инициализирована позже")
            
    except Exception as e:
        logger.error(f"Ошибка при инициализации: {e}")
        recommender_system = None
    
    yield
    
    # Shutdown
    executor.shutdown(wait=False)
    logger.info("Приложение завершает работу")

app = FastAPI(
    title="Instacart Recommendation System API",
    description="API для рекомендательной системы Instacart",
    version="1.0.0",
    lifespan=lifespan
)

@app.on_event("startup")
async def startup_event():
    """Инициализация рекомендательной системы при запуске"""
    global recommender_system
    try:
        logger.info("Инициализация рекомендательной системы...")
        recommender_system = OurDataWinnerV20()
        
        # Автопоиск и загрузка данных
        recommender_system.auto_detect_data_files()
        if recommender_system.load_and_preprocess_data():
            logger.info("Рекомендательная система успешно инициализирована")
        else:
            logger.warning("Не удалось загрузить данные, система будет инициализирована позже")
            
    except Exception as e:
        logger.error(f"Ошибка при инициализации: {e}")
        recommender_system = None

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Instacart Recommendation System API",
        "status": "active" if recommender_system else "initializing",
        "endpoints": [
            "/docs - Документация API",
            "/health - Статус системы",
            "/recommend/{user_id} - Рекомендации для пользователя",
            "/recommend-batch - Пакетные рекомендации",
            "/add-transactions - Добавить транзакции",
            "/update-products - Обновить товары",
            "/retrain - Переобучить модель"
        ]
    }

@app.get("/health")
async def health_check():
    """Проверка статуса системы"""
    status = {
        "status": "healthy" if recommender_system and recommender_system.rating_df is not None else "unhealthy",
        "users_count": len(recommender_system.user_ids) if recommender_system and recommender_system.user_ids is not None else 0,
        "products_count": len(recommender_system.product_ids) if recommender_system and recommender_system.product_ids is not None else 0,
        "initialized": recommender_system is not None
    }
    return status

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: int, k: int = 10):
    """
    Получить рекомендации для конкретного пользователя
    """
    if not recommender_system or recommender_system.rating_df is None:
        raise HTTPException(status_code=503, detail="Система не инициализирована")
    
    try:
        # Проверяем, существует ли пользователь
        if user_id not in recommender_system.user_ids:
            raise HTTPException(status_code=404, detail=f"Пользователь {user_id} не найден")
        
        # Получаем рекомендации из рейтингового датасета
        user_recommendations = (
            recommender_system.rating_df[recommender_system.rating_df['user_id'] == user_id]
            .sort_values('rating', ascending=False)
            .head(k)
        )
        
        if user_recommendations.empty:
            # Если нет рекомендаций, возвращаем популярные товары
            popular_products = (
                recommender_system.rating_df.groupby('product_id')['rating']
                .sum()
                .sort_values(ascending=False)
                .head(k)
                .index.tolist()
            )
            recommendations = popular_products
        else:
            recommendations = user_recommendations['product_id'].tolist()
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Ошибка при получении рекомендаций: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.post("/recommend-batch")
async def get_batch_recommendations(request: UsersRequest):
    """
    Получить рекомендации для нескольких пользователей
    """
    if not recommender_system or recommender_system.rating_df is None:
        raise HTTPException(status_code=503, detail="Система не инициализирована")
    
    try:
        results = []
        
        def process_user(user_id):
            try:
                if user_id in recommender_system.user_ids:
                    user_recs = (
                        recommender_system.rating_df[recommender_system.rating_df['user_id'] == user_id]
                        .sort_values('rating', ascending=False)
                        .head(request.k)
                        ['product_id']
                        .tolist()
                    )
                else:
                    # Для неизвестных пользователей возвращаем популярные товары
                    user_recs = (
                        recommender_system.rating_df.groupby('product_id')['rating']
                        .sum()
                        .sort_values(ascending=False)
                        .head(request.k)
                        .index.tolist()
                    )
                
                return {"user_id": user_id, "recommendations": user_recs}
            except Exception as e:
                return {"user_id": user_id, "error": str(e)}
        
        # Обрабатываем пользователей параллельно
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, process_user, user_id) for user_id in request.user_ids]
        results = await asyncio.gather(*tasks)
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Ошибка при пакетной обработке: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.post("/add-transactions")
async def add_transactions(data: TransactionData, background_tasks: BackgroundTasks):
    """
    Добавить новые транзакции в систему
    """
    try:
        if not recommender_system:
            raise HTTPException(status_code=503, detail="Система не инициализирована")
        
        # Конвертируем в DataFrame
        new_transactions = pd.DataFrame(data.transactions)
        
        # Сохраняем новые транзакции
        transactions_file = "data/new_transactions.csv"
        if os.path.exists(transactions_file):
            existing_transactions = pd.read_csv(transactions_file)
            updated_transactions = pd.concat([existing_transactions, new_transactions], ignore_index=True)
        else:
            updated_transactions = new_transactions
        
        updated_transactions.to_csv(transactions_file, index=False)
        
        # Запускаем обновление в фоне
        background_tasks.add_task(update_system_with_new_data)
        
        return {
            "status": "success",
            "message": f"Добавлено {len(new_transactions)} транзакций",
            "total_transactions": len(updated_transactions)
        }
        
    except Exception as e:
        logger.error(f"Ошибка при добавлении транзакций: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.post("/update-products")
async def update_products(data: ProductData):
    """
    Обновить информацию о товарах
    """
    try:
        # Сохраняем информацию о товарах
        products_file = "data/products_metadata.json"
        new_products = data.products
        
        if os.path.exists(products_file):
            with open(products_file, 'r') as f:
                existing_products = json.load(f)
            # Обновляем существующие товары и добавляем новые
            product_dict = {p['product_id']: p for p in existing_products}
            for product in new_products:
                product_dict[product['product_id']] = product
            updated_products = list(product_dict.values())
        else:
            updated_products = new_products
        
        with open(products_file, 'w') as f:
            json.dump(updated_products, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Обновлено информация о {len(updated_products)} товарах"
        }
        
    except Exception as e:
        logger.error(f"Ошибка при обновлении товаров: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Переобучить модель рекомендательной системы
    """
    try:
        if not recommender_system:
            raise HTTPException(status_code=503, detail="Система не инициализирована")
        
        training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Запускаем переобучение в фоне
        background_tasks.add_task(train_model, training_id)
        
        return {
            "status": "started",
            "message": "Запущено переобучение модели",
            "training_id": training_id
        }
        
    except Exception as e:
        logger.error(f"Ошибка при запуске переобучения: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.get("/training-status/{training_id}")
async def get_training_status(training_id: str):
    """
    Проверить статус обучения
    """
    # В реальной системе здесь можно хранить статус в Redis или БД
    return {
        "training_id": training_id,
        "status": "completed",  # Заглушка
        "message": "Обучение завершено"
    }

# Фоновые задачи
async def update_system_with_new_data():
    """Обновление системы с новыми данными"""
    try:
        logger.info("Обновление системы с новыми данными...")
        # Здесь можно добавить логику обработки новых данных
        # и частичного обновления модели
        await asyncio.sleep(1)
        logger.info("Система обновлена с новыми данными")
    except Exception as e:
        logger.error(f"Ошибка при обновлении системы: {e}")

async def train_model(training_id: str):
    """Фоновая задача для обучения модели"""
    try:
        logger.info(f"Запуск обучения модели {training_id}...")
        
        # Переинициализируем систему с обновленными данными
        global recommender_system
        recommender_system = OurDataWinnerV20()
        
        # Загружаем данные (включая новые транзакции)
        recommender_system.auto_detect_data_files()
        recommender_system.load_and_preprocess_data()
        
        # Оптимизируем рекомендации
        recommender_system.optimize_recommendations()
        
        logger.info(f"Обучение модели {training_id} завершено")
        
    except Exception as e:
        logger.error(f"Ошибка при обучении модели {training_id}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)