import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, lil_matrix
import logging
import os
from tqdm import tqdm
from surprise import Dataset, Reader, SVD, KNNBaseline
from surprise.model_selection import train_test_split as surprise_train_test_split
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class EnhancedKNNRecommender:
    def __init__(self):
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.item_id_to_index = {}
        self.user_id_to_index = {}
        self.index_to_item_id = {}
        self.index_to_user_id = {}
    
    def create_sparse_matrix(self, rating_df):
        """Создание разреженной матрицы пользователь-товар"""
        logger.info("Создание разреженной матрицы пользователь-товар...")
        
        unique_users = sorted(rating_df['user_id'].unique())
        unique_items = sorted(rating_df['product_id'].unique())
        
        self.user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_to_index = {pid: idx for idx, pid in enumerate(unique_items)}
        self.index_to_user_id = {idx: uid for uid, idx in self.user_id_to_index.items()}
        self.index_to_item_id = {idx: pid for pid, idx in self.item_id_to_index.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        matrix = lil_matrix((n_users, n_items), dtype=np.float32)
        
        for _, row in rating_df.iterrows():
            user_idx = self.user_id_to_index[row['user_id']]
            item_idx = self.item_id_to_index[row['product_id']]
            matrix[user_idx, item_idx] = row['rating']
        
        self.user_item_matrix = matrix.tocsr()
        logger.info(f"Матрица создана: {n_users} пользователей × {n_items} товаров")
        return self.user_item_matrix
    
    def compress_dataset(self, rating_df, min_user_interactions=1, min_item_interactions=1):
        """Сжатие датасета с более мягкими фильтрами"""
        logger.info("Сжатие датасета...")
        
        user_counts = rating_df['user_id'].value_counts()
        item_counts = rating_df['product_id'].value_counts()
        
        # Более мягкие фильтры чтобы сохранить всех пользователей
        active_users = user_counts[user_counts >= min_user_interactions].index
        popular_items = item_counts[item_counts >= min_item_interactions].index
        
        compressed_df = rating_df[
            (rating_df['user_id'].isin(active_users)) & 
            (rating_df['product_id'].isin(popular_items))
        ]
        
        logger.info(f"Сжатие: {len(rating_df)} → {len(compressed_df)} записей "
                f"({len(active_users)} пользователей, {len(popular_items)} товаров)")
        
        return compressed_df
        
    def build_similarity_matrix(self, method='cosine', k=50):
        """Построение матрицы схожести товаров"""
        logger.info(f"Построение матрицы схожести (method={method}, k={k})...")
        
        if self.user_item_matrix is None:
            raise ValueError("Сначала создайте матрицу пользователь-товар")
        
        item_user_matrix = self.user_item_matrix.T
        item_user_matrix_norm = normalize(item_user_matrix, norm='l2', axis=1)
        
        knn = NearestNeighbors(
            n_neighbors=min(k + 1, item_user_matrix_norm.shape[0]),
            metric=method,
            algorithm='brute' if method == 'cosine' else 'auto'
        )
        
        knn.fit(item_user_matrix_norm)
        distances, indices = knn.kneighbors(item_user_matrix_norm)
        
        n_items = item_user_matrix_norm.shape[0]
        similarity_matrix = lil_matrix((n_items, n_items), dtype=np.float32)
        
        for i in range(n_items):
            for j_idx, dist in zip(indices[i], distances[i]):
                if i != j_idx:
                    similarity = 1.0 - dist
                    similarity_matrix[i, j_idx] = similarity
        
        self.item_similarity_matrix = similarity_matrix.tocsr()
        logger.info("Матрица схожести построена")
        return self.item_similarity_matrix
    
    def knn_predict(self, user_id, top_n=10):
        """Предсказание рейтингов для пользователя с помощью KNN"""
        if user_id not in self.user_id_to_index:
            return None
        
        user_idx = self.user_id_to_index[user_id]
        user_vector = self.user_item_matrix[user_idx].toarray().flatten()
        rated_items = user_vector.nonzero()[0]
        
        if len(rated_items) == 0:
            return None
        
        predictions = np.zeros(self.user_item_matrix.shape[1])
        
        for item_idx in range(self.user_item_matrix.shape[1]):
            if user_vector[item_idx] == 0:
                similar_items = self.item_similarity_matrix[item_idx].indices
                rated_similar_items = np.intersect1d(similar_items, rated_items)
                
                if len(rated_similar_items) > 0:
                    similarities = self.item_similarity_matrix[item_idx, rated_similar_items].toarray().flatten()
                    ratings = user_vector[rated_similar_items]
                    
                    if np.sum(similarities) > 0:
                        predictions[item_idx] = np.sum(similarities * ratings) / np.sum(similarities)
        
        top_indices = np.argsort(predictions)[::-1][:top_n]
        results = []
        
        for idx in top_indices:
            if predictions[idx] > 0:
                results.append({
                    'product_id': self.index_to_item_id[idx],
                    'predicted_rating': predictions[idx],
                    'rank': len(results) + 1
                })
        
        return results
    
    def batch_knn_predict(self, user_ids, top_n=10):
        """Пакетное предсказание для нескольких пользователей"""
        all_predictions = []
        
        for user_id in user_ids:
            predictions = self.knn_predict(user_id, top_n)
            if predictions:
                for pred in predictions:
                    all_predictions.append({
                        'user_id': user_id,
                        'product_id': pred['product_id'],
                        'rating_knn': pred['predicted_rating'],
                        'rank': pred['rank']
                    })
        
        return pd.DataFrame(all_predictions)

class AdvancedRatingSystem:
    def __init__(self):
        self.rating_data = {}
        self.feature_weights = {
            'purchase_exists': 2.0,
            'purchase_count': 0.1,
            'reordered': 3.0,
            'morning_purchase': 0.5,
            'evening_purchase': 0.5,
            'global_popularity': 1.0,
            'early_cart_position': 1.0,
            'late_cart_position': -0.3,
            'experimental_user': 0.3,
            'loyal_user': 0.5,
            'time_since_last_order': -0.2,
            'order_size_ratio': 0.4
        }
        self.user_stats = {}
        self.product_stats = {}
    
    def calculate_ratings(self, df):
        logger.info("Сверхбыстрый расчет рейтингов по 12 фичам...")
        
        self._calculate_preliminary_stats(df)
        
        agg_dict = {
            'purchase_count': ('product_id', 'size'),
            'reordered_sum': ('reordered', 'sum'),
            'avg_cart_position': ('add_to_cart_order', 'mean'),
            'avg_time_gap': ('time_since_last_order', 'mean'),
            'avg_size_ratio': ('order_size_ratio', 'mean'),
            'hours': ('order_hour_of_day', lambda x: list(x))
        }
        
        agg_spec = {key: val for key, val in agg_dict.items()}
        user_product_stats = df.groupby(['user_id', 'product_id']).agg(**agg_spec).reset_index()
        
        def calculate_time_features(hours):
            morning = sum(1 for h in hours if 6 <= h <= 12) if hours else 0
            evening = sum(1 for h in hours if 18 <= h <= 24) if hours else 0
            return morning, evening
        
        time_features = user_product_stats['hours'].apply(calculate_time_features)
        user_product_stats['morning_count'] = time_features.apply(lambda x: x[0])
        user_product_stats['evening_count'] = time_features.apply(lambda x: x[1])
        
        user_product_stats = user_product_stats.merge(
            self.user_stats[['user_id', 'unique_products', 'reorder_rate']], 
            on='user_id', how='left'
        )
        
        user_product_stats = user_product_stats.merge(
            self.product_stats[['product_id', 'global_popularity']], 
            on='product_id', how='left'
        )
        
        rating = np.zeros(len(user_product_stats))
        
        rating += self.feature_weights['purchase_exists'] * np.minimum(user_product_stats['purchase_count'], 1)
        rating += self.feature_weights['purchase_count'] * user_product_stats['purchase_count']
        rating += self.feature_weights['reordered'] * np.minimum(user_product_stats['reordered_sum'], 1)
        rating += self.feature_weights['morning_purchase'] * user_product_stats['morning_count']
        rating += self.feature_weights['evening_purchase'] * user_product_stats['evening_count']
        rating += self.feature_weights['global_popularity'] * user_product_stats['global_popularity'].fillna(0)
        rating += np.where(user_product_stats['avg_cart_position'] <= 5, self.feature_weights['early_cart_position'], 0)
        rating += np.where(user_product_stats['avg_cart_position'] >= 15, self.feature_weights['late_cart_position'], 0)
        rating += np.where(user_product_stats['unique_products'] > 50, self.feature_weights['experimental_user'], 0)
        rating += np.where(user_product_stats['reorder_rate'] > 0.7, self.feature_weights['loyal_user'], 0)
        rating += self.feature_weights['time_since_last_order'] * np.minimum(user_product_stats['avg_time_gap'].fillna(0) / 24.0, 1.0)
        rating += self.feature_weights['order_size_ratio'] * user_product_stats['avg_size_ratio'].fillna(1.0)
        
        rating = np.maximum(rating, 0.0)
        
        rating_df = pd.DataFrame({
            'user_id': user_product_stats['user_id'],
            'product_id': user_product_stats['product_id'],
            'rating': rating
        })
        
        logger.info(f"Создан датасет рейтингов: {len(rating_df)} записей")
        return rating_df
    
    def _calculate_preliminary_stats(self, df):
        logger.info("Расчет предварительной статистики...")
        
        self.user_stats = df.groupby('user_id').agg({
            'product_id': 'nunique',
            'reordered': 'mean',
            'order_number': 'max',
            'order_size': 'mean'
        }).reset_index()
        self.user_stats.columns = ['user_id', 'unique_products', 'reorder_rate', 'total_orders', 'avg_order_size']
        
        self.product_stats = df.groupby('product_id').agg({
            'user_id': 'nunique',
            'reordered': 'mean',
            'add_to_cart_order': 'mean'
        }).reset_index()
        self.product_stats.columns = ['product_id', 'unique_users', 'reorder_rate', 'avg_cart_position']
        
        product_popularity = df['product_id'].value_counts()
        max_popularity = product_popularity.max()
        self.product_stats['global_popularity'] = self.product_stats['product_id'].map(
            lambda x: product_popularity.get(x, 0) / max_popularity
        )

class RecommenderOptimizer:
    def __init__(self, rating_df):
        self.rating_df = rating_df
        self.best_knn_params = {'k': 250, 'method': 'cosine'}
        self.best_svd_params = {'n_factors': 480}
        self.best_hybrid_weights = (0.6, 0.4)
        self.data = None
        self.trainset = None
        self.testset = None
        self.knn_recommender = EnhancedKNNRecommender()
        self.compressed_df = None
    
    def prepare_fast_optimization(self):
        """Быстрая подготовка для оптимизации"""
        logger.info("🚀 Быстрая подготовка для оптимизации...")
        
        # Используем подвыборку для быстрой оптимизации (10% данных)
        sample_size = min(500000, len(self.rating_df))
        optimization_df = self.rating_df.sample(sample_size, random_state=42)
        
        # Сжимаем с минимальными фильтрами чтобы сохранить всех пользователей
        self.compressed_df = self.knn_recommender.compress_dataset(
            optimization_df, 
            min_user_interactions=1, 
            min_item_interactions=1
        )
        
        logger.info(f"Подвыборка для оптимизации: {len(optimization_df)} записей")
        logger.info(f"Сжатый датасет: {len(self.compressed_df)} записей")
        
        # Создаем матрицу для KNN
        self.knn_recommender.create_sparse_matrix(self.compressed_df)
        
        # Подготавливаем данные для Surprise
        self.prepare_surprise_data(test_size=0.2)
        
        logger.info("✅ Быстрая подготовка завершена")
    
    def prepare_surprise_data(self, test_size=0.2):
        
        # Используем подвыборку для Surprise чтобы ускорить оптимизацию
        surprise_sample = self.rating_df.sample(min(100000, len(self.rating_df)), random_state=42)
        
        surprise_df = surprise_sample.rename(columns={
            'user_id': 'userID',
            'product_id': 'itemID'
        })
        
        min_rating = surprise_df['rating'].min()
        max_rating = surprise_df['rating'].max()
        
        reader = Reader(rating_scale=(min_rating, max_rating))
        self.data = Dataset.load_from_df(surprise_df[['userID', 'itemID', 'rating']], reader)
        
        self.trainset, self.testset = surprise_train_test_split(self.data, test_size=test_size, random_state=42)
        return True
        
    def optimize_svd(self, n_factors_list=[80, 120, 200, 360]):
                
        logger.info("Оптимизация SVD параметров...")
        
        best_rmse = float('inf')
        best_n_factors = 50
        
        for n_factors in tqdm(n_factors_list, desc="SVD Optimization"):
            try:
                algo = SVD(n_factors=n_factors, biased=True, random_state=999, verbose=False)
                
                algo.fit(self.trainset)
                predictions = algo.test(self.testset)
                
                rmse = np.sqrt(np.mean([(pred.r_ui - pred.est) ** 2 for pred in predictions]))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_n_factors = n_factors
                    
            except Exception as e:
                logger.warning(f"Ошибка для n_factors={n_factors}: {e}")
                continue
        
        self.best_svd_params = {'n_factors': best_n_factors}
        logger.info(f"🎯 Лучшие SVD параметры: n_factors={best_n_factors}, RMSE={best_rmse:.4f}")
        
        return best_n_factors, best_rmse

    def optimize_knn_on_compressed_data(self, k_list=[20, 50, 100]):
        """KNN оптимизация на сжатом датасете"""
        logger.info("KNN оптимизация на сжатом датасете...")
        
        if self.compressed_df is None:
            self.compressed_df = self.knn_recommender.compress_dataset(
                self.rating_df, min_user_interactions=1, min_item_interactions=1
            )
        
        self.knn_recommender.create_sparse_matrix(self.compressed_df)
        
        best_rmse = float('inf')
        best_k = 50
        
        for k in tqdm(k_list, desc="KNN Optimization on compressed data"):
            try:
                self.knn_recommender.build_similarity_matrix(k=k)
                
                # Быстрая оценка на маленьком подмножестве
                sample_users = self.compressed_df['user_id'].unique()[:50]
                knn_predictions = self.knn_recommender.batch_knn_predict(sample_users, 10)
                
                if len(knn_predictions) > 0:
                    rmse = self._evaluate_knn_predictions(knn_predictions)
                    
                    if rmse < best_rmse and rmse != float('inf'):
                        best_rmse = rmse
                        best_k = k
                        
            except Exception as e:
                logger.warning(f"Ошибка для k={k}: {e}")
                continue
        
        self.best_knn_params['k'] = best_k
        logger.info(f"🎯 Лучший KNN: k={best_k}, RMSE={best_rmse:.4f}")
        return best_k, best_rmse

    def _evaluate_knn_predictions(self, knn_predictions):
        """Оценка KNN предсказаний с улучшенной обработкой"""
        if len(knn_predictions) == 0:
            return float('inf')
        
        # Используем inner join для гарантии совпадения
        merged = knn_predictions.merge(
            self.rating_df, 
            on=['user_id', 'product_id'],
            how='inner'
        )
        
        if len(merged) == 0:
            return float('inf')
        
        # Убедимся, что есть действительные значения для сравнения
        valid_data = merged.dropna(subset=['rating', 'rating_knn'])
        if len(valid_data) == 0:
            return float('inf')
        
        return np.sqrt(mean_squared_error(valid_data['rating'], valid_data['rating_knn']))

    def optimize_hybrid_weights(self):
        """Оптимизация весов гибридной модели"""
        logger.info("Оптимизация гибридных весов...")
        
        best_score = 0
        best_weights = (0.6, 0.4)  # Начинаем с равных весов
        
        for svd_weight in [0.4, 0.5, 0.6, 0.7]:
            knn_weight = 1.0 - svd_weight
            
            # Используем очень маленькое подмножество для быстрой оценки
            sample_users = self.compressed_df['user_id'].unique()[:20]
            hybrid_score = self._evaluate_hybrid_weights(sample_users, svd_weight, knn_weight)
            
            if hybrid_score > best_score:
                best_score = hybrid_score
                best_weights = (svd_weight, knn_weight)
        
        self.best_hybrid_weights = best_weights
        logger.info(f"🎯 Лучшие веса: SVD={best_weights[0]}, KNN={best_weights[1]}")
        return best_weights

    def _evaluate_hybrid_weights(self, user_ids, svd_weight, knn_weight):
        """Оценка гибридных весов"""
        _, svd_algo = self.train_final_models()
        knn_predictions = self.knn_recommender.batch_knn_predict(user_ids, 10)
        
        scores = []
        
        for user_id in user_ids:
            try:
                user_inner_id = svd_algo.trainset.to_inner_uid(str(user_id))
                user_bias = svd_algo.bu[user_inner_id]
                user_factors = svd_algo.pu[user_inner_id]
                svd_ratings = svd_algo.bi + user_bias + np.dot(svd_algo.qi, user_factors)
                
                user_knn = knn_predictions[knn_predictions['user_id'] == user_id]
                
                hybrid_scores = []
                for item_inner_id, svd_rating in enumerate(svd_ratings):
                    try:
                        item_id = int(svd_algo.trainset.to_raw_iid(item_inner_id))
                        knn_rating = 0
                        
                        knn_match = user_knn[user_knn['product_id'] == item_id]
                        if not knn_match.empty:
                            knn_rating = knn_match['rating_knn'].values[0]
                        
                        hybrid_rating = (svd_weight * svd_rating + 
                                       knn_weight * knn_rating if knn_rating > 0 else svd_rating)
                        hybrid_scores.append(hybrid_rating)
                        
                    except:
                        continue
                
                if hybrid_scores:
                    scores.append(max(hybrid_scores))
                    
            except:
                continue
        
        return np.mean(scores) if scores else 0

    def train_final_models(self):
                  
        logger.info("Обучение финальной SVD модели...")
        
        full_trainset = self.data.build_full_trainset()
        
        svd_algo = SVD(n_factors=self.best_svd_params['n_factors'], biased=True, 
                      random_state=999, verbose=False)
        svd_algo.fit(full_trainset)
        
        return None, svd_algo
    
    def create_simple_submission(self, n_recommendations=10):
        """Создание простого submission файла на основе рейтингов"""
        logger.info("Создание submission файла...")
        
        # Просто сортируем по рейтингу и группируем по пользователям
        submission_df = (
            self.rating_df
            .sort_values(['user_id', 'rating'], ascending=[True, False])
            .groupby('user_id')
            .head(n_recommendations)
            .groupby('user_id')['product_id']
            .apply(lambda x: ' '.join(map(str, x)))
            .reset_index()
            .rename(columns={'product_id': 'product_id'})
        )
        
        submission_df.to_csv('submission.csv', index=False)
        logger.info(f"✅ Submission файл создан: {len(submission_df)} пользователей")
        
        return True

class OurDataWinnerV20:
    def __init__(self):
        self.rating_system = AdvancedRatingSystem()
        self.rating_df = None
        self.model = None
        self.user_ids = None
        self.product_ids = None
        self.user_id_to_idx = {}
        self.product_to_idx = {}
        self.data_files = {}
        self.column_mapping = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.recommender_optimizer = None
        logger.info(f"Используемое устройство: {self.device}")
    
    def auto_detect_data_files(self):
        logger.info("Автопоиск файлов данных...")
        
        possible_files = {
            'transactions': [
                'transactions.csv'
            ]
        }
        
        detected_files = {}
        
        for data_type, filenames in possible_files.items():
            for filename in filenames:
                if os.path.exists(filename):
                    detected_files[data_type] = filename
                    logger.info(f"Найден файл {data_type}: {filename}")
                    break
                data_path = f"data/{filename}"
                if os.path.exists(data_path):
                    detected_files[data_type] = data_path
                    logger.info(f"Найден файл {data_type}: {data_path}")
                    break
        
        self.data_files = detected_files
        return detected_files
    
    def detect_column_mapping(self, file_path):
        logger.info(f"Определение соответствия колонок для {file_path}...")
        
        try:
            sample = pd.read_csv(file_path, nrows=1000)
            columns = sample.columns.tolist()
            
            column_patterns = {
                'user_id': ['user_id', 'userid', 'user', 'customer_id', 'customerid', 'customer', 'uid'],
                'product_id': ['product_id', 'productid', 'product', 'item_id', 'itemid', 'item', 'pid'],
                'order_id': ['order_id', 'orderid', 'order', 'basket_id', 'basketid', 'order_number'],
                'order_number': ['order_number', 'ordernumber', 'order_seq', 'sequence', 'order_sequence'],
                'reordered': ['reordered', 'reorder', 'repeat', 'repurchased'],
                'add_to_cart_order': ['add_to_cart_order', 'cart_order', 'add_order'],
                'order_dow': ['order_dow', 'day_of_week', 'dow', 'weekday'],
                'order_hour_of_day': ['order_hour_of_day', 'hour_of_day', 'hour', 'order_hour']
            }
            
            mapping = {}
            for standard_name, patterns in column_patterns.items():
                for pattern in patterns:
                    if pattern in columns:
                        mapping[standard_name] = pattern
                        logger.info(f"Сопоставлено {standard_name} -> {pattern}")
                        break
            
            if 'user_id' not in mapping and len(columns) > 0:
                mapping['user_id'] = columns[0]
            if 'product_id' not in mapping and len(columns) > 1:
                mapping['product_id'] = columns[1]
            
            self.column_mapping = mapping
            logger.info(f"Финальное соответствие колонок: {mapping}")
            return mapping
            
        except Exception as e:
            logger.error(f"Ошибка при определении соответствия колонок: {e}")
            return {}
    
    def load_and_preprocess_data(self):
        logger.info("🚀 Загрузка и предобработка данных...")
        
        if not self.data_files.get('transactions'):
            logger.error("Данные транзакций не найдены!")
            return False
        
        try:
            self.column_mapping = self.detect_column_mapping(self.data_files['transactions'])
            
            if not self.column_mapping:
                logger.error("Не удалось определить соответствие колонок!")
                return False
            
            usecols = list(self.column_mapping.values())
            dtype = {}
            
            for std_col, actual_col in self.column_mapping.items():
                if std_col in ['user_id', 'product_id', 'order_id']:
                    dtype[actual_col] = 'int32'
                elif std_col in ['order_number', 'add_to_cart_order', 'order_dow', 'order_hour_of_day']:
                    dtype[actual_col] = 'int16'
                elif std_col == 'reordered':
                    dtype[actual_col] = 'int8'
            
            logger.info("Загрузка данных...")
            transactions = pd.read_csv(
                self.data_files['transactions'],
                usecols=usecols,
                dtype=dtype
            )
            
            reverse_mapping = {v: k for k, v in self.column_mapping.items()}
            transactions = transactions.rename(columns=reverse_mapping)
            
            logger.info(f"✅ Загружено {len(transactions):,} транзакций")
            
            transactions = self.optimize_data_types(transactions)
            transactions = self.add_temporal_features(transactions)
            self.create_mappings(transactions)
            self.rating_df = self.rating_system.calculate_ratings(transactions)
            
            os.makedirs('processed_data', exist_ok=True)
            transactions.to_parquet('processed_data/transactions_processed.parquet', index=False)
            self.rating_df.to_parquet('processed_data/ratings_dataset.parquet', index=False)
            
            logger.info("✅ Данные загружены и обработаны")
            logger.info(f"✅ Создан рейтинговый датасет: {len(self.rating_df)} записей")
            
            self._show_rating_stats()
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def optimize_data_types(self, df):
        dtype_optimization = {
            'user_id': 'int32',
            'product_id': 'int32', 
            'order_id': 'int32',
            'order_number': 'int16',
            'order_dow': 'int8',
            'order_hour_of_day': 'int8',
            'reordered': 'int8',
            'add_to_cart_order': 'int16'
        }
        
        for col, target_dtype in dtype_optimization.items():
            if col in df.columns:
                df[col] = df[col].astype(target_dtype)
        
        return df
    
    def add_temporal_features(self, df):
        logger.info("Добавление временных фич...")
        
        try:
            if 'order_number' in df.columns:
                df = df.sort_values(['user_id', 'order_number'])
                df['time_since_last_order'] = df.groupby('user_id')['order_number'].diff().fillna(0)
            
            if 'order_id' in df.columns:
                order_sizes = df.groupby(['user_id', 'order_id']).size().reset_index(name='order_size')
                df = df.merge(order_sizes, on=['user_id', 'order_id'], how='left')
                
                user_avg_order = df.groupby('user_id')['order_size'].mean().reset_index(name='avg_user_order_size')
                df = df.merge(user_avg_order, on='user_id', how='left')
                df['order_size_ratio'] = df['order_size'] / df['avg_user_order_size'].replace(0, 1)
            
            logger.info("✅ Временные фичи добавлены")
            return df
            
        except Exception as e:
            logger.warning(f"Не удалось добавить все временные фичи: {e}")
            return df
    
    def create_mappings(self, df):
        logger.info("Создание маппингов...")
        
        self.user_ids = df['user_id'].unique()
        self.product_ids = df['product_id'].unique()
        
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.product_to_idx = {pid: idx for idx, pid in enumerate(self.product_ids)}
        
        logger.info(f"✅ Маппинги созданы: {len(self.user_ids)} пользователей, {len(self.product_ids)} товаров")
    
    def _show_rating_stats(self):
        if self.rating_df is not None:
            logger.info("Статистика рейтингов:")
            logger.info(f"  Всего записей: {len(self.rating_df):,}")
            logger.info(f"  Уникальных пользователей: {self.rating_df['user_id'].nunique()}")
            logger.info(f"  Уникальных товаров: {self.rating_df['product_id'].nunique()}")
            logger.info(f"  Средний рейтинг: {self.rating_df['rating'].mean():.3f}")
            logger.info(f"  Минимальный рейтинг: {self.rating_df['rating'].min():.3f}")
            logger.info(f"  Максимальный рейтинг: {self.rating_df['rating'].max():.3f}")
            logger.info(f"  Медианный рейтинг: {self.rating_df['rating'].median():.3f}")
    
    def optimize_recommendations(self, n_recommendations=10):
        logger.info("🎯 Быстрая гибридная оптимизация SVD + KNN...")
        
        if self.rating_df is None:
            logger.error("Рейтинговый датасет не загружен!")
            return False
        
        self.recommender_optimizer = RecommenderOptimizer(self.rating_df)
        
        # Быстрая подготовка
        self.recommender_optimizer.prepare_fast_optimization()
        
       
        # Оптимизация SVD
        best_n_factors, svd_rmse = self.recommender_optimizer.optimize_svd()
        self.recommender_optimizer.best_svd_params = {'n_factors': best_n_factors}
        
        # Быстрая оптимизация KNN
        best_k, knn_rmse = self.recommender_optimizer.optimize_knn_on_compressed_data()
        
        # Оптимизация гибридных весов
        best_weights = self.recommender_optimizer.optimize_hybrid_weights()
        
        logger.info(f"✅ Гибридная оптимизация завершена!")
        logger.info(f"   SVD: n_factors={best_n_factors}, RMSE={svd_rmse:.4f}")
        logger.info(f"   KNN: k={best_k}, RMSE={knn_rmse:.4f}")
        logger.info(f"   Веса: SVD={best_weights[0]}, KNN={best_weights[1]}")
            
        
        
        return True
    
    def create_submission_file(self, n_recommendations=10):
        logger.info("Создание submission файла...")
        
        # Используем простой метод для создания submission файла
        if self.recommender_optimizer is not None:
            success = self.recommender_optimizer.create_simple_submission(n_recommendations)
        else:
            # Создаем простой submission на основе рейтингов
            submission_df = (
                self.rating_df
                .sort_values(['user_id', 'rating'], ascending=[True, False])
                .groupby('user_id')
                .head(n_recommendations)
                .groupby('user_id')['product_id']
                .apply(lambda x: ' '.join(map(str, x)))
                .reset_index()
                .rename(columns={'product_id': 'products'})
            )
            
            submission_df.to_csv('submission.csv', index=False)
            success = True
        
        if success:
            logger.info("✅ Submission файл успешно создан!")
            return True
        else:
            logger.error("❌ Ошибка при создании submission файла!")
            return False


def main():
    logger.info("🚀 Запуск OurDataWinnerV20...")
    
    recommender = OurDataWinnerV20()
    
    # Автопоиск файлов
    recommender.auto_detect_data_files()
    
    # Загрузка и предобработка данных
    if not recommender.load_and_preprocess_data():
        logger.error("Не удалось загрузить данные!")
        return
    
    # Быстрая оптимизация
    if not recommender.optimize_recommendations():
        logger.error("Не удалось оптимизировать рекомендации!")
        return
    
    # Создание submission файла
    if not recommender.create_submission_file():
        logger.error("Не удалось создать submission файл!")
        return
    
    logger.info("🎉 Программа успешно завершена!")

if __name__ == "__main__":
    main()