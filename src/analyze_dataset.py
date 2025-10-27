#!/usr/bin/env python3
"""
Анализ датасета и проблем обучения
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = 'dataset.csv'


def analyze():
    """Анализировать датасет"""
    logger.info("=" * 80)
    logger.info("📊 АНАЛИЗ ДАТАСЕТА")
    logger.info("=" * 80)

    df = pd.read_csv(DATASET_PATH)
    logger.info(f"\n📝 Базовая информация:")
    logger.info(f"   Всего строк: {len(df)}")
    logger.info(f"   Actionable=1: {(df['actionable'] == 1).sum()}")
    logger.info(f"   Actionable=0: {(df['actionable'] == 0).sum()}")

    # Статистика по длине текста
    logger.info(f"\n📏 Длина текстов:")
    df['text_len'] = df['text'].str.len()
    logger.info(f"   Mean: {df['text_len'].mean():.1f}")
    logger.info(f"   Median: {df['text_len'].median():.1f}")
    logger.info(f"   Min: {df['text_len'].min()}")
    logger.info(f"   Max: {df['text_len'].max()}")

    # Примеры
    logger.info(f"\n📌 Примеры ACTIONABLE=1:")
    for i, text in enumerate(df[df['actionable'] == 1]['text'].head(5)):
        logger.info(f"   {i + 1}. {text[:70]}")

    logger.info(f"\n📌 Примеры ACTIONABLE=0:")
    for i, text in enumerate(df[df['actionable'] == 0]['text'].head(5)):
        logger.info(f"   {i + 1}. {text[:70]}")

    # Уникальные слова
    logger.info(f"\n🔤 Анализ слов:")
    all_texts_1 = ' '.join(df[df['actionable'] == 1]['text'].str.lower())
    all_texts_0 = ' '.join(df[df['actionable'] == 0]['text'].str.lower())

    words_1 = set(all_texts_1.split())
    words_0 = set(all_texts_0.split())

    logger.info(f"   Уникальные слова в классе 1: {len(words_1)}")
    logger.info(f"   Уникальные слова в классе 0: {len(words_0)}")
    logger.info(f"   Общие слова: {len(words_1 & words_0)}")
    logger.info(f"   Только в классе 1: {len(words_1 - words_0)}")
    logger.info(f"   Только в классе 0: {len(words_0 - words_1)}")

    # Слова специфичные для класса 1
    logger.info(f"\n🎯 Топ слова для ACTIONABLE=1:")
    words_1_only = list(words_1 - words_0)[:20]
    logger.info(f"   {words_1_only}")

    logger.info(f"\n🎯 Топ слова для ACTIONABLE=0:")
    words_0_only = list(words_0 - words_1)[:20]
    logger.info(f"   {words_0_only}")

    # Векторизация и тестирование
    logger.info(f"\n🧪 Тест векторизации:")
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(df['text'].str.lower())
    logger.info(f"   Shape: {X.shape}")
    logger.info(f"   Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}")

    # Простой тест: могут ли KNN классифицировать?
    logger.info(f"\n🔍 Тест классификации (KNN):")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, df['actionable'], test_size=0.2, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"   KNN (5-fold CV) accuracy: {scores.mean() * 100:.2f}% ± {scores.std() * 100:.2f}%")

    # Random Forest тест
    logger.info(f"\n🔍 Тест классификации (RandomForest):")
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"   RF (5-fold CV) accuracy: {scores.mean() * 100:.2f}% ± {scores.std() * 100:.2f}%")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    analyze()