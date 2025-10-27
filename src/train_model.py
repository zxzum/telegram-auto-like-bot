#!/usr/bin/env python3
"""
Обучение ML модели - ФИНАЛЬНАЯ ВЕРСИЯ
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path('models')
DATASET_PATH = Path('dataset.csv')


def main():
    logger.info("=" * 80)
    logger.info("🚀 ОБУЧЕНИЕ ML МОДЕЛИ - ФИНАЛЬНАЯ ВЕРСИЯ")
    logger.info("=" * 80)

    if not DATASET_PATH.exists():
        logger.error(f"❌ Датасет не найден: {DATASET_PATH}")
        return

    # Загрузка
    logger.info("\n📂 Загрузка датасета...")
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"   Строк: {len(df)}")
    logger.info(f"   Actionable=1: {(df['actionable'] == 1).sum()}")
    logger.info(f"   Actionable=0: {(df['actionable'] == 0).sum()}")

    X = df['text'].str.lower()
    y = df['actionable'].values

    # Split
    logger.info("\n✂️  Разделение данных...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    # Векторизация - МАКСИМАЛЬНО АГРЕССИВНАЯ
    logger.info("\n📝 Векторизация (TF-IDF, максимум признаков)...")
    vectorizer = TfidfVectorizer(
        max_features=2000,  # БОЛЬШЕ признаков
        ngram_range=(1, 3),  # Триграммы тоже!
        min_df=1,  # Даже одиночные слова
        max_df=0.99,
        lowercase=True,
        token_pattern=r'(?u)\b\w+\b',  # Все слова
        strip_accents='unicode',
        analyzer='word'
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    logger.info(f"   Shape: {X_train_vec.shape}")
    logger.info(f"   Features: {len(vectorizer.get_feature_names_out())}")

    # Обучение - МАКСИМУМ ПАРАМЕТРОВ
    logger.info("\n🧠 Обучение RandomForest (максимум деревьев)...")
    model = RandomForestClassifier(
        n_estimators=500,  # МНОГО деревьев
        max_depth=25,  # ГЛУБЖЕ
        min_samples_split=2,  # Меньше ограничений
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample',  # ЛУЧШЕ балансировка
        bootstrap=True,
        oob_score=True
    )

    model.fit(X_train_vec, y_train)

    # Оценка
    logger.info("\n📈 РЕЗУЛЬТАТЫ:")
    train_score = model.score(X_train_vec, y_train)
    test_score = model.score(X_test_vec, y_test)
    oob_score = model.oob_score_

    logger.info(f"   ✅ Train accuracy: {train_score * 100:.2f}%")
    logger.info(f"   ✅ Test accuracy: {test_score * 100:.2f}%")
    logger.info(f"   ✅ OOB score: {oob_score * 100:.2f}%")

    y_pred = model.predict(X_test_vec)

    logger.info("\n" + classification_report(y_test, y_pred,
                                             target_names=['Non-actionable', 'Actionable']))

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    # Важные признаки
    logger.info("\n📊 Топ 30 признаков:")
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = model.feature_importances_

    top_indices = np.argsort(feature_importance)[-30:][::-1]
    for i, idx in enumerate(top_indices, 1):
        importance = feature_importance[idx]
        logger.info(f"   {i:2d}. {feature_names[idx]:30} → {importance:.4f}")

    # Сохранение
    logger.info(f"\n💾 Сохранение модели...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / 'model.pkl')
    joblib.dump(vectorizer, MODEL_DIR / 'vectorizer.pkl')
    logger.info(f"   ✅ model.pkl сохранена ({MODEL_DIR / 'model.pkl'})")
    logger.info(f"   ✅ vectorizer.pkl сохранена ({MODEL_DIR / 'vectorizer.pkl'})")

    # Тест на примерах
    logger.info("\n🧪 ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ:")
    test_texts = [
        ("Лабораторная работа №2 - дата: 13.02", 1, "норм задание с датой"),
        ("Типовой расчет №1: ДМ Дата: 24:10", 1, "типовик с датой"),
        ("Коллоквиум завтра в 15:30", 1, "коллоквиум с временем"),
        ("сыншалавы", 0, "бессмысленный текст"),
        ("хахахаха", 0, "смех"),
        ("дура", 0, "оскорбление"),
        ("Завтра в 15:00 будет запись", 0, "информация о записи"),
        ("Откроется запись на экзамены", 0, "откроется - не actionable"),
        ("перенос по расписанию", 0, "перенос - не actionable"),
        ("Досдача Коллок №3 и №4 - оп | дата - 11.3", 1, "досдача с датой"),
        ("ахахаха", 0, "смех 2"),
        ("блять", 0, "ругань"),
    ]

    correct = 0
    for text, expected, description in test_texts:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        is_correct = pred == expected
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        confidence = proba[1]
        pred_name = "✓" if pred == 1 else "✗"

        logger.info(f"   {status} [{pred_name}] {text[:40]:40} → {confidence * 100:5.1f}% | {description}")

    logger.info(f"\n   🎯 Точность на примерах: {correct}/{len(test_texts)} ({correct * 100 // len(test_texts)}%)")

    logger.info("\n" + "=" * 80)
    logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()