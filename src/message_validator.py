import logging
from pathlib import Path
from typing import Tuple, Dict
import re
import joblib

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 10

# 🎯 МИНИМАЛЬНЫЕ ПРАВИЛА - только самые важные индикаторы

# Слова которые ВСЕГДА означают actionable=1
STRONG_ACTIONABLE_KEYWORDS = [
    r'\b(лабораторн|типовик|типовая|коллоквиум|контрольн|экзамен|зачет|курсовая|практик|семинар|досдач|расчет)\b',
]

# Слова которые ВСЕГДА означают actionable=0
STRONG_NON_ACTIONABLE_KEYWORDS = [
    r'\b(откроется|открыли|регистрация|запись|информация|уведомл|напомин|список|перенос|отмен)\b',
]

# Паттерны дат (очень важно!)
DATE_PATTERNS = [
    r'\d{1,2}[.\-/]\d{1,2}',  # 24.10, 24-10, 24/10
    r'\d{1,2}:\d{1,2}',  # 24:10 (время)
    r'\d{1,2}\s*(?:янв|фев|мар|апр|май|июн|июл|авг|сен|окт|ноя|дек)',  # 24 октября
    r'(?:завтра|послезавтра|сегодня)',  # временные слова
]

# Шумовые паттерны (НЕ actionable)
NOISE_PATTERNS = [
    r'^[а-яё]{2,4}$',  # Очень короткие слова
    r'^[ахе]+$',  # ааааа, ехехе
    r'(ха+|ху+|хе+)[\s\.]*$',  # смех в конце
]


class MLValidator:
    """ML классификатор с гибридным подходом"""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.vectorizer = None
        self.is_trained = False

        self.load_model()

    def load_model(self):
        """Загрузить обученную модель"""
        model_path = self.model_dir / 'model.pkl'
        vectorizer_path = self.model_dir / 'vectorizer.pkl'

        if model_path.exists() and vectorizer_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.is_trained = True
                logger.info(f"✅ ML модель загружена")
            except Exception as e:
                logger.error(f"❌ Ошибка при загрузке модели: {e}")
                self.is_trained = False
        else:
            logger.error(f"❌ Модель не найдена в {self.model_dir}")
            self.is_trained = False

    def is_noise(self, text: str) -> bool:
        """Проверить не бессмысленный ли текст"""
        text_lower = text.lower().strip()

        for pattern in NOISE_PATTERNS:
            if re.search(pattern, text_lower):
                return True

        return False

    def predict(self, text: str) -> Tuple[bool, float, str]:
        """
        Предсказание с гибридным подходом
        Returns: (should_react, confidence, reason)
        """
        if not self.is_trained:
            return False, 0.0, "ML модель не загружена"

        text_lower = text.lower()

        # 1️⃣ ФИЛЬТРЫ
        if len(text.strip()) < MIN_TEXT_LENGTH:
            return False, 0.0, f"Текст слишком коротко ({len(text)} < {MIN_TEXT_LENGTH})"

        if self.is_noise(text):
            return False, 0.0, "Обнаружен шум (бессмысленный текст)"

        # 2️⃣ СИЛЬНЫЕ ПРАВИЛА - они важнее ML

        # Проверка на сильные NON-ACTIONABLE слова
        for pattern in STRONG_NON_ACTIONABLE_KEYWORDS:
            if re.search(pattern, text_lower):
                logger.debug(f"⚠️  Сильное правило NON-ACTIONABLE: {pattern}")
                return False, 0.1, f"Найдено слово '{pattern}'"

        # Проверка на сильные ACTIONABLE слова + дата
        has_assignment = False
        assignment_found = None
        for pattern in STRONG_ACTIONABLE_KEYWORDS:
            if re.search(pattern, text_lower):
                has_assignment = True
                assignment_found = pattern
                break

        has_date = False
        date_found = None
        for pattern in DATE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                has_date = True
                date_found = match.group(0)
                break

        # Если есть задание + дата = ОЧЕНЬ вероятно actionable
        if has_assignment and has_date:
            logger.debug(f"✅ Сильное правило ACTIONABLE: задание + дата")
            return True, 0.95, f"Найдено задание + дата ({date_found})"

        # Если только задание - нужна ML для проверки
        if has_assignment:
            logger.debug(f"📊 Есть задание, используем ML для проверки")
            try:
                X = self.vectorizer.transform([text_lower])
                prediction = self.model.predict(X)[0]
                probabilities = self.model.predict_proba(X)[0]
                confidence = probabilities[1]

                if confidence > 0.6:
                    return True, confidence, "ML подтвердило (задание есть)"
                else:
                    return False, confidence, "ML сомневается (несмотря на задание)"
            except Exception as e:
                logger.error(f"❌ Ошибка ML: {e}")
                return True, 0.7, "Ошибка ML но есть задание"

        # 3️⃣ ЕСЛИ НЕТ СИЛЬНЫХ ПРАВИЛ - используем ML
        try:
            X = self.vectorizer.transform([text_lower])
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]

            if prediction == 1:
                confidence = probabilities[1]
                reason = f"ML решение (confidence: {confidence * 100:.1f}%)"
                return True, confidence, reason
            else:
                confidence = probabilities[0]
                reason = f"ML решение (confidence: {confidence * 100:.1f}%)"
                return False, confidence, reason

        except Exception as e:
            logger.error(f"❌ Ошибка ML: {e}")
            return False, 0.0, f"Ошибка: {e}"


class MessageValidator:
    """Валидатор с гибридным подходом"""

    def __init__(self):
        self.ml_validator = MLValidator()

        if not self.ml_validator.is_trained:
            raise RuntimeError("❌ ML модель не загружена!")

    def calculate_message_score(self, text: str) -> Tuple[bool, Dict]:
        """Оценка сообщения"""
        if not text or len(text.strip()) < 2:
            return False, {'error': 'Пустое', 'confidence': 0.0, 'reason': 'Пустой текст'}

        should_react, confidence, reason = self.ml_validator.predict(text)

        score_details = {
            'confidence': confidence,
            'prediction': 'actionable' if should_react else 'non-actionable',
            'reason': reason,
        }

        if should_react:
            logger.info(f"✅ ACTIONABLE ({confidence * 100:.1f}%): {text[:60]} | {reason}")
        else:
            logger.debug(f"❌ NON-ACTIONABLE ({confidence * 100:.1f}%): {text[:60]} | {reason}")

        return should_react, score_details

    def should_react(self, text: str) -> bool:
        """Простой интерфейс"""
        should_react, _ = self.calculate_message_score(text)
        return should_react