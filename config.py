"""
Configuration management for the Telegram User Bot
"""
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()


class Config:
    """Bot configuration class"""

    # Telegram API
    API_ID: int = int(os.getenv('TELEGRAM_API_ID', '0'))
    API_HASH: str = os.getenv('TELEGRAM_API_HASH', '')
    PHONE: str = os.getenv('TELEGRAM_PHONE', '')

    # Chat settings
    CHAT_ID: int = int(os.getenv('TELEGRAM_CHAT_ID', '0'))
    TOPIC_ID: int = int(os.getenv('TELEGRAM_TOPIC_ID', '0'))  # NEW: Topic/Thread ID
    ALLOWED_USER_IDS: List[int] = [
        int(uid.strip()) for uid in os.getenv('ALLOWED_USER_IDS', '').split(',')
        if uid.strip()
    ]

    # Keywords for filtering
    KEYWORDS: List[str] = [
        'лабораторная',
        'лабораторка',
        'сдача',
        'работа',
        'коллоквиум',
        'типовик',
        'типовая работа',
        'экзамен',
        'зачёт',
        'зачет',
        'проект',
        'тест',
        'отчёт',
        'отчет',
        'защита',
        'исследовательская',
        'практикум',
        'контрольная',
        'домашнее задание',
        'дз'
    ]

    # Reactions
    REACTIONS: List[str] = ['👍', '❤️', '🐬']

    # Safety settings
    RATE_LIMIT_DELAY: float = 2.0  # Seconds between reactions
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 5.0
    REQUEST_TIMEOUT: int = 30
    IDLE_TIMEOUT: int = 300  # 5 minutes

    # Session
    SESSION_PATH: str = '/app/sessions/user_bot'

    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_FILE: str = '/app/logs/bot.log'

    # Validation
    @staticmethod
    def validate():
        """Validate configuration"""
        errors = []

        if not Config.API_ID or Config.API_ID == 0:
            errors.append("TELEGRAM_API_ID is not set or invalid")

        if not Config.API_HASH:
            errors.append("TELEGRAM_API_HASH is not set")

        if not Config.PHONE:
            errors.append("TELEGRAM_PHONE is not set")

        if not Config.CHAT_ID or Config.CHAT_ID == 0:
            errors.append("TELEGRAM_CHAT_ID is not set or invalid")

        if not Config.TOPIC_ID or Config.TOPIC_ID == 0:
            errors.append("TELEGRAM_TOPIC_ID is not set or invalid (get from link like https://t.me/c/2584565460/2683)")

        if not Config.ALLOWED_USER_IDS:
            errors.append("ALLOWED_USER_IDS is not set or empty")

        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  • {e}" for e in errors))

        return True