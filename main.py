import asyncio
import logging
import random
from telethon import TelegramClient, events
from telethon.tl.functions.messages import SendReactionRequest
from telethon.tl.types import ReactionEmoji
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Настройки логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Переменные окружения
API_ID = int(os.getenv('TELEGRAM_API_ID'))
API_HASH = os.getenv('TELEGRAM_API_HASH')
CHAT_ID = int(os.getenv('TELEGRAM_CHAT_ID'))
PHONE_NUMBER = os.getenv('TELEGRAM_PHONE')

# ID пользователей, на чьи сообщения реагировать (через запятую)
ALLOWED_USER_IDS = [int(uid.strip()) for uid in os.getenv('ALLOWED_USER_IDS', '').split(',') if uid.strip()]

# Ключевые слова для фильтрации
KEYWORDS = [
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
    'практикум'
]

# Реакции (эмодзи)
REACTIONS = ['👍', '❤️', '🐬']

# Создаём директорию для сессий если её нет
sessions_dir = Path('/app/sessions')
sessions_dir.mkdir(exist_ok=True)

session_path = str(sessions_dir / 'user_bot_session')

# Создаём клиент с корректным путём
client = TelegramClient(session_path, API_ID, API_HASH)


def contains_keyword(text):
    """Проверяет наличие ключевых слов в тексте"""
    if not text:
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in KEYWORDS)


@client.on(events.NewMessage(chats=CHAT_ID))
async def handle_new_message(event):
    """Обработчик новых сообщений в чате"""
    try:
        message = event.message

        # Пропускаем свои сообщения
        if message.out:
            return

        # Пропускаем сообщения от недопустимых пользователей
        if ALLOWED_USER_IDS and message.sender_id not in ALLOWED_USER_IDS:
            logger.debug(f"Сообщение от пользователя {message.sender_id} - пропущено (не в списке)")
            return

        # Проверяем наличие ключевых слов
        if not contains_keyword(message.text):
            logger.debug(f"Сообщение {message.id} не содержит ключевых слов")
            return

        logger.info(f"✨ Обнаружен пост от пользователя {message.sender_id} с ключевым словом! ID: {message.id}")

        # Выбираем случайную реакцию
        reaction = random.choice(REACTIONS)

        # Ставим реакцию
        await client(SendReactionRequest(
            peer=CHAT_ID,
            msg_id=message.id,
            reaction=[ReactionEmoji(emoticon=reaction)]
        ))
        logger.info(f"✅ Реакция {reaction} поставлена на пост {message.id}")

    except Exception as e:
        logger.error(f"❌ Ошибка при обработке сообщения: {e}")


async def main():
    """Главная функция"""
    try:
        logger.info("🚀 Запуск Telegram User Bot...")

        await client.start(phone=PHONE_NUMBER)

        logger.info("✅ Бот успешно запущен!")
        logger.info(f"📱 Мониторю чат ID: {CHAT_ID}")

        if ALLOWED_USER_IDS:
            logger.info(f"👤 Реагирую на сообщения от пользователей: {ALLOWED_USER_IDS}")
        else:
            logger.warning("⚠️  Список допустимых пользователей пуст! Реакции не будут ставиться.")

        logger.info(f"🔑 Ключевые слова: {', '.join(KEYWORDS)}")
        logger.info(f"😊 Случайные реакции: {', '.join(REACTIONS)}")
        logger.info("🔔 Жду новые сообщения...")

        await client.run_until_disconnected()

    except Exception as e:
        logger.error(f"❌ Ошибка при запуске: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(main())