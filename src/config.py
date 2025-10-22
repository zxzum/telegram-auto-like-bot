import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Telegram API credentials
API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')
PHONE_NUMBER = os.getenv('PHONE_NUMBER')  # Добавлен номер телефона

# Target chat ID to monitor
CHAT_ID = int(os.getenv('CHAT_ID'))
TOPIC_ID = int(os.getenv('TOPIC_ID'))

# Session name
SESSION_NAME = os.getenv('SESSION_NAME', 'userbot')

# Validate required configuration
if not API_ID or not API_HASH or not CHAT_ID or not PHONE_NUMBER:
    raise ValueError("Required environment variables are missing. "
                    "Please check your .env file.")