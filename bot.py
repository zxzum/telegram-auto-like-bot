"""Professional Telegram User Bot with Pyrogram - No Telethon Parameters"""
import asyncio
import random
import logging
from pathlib import Path
from typing import Optional

from pyrogram import Client, filters
from pyrogram.errors import FloodWait, Unauthorized

from config import Config
from utils.logger import setup_logger
from utils.safety import RateLimiter, RetryManager, IdleDetector

logger = setup_logger('TelegramBot', Config.LOG_FILE, Config.LOG_LEVEL)

try:
    Config.validate()
except ValueError as e:
    logger.error(f"❌ Config error: {e}")
    exit(1)

rate_limiter = RateLimiter(Config.RATE_LIMIT_DELAY)
retry_manager = RetryManager(Config.MAX_RETRIES, Config.RETRY_DELAY)
idle_detector = IdleDetector(Config.IDLE_TIMEOUT)

Path(Config.SESSION_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(Config.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

# PYROGRAM ONLY - no Telethon parameters!
app = Client(
    name=Config.SESSION_PATH,
    api_id=Config.API_ID,
    api_hash=Config.API_HASH,
    phone_number=Config.PHONE,
    workdir=str(Path(Config.SESSION_PATH).parent),
)


def contains_keyword(text: Optional[str]) -> bool:
    if not text:
        return False
    return any(keyword in text.lower() for keyword in Config.KEYWORDS)


async def send_reaction(message_id: int, emoji: str) -> bool:
    try:
        await rate_limiter.wait()
        
        async def send_react():
            await app.send_reaction(
                chat_id=Config.CHAT_ID,
                message_id=message_id,
                emoji=emoji
            )
        
        await retry_manager.execute(send_react(), f"Reaction {emoji}")
        logger.info(f"✅ Reaction {emoji} -> message {message_id}")
        idle_detector.ping()
        return True
        
    except FloodWait as e:
        logger.warning(f"⚠️  Rate limit! Wait {e.value}s...")
        await asyncio.sleep(e.value + 1)
        return False
    except Unauthorized:
        logger.error("❌ Session expired!")
        return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


@app.on_message(filters.group & filters.incoming)
async def handle_message(client: Client, message):
    try:
        if message.from_user and message.from_user.is_self:
            return
        
        if message.chat.id != Config.CHAT_ID:
            return
        
        if hasattr(message, 'topic_id') and message.topic_id != Config.TOPIC_ID:
            return
        
        sender_id = message.from_user.id if message.from_user else None
        if not sender_id or sender_id not in Config.ALLOWED_USER_IDS:
            return
        
        message_text = message.text or message.caption
        if not contains_keyword(message_text):
            return
        
        logger.info(f"✨ Message from {sender_id}: {message_text[:50]}...")
        emoji = random.choice(Config.REACTIONS)
        await send_reaction(message.id, emoji)
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)


async def health_check():
    while True:
        try:
            await asyncio.sleep(60)
            if idle_detector.is_idle():
                logger.warning(f"⚠️  Idle {idle_detector.get_idle_time():.0f}s")
        except Exception as e:
            logger.error(f"❌ Health: {e}")


async def start_bot():
    try:
        logger.info("🚀 Starting Bot (Pyrogram)...")
        logger.info(f"Chat: {Config.CHAT_ID}, Topic: {Config.TOPIC_ID}")
        logger.info(f"Users: {Config.ALLOWED_USER_IDS}")
        logger.info(f"Keywords: {len(Config.KEYWORDS)}, Reactions: {', '.join(Config.REACTIONS)}")
        
        await app.start()
        me = await app.get_me()
        logger.info(f"✅ Started as {me.first_name}")
        
        idle_detector.ping()
        asyncio.create_task(health_check())
        
        logger.info("🔔 Listening...")
        await app.run_until_disconnected()
        
    except Unauthorized:
        logger.error("❌ Session expired!")
        session_file = Path(f"{Config.SESSION_PATH}.session")
        if session_file.exists():
            session_file.unlink()
        exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal: {e}", exc_info=True)
        exit(1)


def main():
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        logger.info("⏹️  Stopped")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()