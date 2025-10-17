"""Professional Telegram User Bot with Pyrogram - Optimized"""
import asyncio
import random
import logging
from pathlib import Path
from typing import Optional

from pyrogram import Client, filters, idle
from pyrogram.errors import FloodWait, Unauthorized

from config import Config
from utils.logger import setup_logger
from utils.safety import RateLimiter, RetryManager

logger = setup_logger('TelegramBot', Config.LOG_FILE, Config.LOG_LEVEL)

try:
    Config.validate()
except ValueError as e:
    logger.error(f"❌ Config error: {e}")
    exit(1)

rate_limiter = RateLimiter(Config.RATE_LIMIT_DELAY)
retry_manager = RetryManager(Config.MAX_RETRIES, Config.RETRY_DELAY)

Path(Config.SESSION_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(Config.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

app = Client(
    name=Config.SESSION_PATH,
    api_id=Config.API_ID,
    api_hash=Config.API_HASH,
    phone_number=Config.PHONE,
    workdir=str(Path(Config.SESSION_PATH).parent),
)


def contains_keyword(text: Optional[str]) -> bool:
    """Check if text contains any keyword"""
    if not text:
        return False
    return any(keyword in text.lower() for keyword in Config.KEYWORDS)


async def send_reaction(message_id: int, emoji: str) -> bool:
    """Send reaction to a message with safety checks"""
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
        return True

    except FloodWait as e:
        logger.warning(f"⚠️  Rate limit! Waiting {e.value}s...")
        await asyncio.sleep(e.value + 1)
        return False
    except Unauthorized:
        logger.error("❌ Session expired!")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to send reaction: {e}")
        return False


@app.on_message(filters.group & filters.incoming & filters.chat(Config.CHAT_ID))
async def handle_message(client: Client, message):
    """Only listen to messages in target chat and topic"""
    try:
        # Skip own messages
        if message.from_user and message.from_user.is_self:
            return

        # Skip if not in target topic
        if hasattr(message, 'topic_id') and message.topic_id != Config.TOPIC_ID:
            return

        # Check user ID
        sender_id = message.from_user.id if message.from_user else None
        if not sender_id or sender_id not in Config.ALLOWED_USER_IDS:
            return

        # Check keywords
        message_text = message.text or message.caption
        if not contains_keyword(message_text):
            return

        logger.info(f"✨ New message: {message_text[:50]}...")
        emoji = random.choice(Config.REACTIONS)
        await send_reaction(message.id, emoji)

    except Exception as e:
        logger.error(f"❌ Error handling message: {e}", exc_info=True)


async def health_check():
    """Periodic health check - only log errors"""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            me = await app.get_me()
            logger.debug(f"✅ Health check passed: {me.first_name} online")
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")


async def start_bot():
    """Start the bot"""
    try:
        logger.info("🚀 Starting Bot (Pyrogram)...")
        logger.info(f"   Chat: {Config.CHAT_ID}")
        logger.info(f"   Topic: {Config.TOPIC_ID}")
        logger.info(f"   Users: {Config.ALLOWED_USER_IDS}")
        logger.info(f"   Keywords: {len(Config.KEYWORDS)}")
        logger.info(f"   Reactions: {', '.join(Config.REACTIONS)}")
        logger.info(f"   Rate limit: {Config.RATE_LIMIT_DELAY}s")

        await app.start()
        me = await app.get_me()
        logger.info(f"✅ Bot started as {me.first_name}")
        logger.info("🔔 Listening for messages...")

        asyncio.create_task(health_check())
        await idle()

    except Unauthorized:
        logger.error("❌ Session expired! Deleting session...")
        session_file = Path(f"{Config.SESSION_PATH}.session")
        if session_file.exists():
            session_file.unlink()
        exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        exit(1)


def main():
    """Main entry point"""
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        logger.info("⏹️  Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        exit(1)


if __name__ == '__main__':
    main()