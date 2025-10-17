"""
Professional Telegram User Bot with safety protections
"""
import asyncio
import random
import logging
from pathlib import Path
from typing import Optional

from pyrogram import Client, filters
from pyrogram.errors import (
    FloodWait,
    AuthKeyUnregistered,
    SessionPasswordNeeded,
    PasswordHashInvalid
)

from config import Config
from utils.logger import setup_logger
from utils.safety import RateLimiter, RetryManager, IdleDetector

# Setup logger
logger = setup_logger('TelegramBot', Config.LOG_FILE, Config.LOG_LEVEL)

# Validate config
try:
    Config.validate()
except ValueError as e:
    logger.error(f"❌ Configuration error: {e}")
    exit(1)

# Safety components
rate_limiter = RateLimiter(Config.RATE_LIMIT_DELAY)
retry_manager = RetryManager(Config.MAX_RETRIES, Config.RETRY_DELAY)
idle_detector = IdleDetector(Config.IDLE_TIMEOUT)

# Create session directory
Path(Config.SESSION_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(Config.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

# Initialize Pyrogram client
app = Client(
    name=Config.SESSION_PATH,
    api_id=Config.API_ID,
    api_hash=Config.API_HASH,
    phone_number=Config.PHONE,
    workdir=str(Path(Config.SESSION_PATH).parent),
    sleep_threshold=Config.REQUEST_TIMEOUT,
    no_updates=False,  # Enable updates
    take_out=False,  # Don't use takeout
)


def contains_keyword(text: Optional[str]) -> bool:
    """Check if text contains any keyword"""
    if not text:
        return False

    text_lower = text.lower()
    return any(keyword in text_lower for keyword in Config.KEYWORDS)


async def send_reaction(message_id: int, emoji: str) -> bool:
    """
    Send reaction to a message with safety checks

    Args:
        message_id: Message ID
        emoji: Reaction emoji

    Returns:
        True if successful, False otherwise
    """
    try:
        # Apply rate limiting
        await rate_limiter.wait()

        # Send reaction with retry
        async def send_react():
            await app.send_reaction(
                chat_id=Config.CHAT_ID,
                message_id=message_id,
                emoji=emoji
            )

        await retry_manager.execute(send_react(), f"Send reaction {emoji}")
        logger.info(f"✅ Reaction {emoji} sent to message {message_id}")
        idle_detector.ping()
        return True

    except FloodWait as e:
        logger.warning(f"⚠️  Rate limit hit! Waiting {e.value}s...")
        await asyncio.sleep(e.value + 1)
        return False

    except AuthKeyUnregistered:
        logger.error("❌ Session expired! Need to re-authorize.")
        return False

    except Exception as e:
        logger.error(f"❌ Failed to send reaction: {e}")
        return False


@app.on_message(filters.group & filters.incoming)
async def handle_message(client: Client, message):
    """
    Handle incoming messages with full validation
    """
    try:
        # Safety: Skip own messages
        if message.from_user and message.from_user.is_self:
            return

        # Safety: Check user ID
        sender_id = message.from_user.id if message.from_user else None
        if not sender_id or sender_id not in Config.ALLOWED_USER_IDS:
            logger.debug(f"⏭️  Message from {sender_id} skipped (not in allowed users)")
            return

        # Safety: Check message text
        message_text = message.text or message.caption
        if not contains_keyword(message_text):
            logger.debug(f"⏭️  Message {message.id} has no keywords")
            return

        logger.info(
            f"✨ New message detected!\n"
            f"   From: {message.from_user.first_name} ({sender_id})\n"
            f"   ID: {message.id}\n"
            f"   Text: {message_text[:50]}..."
        )

        # Select random reaction
        emoji = random.choice(Config.REACTIONS)

        # Send reaction
        success = await send_reaction(message.id, emoji)

        if not success:
            logger.warning(f"⚠️  Reaction failed, skipping...")

    except Exception as e:
        logger.error(f"❌ Error handling message: {e}", exc_info=True)


async def health_check():
    """
    Periodic health check
    """
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute

            # Check idle status
            if idle_detector.is_idle():
                idle_time = idle_detector.get_idle_time()
                logger.warning(
                    f"⚠️  Bot is idle for {idle_time:.0f}s\n"
                    f"   This might indicate connection issues"
                )

            # Check connection
            try:
                me = await app.get_me()
                logger.debug(f"✅ Health check passed: {me.first_name} is online")
            except Exception as e:
                logger.error(f"❌ Health check failed: {e}")

        except Exception as e:
            logger.error(f"❌ Error in health check: {e}", exc_info=True)


async def start_bot():
    """Start the bot with all safety measures"""
    try:
        logger.info("🚀 Starting Telegram User Bot...")
        logger.info(f"   Chat ID: {Config.CHAT_ID}")
        logger.info(f"   Allowed users: {Config.ALLOWED_USER_IDS}")
        logger.info(f"   Keywords: {len(Config.KEYWORDS)} keywords loaded")
        logger.info(f"   Reactions: {', '.join(Config.REACTIONS)}")
        logger.info(f"   Rate limit: {Config.RATE_LIMIT_DELAY}s between reactions")

        # Start client
        await app.start()
        me = await app.get_me()
        logger.info(f"✅ Bot started successfully as {me.first_name}")

        # Update idle detector
        idle_detector.ping()

        # Start health check task
        health_task = asyncio.create_task(health_check())

        # Keep running
        logger.info("🔔 Waiting for messages...")
        await app.run_until_disconnected()

    except SessionPasswordNeeded:
        logger.error("❌ Two-factor authentication is enabled!")
        logger.error("   Please disable 2FA in Telegram settings or provide password in .env")
        exit(1)

    except PasswordHashInvalid:
        logger.error("❌ Wrong two-factor authentication password!")
        exit(1)

    except AuthKeyUnregistered:
        logger.error("❌ Session expired! Deleting session file...")
        logger.error("   Please run the bot again to re-authorize")
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