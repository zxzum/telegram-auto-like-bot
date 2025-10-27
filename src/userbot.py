import asyncio
import random
import logging
from telethon import TelegramClient
from telethon import events
from src import config
from src.message_validator import MessageValidator

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

REACTIONS = ['👍', '❤️', '🐬']
MESSAGE_INTERVAL = 2
last_action_time = 0


class UserBot:
    def __init__(self):
        self.client = TelegramClient(
            config.SESSION_NAME,
            config.API_ID,
            config.API_HASH
        )
        self.validator = MessageValidator()

    async def is_rate_limited(self):
        """Flood control"""
        global last_action_time
        current_time = asyncio.get_event_loop().time()

        if current_time - last_action_time < MESSAGE_INTERVAL:
            wait_time = MESSAGE_INTERVAL - (current_time - last_action_time)
            logger.info(f"⏱️  Rate limiting: {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        last_action_time = asyncio.get_event_loop().time()
        return False

    async def add_random_reaction(self, event):
        """Add reaction"""
        try:
            message_text = event.message.text or ""

            should_react, score_details = self.validator.calculate_message_score(message_text)

            if not should_react:
                return

            delay = random.uniform(1, 2)
            logger.info(f"⏱️  Waiting {delay:.2f}s...")
            await asyncio.sleep(delay)

            if await self.is_rate_limited():
                return

            reaction = random.choice(REACTIONS)

            try:
                from telethon.tl.functions.messages import SendReactionRequest
                from telethon.tl.types import ReactionEmoji

                reaction_obj = ReactionEmoji(emoticon=reaction)

                await self.client(SendReactionRequest(
                    peer=event.chat_id,
                    msg_id=event.message.id,
                    reaction=[reaction_obj]
                ))
                logger.info(f"✅ Reaction {reaction} to msg {event.message.id}")

            except Exception as e:
                logger.error(f"❌ Reaction failed: {e}")
                try:
                    alt_reaction = '👍'
                    reaction_obj = ReactionEmoji(emoticon=alt_reaction)
                    await self.client(SendReactionRequest(
                        peer=event.chat_id,
                        msg_id=event.message.id,
                        reaction=[reaction_obj]
                    ))
                    logger.info(f"✅ Fallback reaction {alt_reaction}")
                except Exception as e2:
                    logger.error(f"❌ Fallback failed: {e2}")

        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)

    async def start(self):
        """Start"""
        await self.client.start()

        @self.client.on(events.NewMessage(chats=config.CHAT_ID))
        async def handler(event):
            message = event.message

            topic_id = None
            if message.reply_to and hasattr(message.reply_to, 'forum_topic') and message.reply_to.forum_topic:
                topic_id = message.reply_to.reply_to_msg_id

            if topic_id and topic_id != config.TOPIC_ID:
                return

            if topic_id is None and config.TOPIC_ID is not None:
                return

            await self.add_random_reaction(event)

        logger.info(f"🚀 Userbot started")
        logger.info(f"   Chat: {config.CHAT_ID}, Topic: {config.TOPIC_ID}")
        logger.info(f"   Mode: Hybrid (ML + Rules)")

        await self.client.run_until_disconnected()