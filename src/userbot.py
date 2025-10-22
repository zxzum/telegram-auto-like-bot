import asyncio
import random
import logging
import re
from telethon import TelegramClient
from telethon import events
from src import config

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Reactions to use
REACTIONS = ['👍', '❤️', '🐬']

# Flood control settings
MESSAGE_INTERVAL = 2  # Minimum time between actions (seconds)
last_action_time = 0

# Patterns for assignment types (лабораторная, типовик, коллоквиум, и т.д.)
ASSIGNMENT_TYPES = [
    r'лабораторн',  # лабораторная, лабораторные, лаба, лаб
    r'типовик',  # типовик, типовая работа
    r'типов',  # типовая
    r'расчет',  # расчетная работа
    r'коллоквиум',  # коллоквиум
    r'контрольн',  # контрольная
    r'экзамен',  # экзамен
    r'зачет',  # зачет
]

# Patterns that indicate "opening registration" (which we should IGNORE)
IGNORE_PATTERNS = [
    r'откроется.*запись',  # откроется запись
    r'откроется.*регистрация',  # откроется регистрация
    r'открыли.*запись',  # открыли запись
    r'открыли.*регистрацию',  # открыли регистрацию
]

# Date pattern to check if message contains a date
DATE_PATTERN = r'\d{1,2}[.\-/]\d{1,2}'  # Matches dates like 12.09, 12-09, 12/09


class MessageValidator:
    """Validates if a message should receive a reaction"""

    @staticmethod
    def should_react(text: str) -> bool:
        """
        Determines if the message should receive a reaction

        Returns True if:
        - Message contains an assignment type keyword
        - Message contains a date
        - Message does NOT contain "opening registration" patterns
        """

        if not text:
            return False

        text_lower = text.lower()

        # Check if message contains "opening registration" pattern - if yes, ignore it
        for ignore_pattern in IGNORE_PATTERNS:
            if re.search(ignore_pattern, text_lower):
                logger.info(f"Ignoring message - matches ignore pattern: {ignore_pattern}")
                return False

        # Check if message contains an assignment type
        has_assignment = False
        for assignment_type in ASSIGNMENT_TYPES:
            if re.search(assignment_type, text_lower):
                has_assignment = True
                logger.info(f"Found assignment type: {assignment_type}")
                break

        if not has_assignment:
            logger.debug("Message does not contain assignment type keywords")
            return False

        # Check if message contains a date
        if not re.search(DATE_PATTERN, text):
            logger.debug("Message does not contain a date")
            return False

        logger.info("Message validation passed - will react")
        return True


class UserBot:
    def __init__(self):
        # Initialize the client
        self.client = TelegramClient(
            config.SESSION_NAME,
            config.API_ID,
            config.API_HASH
        )

    async def is_rate_limited(self):
        """Flood control to prevent spam detection"""
        global last_action_time
        current_time = asyncio.get_event_loop().time()

        if current_time - last_action_time < MESSAGE_INTERVAL:
            wait_time = MESSAGE_INTERVAL - (current_time - last_action_time)
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)

        last_action_time = asyncio.get_event_loop().time()
        return False

    async def add_random_reaction(self, event):
        """Add a random reaction to a message with delay"""
        try:
            # Validate if we should react to this message
            message_text = event.message.text or ""

            if not MessageValidator.should_react(message_text):
                logger.debug(f"Skipping message: {message_text[:100]}")
                return

            # Add random delay (1-2 seconds) for natural behavior
            delay = random.uniform(1, 2)
            logger.info(f"Waiting {delay:.2f} seconds before reaction...")
            await asyncio.sleep(delay)

            # Check rate limiting
            if await self.is_rate_limited():
                return

            # Choose random reaction
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
                logger.info(f"✓ Added reaction {reaction} to message {event.message.id}")

            except Exception as e:
                logger.debug(f"Reaction {reaction} failed, trying alternative: {e}")
                try:
                    alt_reaction = '👍'
                    reaction_obj = ReactionEmoji(emoticon=alt_reaction)
                    await self.client(SendReactionRequest(
                        peer=event.chat_id,
                        msg_id=event.message.id,
                        reaction=[reaction_obj]
                    ))
                    logger.info(f"✓ Added alternative reaction {alt_reaction} to message {event.message.id}")
                except Exception as e2:
                    logger.error(f"✗ Failed to add alternative reaction: {e2}")

        except Exception as e:
            logger.error(f"✗ Failed to process message: {e}")

    async def start(self):
        """Start the userbot"""
        # Connect to Telegram
        await self.client.start()

        # Register event handler for new messages in the target chat
        @self.client.on(events.NewMessage(chats=config.CHAT_ID))
        async def handler(event):
            message = event.message

            # Для форумов проверяем reply_to
            topic_id = None
            if message.reply_to and hasattr(message.reply_to, 'forum_topic') and message.reply_to.forum_topic:
                topic_id = message.reply_to.reply_to_msg_id
                logger.info(f"✓ Found forum topic: {topic_id}")

            logger.info(f"Topic ID: {topic_id}, Expected: {config.TOPIC_ID}")

            # Если это сообщение в другой теме - пропускаем
            if topic_id and topic_id != config.TOPIC_ID:
                logger.info(f"❌ Skipping - wrong topic {topic_id}")
                return

            # Если topic_id None и мы ожидаем конкретную тему - пропускаем
            if topic_id is None and config.TOPIC_ID is not None:
                logger.info(f"❌ Skipping - no forum topic found")
                return

            logger.info(f"✅ Processing message...")
            await self.add_random_reaction(event)

        logger.info(f"Userbot started, monitoring chat {config.CHAT_ID}, topic {config.TOPIC_ID}")
        logger.info("Patterns for reactions:")
        logger.info(f"  - Assignment types: {', '.join(ASSIGNMENT_TYPES)}")
        logger.info(f"  - Ignore patterns: {', '.join(IGNORE_PATTERNS)}")
        logger.info(f"  - Date pattern: {DATE_PATTERN}")

        # Run the client until disconnected
        await self.client.run_until_disconnected()