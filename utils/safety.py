"""
Safety protections for the bot
"""
import asyncio
import time
from typing import Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter to prevent spamming"""

    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.last_action_time: float = 0.0

    async def wait(self):
        """Wait before next action"""
        elapsed = time.time() - self.last_action_time
        if elapsed < self.delay:
            wait_time = self.delay - elapsed
            logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        self.last_action_time = time.time()


class RetryManager:
    """Retry mechanism with exponential backoff"""

    def __init__(self, max_retries: int = 3, base_delay: float = 5.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def execute(self, coro, operation_name: str = "operation"):
        """Execute coroutine with retries"""
        for attempt in range(1, self.max_retries + 1):
            try:
                return await coro
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(f"❌ {operation_name} failed after {self.max_retries} attempts: {e}")
                    raise

                delay = self.base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"⚠️  {operation_name} attempt {attempt}/{self.max_retries} failed: {e}\n"
                    f"   Retrying in {delay:.0f}s..."
                )
                await asyncio.sleep(delay)


class IdleDetector:
    """Detect if bot is idle too long"""

    def __init__(self, timeout: int = 300):
        self.timeout = timeout
        self.last_activity: Optional[datetime] = None

    def ping(self):
        """Update last activity time"""
        self.last_activity = datetime.now()

    def is_idle(self) -> bool:
        """Check if bot is idle"""
        if not self.last_activity:
            return False

        idle_time = (datetime.now() - self.last_activity).total_seconds()
        return idle_time > self.timeout

    def get_idle_time(self) -> float:
        """Get idle time in seconds"""
        if not self.last_activity:
            return 0.0
        return (datetime.now() - self.last_activity).total_seconds()