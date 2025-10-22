import os
import sys
import asyncio
from telethon import TelegramClient
from telethon.errors.rpcerrorlist import UpdateAppToLoginError, PhoneNumberInvalidError, SessionPasswordNeededError
from dotenv import load_dotenv

# Импортируем переменные из .env
load_dotenv()

API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')
PHONE_NUMBER = os.getenv('PHONE_NUMBER')
SESSION_NAME = os.getenv('SESSION_NAME', '/app/sessions/userbot')


async def login():
    """Create a new session file with 2FA support"""
    try:
        client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

        print(f"Attempting to connect with phone number: {PHONE_NUMBER}")
        await client.connect()

        if await client.is_user_authorized():
            print("You are already authorized, session file exists and is valid.")
        else:
            print("Sending authentication code request...")
            await client.send_code_request(PHONE_NUMBER)

            code = input("Enter the code you received: ")

            try:
                await client.sign_in(PHONE_NUMBER, code)
            except SessionPasswordNeededError:
                # Двухфакторная аутентификация включена
                print("\nTwo-factor authentication is enabled.")
                password = input("Please enter your 2FA password: ")
                await client.sign_in(password=password)

            print("Session saved successfully!")

        await client.disconnect()
        return 0

    except UpdateAppToLoginError:
        print("\nError: Your API application needs to be updated.")
        print("Please go to https://my.telegram.org/apps and create new API keys")
        print("Then update your .env file with the new API_ID and API_HASH")
        return 1

    except PhoneNumberInvalidError:
        print(f"\nError: The phone number '{PHONE_NUMBER}' is invalid.")
        print("Please check your phone number format in .env file. It should include country code (e.g., +79123456789)")
        return 1

    except Exception as e:
        print(f"\nError during authentication: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(login()))