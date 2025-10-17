# 🤖 Telegram User Bot - Professional Edition

Профессиональный, безопасный и быстрый Telegram User Bot на **Pyrogram** с полной автоматизацией и защитами от блокировки.

## ⚡ Особенности

- ✅ **Pyrogram** - легкая, быстрая и безопасная библиотека
- ✅ **Безопасность** - защита от флуда, переполнения, блокировки
- ✅ **Rate Limiting** - контролируемая отправка реакций
- ✅ **Автоматический retry** - восстановление при сбоях
- ✅ **Health Check** - мониторинг состояния бота
- ✅ **Профессиональное логирование** - полная история действий
- ✅ **Фильтрация** - по ID пользователей и ключевым словам
- ✅ **Случайные реакции** - 👍 ❤️ 🐬
- ✅ **Docker** - готов к развёртыванию

## 📋 Требования

- Docker и Docker Compose
- Telegram аккаунт (не бот)
- API ID и API HASH
- Python 3.11+ (для локального запуска)

## 🚀 Быстрый старт

### 1. Получить API ключи

1. Перейти на https://my.telegram.org/auth/login
2. Войти со своего номера телефона
3. Перейти в "API development tools"
4. Скопировать `API ID` и `API HASH`

### 2. Получить ID чата

Отправить сообщение боту @userinfobot в целевой чат.

### 3. Получить ID пользователя

Отправить сообщение боту @userinfobot.

### 4. Настроить проект

```bash
git clone https://github.com/zxzum/telegram-auto-like-bot.git
cd telegram-auto-like-bot

cp .env.example .env
nano .env
```

**Заполнить .env:**
```env
TELEGRAM_API_ID=123456789
TELEGRAM_API_HASH=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
TELEGRAM_PHONE=+79991234567
TELEGRAM_CHAT_ID=-1002584565460
ALLOWED_USER_IDS=123456789,987654321
```

### 5. Первый запуск (авторизация)

```bash
docker-compose run --rm telegram-bot
```

Введите код подтверждения из Telegram.

### 6. Запустить в фоне

```bash
docker-compose up -d

# Проверить логи
docker-compose logs -f
```

## 📝 Переменные окружения

| Переменная | Описание | Пример |
|-----------|---------|--------|
| `TELEGRAM_API_ID` | API ID | `123456789` |
| `TELEGRAM_API_HASH` | API Hash (32 символа) | `a1b2c3d4...` |
| `TELEGRAM_PHONE` | Номер телефона | `+79991234567` |
| `TELEGRAM_CHAT_ID` | ID целевой группы | `-1002584565460` |
| `ALLOWED_USER_IDS` | ID пользователей (через запятую) | `123,456,789` |
| `LOG_LEVEL` | Уровень логирования | `INFO` |
| `RATE_LIMIT_DELAY` | Задержка между реакциями | `2.0` |
| `MAX_RETRIES` | Максимум попыток | `3` |
| `REQUEST_TIMEOUT` | Таймаут запроса | `30` |
| `IDLE_TIMEOUT` | Таймаут неактивности | `300` |

## 🛠️ Команды

### Docker команды

```bash
# Запустить в фоне
docker-compose up -d

# Остановить
docker-compose down

# Просмотр логов (в реальном времени)
docker-compose logs -f

# Просмотр последних 100 строк
docker-compose logs -f --tail=100

# Перезагрузить
docker-compose restart telegram-bot

# Пересоздать с новым образом
docker-compose up -d --force-recreate

# Удалить всё (включая данные)
docker-compose down -v
```

### SSH команды на сервере

```bash
# Обновить код
cd /path/to/telegram-auto-like-bot
git pull origin main

# Пересоздать контейнер
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Переавторизация (если сессия сломалась)
rm -rf sessions/
docker-compose run --rm telegram-bot
docker-compose up -d
```

## 🔒 Безопасность

### Что делает бот защищённым?

1. **Rate Limiting** - минимум 2 секунды между реакциями (по умолчанию)
2. **Retry Mechanism** - автоматическое восстановление при сбоях с exponential backoff
3. **FloodWait Detection** - обнаружение лимита Telegram и ожидание
4. **Health Check** - мониторинг состояния каждую минуту
5. **Session Management** - безопасное управление сессией
6. **User Filtering** - реагирует только на сообщения от конкретных пользователей
7. **Keyword Filtering** - игнорирует ненужные сообщения
8. **Timeout Protection** - обнаружение "зависания" бота

### Рекомендации

- ✅ Не устанавливайте Rate Limit ниже 1.0 второй
- ✅ Используйте Pyrogram вместо других библиотек (безопаснее)
- ✅ Не даёте никому API KEY и HASH
- ✅ Регулярно проверяйте логи
- ✅ Используйте Docker для изоляции

## 📊 Логирование

### Примеры логов

```
2025-10-17 12:34:56 - TelegramBot - INFO - 🚀 Starting Telegram User Bot...
2025-10-17 12:34:56 - TelegramBot - INFO - ✅ Bot started successfully as John Doe
2025-10-17 12:34:56 - TelegramBot - INFO - 🔔 Waiting for messages...
2025-10-17 12:35:12 - TelegramBot - INFO - ✨ New message detected!
2025-10-17 12:35:12 - TelegramBot - INFO - ✅ Reaction 👍 sent to message 123456
```

### Файлы логов

```
logs/
└── bot.log  # Все события в один файл
```

## 🚨 Решение проблем

### Бот не видит сообщения

```bash
# Проверить правильный ID чата
docker-compose logs -f

# Убедиться, что ID пользователей в ALLOWED_USER_IDS
```

### Сессия повреждена

```bash
rm -rf sessions/
docker-compose run --rm telegram-bot  # Переавторизоваться
docker-compose up -d
```

### FloodWait ошибка

- Это означает, что Telegram ограничил запросы
- Бот автоматически будет ждать и повторять
- Увеличьте `RATE_LIMIT_DELAY` в .env

### "Two-factor authentication"

- Отключите 2FA в Telegram Settings
- Или предоставьте пароль в .env

## 📖 Структура проекта

```
telegram-auto-like-bot/
├── bot.py                 # Главный файл бота
├── config.py              # Конфигурация
├── requirements.txt       # Зависимости
├── Dockerfile             # Docker образ
├── docker-compose.yml     # Docker Compose
├── .env.example           # Пример конфигурации
├── .gitignore             # Git ignore
├── utils/
│   ├── logger.py          # Логирование
│   └── safety.py          # Безопасность
└── README.md              # Этот файл
```

## 🎯 Ключевые слова (по умолчанию)

```
лабораторная, лабораторка, сдача, работа, коллоквиум,
типовик, типовая работа, экзамен, зачёт, проект,
тест, отчёт, защита, исследовательская, практикум,
контрольная, домашнее задание, дз
```

Добавить новые можно в `config.py` в список `KEYWORDS`.

## 🎲 Реакции (по умолчанию)

- 👍 Лайк
- ❤️ Сердце
- 🐬 Дельфин

Изменить можно в `config.py` в список `REACTIONS`.

## 🔄 Обновление кода

```bash
cd /path/to/telegram-auto-like-bot

# Скачать последний код
git pull origin main

# Пересоздать контейнер
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Проверить логи
docker-compose logs -f
```

## 📞 Контакты & Поддержка

Вопросы? Посмотрите логи:
```bash
docker-compose logs -f telegram-bot
```

## 📄 Лицензия

MIT

## ⚠️ Disclaimer

Этот бот использует User API Telegram. Убедитесь, что:
- Вы согласны с Terms of Service Telegram
- Не используете бота для спама или нарушения правил
- Регулярно проверяете логи
- Понимаете риски использования User API

**Используйте на свой риск!**

---

**Создано с ❤️ для профессионалов! 🚀**