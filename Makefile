.PHONY: build run stop logs shell login clean

# Check if requirements have changed and rebuild if needed
REQUIREMENTS_HASH := $(shell md5sum requirements.txt 2>/dev/null || echo "")

# Docker compose files
COMPOSE_FILES := docker-compose.yml

# Docker compose command with files
DOCKER_COMPOSE := docker-compose -f $(COMPOSE_FILES)

# Service name
SERVICE := userbot

build:
	@echo "Building Docker image..."
	@$(DOCKER_COMPOSE) build

run: check-env
	@echo "Starting userbot..."
	@$(DOCKER_COMPOSE) up -d
	@echo "Userbot is running in the background."

stop:
	@echo "Stopping userbot..."
	@$(DOCKER_COMPOSE) down
	@echo "Userbot stopped."

logs:
	@$(DOCKER_COMPOSE) logs -f $(SERVICE)

shell:
	@$(DOCKER_COMPOSE) exec $(SERVICE) /bin/bash

# Login to create a new session file (only needed once)
login: check-env
	@echo "Starting interactive login session..."
	@echo "This will create or update the session file."
	@$(DOCKER_COMPOSE) run --rm -it $(SERVICE) python -m src.login

clean:
	@echo "Cleaning Docker resources..."
	@$(DOCKER_COMPOSE) down -v --rmi local
	@echo "Clean completed."

check-env:
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found!"; \
		echo "Please create a .env file from .env.example"; \
		exit 1; \
	fi

# Check if requirements have changed and rebuild if needed
requirements-changed:
	@current_hash=$$(md5sum requirements.txt 2>/dev/null || echo ""); \
	if [ "$(REQUIREMENTS_HASH)" != "$$current_hash" ]; then \
		echo "Requirements have changed. Rebuilding..."; \
		$(MAKE) build; \
	else \
		echo "Requirements unchanged."; \
	fi

# Default action when just running make
all: check-env requirements-changed run