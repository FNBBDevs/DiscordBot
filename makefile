VENV := .venv
PYTHON := $(VENV)/bin/python
POETRY := poetry
NAME = BOT
RED = \033[0;32m
FAILED = [1m[91mFAILED[0m[0m
PASSED = [1m[92m[92mPASSED[0m[0m
COLOR_RESET = \033[0m
SHELL=/bin/bash

POUND = \#

colors: ## show all the colors
	@echo $(FAILED)
	@echo $(PASSED)


.PHONY: help
help:
	@echo ""
	@echo "  lint        run the code linters"
	@echo "  format      reformat code"
	@echo "  test        run all the tests"
	@echo ""

.PHONY: install
install:
	pyproject.toml poetry.lock
	$(POETRY) install

.PHONY: lint
lint:
	-$(POETRY) run isort --profile=black --lines-after-imports=2 --check-only $(NAME)
	-$(POETRY) run black $(NAME) --check
	-$(POETRY) run flake8 --ignore=W503,E501 $(NAME)
	-$(POETRY) run mypy $(NAME) --ignore-missing-imports
	-$(POETRY) run bandit -r $(NAME) -s B608

.PHONY: format
format:
	$(POETRY) run isort --profile=black --lines-after-imports=2 $(NAME)
	$(POETRY) run black $(NAME)

.PHONY: build
build:
	$(POETRY) build

.PHONY: run
run:
	python $(NAME)/run.py

.PHONY: fix-yt
fix-yt:
	pip install --upgrade --force-reinstall "git+https://github.com/ytdl-org/youtube-dl.git"

.PHONY: test
test:
	$(POETRY) run pytest ./tests/ --cov-report term-missing --cov-fail-under 100 --cov $(NAME)