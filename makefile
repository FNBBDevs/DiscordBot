VENV := .venv
PYTHON := $(VENV)/bin/python
POETRY := poetry
NAME = BOT

.PHONY: install
install:
	$(POETRY) install

.PHONY: build
build:
	$(POETRY) build

.PHONY: run
run:
	poetry run python BOT/run.py

.PHONY: fix-yt
fix-yt:
	pip install --upgrade --force-reinstall "git+https://github.com/ytdl-org/youtube-dl.git"

.PHONY: test
test:
	$(POETRY) run pytest ./tests/ --cov-report term-missing --cov-fail-under 100 --cov $(NAME)