VENV := .venv
PYTHON := $(VENV)/bin/python
POETRY := poetry
NAME = BOT
FAILED = [1m[91mFAILED[0m[0m
PASSED = [1m[92m[92mPASSED[0m[0m
define newline


endef

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
.SILENT: lint
lint:
	-$(POETRY) run pylint --reports yes $(NAME)
	-$(POETRY) run black $(NAME) --check
	-$(POETRY) run flake8 --ignore=W503,E501 $(NAME)
	-$(POETRY) run isort --profile=black --lines-after-imports=2 --check-only $(NAME)
	-$(POETRY) run mypy $(NAME) --ignore-missing-imports
	-$(POETRY) run bandit -r $(NAME) -s B608

.PHONY: lint-silent
.SILENT: lint-silent
lint-silent:
	-($(POETRY) run isort --profile=black --lines-after-imports=2 --check-only $(NAME) || @echo isort:..................................................... $(FAILED))
	-$(POETRY) run black $(NAME) --check > lint.txt || @echo black:..................................................... $(FAILED)
	-$(POETRY) run flake8 --ignore=W503,E501 $(NAME) > lint.txt || @echo flake8:..................................................... $(FAILED)
	-$(POETRY) run mypy $(NAME) --ignore-missing-imports > lint.txt || @echo mypy:..................................................... $(FAILED)
	-$(POETRY) run bandit -r $(NAME) -s B608 > lint.txt || @echo bandit:..................................................... $(FAILED)

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