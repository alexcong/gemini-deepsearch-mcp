# Make an alias for all to help, so that when make is called without arguments, it prints the help message.
.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests dev

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path, defaulting to all tests.
TEST_FILE ?= tests/

# Run unit tests using pytest, excluding tests marked as 'trio'.
# Allows specifying a single test file via TEST_FILE variable (e.g., make test TEST_FILE=tests/test_app.py).
test:
	uv run --with-editable . pytest $(TEST_FILE) -k "not trio"

# Run unit tests in watch mode using ptw (pytest-watch).
# Automatically re-runs tests when files change and updates snapshots.
test_watch:
	uv run --with-editable . ptw --snapshot-update --now . -- -vv tests/

# Run unit tests with profiling and generate an SVG output of the profile.
test_profile:
	uv run --with-editable . pytest -vv tests/ --profile-svg

# Run extended tests, typically longer-running or more comprehensive tests.
# Uses the --only-extended flag (assuming a custom pytest marker or similar).
extended_tests:
	uv run --with-editable . pytest --only-extended $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files to be processed by linters/formatters.
PYTHON_FILES=src/
# Define the mypy cache directory.
MYPY_CACHE=.mypy_cache

# Define phony targets for combined linting and formatting operations.
# `lint format: PYTHON_FILES=.` sets PYTHON_FILES for both lint and format if called as `make lint format`.
lint format: PYTHON_FILES=.
# `lint_diff format_diff: PYTHON_FILES=$(...)` sets PYTHON_FILES to only git-diffed files for lint_diff and format_diff.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
# Lint only the main package source files.
lint_package: PYTHON_FILES=src
# Lint only the test files.
lint_tests: PYTHON_FILES=tests
# Use a separate mypy cache for test linting.
lint_tests: MYPY_CACHE=.mypy_cache_test

# Common recipe for various linting targets.
# Runs ruff check, ruff format (in diff mode), ruff import sorting check, and mypy.
lint lint_diff lint_package lint_tests:
	uv run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

# Common recipe for formatting targets.
# Runs ruff formatter and ruff import sorting with auto-fix.
format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)

# Check for spelling errors using codespell.
spell_check:
	codespell --toml pyproject.toml

# Fix spelling errors automatically using codespell.
spell_fix:
	codespell --toml pyproject.toml -w

######################
# HELP
######################

# Display a help message with common make targets.
help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'test_mcp                     - test MCP stdio server (simple)'
	@echo 'test_mcp_full                - test full MCP stdio protocol'
	@echo 'dev                          - start LangGraph development server'
	@echo 'local                        - start MCP stdio server'

# Start the LangGraph development server.
dev:
	@echo "Starting development server..."
	@uv run langgraph dev

# Start the local MCP (Model Context Protocol) server using stdio.
local:
	@echo "Starting local MCP server..."
	@uv run python main.py

# Run a simple test script for the MCP stdio server.
test_mcp:
	@echo "Testing MCP stdio server..."
	@uv run python tests/test_simple_mcp.py

# Inspect the local MCP server using the MCP inspector tool.
# This typically involves running the server and having the inspector connect to it.
inspect:
	@echo "Inspecting local MCP server..."
	@npx @modelcontextprotocol/inspector uv --directory `pwd` run python main.py
