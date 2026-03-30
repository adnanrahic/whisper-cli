.PHONY: install install-dev build clean test test-unit test-integration smoke-test help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	pip install torch torchaudio
	pip install -e .

install-dev: ## Install project with dev dependencies
	pip install torch torchaudio
	pip install -e ".[dev]"
	pip install pyinstaller

build: ## Build standalone binary
	pyinstaller whisper-cli.spec --noconfirm
	@echo ""
	@echo "Binary created at ./dist/whisper-cli"
	@du -sh ./dist/whisper-cli

clean: ## Remove build artifacts
	rm -rf build/ dist/

test: test-unit test-integration ## Run all tests

test-unit: ## Run unit tests (fast, no model download)
	pytest tests/test_formatter.py tests/test_cli.py -v

test-integration: ## Run integration tests (downloads tiny model on first run)
	pytest tests/test_transcriber.py -v

smoke-test: ## Smoke test the built binary
	@test -f ./dist/whisper-cli || (echo "Binary not found. Run 'make build' first." && exit 1)
	./dist/whisper-cli --version
	./dist/whisper-cli --help
