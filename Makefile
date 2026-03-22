.PHONY: install hooks clean

install:
	@echo "→ Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "→ Installing pre-commit hooks..."
	pre-commit install
	@echo ""
	@echo "⚠️  macOS only: if XGBoost fails to import, run:"
	@echo "   brew install libomp"
	@echo ""
	@echo "✓ Environment ready. Run: jupyter lab"

hooks:
	@echo "→ Running pre-commit on all files..."
	pre-commit run --all-files

clean:
	@echo "→ Removing compiled Python files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@echo "✓ Clean done."
