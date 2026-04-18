.PHONY: install clean train status logs

# Setup the environment and pre-commit hooks
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pre-commit install
	@mkdir -p logs models results figures

# Submit the training job to the Slurm queue
train:
	@echo "Submitting job to Slurm..."
	sbatch train_launcher.sh

# Check the status of your jobs in the queue
status:
	squeue -u jccomas

# Follow the error logs in real-time
logs:
	tail -f logs/*.err

# Clean up temporary Python files and checkpoints
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@echo "Clean up complete."
