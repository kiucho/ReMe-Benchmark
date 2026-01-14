import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
RESULTS_DIR = os.getenv("RESULTS_DIR") or f"{BASE_DIR}/results"

# Experiment name for organizing results (can be set via environment variable or programmatically)
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", None)
