# monitoring/logger.py

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_production_logger(name="lc-8_credit_scoring_api", log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:  # Avoid duplicates
        return logger
    
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "predictions.log"),
        maxBytes=10_000_000,  # 10 MB per file
        backupCount=5,        # Keep 5 rotated files
    )
            
    console_handler = logging.StreamHandler()

    # Both handlers use the same format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger