import logging

logger = logging.getLogger("ianvs")
logger.setLevel(logging.INFO)

# Prevent multiple handlers from being added during repeated imports
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)
