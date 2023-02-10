import logging
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp'+os.sep+__name__ + '.log')

from logging.handlers import RotatingFileHandler
rotating_file_handler = RotatingFileHandler(filepath, maxBytes=100*2024, backupCount=3)
rotating_file_handler.setFormatter(formatter)
rotating_file_handler.setLevel(logging.DEBUG)

logger.addHandler(rotating_file_handler)
