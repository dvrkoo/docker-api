import logging
import logging.handlers
import os
import sys

# Configure rotating file handler
rotating = logging.handlers.RotatingFileHandler(
    os.getenv("LOG_FILE", "app.log"),
    maxBytes=1024 * 1024,  # 1MB per file
    backupCount=3,  # Keep 3 backup files
    encoding='utf-8'
)

# Configure stdout handler for console output
stdout = logging.StreamHandler(sys.stdout)

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    handlers=[rotating, stdout],
)

# Get logger instance
logger = logging.getLogger(__name__)

# Log initial setup message
logger.info('Logging system initialized')
