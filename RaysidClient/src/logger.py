import logging

logger = logging.getLogger("RaysidClient")
logger.setLevel(logging.INFO)

logging.basicConfig(
    format="%(name)s - %(levelname)s: %(message)s"
)