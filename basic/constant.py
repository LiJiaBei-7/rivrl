import os
import logging

ROOT_PATH = os.path.join(os.environ['HOME'], 'workplace', 'disk3',
                         'wyb', 'VisualSearch_hybrid')
# ROOT_PATH="/media/daniel/disk3/daniel/VisualSearch"

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)
# logger.setLevel(logging.INFO)

