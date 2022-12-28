import logging
logging.basicConfig(level=logging.INFO)

from datetime import datetime
from typing import Optional


def get_curr_time_stamp():
    """
    """
    curr = datetime.now()
    return curr.strftime("%H:%M:%S")

def info(msg):
    
    # 
    time_stamp = get_curr_time_stamp()
    logging.info(f"{time_stamp} - {msg}")