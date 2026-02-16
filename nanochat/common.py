"""
Common utilities for nanochat.
"""

import os
import re
import logging

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool ANSI Shadow font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
███╗   ██╗ █████╗ ███╗   ██╗ ██████╗  ██████╗██╗  ██╗ █████╗ ████████╗     ██╗ █████╗ ██╗  ██╗
████╗  ██║██╔══██╗████╗  ██║██╔═══██╗██╔════╝██║  ██║██╔══██╗╚══██╔══╝     ██║██╔══██╗╚██╗██╔╝
██╔██╗ ██║███████║██╔██╗ ██║██║   ██║██║     ███████║███████║   ██║        ██║███████║ ╚███╔╝ 
██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║██║     ██╔══██║██╔══██║   ██║   ██   ██║██╔══██║ ██╔██╗ 
██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝╚██████╗██║  ██║██║  ██║   ██║██╗╚█████╔╝██║  ██║██╔╝ ██╗
╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝╚═╝ ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
                                                                                              
"""
    print0(banner)

