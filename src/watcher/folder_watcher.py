"""
Folder watcher: poll-based file stability detection.

Watches for video download completion by polling file size until stable.
Simpler than full Watchdog; sufficient for sequential pipelines.
"""

import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def wait_for_file_stable(file_path: Path, timeout: float = 300.0,
                         poll_interval: float = 2.0) -> bool:
    """Poll until file exists and size stabilizes.

    Args:
        file_path: Path to the downloading file
        timeout: Max seconds to wait
        poll_interval: Check every N seconds

    Returns:
        True if file stabilized, False on timeout
    """
    start = time.time()
    last_size = -1
    stable_count = 0

    while time.time() - start < timeout:
        if file_path.exists():
            current_size = file_path.stat().st_size
            if current_size > 0 and current_size == last_size:
                stable_count += 1
                if stable_count >= 2:  # Two consecutive stable checks
                    logger.info("File stable: %s (%d bytes)", file_path, current_size)
                    return True
            else:
                stable_count = 0
            last_size = current_size
        time.sleep(poll_interval)

    logger.error("Timeout waiting for %s (%.0fs)", file_path, timeout)
    return False
