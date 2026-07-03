"""
Network diagnostics: check MLB.com reachability, proxy configuration.
"""

import urllib.request
import socket
import logging
from typing import Optional

from src.config import AppConfig

logger = logging.getLogger(__name__)


def check_url_reachable(url: str, timeout: int = 10) -> bool:
    """Test if a URL is reachable via HTTP HEAD request."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception as e:
        logger.debug("URL check failed for %s: %s", url, e)
        return False


def check_mlb_reachable(config: AppConfig, timeout: int = 10) -> bool:
    """Test if MLB.com is reachable."""
    return check_url_reachable(config.network.mlb_test_url, timeout)


def diagnose_network(config: AppConfig) -> dict:
    """Full network diagnosis. Returns a report dict."""
    mlb_ok = check_mlb_reachable(config)

    diagnosis = {
        "mlb_reachable": mlb_ok,
        "proxy_configured": bool(config.proxy_url),
        "recommended_action": "",
    }

    if mlb_ok:
        diagnosis["recommended_action"] = "Network OK — MLB.com is reachable."
    elif config.proxy_url:
        diagnosis["recommended_action"] = (
            f"MLB.com not directly reachable. "
            f"Will try proxy: {config.proxy_url}"
        )
    else:
        diagnosis["recommended_action"] = (
            "MLB.com is NOT reachable and no proxy configured. "
            "Set PROXY_URL in .env if you have a local proxy (Clash/V2Ray). "
            "The pipeline will still attempt the download but may fail."
        )

    return diagnosis


def configure_proxy(proxy_url: Optional[str]) -> dict:
    """Build yt-dlp compatible proxy dict."""
    if not proxy_url:
        return {}
    return {"http": proxy_url, "https": proxy_url}
