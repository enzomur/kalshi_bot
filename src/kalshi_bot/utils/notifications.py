"""Desktop notifications for trade alerts."""

from __future__ import annotations

import subprocess

from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


def send_notification(title: str, message: str) -> bool:
    """
    Send a macOS desktop notification.

    Args:
        title: Notification title
        message: Notification body text

    Returns:
        True if notification was sent successfully, False otherwise
    """
    try:
        safe_title = title.replace('\\', '\\\\').replace('"', '\\"')
        safe_message = message.replace('\\', '\\\\').replace('"', '\\"')
        script = f'display notification "{safe_message}" with title "{safe_title}"'
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            timeout=5,
        )
        logger.debug(f"Notification sent: {title}")
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Notification timed out")
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to send notification: {e}")
        return False
    except Exception as e:
        logger.warning(f"Notification error: {e}")
        return False
