"""Slack incoming webhook notifier for GitHub Actions and bots."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


def slack_post(
    text: str,
    *,
    webhook_url: str | None = None,
    blocks: list[dict[str, Any]] | None = None,
) -> None:
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        print("No Slack webhook URL configured; skipping.", file=sys.stderr)
        return

    payload: dict[str, Any] = {"text": text}
    if blocks:
        payload["blocks"] = blocks

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read()
    except urllib.error.URLError as exc:
        print(f"Slack notification failed: {exc}", file=sys.stderr)
        raise


def slack_digest(title: str, body: str, *, webhook_url: str | None = None) -> None:
    slack_post(
        title,
        webhook_url=webhook_url,
        blocks=[
            {"type": "header", "text": {"type": "plain_text", "text": title[:150]}},
            {"type": "section", "text": {"type": "mrkdwn", "text": body[:3000]}},
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Post a Slack notification")
    parser.add_argument("--text", required=True)
    parser.add_argument("--webhook-env", default="SLACK_WEBHOOK_URL")
    parser.add_argument("--digest", action="store_true")
    parser.add_argument("--title", default="Notification")
    args = parser.parse_args()

    webhook = os.environ.get(args.webhook_env)
    if args.digest:
        slack_digest(args.title, args.text, webhook_url=webhook)
    else:
        slack_post(args.text, webhook_url=webhook)


if __name__ == "__main__":
    main()
