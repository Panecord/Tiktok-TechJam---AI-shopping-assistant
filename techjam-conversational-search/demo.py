"""Small terminal interface for demonstrating a Track 4 multi-turn session."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from starter.agent import Agent, VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an interactive Shopping Copilot demo")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--profile-tags",
        default="comfort,fit,durability",
        help="Comma-separated anonymized preference tags",
    )
    args = parser.parse_args()

    tags = [tag.strip() for tag in args.profile_tags.split(",") if tag.strip()]
    agent = Agent(Path(args.catalog))
    session_id = f"demo_{uuid.uuid4().hex}"
    agent.reset(session_id, {"preference_tags": tags})

    print(f"Shopping Copilot {VERSION} | profile tags: {', '.join(tags) or 'none'}")
    print("Describe what you want. Type 'quit' to stop.\n")
    for turn in range(1, 11):
        message = input(f"You [{turn}/10]: ").strip()
        if message.lower() in {"quit", "exit"}:
            break
        response = agent.respond(session_id, message, turn, top_k=10)
        print(f"Copilot: {response['message']}")
        for rank, item in enumerate(response["recommendations"], start=1):
            asin = item["parent_asin"]
            product = agent.products.get(asin, {})
            title = str(product.get("title") or "Untitled product")
            print(f"  {rank:>2}. {asin} | {title[:100]}")
        route = getattr(agent, "_last_route", {})
        print(
            f"  [route={route.get('intent', 'unknown')}; "
            f"ask_attribute={response['ask_attribute']}; "
            f"tokens={response['usage'].get('prompt_tokens', 0) + response['usage'].get('completion_tokens', 0)}]\n"
        )


if __name__ == "__main__":
    main()
