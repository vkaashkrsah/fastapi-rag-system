"""Live CLI demonstration; requires running services and makes billable model calls."""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx


def main() -> None:
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise SystemExit("Set API_KEY to the API secret from your .env first.")
    base = os.environ.get("BASE_URL", "http://localhost:8000")
    examples = Path(__file__).parent
    with httpx.Client(base_url=base, headers={"Authorization": f"Bearer {api_key}"},
                      timeout=180) as client:
        document_ids = []
        for filename, strategy in (("handbook.txt", "fixed"), ("handbook.pdf", "paragraph")):
            with (examples / filename).open("rb") as file:
                response = client.post("/api/v1/documents", files={"file": (filename, file)},
                                       data={"strategy": strategy})
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
            document_ids.append(response.json()["document_id"])
        session_id = str(uuid4())
        # Two weeks ahead avoids past dates; a random minute reduces repeated-demo collisions.
        future = datetime.now() + timedelta(days=14)
        minute = uuid4().int % 60
        when = f"{future.date().isoformat()} at 14:{minute:02d}"
        messages = [
            "What skills are useful for the internship?",
            "Which of those relate to document search?",
            "What is the exact salary?",
            "I want to book an interview. My name is Demo Candidate, email demo@example.com.",
            when,
            "confirm",
        ]
        for message in messages:
            payload = {"session_id": session_id, "request_id": str(uuid4()),
                       "message": message, "document_ids": document_ids}
            response = client.post("/api/v1/chat", json=payload)
            response.raise_for_status()
            print(f"\nUSER: {message}")
            print(json.dumps(response.json(), indent=2))
        replay = client.post("/api/v1/chat", json=payload)
        replay.raise_for_status()
        assert replay.json() == response.json(), "Replay should match the original response"
        print("\nReplay matched the last response.")
        if response.json().get("booking") is None:
            raise SystemExit("Booking was not completed. Inspect the conversation above.")
        print("Live demo completed: documents, follow-up questions, and confirmed booking.")


if __name__ == "__main__":
    main()
