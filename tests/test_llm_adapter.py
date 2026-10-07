"""Exercise the real OpenAI SDK adapter with HTTP responses, without billing."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from openai import OpenAI

from app.llm import OpenAILanguageModel
from app.schemas import SessionState, Source


def test_real_sdk_contract_for_embeddings_plan_and_answer(service):
    requests = []

    def transport(request):
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.1, 0.0]}],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                },
            )
        schema_name = payload["text"]["format"]["name"]
        result = (
            {
                "intent": "booking",
                "standalone_query": "",
                "name": "Vikash",
                "email": None,
                "date": None,
                "time": None,
                "confirm": False,
            }
            if schema_name == "Plan"
            else {"answer": "Python is useful.", "source_indices": [0]}
        )
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": json.dumps(result), "annotations": []}
                        ],
                    }
                ],
            },
        )

    adapter = OpenAILanguageModel(service.settings)
    adapter.client.close()
    adapter.client = OpenAI(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(transport))
    )
    try:
        assert adapter.embed(["hello"]) == [[1.0, 0.1, 0.0]]
        result = adapter.plan("I am Vikash", SessionState(), datetime.now(ZoneInfo("Asia/Kolkata")))
        assert result.name == "Vikash"
        source = Source(document_id="doc", filename="a.txt", chunk_index=0, score=1, text="Python")
        assert adapter.answer("Skills?", [source]).source_indices == [0]
        assert requests[0][1]["dimensions"] == 3
        assert requests[1][1]["store"] is False
        assert requests[1][1]["text"]["format"]["strict"] is True
    finally:
        adapter.client.close()
