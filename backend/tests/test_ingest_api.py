import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ["VECTORSTORE_BACKEND"] = "memory"

from backend.app.main import app  # noqa: E402
from backend.common.vectorstore import InMemoryVectorStore  # noqa: E402


@pytest.fixture(autouse=True)
def reset_memory_store():
    InMemoryVectorStore.reset()
    yield
    InMemoryVectorStore.reset()


def test_ingest_endpoint_returns_inserted_count():
    html = """
    <html>
      <body>
        <h1>Python Documentation</h1>
        <p>Python makes it easy to create REST APIs.</p>
      </body>
    </html>
    """

    class DummyResponse:
        def __init__(self, text: str):
            self.text = text
            self.status_code = 200

        def raise_for_status(self):
            return None

    client = TestClient(app)
    with patch("backend.ingest.loaders.requests.get", return_value=DummyResponse(html)):
        res = client.post("/ingest", json={"urls": ["https://docs.python.org/3/"]})
    assert res.status_code == 200
    payload = res.json()
    assert payload["inserted"] == 1


def test_query_filters_by_sources(monkeypatch):
    client = TestClient(app)

    class DummyResponse:
        def __init__(self, text: str):
            self.text = text
            self.status_code = 200

        def raise_for_status(self):
            return None

    def fake_get(url, *args, **kwargs):
        if "whatsnew" in url:
            return DummyResponse(
                "<html><body><p>Python 3.14 introduces new pattern matching goodies.</p></body></html>"
            )
        return DummyResponse("<html><body><p>Unrelated page about something else.</p></body></html>")

    with patch("backend.ingest.loaders.requests.get", side_effect=fake_get):
        res1 = client.post("/ingest", json={"urls": ["https://docs.python.org/3/whatsnew/3.14.html"]})
        res2 = client.post("/ingest", json={"urls": ["https://example.com/other"]})
    assert res1.status_code == 200
    assert res2.status_code == 200

    class DummyLLMResponse:
        def __init__(self, content: str):
            self.content = content

    from backend.app.main import llm as app_llm

    monkeypatch.setattr(
        app_llm.__class__,
        "invoke",
        lambda self, *args, **kwargs: DummyLLMResponse("ok"),
        raising=False,
    )

    payload = {
        "query": "What is new in Python 3.14?",
        "top_k": 5,
        "with_answer": True,
        "sources_filter": ["https://docs.python.org/3/whatsnew/3.14.html"],
    }
    res = client.post("/query", json=payload)
    assert res.status_code == 200

    data = res.json()
    assert data["results"], "Expected at least one result from filtered sources"
    assert all(r["metadata"].get("source") == "https://docs.python.org/3/whatsnew/3.14.html" for r in data["results"])
