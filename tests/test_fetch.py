import unittest
from unittest.mock import Mock, patch

import httpx

from agent.fetch import Paper, _s2_request, fetch_all


def make_paper(identifier: str, source: str, title: str) -> Paper:
    return Paper(
        id=identifier,
        title=title,
        abstract="abstract",
        authors=[],
        url="https://example.com",
        source=source,
        published_date="2026-07-13",
    )


class FetchTests(unittest.TestCase):
    def test_fetch_all_deduplicates_canonical_identity_across_sources(self):
        arxiv = make_paper("arxiv:2607.01234", "arxiv", "arXiv copy")
        huggingface = make_paper(
            "arxiv:2607.01234", "huggingface", "Hugging Face copy"
        )
        semantic_scholar = make_paper("s2:other", "semantic_scholar", "Other")
        with patch("agent.fetch.fetch_arxiv", return_value=[arxiv]), patch(
            "agent.fetch.fetch_semantic_scholar", return_value=[semantic_scholar]
        ), patch("agent.fetch.fetch_huggingface", return_value=[huggingface]):
            result = fetch_all()

        self.assertEqual([item.id for item in result], ["arxiv:2607.01234", "s2:other"])
        self.assertEqual(result[0].source, "arxiv")

    def test_semantic_scholar_retries_transient_rate_limit(self):
        request = httpx.Request("GET", "https://example.com")
        client = Mock()
        client.get.side_effect = [
            httpx.Response(429, request=request, headers={"retry-after": "0"}),
            httpx.Response(200, request=request, json={"data": []}),
        ]
        with patch("agent.fetch.time.sleep"):
            self.assertEqual(_s2_request(client, {}, {}), {"data": []})
        self.assertEqual(client.get.call_count, 2)

    def test_semantic_scholar_does_not_retry_authentication_error(self):
        request = httpx.Request("GET", "https://example.com")
        client = Mock()
        client.get.return_value = httpx.Response(401, request=request)
        with self.assertRaises(httpx.HTTPStatusError):
            _s2_request(client, {}, {})
        self.assertEqual(client.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
