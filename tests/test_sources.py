import datetime
from contextlib import nullcontext
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from agent.fetch import Paper, fetch_all
from agent.sources import _request_json, fetch_alphaxiv

CUTOFF = datetime.datetime(2026, 11, 9, tzinfo=datetime.timezone.utc)
FIXTURE = Path(__file__).parent / "fixtures" / "alphaxiv_loopfill.json"


def alpha_paper(identifier, date="2026-11-14T00:00:00Z", **kwargs):
    return {
        "universal_paper_id": identifier,
        "title": "Efficient speculative decoding for language models " + identifier,
        "abstract": "Drafting and verification for faster inference.",
        "authors": ["A Researcher"],
        "publication_date": date,
        "first_publication_date": date,
        **kwargs,
    }


class AlphaXivTests(unittest.TestCase):
    def fetch(self, pages, page_size=100):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"papers": pages[int(request.url.params["pageNum"])]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client, patch(
            "agent.sources.httpx.Client", return_value=nullcontext(client)
        ), patch("agent.sources._cutoff", return_value=CUTOFF), patch(
            "agent.sources.PAGE_SIZE", page_size
        ):
            return fetch_alphaxiv(lookback_hours=168), requests

    def test_live_loopfill_fixture_keeps_exclusive_slug_and_metadata(self):
        # Public feed response captured 2026-11-16 for the user's acceptance case.
        fixture = json.loads(FIXTURE.read_text())
        papers, requests = self.fetch([[fixture]])
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        slug = "2611.loopfill-speculative-decoding-looped-transformers"
        self.assertEqual(paper.id, "alphaxiv:" + slug)
        self.assertEqual(paper.url, "https://www.alphaxiv.org/abs/" + slug)
        self.assertIn("multi-token suffix drafts", paper.abstract)
        self.assertEqual(paper.authors, fixture["authors"])
        self.assertEqual(paper.external_ids, {})
        self.assertEqual(requests[0].url.params["sort"], "Recent")
        self.assertIn("cs.CL", json.loads(requests[0].url.params["topics"]))

    def test_paginates_deduplicates_and_stops_at_old_page(self):
        shared = alpha_paper("2611.12345v2")
        pages = [
            [shared, alpha_paper("2611.exclusive")],
            [shared, alpha_paper("2611.another")],
            [alpha_paper("2610.old", "2026-10-20T00:00:00Z")],
        ]
        papers, requests = self.fetch(pages, page_size=2)
        self.assertEqual(len(papers), 3)
        self.assertEqual(papers[0].id, "arxiv:2611.12345")
        self.assertEqual(len(requests), 3)

    def test_skips_blogs_old_revisions_and_unknown_dates(self):
        papers, _ = self.fetch([[
            alpha_paper("2611.blog", external_blog={"body_blob_id": "blog"}),
            alpha_paper("2611.link", external_link={"cover_blob_id": "link"}),
            alpha_paper("2610.old", first_publication_date="2026-10-01T00:00:00Z"),
            alpha_paper("2611.undated", date="invalid"),
        ]])
        self.assertEqual(papers, [])

    def test_schema_change_does_not_look_like_an_empty_feed(self):
        with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"results": []})
        )) as client, patch("agent.sources.httpx.Client", return_value=nullcontext(client)):
            with self.assertRaisesRegex(ValueError, "papers"):
                fetch_alphaxiv()

    def test_unrelated_category_papers_do_not_expand_the_scoring_queue(self):
        unrelated = alpha_paper("2611.unrelated", title="A taxonomy of ancient pottery", abstract="Pottery from the Bronze Age.")
        relevant = alpha_paper("2611.relevant")
        papers, _ = self.fetch([[unrelated, relevant]])
        self.assertEqual([paper.source_id for paper in papers], ["2611.relevant"])


class SourceIntegrationTests(unittest.TestCase):
    def test_a_title_and_doi_bridge_merges_two_prior_records(self):
        first = Paper("arxiv:2611.12345", "An earlier title for the same research paper", "", [], "", "arxiv", "2026-11-14")
        first.external_ids = {"ArXiv": "2611.12345", "DOI": "10.1234/bridge"}
        second = Paper("semantic_scholar:local", "A revised title for the same research paper", "", [], "", "semantic_scholar", "2026-11-14")
        bridge = Paper("doi:10.1234/bridge", second.title, "", [], "", "alphaxiv", "2026-11-14")
        with patch("agent.fetch.fetch_arxiv", return_value=[first]), patch(
            "agent.fetch.fetch_semantic_scholar", return_value=[second]
        ), patch("agent.fetch.fetch_huggingface", return_value=[]), patch(
            "agent.sources.fetch_alphaxiv", return_value=[bridge]
        ):
            self.assertEqual(fetch_all(), [first])
        self.assertIn(second.id, first.identity_aliases)
        self.assertIn(bridge.id, first.identity_aliases)

    def test_alphaxiv_receives_backfill_window_and_merges_title_aliases(self):
        title = "LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers"
        existing = Paper("arxiv:2611.12345", title.upper(), "abstract", [], "https://example.com", "arxiv", "2026-11-14")
        alpha = Paper("alphaxiv:2611.loopfill", title, "abstract", [], "https://example.com", "alphaxiv", "2026-11-14")
        with patch("agent.fetch.fetch_arxiv", return_value=[existing]), patch(
            "agent.fetch.fetch_semantic_scholar", return_value=[]
        ), patch("agent.fetch.fetch_huggingface", return_value=[]), patch(
            "agent.sources.fetch_alphaxiv", return_value=[alpha]
        ) as fetch_alpha:
            papers = fetch_all(lookback_hours=720)
        self.assertEqual(papers, [existing])
        self.assertIn(alpha.id, existing.identity_aliases)
        fetch_alpha.assert_called_once_with(lookback_hours=720)

    def test_optional_source_outage_keeps_other_results(self):
        existing = Paper("arxiv:2611.12345", "Existing result", "", [], "", "arxiv", "2026-11-14")
        with patch("agent.fetch.fetch_arxiv", return_value=[existing]), patch(
            "agent.fetch.fetch_semantic_scholar", return_value=[]
        ), patch("agent.fetch.fetch_huggingface", return_value=[]), patch(
            "agent.sources.fetch_alphaxiv", side_effect=ValueError("schema changed")
        ):
            self.assertEqual(fetch_all(), [existing])

    def test_transient_requests_retry_but_authentication_does_not(self):
        attempts = []

        def respond(request):
            attempts.append(request)
            return httpx.Response(429, headers={"Retry-After": "0"}) if len(attempts) == 1 else httpx.Response(200, json={"papers": []})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client, patch("agent.sources.time.sleep"):
            self.assertEqual(_request_json(client, "https://example.com", {}), {"papers": []})
        self.assertEqual(len(attempts), 2)
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401))) as client, patch("agent.sources.time.sleep") as sleep:
            with self.assertRaises(httpx.HTTPStatusError):
                _request_json(client, "https://example.com", {})
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
