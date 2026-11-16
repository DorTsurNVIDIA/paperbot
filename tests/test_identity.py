import unittest

from agent.identity import (
    canonical_paper_id,
    canonicalize_stored_id,
    normalize_arxiv_id,
    title_identity_key,
)


class IdentityTests(unittest.TestCase):
    def test_alphaxiv_mirrors_and_exclusive_slugs_use_distinct_namespaces(self):
        self.assertEqual(canonical_paper_id("alphaxiv", "2611.12345v2"), "arxiv:2611.12345")
        self.assertEqual(canonical_paper_id("alphaxiv", "2611.loopfill"), "alphaxiv:2611.loopfill")

    def test_full_titles_match_without_fuzzy_matching(self):
        title = "LoopFill: Pipelined Multi-Token Speculative Decoding"
        self.assertEqual(title_identity_key(title), title_identity_key(title.upper().replace("-", " ")))
        self.assertNotEqual(title_identity_key(title), title_identity_key(title + " II"))
        self.assertEqual(title_identity_key("Introduction"), "")

    def test_doi_urls_share_an_identity(self):
        self.assertEqual(
            canonical_paper_id("openreview", "local", {"DOI": "https://doi.org/10.1234/EXAMPLE"}),
            "doi:10.1234/example",
        )

    def test_arxiv_versions_share_one_identity(self):
        self.assertEqual(normalize_arxiv_id("2607.01234v3"), "2607.01234")
        self.assertEqual(
            canonical_paper_id("arxiv", "https://arxiv.org/abs/2607.01234v3"),
            "arxiv:2607.01234",
        )

    def test_huggingface_arxiv_id_uses_arxiv_namespace(self):
        self.assertEqual(
            canonical_paper_id("huggingface", "2607.01234"),
            "arxiv:2607.01234",
        )

    def test_semantic_scholar_prefers_external_arxiv_id(self):
        self.assertEqual(
            canonical_paper_id(
                "semantic_scholar",
                "s2-local-id",
                {"ArXiv": "2607.01234v2", "DOI": "10.1/example"},
            ),
            "arxiv:2607.01234",
        )

    def test_semantic_scholar_falls_back_to_doi(self):
        self.assertEqual(
            canonical_paper_id(
                "semantic_scholar", "s2-local-id", {"DOI": "10.1/EXAMPLE"}
            ),
            "doi:10.1/example",
        )

    def test_legacy_seen_ids_are_migrated(self):
        self.assertEqual(
            canonicalize_stored_id("arxiv:2607.01234v4"), "arxiv:2607.01234"
        )
        self.assertEqual(
            canonicalize_stored_id("hf:2607.01234"), "arxiv:2607.01234"
        )


if __name__ == "__main__":
    unittest.main()
