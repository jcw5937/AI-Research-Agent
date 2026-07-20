import nodes


class TestNormalizeArxivCategories:
    def test_non_list_input_returns_empty(self):
        assert nodes._normalize_arxiv_categories(None) == []
        assert nodes._normalize_arxiv_categories("cs.AI") == []
        assert nodes._normalize_arxiv_categories(42) == []
        assert nodes._normalize_arxiv_categories({"cat": "cs.AI"}) == []

    def test_filters_non_string_and_empty_items(self):
        raw = ["cs.AI", 123, None, "", "   ", "cs.LG"]
        assert nodes._normalize_arxiv_categories(raw) == ["cs.AI", "cs.LG"]

    def test_valid_list_is_stripped_and_preserved(self):
        raw = ["  cs.AI ", "cs.CL", " q-bio.NC"]
        assert nodes._normalize_arxiv_categories(raw) == ["cs.AI", "cs.CL", "q-bio.NC"]

    def test_empty_list_stays_empty(self):
        assert nodes._normalize_arxiv_categories([]) == []


class TestNormalizeLayer:
    def test_missing_arxiv_categories_defaults_to_empty(self):
        layer = nodes._normalize_layer(
            {"layer_name": "L", "rationale": "r", "search_type": "both"}
        )
        assert layer["arxiv_categories"] == []

    def test_malformed_arxiv_categories_defaults_to_empty(self):
        layer = nodes._normalize_layer(
            {"layer_name": "L", "search_type": "both", "arxiv_categories": "cs.AI"}
        )
        assert layer["arxiv_categories"] == []

    def test_invalid_search_type_falls_back_to_both(self):
        layer = nodes._normalize_layer({"layer_name": "L", "search_type": "bogus"})
        assert layer["search_type"] == "both"

    def test_missing_search_type_defaults_to_both(self):
        layer = nodes._normalize_layer({"layer_name": "L"})
        assert layer["search_type"] == "both"

    def test_full_valid_layer_passes_through(self):
        raw = {
            "layer_name": "State Machine Theory",
            "rationale": "Foundational theory layer.",
            "search_type": "academic",
            "arxiv_categories": ["cs.LG", "cs.FL"],
        }
        layer = nodes._normalize_layer(raw)
        assert layer == {
            "layer_name": "State Machine Theory",
            "rationale": "Foundational theory layer.",
            "search_type": "academic",
            "arxiv_categories": ["cs.LG", "cs.FL"],
        }
