"""Behavioral ranking checks using metadata only, without AI dependencies."""

import pytest

from app.search_service import fuse_rankings, keyword_rank


def photo(identifier, name="photo.jpg", caption="", tags=()):
    return {"id": identifier, "name": name, "caption": caption, "tags": list(tags)}


@pytest.mark.parametrize(
    "query,photos,expected_order",
    [
        (
            "海边日落",
            [
                photo("city", caption="城市夜景与霓虹"),
                photo("partial", caption="海边的清晨"),
                photo("best", caption="海边日落", tags=["海边日落"]),
            ],
            ["best", "partial"],
        ),
        (
            "BEACH SUNSET",
            [
                photo("city", caption="City lights after dark"),
                photo("partial", caption="A quiet beach"),
                photo("best", caption="Beach sunset", tags=["sunset"]),
            ],
            ["best", "partial"],
        ),
    ],
)
def test_bm25_ranks_stronger_chinese_and_english_matches_first(query, photos, expected_order):
    results = keyword_rank(photos, query)
    assert [item["id"] for item in results] == expected_order
    assert results[0]["score"] > results[1]["score"] > 0


def test_keyword_search_returns_nothing_when_no_metadata_matches():
    photos = [photo("a", caption="繁忙城市"), photo("b", caption="Mountain forest")]
    assert keyword_rank(photos, "海边日落") == []
    assert keyword_rank(photos, "beach sunset") == []
    assert keyword_rank(photos, "   ") == []
    assert keyword_rank([], "forest") == []


def test_keyword_ties_have_stable_order_independent_of_input_order():
    photos = [photo("b", caption="beach"), photo("a", caption="beach")]
    forward = keyword_rank(photos, "beach")
    reversed_input = keyword_rank(list(reversed(photos)), "beach")
    assert [item["id"] for item in forward] == ["a", "b"]
    assert forward == reversed_input


def test_rrf_promotes_agreement_between_searches_and_returns_each_photo_once():
    keyword_only = photo("keyword-only")
    semantic_only = photo("semantic-only")
    shared = photo("shared")
    results = fuse_rankings([keyword_only, shared], [semantic_only, shared])
    assert results[0]["id"] == "shared"
    assert results[0]["score"] > results[1]["score"]
    assert len(results) == len({item["id"] for item in results}) == 3
    assert "关键词" in results[0]["match_reason"] and "语义" in results[0]["match_reason"]


def test_rrf_uses_rank_positions_not_incomparable_raw_scores():
    keyword = [dict(photo("keyword-only"), score=900), dict(photo("shared"), score=100)]
    semantic = [dict(photo("semantic-only"), score=0.98), dict(photo("shared"), score=0.8)]
    original = fuse_rankings(keyword, semantic)
    rescaled = fuse_rankings(
        [{**item, "score": item["score"] / 100_000} for item in keyword],
        [{**item, "score": item["score"] * 100_000} for item in semantic],
    )
    assert original == rescaled
    assert original[0]["id"] == "shared"


def test_rrf_ties_are_stable_and_empty_sources_preserve_order():
    a, b = photo("a"), photo("b")
    assert [item["id"] for item in fuse_rankings([b], [a])] == ["a", "b"]
    assert [item["id"] for item in fuse_rankings([a], [b])] == ["a", "b"]
    assert [item["id"] for item in fuse_rankings([], [b, a])] == ["b", "a"]
    assert fuse_rankings([], []) == []
