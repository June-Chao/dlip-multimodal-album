"""Metadata BM25 retrieval and reciprocal-rank fusion with CLIP results."""
from collections import Counter
import math
import re


def tokens(text):
    output = re.findall(r"[a-z0-9]+", text.casefold())
    for word in re.findall(r"[\u3400-\u9fff]+", text):
        output.extend(word)
        output.extend(word[i:i + 2] for i in range(len(word) - 1))
    return output


def keyword_rank(photos, query):
    query_tokens = set(tokens(query))
    if not query_tokens or not photos:
        return []
    documents = [tokens(" ".join([p["name"], p["caption"], *p["tags"], *p["tags"]])) for p in photos]
    counters = [Counter(document) for document in documents]
    frequencies = Counter(token for counter in counters for token in counter)
    average_length = sum(map(len, documents)) / len(documents) or 1
    results = []
    for photo, document, counter in zip(photos, documents, counters):
        score = 0.0
        for token in query_tokens:
            frequency = counter[token]
            if not frequency:
                continue
            idf = math.log(1 + (len(photos) - frequencies[token] + 0.5) / (frequencies[token] + 0.5))
            score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * len(document) / average_length))
        if score > 0:
            results.append({**photo, "score": round(score, 4), "match_reason": "关键词匹配"})
    return sorted(results, key=lambda item: (-item["score"], item["name"], item["id"]))


def fuse_rankings(keyword_results, semantic_results):
    scores, photos, sources = {}, {}, {}
    for source, results in (("关键词", keyword_results), ("语义", semantic_results)):
        for rank, photo in enumerate(results, 1):
            identifier = photo["id"]
            scores[identifier] = scores.get(identifier, 0) + 1 / (60 + rank)
            photos[identifier] = photo
            sources.setdefault(identifier, []).append(source)
    return [{**photos[identifier], "score": round(scores[identifier], 6),
             "match_reason": " + ".join(sources[identifier]) + "匹配"}
            for identifier in sorted(scores, key=lambda key: (-scores[key], key))]
