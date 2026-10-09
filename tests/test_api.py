"""API integration checks that use real image files and an isolated local library.

Run with ``python -m pytest tests/test_api.py``; no model downloads are needed.
"""

from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import importlib.util
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.server import create_app


def image_bytes(color="red", size=(48, 32), image_format="PNG"):
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format=image_format)
    return output.getvalue()


def upload(client, name="photo.png", content=None):
    response = client.post(
        "/api/import",
        files={"files": (name, content or image_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert not result["errors"], result
    assert len(result["imported"]) == 1, result
    return result["imported"][0]


def isolated_app(data_dir):
    return create_app(data_dir=data_dir, model_dir=data_dir / "models")


@pytest.fixture
def client(tmp_path):
    with TestClient(isolated_app(tmp_path)) as test_client:
        yield test_client


def test_empty_library_is_usable_without_loading_ai(client):
    response = client.get("/api/library")
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == 0

    stats = client.get("/api/stats").json()
    assert stats["total"] == stats["favorites"] == stats["albums"] == 0
    assert stats["storage_bytes"] == 0

    system = client.get("/api/system")
    assert system.status_code == 200
    assert {"models", "indexes", "job", "device"} <= system.json().keys()


def test_import_handles_corruption_duplicates_and_same_names(client):
    first_bytes = image_bytes("red")
    response = client.post(
        "/api/import",
        files=[
            ("files", ("same.png", first_bytes, "image/png")),
            ("files", ("broken.png", b"not an image", "image/png")),
            ("files", ("same.png", image_bytes("blue"), "image/png")),
            ("files", ("copy.png", first_bytes, "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["imported"]) == 2
    assert len({photo["id"] for photo in result["imported"]}) == 2
    assert result["duplicates"] == ["copy.png"]
    assert [error["name"] for error in result["errors"]] == ["broken.png"]
    assert result["errors"][0]["error"]

    library = client.get("/api/library").json()
    assert library["total"] == 2
    required_fields = {
        "id", "name", "url", "thumbnail_url", "width", "height", "bytes",
        "format", "created_at", "favorite", "caption", "tags", "album_ids",
    }
    for photo in library["items"]:
        assert required_fields <= photo.keys()
        assert (photo["width"], photo["height"]) == (48, 32)
        assert photo["bytes"] > 0


@pytest.mark.parametrize("image_format,extension", [("JPEG", "jpg"), ("WEBP", "webp"), ("BMP", "bmp")])
def test_supported_image_formats_can_be_imported(client, image_format, extension):
    photo = upload(client, f"picture.{extension}", image_bytes(image_format=image_format))
    assert photo["width"] == 48
    assert client.get(f"/api/photos/{photo['id']}/file").status_code == 200


def test_disallowed_image_format_is_reported_per_file(client):
    response = client.post(
        "/api/import", files={"files": ("animated.gif", image_bytes(image_format="GIF"), "image/gif")}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["imported"] == []
    assert result["errors"][0]["name"] == "animated.gif"
    assert client.get("/api/library").json()["total"] == 0


def test_import_rejects_batches_larger_than_fifty(client):
    response = client.post(
        "/api/import",
        files=[("files", (f"{number}.png", image_bytes(), "image/png")) for number in range(51)],
    )
    assert response.status_code in {400, 413, 422}
    assert client.get("/api/library").json()["total"] == 0


def test_oversize_file_does_not_prevent_other_files_from_importing(client):
    oversized = image_bytes() + bytes(25 * 1024 * 1024)
    response = client.post(
        "/api/import",
        files=[
            ("files", ("too-large.png", oversized, "image/png")),
            ("files", ("valid.png", image_bytes("blue"), "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["imported"]) == 1
    assert result["imported"][0]["name"] == "valid.png"
    assert result["errors"][0]["name"] == "too-large.png"
    assert client.get("/api/library").json()["total"] == 1


def test_photo_metadata_and_deduplication_survive_reopening(tmp_path):
    original = image_bytes("green")
    with TestClient(isolated_app(tmp_path)) as client:
        photo = upload(client, "walk.png", original)
        response = client.patch(
            f"/api/photos/{photo['id']}",
            json={"favorite": True, "caption": "周末在海边散步", "tags": ["旅行", "日落"]},
        )
        assert response.status_code == 200, response.text

    with TestClient(isolated_app(tmp_path)) as client:
        photos = client.get("/api/library", params={"favorite": "true"}).json()["items"]
        assert len(photos) == 1
        assert photos[0]["id"] == photo["id"]
        assert photos[0]["favorite"] is True
        assert photos[0]["caption"] == "周末在海边散步"
        assert set(photos[0]["tags"]) == {"旅行", "日落"}
        response = client.post(
            "/api/import", files={"files": ("renamed.png", original, "image/png")}
        )
        assert response.json()["duplicates"] == ["renamed.png"]
        assert response.json()["imported"] == []
        assert client.get("/api/stats").json()["favorites"] == 1


def test_album_membership_changes_filters_without_duplicating_photos(client):
    photo = upload(client)
    other = upload(client, "other.png", image_bytes("blue"))
    response = client.post("/api/albums", json={"name": "旅行记录", "description": "旅途中的片刻"})
    assert response.status_code in {200, 201}, response.text
    album = response.json()
    membership_url = f"/api/albums/{album['id']}/photos/{photo['id']}"
    for _ in range(2):
        assert client.put(membership_url).json()["ok"] is True

    result = client.get("/api/library", params={"album_id": album["id"]}).json()
    assert result["total"] == 1
    assert result["items"][0]["id"] == photo["id"]
    assert album["id"] in result["items"][0]["album_ids"]
    assert client.get("/api/library").json()["total"] == 2
    albums = client.get("/api/albums").json()["items"]
    assert any(item["id"] == album["id"] and item["name"] == "旅行记录" for item in albums)

    assert client.delete(membership_url).json()["ok"] is True
    assert client.get("/api/library", params={"album_id": album["id"]}).json()["total"] == 0
    assert {p["id"] for p in client.get("/api/library").json()["items"]} == {photo["id"], other["id"]}


def test_keyword_search_matches_chinese_caption_and_tags(client):
    matched = upload(client, "trip.png")
    upload(client, "other.png", image_bytes("blue"))
    client.patch(
        f"/api/photos/{matched['id']}",
        json={"caption": "海边的一天", "tags": ["夏日旅行"], "favorite": True},
    ).raise_for_status()

    for term in ["海边", "夏日旅行", "trip"]:
        result = client.post("/api/search", json={"q": term, "mode": "keyword", "k": 60})
        assert result.status_code == 200, result.text
        assert result.json()["mode"] == "keyword"
        assert [photo["id"] for photo in result.json()["items"]] == [matched["id"]]
        assert result.json()["total"] == 1
        assert result.json()["elapsed_ms"] >= 0

    library = client.get("/api/library", params={"q": "海边"}).json()
    assert [photo["id"] for photo in library["items"]] == [matched["id"]]


def test_orientation_favorites_and_pagination_preserve_counts(client):
    landscape = upload(client, "wide.png", image_bytes("red", (80, 40)))
    portrait = upload(client, "tall.png", image_bytes("green", (40, 80)))
    square = upload(client, "square.png", image_bytes("blue", (40, 40)))
    client.patch(f"/api/photos/{portrait['id']}", json={"favorite": True}).raise_for_status()

    for orientation, expected in [("landscape", landscape), ("portrait", portrait), ("square", square)]:
        result = client.get("/api/library", params={"orientation": orientation}).json()
        assert result["total"] == 1
        assert result["items"][0]["id"] == expected["id"]
    assert client.get("/api/library", params={"favorite": "true"}).json()["total"] == 1
    assert client.get("/api/library", params={"favorite": "true", "orientation": "landscape"}).json()["total"] == 0

    first = client.get("/api/library", params={"page": 1, "page_size": 2}).json()
    second = client.get("/api/library", params={"page": 2, "page_size": 2}).json()
    assert first["total"] == second["total"] == 3
    assert (first["page"], second["page"], first["page_size"]) == (1, 2, 2)
    assert len(first["items"]) == 2 and len(second["items"]) == 1
    assert len({photo["id"] for photo in first["items"] + second["items"]}) == 3


@pytest.mark.parametrize("mode", ["keyword", "hybrid"])
def test_search_applies_all_library_filters(client, mode):
    selected = upload(client, "trip-wide.png", image_bytes("red", (80, 40)))
    tall = upload(client, "trip-tall.png", image_bytes("green", (40, 80)))
    other_album = upload(client, "trip-other.png", image_bytes("blue", (80, 40)))
    not_favorite = upload(client, "trip-unstarred.png", image_bytes("yellow", (80, 40)))
    album = client.post("/api/albums", json={"name": "本次旅行"}).json()
    for photo in [selected, tall, other_album]:
        client.patch(f"/api/photos/{photo['id']}", json={"favorite": True}).raise_for_status()
    for photo in [selected, tall, not_favorite]:
        client.put(f"/api/albums/{album['id']}/photos/{photo['id']}").raise_for_status()

    response = client.post(
        "/api/search",
        json={"q": "trip", "mode": mode, "favorite": True, "album_id": album["id"], "orientation": "landscape"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["total"] == 1
    assert [photo["id"] for photo in response.json()["items"]] == [selected["id"]]


def test_original_and_thumbnail_are_decodable_images(client):
    original = image_bytes("orange", (1600, 900))
    photo = upload(client, "large.png", original)
    response = client.get(f"/api/photos/{photo['id']}/file")
    assert response.status_code == 200
    assert response.content == original
    thumbnail = client.get(f"/api/photos/{photo['id']}/thumbnail")
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"].startswith("image/")
    with Image.open(BytesIO(thumbnail.content)) as image:
        image.verify()


def test_media_routes_reject_unknown_ids_and_local_file_escape(client, tmp_path):
    outside_image = tmp_path / "private.png"
    outside_image.write_bytes(image_bytes())
    private_file = tmp_path / "images" / "private.txt"
    private_file.write_text("private file contents", encoding="utf-8")
    link = tmp_path / "images" / "outside.png"
    link.symlink_to(outside_image)

    for path in [outside_image, private_file, link, tmp_path / "images" / ".." / "private.png"]:
        response = client.get("/image", params={"path": str(path)})
        assert response.status_code in {400, 403, 404, 422}, path
        assert response.content != outside_image.read_bytes()
        assert response.content != b"private file contents"
    for suffix in ["file", "thumbnail"]:
        response = client.get(f"/api/photos/does-not-exist/{suffix}")
        assert response.status_code == 404


def test_startup_discovers_existing_images_and_ignores_invalid_files(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "old-photo.png").write_bytes(image_bytes("purple"))
    (images / "broken.png").write_bytes(b"invalid image")
    (images / "notes.txt").write_text("not a photo", encoding="utf-8")
    with TestClient(isolated_app(tmp_path)) as client:
        library = client.get("/api/library").json()
        assert library["total"] == 1
        assert library["items"][0]["name"] == "old-photo.png"
        assert client.get("/image", params={"path": str(images / "old-photo.png")}).status_code == 200
    with TestClient(isolated_app(tmp_path)) as client:
        assert client.get("/api/library").json()["total"] == 1


def test_ai_unavailable_has_explicit_errors_and_hybrid_keeps_keyword_results(client):
    photo = upload(client, "sunset.png")
    hybrid = client.post("/api/search", json={"q": "sunset", "mode": "hybrid", "k": 60})
    assert hybrid.status_code == 200, hybrid.text
    assert [item["id"] for item in hybrid.json()["items"]] == [photo["id"]]
    assert hybrid.json()["mode"] == "keyword"
    assert hybrid.json()["warning"]

    semantic = client.post("/api/search", json={"q": "sunset", "mode": "semantic"})
    assert semantic.status_code == 503, semantic.text
    assert semantic.json()["detail"]
    index = client.post("/api/index", json={"languages": ["en"], "force": False})
    assert index.status_code == 503, index.text
    assert index.json()["detail"]


def test_import_does_not_treat_replaced_file_as_original_duplicate(client):
    original = image_bytes("red")
    photo = upload(client, "red.png", original)
    stored_path = client.app.state.library.path(photo["id"])
    stored_path.write_bytes(image_bytes("blue"))

    response = client.post(
        "/api/import", files={"files": ("red-again.png", original, "image/png")}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["duplicates"] == []
    assert len(result["imported"]) == 1
    restored = client.get(result["imported"][0]["url"])
    assert restored.content == original


@pytest.mark.parametrize("endpoint", ["file", "thumbnail", "legacy"])
def test_media_revalidates_files_that_changed_since_import(client, endpoint):
    photo = upload(client)
    stored_path = client.app.state.library.path(photo["id"])
    stored_path.write_bytes(b"a replaced file is no longer an image")

    # Every endpoint must handle a stale catalog entry, regardless of which one
    # is called first or whether the user has manually refreshed the library.
    if endpoint == "legacy":
        response = client.get("/image", params={"path": str(stored_path)})
    else:
        response = client.get(f"/api/photos/{photo['id']}/{endpoint}")
    assert response.status_code in {400, 404}, response.text
    assert response.content != stored_path.read_bytes()


def test_legacy_and_new_index_endpoints_share_job_state(tmp_path, monkeypatch):
    entered, release = Event(), Event()

    def blocking_build(**kwargs):
        entered.set()
        assert release.wait(5), "test did not release the fake index build"
        return {"en": SimpleNamespace(ntotal=1), "zh": SimpleNamespace(ntotal=1)}

    model_dir = tmp_path / "models"
    for name in ["clip-vit-base-patch32-local", "chinese-clip-vit-base-patch16-local"]:
        model = model_dir / name
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"placeholder")

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        "api.server.importlib.util.find_spec",
        lambda name: object() if name in {"torch", "transformers", "faiss"} else real_find_spec(name),
    )
    monkeypatch.setitem(sys.modules, "app.indexer", SimpleNamespace(
        build_all_indexes=blocking_build,
        build_index=lambda **kwargs: blocking_build(**kwargs)["en"],
    ))
    monkeypatch.setitem(sys.modules, "app.clip_service", SimpleNamespace(free_model=lambda *args, **kwargs: None))

    with TestClient(isolated_app(tmp_path)) as client, ThreadPoolExecutor(max_workers=1) as pool:
        legacy = pool.submit(client.post, "/rebuild")
        try:
            assert entered.wait(3)
            assert client.get("/api/system").json()["job"]["status"] == "running"
            concurrent = client.post("/api/index", json={"languages": ["en"]})
            assert concurrent.status_code == 409, concurrent.text
        finally:
            release.set()
        assert legacy.result(timeout=3).status_code in {200, 202}


def test_semantic_and_hybrid_api_search_use_real_custom_faiss_index(tmp_path, monkeypatch):
    import faiss
    import numpy as np

    from app import indexer, retriever

    model_dir = tmp_path / "custom-models"
    data_dir = tmp_path / "custom-library"
    for name in ["clip-vit-base-patch32-local", "chinese-clip-vit-base-patch16-local"]:
        model = model_dir / name
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"placeholder")

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        "api.server.importlib.util.find_spec",
        lambda name: object() if name in {"torch", "transformers", "faiss"} else real_find_spec(name),
    )
    embeddings = {}
    encoded_languages = set()
    query_calls = []

    def embed_images(paths, batch_size=16, lang="en", model_dir=None):
        assert Path(model_dir) == tmp_path / "custom-models"
        assert all(Path(path).parent == data_dir / "images" for path in paths)
        encoded_languages.add(lang)
        return [np.asarray([embeddings[path]], dtype=np.float32) for path in paths], []

    def embed_query(text, lang="en", model_dir=None):
        assert text == "beach" and lang == "en"
        assert Path(model_dir) == tmp_path / "custom-models"
        query_calls.append(text)
        return np.asarray([[1, 0]], dtype=np.float32)

    monkeypatch.setattr(indexer, "embed_image_batch", embed_images)
    monkeypatch.setattr(retriever, "embed_text", embed_query)

    with TestClient(create_app(data_dir=data_dir, model_dir=model_dir)) as client:
        semantic_only = upload(client, "coast.png", image_bytes("red"))
        shared = upload(client, "beach.png", image_bytes("blue"))
        other = upload(client, "forest.png", image_bytes("green"))
        for photo, vector in [(semantic_only, [1, 0]), (shared, [0.8, 0.6]), (other, [0, 1])]:
            path = client.app.state.library.path(photo["id"])
            embeddings[str(path)] = vector
        for photo in [shared, other]:
            client.patch(f"/api/photos/{photo['id']}", json={"favorite": True}).raise_for_status()

        built = client.post("/rebuild", params={"force": True})
        assert built.status_code == 200, built.text
        assert built.json()["indexed_en"] == built.json()["indexed_zh"] == 3
        assert encoded_languages == {"en", "zh"}
        snapshot, paths = indexer.load_index("en", index_dir=data_dir / "index")
        assert isinstance(snapshot, faiss.IndexFlatIP)
        assert set(paths) == set(embeddings)

        semantic = client.post("/api/search", json={"q": "beach", "mode": "semantic"})
        assert semantic.status_code == 200, semantic.text
        result = semantic.json()
        assert result["mode"] == "semantic" and result["warning"] is None
        assert [photo["id"] for photo in result["items"]] == [semantic_only["id"], shared["id"], other["id"]]
        assert [photo["score"] for photo in result["items"]] == pytest.approx([1, 0.8, 0])

        hybrid = client.post("/api/search", json={"q": "beach", "mode": "hybrid"}).json()
        assert hybrid["mode"] == "hybrid" and hybrid["warning"] is None
        assert [photo["id"] for photo in hybrid["items"]] == [shared["id"], semantic_only["id"], other["id"]]
        assert "关键词" in hybrid["items"][0]["match_reason"]
        assert "语义" in hybrid["items"][0]["match_reason"]

        filtered = client.post(
            "/api/search", json={"q": "beach", "mode": "hybrid", "favorite": True, "k": 1}
        ).json()
        assert filtered["total"] == 2
        assert [photo["id"] for photo in filtered["items"]] == [shared["id"]]
        assert filtered["items"][0]["favorite"] is True
        assert len(query_calls) == 3
