"""Exercise real FAISS snapshots with tiny deterministic model substitutes."""
import json
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pytest
from PIL import Image

from app import clip_service, indexer


@pytest.fixture
def library(tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    indexes = tmp_path / "index"
    photos.mkdir()
    indexes.mkdir()
    monkeypatch.setattr(indexer, "IMAGE_DIR", photos)
    monkeypatch.setattr(indexer, "INDEX_DIR", indexes)
    for constant, name in (("INDEX_FILE", "images.index"), ("META_FILE", "meta.json"),
                           ("INDEX_FILE_ZH", "images_zh.index"), ("META_FILE_ZH", "meta_zh.json")):
        monkeypatch.setattr(indexer, constant, indexes / name)
    monkeypatch.setattr(indexer, "CLIP_MODEL_NAME", str(tmp_path / "en-model"))
    monkeypatch.setattr(indexer, "CHINESE_CLIP_MODEL_NAME", str(tmp_path / "zh-model"))
    calls = []

    def embed(paths, batch_size=16, lang="en"):
        calls.append((lang, list(paths)))
        vectors, failed = [], []
        for path in paths:
            if Path(path).name == "broken.jpg":
                failed.append((path, "cannot decode image"))
                continue
            value = np.array([[len(Path(path).read_bytes()), len(Path(path).name), 1]], dtype="float32")
            vectors.append(value / np.linalg.norm(value))
        return vectors, failed

    monkeypatch.setattr(indexer, "embed_image_batch", embed)
    return SimpleNamespace(photos=photos, indexes=indexes, calls=calls, embed=embed)


def photo(library, name, content=b"image"):
    path = library.photos / name
    path.write_bytes(content)
    return str(path.resolve())


def metadata(library, lang="en"):
    name = "meta_zh.json" if lang == "zh" else "meta.json"
    return json.loads((library.indexes / name).read_text())


def test_incremental_add_change_delete_reuses_unchanged_vectors(library):
    a, b = photo(library, "a.jpg"), photo(library, "b.jpg")
    initial = indexer.build_index(batch_size=1)
    assert isinstance(initial, faiss.IndexFlatIP)
    first_vector = initial.reconstruct(0).copy()
    library.calls.clear()

    # Unchanged updates perform no encoding and return the same interface.
    unchanged = indexer.build_index()
    assert unchanged.ntotal == 2
    assert library.calls == []
    committed_generation = metadata(library)["index_file"]
    assert indexer.build_index().ntotal == 2
    assert metadata(library)["index_file"] == committed_generation

    Path(b).write_bytes(b"changed image data")
    c = photo(library, "c.jpg")
    progress = []
    updated = indexer.build_index(progress=progress.append)
    assert library.calls == [("en", [b, c])]
    assert updated.ntotal == 3
    np.testing.assert_array_equal(updated.reconstruct(0), first_vector)
    assert progress[-1]["reused"] == 1
    assert progress[-1]["processed"] == progress[-1]["total"] == 3
    assert metadata(library)["updated_at"]

    library.calls.clear()
    Path(a).unlink()
    assert indexer.build_index().ntotal == 2
    assert indexer.load_index()[1] == [b, c]
    assert library.calls == []


def test_force_reencodes_and_languages_do_not_share_vectors(library):
    path = photo(library, "a.jpg")
    indexer.build_index(lang="en")
    indexer.build_index(lang="zh")
    assert library.calls == [("en", [path]), ("zh", [path])]
    assert metadata(library, "en")["index_file"] != metadata(library, "zh")["index_file"]
    library.calls.clear()
    indexer.build_index(force=True, lang="zh")
    assert library.calls == [("zh", [path])]
    assert indexer.load_index("en")[1] == [path]
    assert indexer.load_index("zh")[1] == [path]


def test_model_replacement_invalidates_reused_vectors(library):
    path = photo(library, "a.jpg")
    indexer.build_index()
    model = Path(indexer.CLIP_MODEL_NAME)
    model.mkdir()
    (model / "config.json").write_text('{"model": "replacement"}')
    library.calls.clear()
    indexer.build_index()
    assert library.calls == [("en", [path])]


def test_readers_keep_old_snapshot_until_successful_publication(library, monkeypatch):
    first = photo(library, "a.jpg")
    indexer.build_index()
    second = photo(library, "b.jpg")

    def embed_while_reading(paths, batch_size=16, lang="en"):
        assert indexer.load_index()[1] == [first]
        return library.embed(paths, batch_size, lang)

    monkeypatch.setattr(indexer, "embed_image_batch", embed_while_reading)
    indexer.build_index()
    assert indexer.load_index()[1] == [first, second]


def test_failed_metadata_commit_preserves_prior_snapshot(library, monkeypatch):
    first = photo(library, "a.jpg")
    indexer.build_index()
    old_meta = metadata(library)
    generations = set(library.indexes.glob("*.index"))
    photo(library, "b.jpg")
    original_write = indexer._atomic_write

    def fail_metadata(path, data):
        if path == indexer.META_FILE:
            raise PermissionError("simulated commit failure")
        original_write(path, data)

    monkeypatch.setattr(indexer, "_atomic_write", fail_metadata)
    with pytest.raises(PermissionError, match="simulated"):
        indexer.build_index()
    assert metadata(library) == old_meta
    assert indexer.load_index()[1] == [first]
    assert set(library.indexes.glob("*.index")) == generations


def test_failed_inference_preserves_prior_snapshot(library, monkeypatch):
    first = photo(library, "a.jpg")
    indexer.build_index()
    photo(library, "b.jpg")

    def fail(*args, **kwargs):
        raise FileNotFoundError("missing local model")

    monkeypatch.setattr(indexer, "embed_image_batch", fail)
    with pytest.raises(FileNotFoundError, match="missing local model"):
        indexer.build_index()
    assert indexer.load_index()[1] == [first]


def test_bad_images_do_not_misalign_vectors_and_paths(library):
    a = photo(library, "a.jpg")
    photo(library, "broken.jpg")
    c = photo(library, "c.jpg")
    result = indexer.build_index()
    assert result.ntotal == 2
    assert indexer.load_index()[1] == [a, c]
    expected, _ = library.embed([c])
    np.testing.assert_allclose(result.reconstruct(1), expected[0][0])
    assert metadata(library)["failed"][0]["path"].endswith("broken.jpg")


def test_deleting_all_photos_publishes_an_empty_index(library):
    path = photo(library, "a.jpg")
    indexer.build_index()
    library.calls.clear()
    Path(path).unlink()
    result = indexer.build_index()
    assert result.ntotal == 0
    assert indexer.load_index()[1] == []
    assert library.calls == []


def test_legacy_index_is_readable_and_migrates(library):
    path = photo(library, "a.jpg")
    old_index = faiss.IndexFlatIP(3)
    old_index.add(np.array([[1, 0, 0]], dtype="float32"))
    indexer.INDEX_FILE.write_bytes(faiss.serialize_index(old_index).tobytes())
    indexer.META_FILE.write_text(json.dumps({"paths": [path]}))
    assert indexer.load_index()[1] == [path]
    assert indexer.build_index().ntotal == 1
    assert metadata(library)["version"] == 2


def test_loader_rejects_inconsistent_paths(library):
    photo(library, "a.jpg")
    indexer.build_index()
    meta = metadata(library)
    meta["paths"] = []
    indexer.META_FILE.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="不一致"):
        indexer.load_index()


def test_collect_ignores_hidden_and_external_files(library, tmp_path):
    good = photo(library, "a.jpg")
    photo(library, ".hidden.jpg")
    (library.photos / ".private").mkdir()
    (library.photos / ".private" / "x.jpg").write_bytes(b"private")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    (library.photos / "external.jpg").symlink_to(outside)
    assert [str(path) for path in indexer.collect_images()] == [good]


def test_missing_model_is_reported_without_inference_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_service, "CLIP_MODEL_NAME", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="本地模型"):
        clip_service._load_model("en")


def test_tensor_and_structured_transformers_outputs_are_supported():
    tensor = SimpleNamespace(detach=lambda: None)
    assert clip_service._feature_tensor(tensor) is tensor
    assert clip_service._feature_tensor(SimpleNamespace(pooler_output=tensor)) is tensor
    assert clip_service._feature_tensor(SimpleNamespace(image_embeds=tensor)) is tensor
    with pytest.raises(TypeError):
        clip_service._feature_tensor(SimpleNamespace(last_hidden_state=tensor))


def test_image_loading_applies_exif_and_releases_source(tmp_path):
    path = tmp_path / "phone.jpg"
    image = Image.new("RGB", (4, 2), "red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    image.close()
    with clip_service._open_rgb(path) as oriented:
        assert oriented.size == (2, 4)
        assert oriented.mode == "RGB"
        path.unlink()
        assert oriented.getpixel((0, 0))[0] > 200


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_device_auto_detection(monkeypatch, cuda, mps, expected):
    import sys
    import config
    monkeypatch.delenv("FORCE_CPU", raising=False)
    monkeypatch.delenv("DEVICE", raising=False)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                                 backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert config.get_device() == expected
    assert config.DEVICE == expected


def test_force_cpu_takes_precedence_without_importing_torch(monkeypatch):
    import sys
    import config
    monkeypatch.setenv("FORCE_CPU", "1")
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setitem(sys.modules, "torch", None)
    assert config.get_device() == "cpu"


def test_explicit_unavailable_device_is_actionable(monkeypatch):
    import sys
    import config
    monkeypatch.delenv("FORCE_CPU", raising=False)
    monkeypatch.setenv("DEVICE", "cuda")
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                 backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(RuntimeError, match="CUDA 不可用"):
        config.get_device()


def test_explicit_library_paths_isolate_builds_and_searches(tmp_path, monkeypatch):
    from app import retriever
    libraries = []
    calls = []
    original_image_dir, original_index_dir = indexer.IMAGE_DIR, indexer.INDEX_DIR

    def embed_images(paths, batch_size=16, lang="en", model_dir=None):
        calls.append((list(paths), model_dir))
        return [np.array([[1, 0, 0]], dtype="float32") for _ in paths], []

    def embed_query(text, lang="en", model_dir=None):
        calls.append((text, model_dir))
        return np.array([[1, 0, 0]], dtype="float32")

    monkeypatch.setattr(indexer, "embed_image_batch", embed_images)
    monkeypatch.setattr(retriever, "embed_text", embed_query)
    for name in ("first", "second"):
        root = tmp_path / name
        images, indexes, models = root / "images", root / "index", root / "models"
        images.mkdir(parents=True)
        image = images / "photo.jpg"
        image.write_bytes(name.encode())
        libraries.append((images, indexes, models, image))
        assert indexer.build_index(image_dir=images, index_dir=indexes, model_dir=models).ntotal == 1
        meta = json.loads((indexes / "meta.json").read_text())
        assert meta["model"]["path"] == str(models / Path(indexer.CLIP_MODEL_NAME).name)

    assert len(calls) == 2
    for images, indexes, models, image in libraries:
        # A second pass must reuse only this library's committed vectors.
        assert indexer.build_index(image_dir=images, index_dir=indexes, model_dir=models).ntotal == 1
        assert indexer.load_index(index_dir=indexes)[1] == [str(image)]
    assert len(calls) == 2
    for images, indexes, models, image in libraries:
        results = retriever.search_by_text("beach", index_dir=indexes, model_dir=models)
        assert [result["path"] for result in results] == [str(image)]
        assert calls[-1] == ("beach", models)
    assert indexer.IMAGE_DIR == original_image_dir
    assert indexer.INDEX_DIR == original_index_dir


def test_model_cache_and_release_are_isolated_by_model_directory(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(clip_service, "_models", {})
    loaded = []

    def load(lang, model_dir=None):
        pair = (object(), object())
        loaded.append((lang, model_dir))
        return pair

    monkeypatch.setattr(clip_service, "_load_model", load)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                 backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    first, second = tmp_path / "first", tmp_path / "second"
    model_a = clip_service.get_model("en", model_dir=first)
    model_b = clip_service.get_model("en", model_dir=second)
    assert model_a is not model_b
    assert clip_service.get_model("en", model_dir=first) is model_a
    assert len(loaded) == 2
    clip_service.free_model("en", model_dir=first)
    assert clip_service.get_model("en", model_dir=second) is model_b
    assert clip_service.get_model("en", model_dir=first) is not model_a
    assert len(loaded) == 3
