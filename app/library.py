"""Persistent, model-independent photo library for the local album."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import uuid

from PIL import Image, ImageOps, UnidentifiedImageError

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 40_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def inspect_image(path: Path):
    try:
        with Image.open(path) as image:
            fmt = image.format
            if fmt not in FORMATS:
                raise ValueError("仅支持 JPEG、PNG、WEBP、BMP 图片")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("图片超过 4000 万像素，请缩小后导入")
            image.verify()
        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image)
            oriented.load()
            width, height = oriented.size
            oriented.close()
        return width, height, fmt
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ValueError("图片已损坏或格式不受支持") from exc


def file_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Library:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.thumbnail_dir = self.data_dir / "thumbnails"
        self.lock = threading.RLock()
        for directory in (self.data_dir, self.image_dir, self.thumbnail_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "library.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS photos (
                    id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL, sha256 TEXT NOT NULL,
                    width INTEGER NOT NULL, height INTEGER NOT NULL,
                    bytes INTEGER NOT NULL, format TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL, created_at TEXT NOT NULL,
                    favorite INTEGER NOT NULL DEFAULT 0,
                    caption TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS photos_hash ON photos(sha256);
                CREATE TABLE IF NOT EXISTS albums (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS album_photos (
                    album_id TEXT REFERENCES albums(id) ON DELETE CASCADE,
                    photo_id TEXT REFERENCES photos(id) ON DELETE CASCADE,
                    PRIMARY KEY(album_id, photo_id)
                );
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def safe_path(self, relative: str):
        path = (self.image_dir / relative).resolve()
        if not path.is_relative_to(self.image_dir.resolve()) or not path.is_file():
            raise KeyError("照片不存在")
        return path

    def _insert(self, db, path, name=None):
        width, height, fmt = inspect_image(path)
        stat = path.stat()
        identifier = uuid.uuid4().hex
        db.execute(
            "INSERT INTO photos(id,path,name,sha256,width,height,bytes,format,mtime_ns,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (identifier, path.relative_to(self.image_dir).as_posix(), name or path.name,
             file_hash(path), width, height, stat.st_size, fmt, stat.st_mtime_ns,
             datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()),
        )
        return identifier

    def sync(self):
        """Discover externally added/changed files; preserve annotations for known paths."""
        errors = []
        with self.lock, self.connect() as db:
            known = {row["path"]: row for row in db.execute("SELECT * FROM photos")}
            found = set()
            for path in sorted(self.image_dir.rglob("*")):
                if path.suffix.lower() not in EXTENSIONS or any(p.startswith(".") for p in path.relative_to(self.image_dir).parts):
                    continue
                if not path.is_file() or not path.resolve().is_relative_to(self.image_dir.resolve()):
                    continue
                relative = path.relative_to(self.image_dir).as_posix()
                found.add(relative)
                old = known.get(relative)
                try:
                    stat = path.stat()
                    if old and old["mtime_ns"] == stat.st_mtime_ns and old["bytes"] == stat.st_size:
                        continue
                    if old:
                        width, height, fmt = inspect_image(path)
                        db.execute("UPDATE photos SET sha256=?,width=?,height=?,bytes=?,format=?,mtime_ns=? WHERE id=?",
                                   (file_hash(path), width, height, stat.st_size, fmt, stat.st_mtime_ns, old["id"]))
                    else:
                        self._insert(db, path)
                except (ValueError, OSError) as exc:
                    errors.append({"name": path.name, "error": str(exc)})
                    if old:
                        db.execute("DELETE FROM photos WHERE id=?", (old["id"],))
            for relative in known.keys() - found:
                db.execute("DELETE FROM photos WHERE path=?", (relative,))
        return errors

    def import_file(self, source: Path, filename: str):
        if source.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("每张图片最大 25 MB")
        _, _, fmt = inspect_image(source)
        digest = file_hash(source)
        with self.lock, self.connect() as db:
            existing = db.execute("SELECT id,path FROM photos WHERE sha256=?", (digest,)).fetchall()
            for row in existing:
                try:
                    current_path = self.safe_path(row["path"])
                    # A user may replace files in the folder between refreshes.
                    if file_hash(current_path) == digest:
                        return None
                except KeyError:
                    continue
            # Keep client filenames as display text only, never as a filesystem path.
            name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()[:240] or "未命名照片"
            destination = self.image_dir / (uuid.uuid4().hex + FORMATS[fmt])
            os.replace(source, destination)
            try:
                identifier = self._insert(db, destination, name)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        return self.get(identifier)

    def _serialize(self, row, album_ids):
        return {
            "id": row["id"], "name": row["name"],
            "url": f"/api/photos/{row['id']}/file",
            "thumbnail_url": f"/api/photos/{row['id']}/thumbnail?v={row['mtime_ns']}",
            "width": row["width"], "height": row["height"], "bytes": row["bytes"],
            "format": row["format"], "created_at": row["created_at"],
            "favorite": bool(row["favorite"]), "caption": row["caption"],
            "tags": json.loads(row["tags"]), "album_ids": album_ids,
        }

    def all(self, *, favorite=False, album_id=None, orientation=None, sort="newest"):
        clauses, params = [], []
        if favorite:
            clauses.append("favorite=1")
        if album_id:
            clauses.append("id IN (SELECT photo_id FROM album_photos WHERE album_id=?)")
            params.append(album_id)
        if orientation == "landscape":
            clauses.append("width > height")
        elif orientation == "portrait":
            clauses.append("width < height")
        elif orientation == "square":
            clauses.append("width = height")
        order = {"newest": "created_at DESC,id", "oldest": "created_at ASC,id", "name": "name COLLATE NOCASE,id"}[sort]
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as db:
            rows = db.execute("SELECT * FROM photos" + where + " ORDER BY " + order, params).fetchall()
            memberships = {}
            for membership in db.execute("SELECT album_id,photo_id FROM album_photos"):
                memberships.setdefault(membership["photo_id"], []).append(membership["album_id"])
        return [self._serialize(row, memberships.get(row["id"], [])) for row in rows]

    def get(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT * FROM photos WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError("照片不存在")
            albums = [a[0] for a in db.execute("SELECT album_id FROM album_photos WHERE photo_id=?", (identifier,))]
            return self._serialize(row, albums)

    def path(self, identifier):
        with self.lock, self.connect() as db:
            row = db.execute("SELECT * FROM photos WHERE id=?", (identifier,)).fetchone()
            if not row:
                raise KeyError("照片不存在")
            path = self.safe_path(row["path"])
            stat = path.stat()
            if row["mtime_ns"] != stat.st_mtime_ns or row["bytes"] != stat.st_size:
                try:
                    width, height, fmt = inspect_image(path)
                except (ValueError, OSError) as exc:
                    raise KeyError("照片已被移动或替换为无效文件，请刷新图库") from exc
                db.execute("UPDATE photos SET sha256=?,width=?,height=?,bytes=?,format=?,mtime_ns=? WHERE id=?",
                           (file_hash(path), width, height, stat.st_size, fmt, stat.st_mtime_ns, identifier))
            return path

    def by_path(self):
        with self.connect() as db:
            return {str((self.image_dir / row["path"]).resolve()): row["id"]
                    for row in db.execute("SELECT id,path FROM photos")}

    def update(self, identifier, changes):
        self.get(identifier)
        assignments, params = [], []
        for key in ("favorite", "caption", "tags"):
            if key not in changes:
                continue
            value = changes[key]
            if key == "tags":
                value = json.dumps(list(dict.fromkeys(tag.strip() for tag in value if tag.strip())), ensure_ascii=False)
            elif key == "favorite":
                value = int(value)
            else:
                value = value.strip()
            assignments.append(key + "=?")
            params.append(value)
        if assignments:
            with self.lock, self.connect() as db:
                db.execute("UPDATE photos SET " + ",".join(assignments) + " WHERE id=?", params + [identifier])
        return self.get(identifier)

    def albums(self):
        with self.connect() as db:
            result = []
            for row in db.execute("SELECT * FROM albums ORDER BY created_at DESC,id"):
                members = db.execute("SELECT photo_id FROM album_photos WHERE album_id=? ORDER BY photo_id", (row["id"],)).fetchall()
                result.append({**dict(row), "count": len(members),
                               "cover_url": f"/api/photos/{members[0][0]}/thumbnail" if members else None})
            return result

    def create_album(self, name, description=""):
        identifier = uuid.uuid4().hex
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO albums VALUES(?,?,?,?)", (identifier, name.strip(), description.strip(), now()))
        return next(album for album in self.albums() if album["id"] == identifier)

    def membership(self, album_id, photo_id, add=True):
        self.get(photo_id)
        with self.lock, self.connect() as db:
            if not db.execute("SELECT id FROM albums WHERE id=?", (album_id,)).fetchone():
                raise KeyError("相册不存在")
            if add:
                db.execute("INSERT OR IGNORE INTO album_photos VALUES(?,?)", (album_id, photo_id))
            else:
                db.execute("DELETE FROM album_photos WHERE album_id=? AND photo_id=?", (album_id, photo_id))

    def stats(self):
        with self.connect() as db:
            row = db.execute("SELECT COUNT(*),COALESCE(SUM(favorite),0),COALESCE(SUM(bytes),0),"
                             "COALESCE(SUM(width>height),0),COALESCE(SUM(width<height),0),COALESCE(SUM(width=height),0) FROM photos").fetchone()
            albums = db.execute("SELECT COUNT(*) FROM albums").fetchone()[0]
        return {"total": row[0], "favorites": row[1], "storage_bytes": row[2], "albums": albums,
                "orientations": dict(zip(("landscape", "portrait", "square"), row[3:]))}

    def thumbnail(self, identifier):
        with self.lock:
            path = self.path(identifier)
            target = self.thumbnail_dir / f"{identifier}-{path.stat().st_mtime_ns}.jpg"
            if not target.exists():
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    image.thumbnail((720, 720), Image.Resampling.LANCZOS)
                    temporary = target.with_suffix(".tmp")
                    try:
                        image.save(temporary, format="JPEG", quality=85, optimize=True)
                        os.replace(temporary, target)
                    finally:
                        image.close()
                        temporary.unlink(missing_ok=True)
            return target
