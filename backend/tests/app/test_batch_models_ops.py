"""Batch/ZIP safety, background jobs, model governance, providers, retention and migrations."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest

from app.services.batch import ZipRejected, safe_extract_images

from .conftest import run_jobs


# ----------------------------------------------------------------------------------------------- ZIP safety
def _zip(path: Path, entries: dict[str, bytes], compression=zipfile.ZIP_DEFLATED) -> Path:
    with zipfile.ZipFile(path, "w", compression) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return path


def _out(tmp_path, name="out"):
    d = tmp_path / name
    d.mkdir()
    return d


def test_zip_bomb_rejected(tmp_path):
    # 1.5 MB of zeros (under the 2 MB per-file limit) compresses ~1000:1 -> whole archive rejected
    bomb = _zip(tmp_path / "bomb.zip", {"a.png": b"\x00" * (1536 * 1024)})
    with pytest.raises(ZipRejected, match="zip bomb"):
        safe_extract_images(bomb, _out(tmp_path))
    huge = _zip(tmp_path / "huge.zip", {"b.png": b"\x00" * (20 * 1024 * 1024)})
    with pytest.raises(ZipRejected):
        safe_extract_images(huge, _out(tmp_path, "o2"))


def test_zip_file_count_limit(tmp_path, monkeypatch, demo_images):
    from app import config

    monkeypatch.setattr(config.get_settings(), "max_zip_files", 3)
    z = _zip(tmp_path / "many.zip", {f"{i}.png": demo_images[0] for i in range(5)}, zipfile.ZIP_STORED)
    with pytest.raises(ZipRejected, match="limit"):
        safe_extract_images(z, _out(tmp_path))


def test_zip_path_traversal_is_flattened(tmp_path, demo_images):
    z = _zip(tmp_path / "trav.zip", {"../../evil.png": demo_images[0], "/abs/x.png": demo_images[0],
                                     "ok/dir/good.png": demo_images[1]})
    out = _out(tmp_path)
    files, skipped = safe_extract_images(z, out)
    assert len(files) == 3 and not skipped
    for f in files:
        assert Path(f).resolve().is_relative_to(out.resolve())
    assert not (tmp_path / "evil.png").exists()


def test_zip_symlink_entries_skipped(tmp_path, demo_images):
    p = tmp_path / "link.zip"
    with zipfile.ZipFile(p, "w") as z:
        info = zipfile.ZipInfo("link.png")
        info.external_attr = (0o120777 << 16)  # S_IFLNK
        z.writestr(info, "/etc/passwd")
        z.writestr("real.png", demo_images[0])
    files, skipped = safe_extract_images(p, _out(tmp_path))
    assert [Path(f).name.split("_", 1)[1] for f in files] == ["real.png"]
    assert any("symbolic link" in s for s in skipped)


def test_zip_non_images_ignored_and_bad_zip(tmp_path, demo_images):
    z = _zip(tmp_path / "mixed.zip", {"readme.txt": b"hi", "run.exe": b"MZ", "a.png": demo_images[0]})
    files, skipped = safe_extract_images(z, _out(tmp_path, "o1"))
    assert len(files) == 1 and len(skipped) == 2
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"PK\x03\x04garbage")
    with pytest.raises(ZipRejected):
        safe_extract_images(bad, _out(tmp_path, "o2"))


# ----------------------------------------------------------------------------------------------- batch jobs
def test_batch_with_zip_and_images(inspector, demo_images):
    normal, anomalous = demo_images
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("b1.png", normal)
        z.writestr("b2.png", anomalous)
        z.writestr("notes.txt", b"ignored")
    r = inspector.post("/api/batches", data={"product_key": "demo-pcb", "batch_code": "T-1", "force": "true"},
                       files=[("files", ("lot.zip", buf.getvalue(), "application/zip")),
                              ("files", ("c.bin", b"not an image", "image/png"))])
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]
    run_jobs()
    job = inspector.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "succeeded", job
    res = inspector.get(f"/api/batches/{job_id}/results").json()
    items = res["items"] if isinstance(res, dict) else res
    decisions = [i["decision"] for i in items]
    assert len(items) >= 2 and all(d in ("PASS", "FAIL", "REVIEW") for d in decisions)


def test_job_cancel(inspector, demo_images):
    r = inspector.post("/api/batches", data={"product_key": "demo-pcb", "force": "true"},
                       files=[("files", ("x.png", demo_images[0], "image/png"))])
    jid = r.json()["id"]
    assert inspector.post(f"/api/jobs/{jid}/cancel").status_code == 200
    run_jobs()
    assert inspector.get(f"/api/jobs/{jid}").json()["status"] == "cancelled"


def test_retention_dry_run_changes_nothing(admin):
    before = admin.get("/api/inspections", params={"page_size": 1}).json()["total"]
    r = admin.post("/api/retention/run", params={"dry_run": "true"})
    assert r.status_code == 202
    run_jobs()
    job = admin.get(f"/api/jobs/{r.json()['id']}").json()
    assert job["status"] == "succeeded"
    assert admin.get("/api/inspections", params={"page_size": 1}).json()["total"] == before


# ----------------------------------------------------------------------------------------------- models
def _models(client, product="demo-pcb"):
    return [m for m in client.get("/api/models").json() if m["product_key"] == product]


def test_only_approved_models_activate_and_rollback(admin):
    ms = _models(admin)
    active = next(m for m in ms if m["active_for"])
    assert active["status"] == "approved"

    # demote → candidate → activation is refused
    # register a second (candidate) version of the same model: activation must be refused until approved
    import shutil

    from app.config import get_settings
    from app.db import SessionLocal
    from app.services import registry

    src = get_settings().models_dir / Path(active["artifact_dir"]).name
    new_dir = src.parent / f"{src.name}-candidate-test"
    shutil.copytree(src, new_dir, dirs_exist_ok=True)
    src = new_dir
    with SessionLocal() as db:
        cand = registry.register_model_dir(db, src, status="candidate", notes="test candidate")
        db.commit()
        cid = cand.id
    r = admin.post("/api/products/demo-pcb/activate", json={"model_id": cid, "reason": "test"})
    assert r.status_code == 409
    assert admin.post(f"/api/models/{cid}/status", json={"status": "approved", "notes": "test"}).status_code == 200
    r = admin.post("/api/products/demo-pcb/activate", json={"model_id": cid, "reason": "test"})
    assert r.status_code == 200, r.text
    # rollback restores the previously active model
    r = admin.post("/api/products/demo-pcb/rollback")
    assert r.status_code == 200 and r.json()["active_model_id"] == active["id"]
    acts = admin.get("/api/products/demo-pcb/activations").json()
    assert len(acts) >= 2


def test_threshold_override_requires_reason_and_is_audited(admin):
    m = next(m for m in _models(admin) if m["active_for"])
    t = m["thresholds"]
    fail, pas = t["fail_threshold"], t["pass_threshold"]
    url = f"/api/models/{m['id']}/thresholds"
    assert admin.put(url, json={"fail_threshold": fail, "pass_threshold": pas, "reason": ""}).status_code == 422
    r = admin.put(url, json={"fail_threshold": fail, "pass_threshold": fail * 1.5, "reason": "invalid ordering"})
    assert r.status_code == 422  # pass threshold must not exceed the reject threshold
    r = admin.put(url, json={"fail_threshold": fail, "pass_threshold": pas, "reason": "line recalibration test"})
    assert r.status_code == 200, r.text
    ev = admin.get("/api/audit", params={"action": "model.thresholds_changed"}).json()
    assert ev and ev[0]["details"]["reason"] == "line recalibration test"
    assert admin.post(f"/api/models/{m['id']}/thresholds/reset").status_code == 200


def test_threshold_override_changes_decisions(admin, inspector, demo_images):
    m = next(m for m in _models(admin) if m["active_for"])
    t = m["thresholds"]
    url = f"/api/models/{m['id']}/thresholds"
    try:
        admin.put(url, json={"fail_threshold": 1e6, "pass_threshold": 1e5, "reason": "test: nothing fails"})
        r = inspector.post("/api/inspections", files={"file": ("ov.png", demo_images[1], "image/png")},
                           data={"product_key": "demo-pcb", "force": "true"}).json()
        assert r["fail_threshold"] == 1e6 and r["decision"] != "FAIL"
    finally:
        admin.put(url, json={"fail_threshold": t["fail_threshold"], "pass_threshold": t["pass_threshold"],
                             "reason": "test: restore"})


def test_model_card_and_metrics_are_real(admin):
    m = next(m for m in _models(admin) if m["active_for"])
    detail = admin.get(f"/api/models/{m['id']}").json()
    tm = detail.get("test_metrics") or {}
    assert "ranking" in tm and 0.0 <= tm["ranking"]["auroc"] <= 1.0
    assert tm["ranking"]["n"] > 0 if "n" in tm["ranking"] else True
    card = admin.get(f"/api/models/{m['id']}/card")
    assert card.status_code == 200 and "SYNTHETIC data" in card.text
    assert f"{tm['ranking']['auroc']:.4f}" in card.text  # card reports the stored metric, not a made-up one


def test_evaluation_job(admin):
    m = next(m for m in _models(admin) if m["active_for"])
    ds = next(d for d in admin.get("/api/datasets").json() if d["product_key"] == "demo-pcb")
    r = admin.post("/api/models/evaluate", json={"dataset_id": ds["id"], "model_ids": [m["id"]], "split": "test"})
    assert r.status_code == 202, r.text
    run_jobs()
    job = admin.get(f"/api/jobs/{r.json()['id']}").json()
    assert job["status"] == "succeeded", job
    ev = admin.get("/api/evaluations", params={"kind": "comparison"}).json() or admin.get("/api/evaluations").json()
    full = admin.get(f"/api/evaluations/{ev[0]['id']}").json()
    assert m["id"] in full["model_ids"]
    # a mismatched dataset/model pair is refused
    other = next(d for d in admin.get("/api/datasets").json() if d["product_key"] == "demo-metal")
    bad = admin.post("/api/models/evaluate", json={"dataset_id": other["id"], "model_ids": [m["id"]]})
    assert bad.status_code == 422


# ----------------------------------------------------------------------------------------------- providers
def test_provider_key_never_returned(admin, inspector):
    r = admin.post("/api/providers", json={"name": "test-openai", "kind": "openai_compatible",
                                           "base_url": "http://127.0.0.1:9/v1", "model": "x",
                                           "api_key": "sk-test-SECRET-1234567890", "timeout_s": 1, "max_retries": 0})
    assert r.status_code in (200, 201), r.text
    listing = admin.get("/api/providers").text
    assert "SECRET-1234567890" not in listing and "sk-test" not in listing
    assert inspector.get("/api/providers").status_code in (200, 403)
    if inspector.get("/api/providers").status_code == 200:
        assert "SECRET" not in inspector.get("/api/providers").text
    pid = r.json()["id"]
    t = admin.post(f"/api/providers/{pid}/test").json()
    assert t["ok"] is False  # unreachable endpoint reported, not crashed
    assert admin.delete(f"/api/providers/{pid}").status_code in (200, 204)


def test_summary_without_ai_is_labelled_template(inspector):
    """With no AI provider the app still works: a deterministic summary that is NOT labelled model-generated."""
    insp = inspector.get("/api/inspections", params={"page_size": 1}).json()["items"][0]
    r = inspector.post(f"/api/inspections/{insp['id']}/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["model_generated"] is False and body["provider"] == "local" and body["cost_usd"] == 0.0
    assert body["text"]


# ----------------------------------------------------------------------------------------------- migrations
def test_alembic_fresh_upgrade(tmp_path):
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    from alembic import command

    backend = Path(__file__).resolve().parents[2]
    url = f"sqlite:///{tmp_path / 'fresh.db'}"
    cfg = Config(str(backend / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    old = os.environ.get("DL_DATABASE_URL")
    os.environ["DL_DATABASE_URL"] = url
    try:
        command.upgrade(cfg, "head")
    finally:
        os.environ["DL_DATABASE_URL"] = old or ""
    eng = create_engine(url)
    tables = set(inspect(eng).get_table_names())
    assert {"users", "inspections", "reviews", "audit_events", "model_versions", "jobs"} <= tables
    assert "password_change_required" in {c["name"] for c in inspect(eng).get_columns("users")}
    eng.dispose()

    os.environ["DL_DATABASE_URL"] = url
    try:
        command.check(cfg)  # migrations exactly match the ORM models (raises on drift)
        command.downgrade(cfg, "base")  # every migration is reversible
        command.upgrade(cfg, "head")
    finally:
        os.environ["DL_DATABASE_URL"] = old or ""


def test_migrations_on_populated_database_with_foreign_keys(app, tmp_path):
    """Upgrades must work on real deployments: populated tables, FK enforcement on (as the app configures)."""
    import shutil

    from alembic.config import Config
    from sqlalchemy import create_engine, event, inspect, text

    from alembic import command
    from app.config import get_settings

    src = Path(get_settings().database_url.removeprefix("sqlite:///"))
    db_copy = tmp_path / "populated.db"
    with create_engine(f"sqlite:///{src}").connect() as c:  # flush WAL into the main file before copying
        c.exec_driver_sql("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(src, db_copy)

    eng = create_engine(f"sqlite:///{db_copy}")

    @event.listens_for(eng, "connect")
    def _fk(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    backend = Path(__file__).resolve().parents[2]
    cfg = Config(str(backend / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend / "alembic"))
    with eng.begin() as conn:
        n_users = conn.execute(text("SELECT count(*) FROM users")).scalar()
        n_audit = conn.execute(text("SELECT count(*) FROM audit_events WHERE actor_id IS NOT NULL")).scalar()
        assert n_users > 0 and n_audit > 0  # rows reference users -> table rebuilds would violate FKs
        cfg.attributes["connection"] = conn
        command.downgrade(cfg, "3ac2e1da3a35")
        command.upgrade(cfg, "head")
    with eng.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM users")).scalar() == n_users
        assert conn.execute(text("PRAGMA foreign_key_check")).fetchall() == []
        tables = set(inspect(conn).get_table_names())
    assert not any(t.startswith("_alembic_tmp") for t in tables)
    eng.dispose()
