"""Startup tasks (migrations, seed users, registry sync) and the demo bootstrap."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import select

from app import audit
from app.config import get_settings
from app.db import SessionLocal, get_engine
from app.models import ModelVersion, Product, User
from app.security import hash_password
from app.services import registry
from app.services.datasets_service import register_dataset, sync_datasets

log = logging.getLogger("defectlens.bootstrap")
BACKEND_DIR = Path(__file__).resolve().parents[2]


def run_migrations() -> None:
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url.replace("%", "%%"))  # configparser escaping
    cfg.attributes["configure_logger"] = False
    with get_engine().begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")


def seed_users() -> None:
    s = get_settings()
    with SessionLocal() as db:
        if db.execute(select(User).limit(1)).first() is None:
            from app.config import WEAK_PASSWORDS

            weak_admin = s.admin_password.lower() in WEAK_PASSWORDS or len(s.admin_password) < 10
            db.add(User(username=s.admin_username, full_name="Administrator", role="admin",
                        password_hash=hash_password(s.admin_password), password_change_required=weak_admin))
            if weak_admin:
                log.warning("admin account uses a weak/default password - change it after first sign-in")
            if s.seed_demo_users and not s.is_production:
                db.add(User(username="inspector", full_name="Demo Inspector", role="inspector",
                            password_hash=hash_password("inspector-demo"), password_change_required=True))
                db.add(User(username="viewer", full_name="Demo Viewer", role="viewer",
                            password_hash=hash_password("viewer-demo"), password_change_required=True))
            audit.record(db, "system.seeded_users", None, "user", "", {"admin": s.admin_username})
            db.commit()
            log.info("created initial admin user '%s'", s.admin_username)


def sync_registry() -> None:
    with SessionLocal() as db:
        sync_datasets(db)
        registry.sync_models(db)
        db.commit()


def bootstrap_demo(algorithms: tuple[str, ...] = ("patchcore", "padim", "pixelstats"), activate: str = "patchcore",
                   n_normal: int = 160, n_anomaly: int = 60) -> list[str]:
    """Generate SYNTHETIC demo datasets, train/evaluate models, approve & activate one per product."""
    from defectml.data.manifest import Dataset
    from defectml.data.synthetic import PRODUCTS, generate_product
    from defectml.evaluation.benchmark import evaluate_model
    from defectml.training import train_on_dataset

    s = get_settings()
    done = []
    for product in PRODUCTS:
        ds_dir = s.datasets_dir / product
        if not (ds_dir / "dataset.json").exists():
            generate_product(product, s.datasets_dir, n_normal=n_normal, n_anomaly=n_anomaly)
        ds = Dataset(ds_dir)
        with SessionLocal() as db:
            register_dataset(db, ds_dir, validate=True)
            db.commit()
        for algo in algorithms:
            out = s.models_dir / f"{product}-{algo}-demo"
            if not (out / "model.json").exists():
                log.info("training %s on %s", algo, product)
                train_on_dataset(ds, algo, out, name=f"{product}-{algo}", config={"device": s.device})
                test = evaluate_model(out, ds, "test")
                test.pop("scores", None)
                (out / "test_metrics.json").write_text(json.dumps(test, indent=2, default=float), encoding="utf-8")
            with SessionLocal() as db:
                m = registry.register_model_dir(db, out, notes="Demo model trained on SYNTHETIC data")
                if m.status == "candidate":
                    admin = db.execute(select(User).where(User.role == "admin").limit(1)).scalar_one_or_none()
                    if admin is not None:
                        registry.set_status(db, m, "approved", admin, "auto-approved demo model (synthetic data)")
                        p = db.get(Product, product)
                        if algo == activate and p is not None and p.active_model_id is None:
                            registry.activate(db, p, m, admin, "demo bootstrap")
                db.commit()
            done.append(f"{product}/{algo}")
    return done


def active_models_summary() -> list[dict]:
    with SessionLocal() as db:
        out = []
        for p in db.execute(select(Product)).scalars():
            m = db.get(ModelVersion, p.active_model_id) if p.active_model_id else None
            out.append({"product": p.key, "model": m.name if m else None})
        return out
