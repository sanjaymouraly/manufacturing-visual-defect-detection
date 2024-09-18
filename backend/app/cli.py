"""DefectLens administration CLI.

    python -m app.cli init-db                 # run migrations + create admin user
    python -m app.cli bootstrap-demo          # synthetic demo datasets + trained/approved/active demo models
    python -m app.cli register-datasets       # register prepared datasets in var/datasets (e.g. VisA)
    python -m app.cli train-visa --category pcb1 [--activate patchcore]
    python -m app.cli create-user NAME --role inspector
    python -m app.cli backup --out backup.zip  /  restore backup.zip
    python -m app.cli retention [--dry-run]
    python -m app.cli worker                  # standalone job worker (DL_WORKER_MODE=external)
    python -m app.cli verify-audit
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
from pathlib import Path


def _init(migrate: bool = True):
    from app.config import get_settings
    from app.db import init_engine
    from app.logging_setup import setup_logging

    s = get_settings()
    setup_logging(s.log_format, s.log_level)
    init_engine(s.database_url)
    if migrate:
        from app.services.bootstrap import run_migrations, seed_users

        run_migrations()
        seed_users()
    return s


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="defectlens", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("init-db")
    p = sp.add_parser("bootstrap-demo")
    p.add_argument("--algorithms", nargs="+", default=["patchcore", "padim", "pixelstats"])
    sp.add_parser("register-datasets")
    p = sp.add_parser("train-visa")
    p.add_argument("--category", default="pcb1")
    p.add_argument("--algorithms", nargs="+", default=["patchcore", "padim", "pixelstats"])
    p.add_argument("--activate", default="patchcore")
    p.add_argument("--max-train-images", type=int, default=None)
    p = sp.add_parser("create-user")
    p.add_argument("username")
    p.add_argument("--role", default="inspector", choices=["admin", "inspector", "viewer"])
    p.add_argument("--full-name", default="")
    p = sp.add_parser("backup")
    p.add_argument("--out", required=True)
    p = sp.add_parser("restore")
    p.add_argument("archive")
    p = sp.add_parser("retention")
    p.add_argument("--dry-run", action="store_true")
    sp.add_parser("worker")
    sp.add_parser("verify-audit")
    a = ap.parse_args(argv)

    if a.cmd == "restore":
        _init(migrate=False)
        from app.services.retention import restore

        print(json.dumps(restore(Path(a.archive))))
        return
    s = _init()
    log = logging.getLogger("defectlens.cli")
    if a.cmd == "init-db":
        from app.services.bootstrap import sync_registry

        sync_registry()
        print("Database initialised.")
    elif a.cmd == "bootstrap-demo":
        from app.services.bootstrap import active_models_summary, bootstrap_demo

        done = bootstrap_demo(tuple(a.algorithms))
        print("Demo models ready:", ", ".join(done))
        print(json.dumps(active_models_summary(), indent=2))
    elif a.cmd == "register-datasets":
        from app.services.bootstrap import sync_registry

        sync_registry()
        print("Datasets and models registered.")
    elif a.cmd == "train-visa":
        from sqlalchemy import select

        from app.db import SessionLocal
        from app.models import Product, User
        from app.services import registry
        from app.services.datasets_service import register_dataset
        from defectml.data.manifest import Dataset
        from defectml.evaluation.benchmark import evaluate_model
        from defectml.training import train_on_dataset

        ds_dir = s.datasets_dir / f"visa-{a.category}"
        if not (ds_dir / "dataset.json").exists():
            sys.exit(f"{ds_dir} not found - run: python -m defectml.cli prepare-visa ... --out {s.datasets_dir}")
        ds = Dataset(ds_dir)
        with SessionLocal() as db:
            register_dataset(db, ds_dir, validate=False)
            db.commit()
        for algo in a.algorithms:
            out = s.models_dir / f"visa-{a.category}-{algo}"
            if not (out / "model.json").exists():
                log.info("training %s on visa-%s", algo, a.category)
                train_on_dataset(ds, algo, out, name=f"visa-{a.category}-{algo}", config={"device": s.device},
                                 max_train_images=a.max_train_images)
                test = evaluate_model(out, ds, "test")
                test.pop("scores", None)
                (out / "test_metrics.json").write_text(json.dumps(test, indent=2, default=float), encoding="utf-8")
            with SessionLocal() as db:
                m = registry.register_model_dir(db, out)
                admin = db.execute(select(User).where(User.role == "admin").limit(1)).scalar_one()
                if algo == a.activate and m.status == "candidate":
                    registry.set_status(db, m, "approved", admin, "approved after held-out evaluation (VisA)")
                    prod = db.get(Product, m.product_key)
                    if prod is not None and prod.active_model_id is None:
                        registry.activate(db, prod, m, admin, "initial VisA model")
                db.commit()
            print(f"visa-{a.category}/{algo}: done")
    elif a.cmd == "create-user":
        from app.db import SessionLocal
        from app.models import User
        from app.security import hash_password

        pw = getpass.getpass("Password: ")
        if len(pw) < 8:
            sys.exit("Password must be at least 8 characters")
        with SessionLocal() as db:
            db.add(User(username=a.username, full_name=a.full_name, role=a.role, password_hash=hash_password(pw)))
            db.commit()
        print(f"User {a.username} created.")
    elif a.cmd == "backup":
        from app.services.retention import backup

        print(f"Backup written to {backup(Path(a.out))}")
    elif a.cmd == "retention":
        from app.services.retention import apply_retention

        print(json.dumps(apply_retention(a.dry_run)))
    elif a.cmd == "worker":
        from app import jobs
        from app.services import batch, datasets_service, ml_jobs, reports, retention  # noqa: F401

        jobs.recover_stale(s.job_stale_seconds)
        w = jobs.Worker(s.worker_poll_seconds)
        print("Worker running (Ctrl+C to stop)")
        try:
            w.run()
        except KeyboardInterrupt:
            w.stop()
    elif a.cmd == "verify-audit":
        from app import audit
        from app.db import SessionLocal

        with SessionLocal() as db:
            print(json.dumps(audit.verify_chain(db)))


if __name__ == "__main__":
    main()
