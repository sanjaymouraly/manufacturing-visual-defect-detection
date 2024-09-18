"""Alembic environment: uses DL_DATABASE_URL / app settings; supports programmatic connections."""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from app import models  # noqa: F401 - register tables
from app.db import Base

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)
target_metadata = Base.metadata

if not config.get_main_option("sqlalchemy.url") or config.get_main_option("sqlalchemy.url").startswith("driver://"):
    from app.config import get_settings

    config.set_main_option("sqlalchemy.url", get_settings().database_url)


def include_object(obj, name, type_, reflected, compare_to):
    # The circular products<->model_versions FKs are only created on PostgreSQL (SQLite cannot ALTER ADD CONSTRAINT).
    if type_ == "foreign_key_constraint" and name in ("fk_product_active_model", "fk_product_secondary_model"):
        return context.get_context().dialect.name != "sqlite"
    return True


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata,
                      literal_binds=True, dialect_opts={"paramstyle": "named"}, render_as_batch=True,
                      include_object=include_object)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True,
                          include_object=include_object)
        with context.begin_transaction():
            context.run_migrations()
        return
    connectable = engine_from_config(config.get_section(config.config_ini_section, {}), prefix="sqlalchemy.",
                                     poolclass=pool.NullPool)
    with connectable.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=True, include_object=include_object)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
