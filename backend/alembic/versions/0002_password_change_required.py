"""users.password_change_required

Flags accounts that still use a weak/default first-run password or an admin-set temporary password, so the
UI can prompt the user to choose their own.

Revision ID: 0002_pw_change_required
Revises: 3ac2e1da3a35
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_pw_change_required"
down_revision: str | Sequence[str] | None = "3ac2e1da3a35"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Plain ADD COLUMN with a constant default is supported natively by SQLite and PostgreSQL. Do NOT use
    # batch_alter_table here: on SQLite it rebuilds the table (copy + DROP TABLE users), which fails with
    # foreign keys enforced as soon as any row references a user (i.e. on every real deployment).
    op.add_column("users", sa.Column("password_change_required", sa.Boolean(), nullable=False,
                                     server_default=sa.false()))


def downgrade() -> None:
    op.drop_column("users", "password_change_required")  # native DROP COLUMN (SQLite >= 3.35, PostgreSQL)
