"""ORM models. Schema changes must be accompanied by an Alembic migration (alembic/versions)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, event
from sqlalchemy import false as sa_false
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="inspector")  # admin | inspector | viewer
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # incremented on logout / password or role change / deactivation -> invalidates all issued tokens
    token_version: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # set for weak/default first-run passwords and admin resets; cleared when the user sets their own password
    password_change_required: Mapped[bool] = mapped_column(Boolean, default=False, server_default=sa_false())


class Product(Base):
    __tablename__ = "products"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, default="")
    active_model_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id", use_alter=True,
                                                                   name="fk_product_active_model"), nullable=True)
    secondary_model_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id", use_alter=True,
                                                                      name="fk_product_secondary_model"), nullable=True)
    policy: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # overrides of global decision policy
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    product_key: Mapped[str] = mapped_column(String(64), index=True)
    algorithm: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer, default=1)
    artifact_dir: Mapped[str] = mapped_column(String(512), unique=True)
    weights_sha256: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(16), default="candidate")  # candidate|approved|archived|rejected
    thresholds: Mapped[dict] = mapped_column(JSON)  # effective thresholds (admin-editable)
    original_thresholds: Mapped[dict] = mapped_column(JSON)  # validation-selected, immutable
    calibration: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    validation_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    test_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    training: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    dataset: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    history: Mapped[list | None] = mapped_column(JSON, nullable=True)
    has_classifier: Mapped[bool] = mapped_column(Boolean, default=False)
    classifier_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    approved_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelActivation(Base):
    __tablename__ = "model_activations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_key: Mapped[str] = mapped_column(String(64), index=True)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    previous_model_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Inspection(Base):
    __tablename__ = "inspections"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    product_key: Mapped[str] = mapped_column(String(64), index=True)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True, index=True)
    model_name: Mapped[str] = mapped_column(String(128), default="")
    algorithm: Mapped[str] = mapped_column(String(32), default="")
    source: Mapped[str] = mapped_column(String(16), default="upload")  # upload|batch|camera|api
    batch_code: Mapped[str] = mapped_column(String(128), default="", index=True)  # production batch / lot
    camera_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    station: Mapped[str] = mapped_column(String(64), default="")
    external_ref: Mapped[str] = mapped_column(String(128), default="")  # MES unit / serial id
    job_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("jobs.id"), nullable=True, index=True)
    operator_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    filename: Mapped[str] = mapped_column(String(255), default="")
    image_sha256: Mapped[str] = mapped_column(String(64), index=True)
    image_path: Mapped[str] = mapped_column(String(512), default="")
    display_path: Mapped[str] = mapped_column(String(512), default="")
    heatmap_path: Mapped[str] = mapped_column(String(512), default="")
    overlay_path: Mapped[str] = mapped_column(String(512), default="")
    mask_path: Mapped[str] = mapped_column(String(512), default="")
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    fail_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    pass_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    seg_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    normalized_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibrated_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    ood_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[str] = mapped_column(String(8), index=True)  # PASS | FAIL | REVIEW
    severity: Mapped[str] = mapped_column(String(10), default="none")
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    regions: Mapped[list] = mapped_column(JSON, default=list)
    quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    defect_type: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    secondary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(8), default="ok")  # ok | error
    error: Mapped[str] = mapped_column(Text, default="")
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    review_status: Mapped[str] = mapped_column(String(10), default="none", index=True)  # none|pending|reviewed
    final_label: Mapped[str | None] = mapped_column(String(10), nullable=True)  # defective | normal
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duplicate_of: Mapped[str | None] = mapped_column(String(32), nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True)
    images_purged: Mapped[bool] = mapped_column(Boolean, default=False)

    reviews: Mapped[list[Review]] = relationship(back_populates="inspection", order_by="Review.created_at")

    __table_args__ = (Index("ix_inspections_product_created", "product_key", "created_at"),)


class Review(Base):
    """Append-only reviewer decisions. The latest review determines ``Inspection.final_label``."""

    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id"), index=True)
    reviewer_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    label: Mapped[str] = mapped_column(String(10))  # defective | normal
    defect_type: Mapped[str] = mapped_column(String(64), default="")
    severity: Mapped[str] = mapped_column(String(10), default="none")
    comment: Mapped[str] = mapped_column(Text, default="")
    regions: Mapped[list] = mapped_column(JSON, default=list)  # reviewer-marked boxes [x,y,w,h] (image px)
    agrees_with_model: Mapped[bool] = mapped_column(Boolean, default=True)
    add_to_feedback: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    inspection: Mapped[Inspection] = relationship(back_populates="reviews")
    reviewer: Mapped[User] = relationship()


class AuditEvent(Base):
    """Append-only, hash-chained audit log (updates/deletes are blocked at ORM level)."""

    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actor_name: Mapped[str] = mapped_column(String(64), default="system")
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(32), default="")
    entity_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    prev_hash: Mapped[str] = mapped_column(String(64), default="")
    hash: Mapped[str] = mapped_column(String(64), default="")


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    type: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(12), default="queued", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=2)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DatasetRecord(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    product_key: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # synthetic | visa | custom | feedback
    path: Mapped[str] = mapped_column(String(512))
    version: Mapped[str] = mapped_column(String(64), default="1")
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    manifest_sha256: Mapped[str] = mapped_column(String(64), default="")
    info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    quality_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    notes: Mapped[str] = mapped_column(Text, default="")


class FeedbackSample(Base):
    """Reviewed inspection queued for possible inclusion in a training dataset (needs admin approval)."""

    __tablename__ = "feedback_samples"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id"), index=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews.id"))
    product_key: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(10))
    defect_type: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(10), default="pending", index=True)  # pending|approved|rejected|used
    decided_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)  # benchmark | robustness | domain_shift | classifier
    product_key: Mapped[str] = mapped_column(String(64), index=True)
    dataset_name: Mapped[str] = mapped_column(String(128), default="")
    model_ids: Mapped[list] = mapped_column(JSON, default=list)
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    artifact_dir: Mapped[str] = mapped_column(String(512), default="")
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AppSetting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)


class ProviderConfig(Base):
    __tablename__ = "provider_configs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    kind: Mapped[str] = mapped_column(String(24))  # openai | anthropic | openrouter | openai_compatible | ollama
    base_url: Mapped[str] = mapped_column(String(255), default="")
    model: Mapped[str] = mapped_column(String(128), default="")
    api_key_enc: Mapped[str] = mapped_column(Text, default="")  # Fernet-encrypted, never returned by the API
    api_key_env: Mapped[str] = mapped_column(String(64), default="")  # alternatively read key from env var
    timeout_s: Mapped[float] = mapped_column(Float, default=30.0)
    max_retries: Mapped[int] = mapped_column(Integer, default=2)
    max_tokens: Mapped[int] = mapped_column(Integer, default=600)
    temperature: Mapped[float] = mapped_column(Float, default=0.2)
    price_in_per_mtok: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_out_per_mtok: Mapped[float | None] = mapped_column(Float, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class LLMUsage(Base):
    __tablename__ = "llm_usage"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider_id: Mapped[int | None] = mapped_column(ForeignKey("provider_configs.id"), nullable=True)
    provider_name: Mapped[str] = mapped_column(String(64), default="local")
    model: Mapped[str] = mapped_column(String(128), default="")
    purpose: Mapped[str] = mapped_column(String(32), default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AppendOnlyViolation(RuntimeError):
    pass


@event.listens_for(AuditEvent, "before_update")
def _audit_no_update(mapper, connection, target):  # pragma: no cover - exercised in tests
    raise AppendOnlyViolation("Audit events are append-only")


@event.listens_for(AuditEvent, "before_delete")
def _audit_no_delete(mapper, connection, target):
    raise AppendOnlyViolation("Audit events are append-only")


@event.listens_for(Review, "before_update")
def _review_no_update(mapper, connection, target):
    raise AppendOnlyViolation("Reviews are append-only; submit a new review to correct a decision")


@event.listens_for(Review, "before_delete")
def _review_no_delete(mapper, connection, target):
    raise AppendOnlyViolation("Reviews are append-only")
