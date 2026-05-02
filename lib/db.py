import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, JSON, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./dev.db"

# check_same_thread=False is required for SQLite when used with FastAPI
# (FastAPI handles requests in multiple threads)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_number = Column(Integer, nullable=False)
    prompt_text = Column(String, nullable=False)
    accuracy_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    results = relationship("EvaluationResult", back_populates="prompt_version")


class TestCase(Base):
    __tablename__ = "test_cases"

    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(String, nullable=False)
    expected_function_name = Column(String, nullable=True)
    expected_params = Column(JSON, nullable=True)

    results = relationship("EvaluationResult", back_populates="test_case")


# failure_type is a plain String (not an Enum) so SQLite can store it without issues.
# Allowed values are enforced in evaluator.py.
class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    passed = Column(Boolean, nullable=False)
    failure_type = Column(String, nullable=True)
    actual_function_name = Column(String, nullable=True)
    actual_params = Column(JSON, nullable=True)
    prompt_version_id = Column(Integer, ForeignKey("prompt_versions.id"), nullable=False)
    test_case_id = Column(Integer, ForeignKey("test_cases.id"), nullable=False)

    prompt_version = relationship("PromptVersion", back_populates="results")
    test_case = relationship("TestCase", back_populates="results")


def create_tables():
    """Create all tables if they don't exist. Called once on app startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and closes it when the request finishes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
