import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, JSON, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./dev.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    tools_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    test_cases = relationship("TestCase", back_populates="scenario", cascade="all, delete-orphan")
    prompt_versions = relationship("PromptVersion", back_populates="scenario", cascade="all, delete-orphan")


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id = Column(Integer, primary_key=True, index=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    prompt_text = Column(String, nullable=False)
    accuracy_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    scenario = relationship("Scenario", back_populates="prompt_versions")
    results = relationship("EvaluationResult", back_populates="prompt_version")


class TestCase(Base):
    __tablename__ = "test_cases"

    id = Column(Integer, primary_key=True, index=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    user_message = Column(String, nullable=False)
    expected_function_name = Column(String, nullable=True)
    expected_params = Column(JSON, nullable=True)

    scenario = relationship("Scenario", back_populates="test_cases")
    results = relationship("EvaluationResult", back_populates="test_case")


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
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
