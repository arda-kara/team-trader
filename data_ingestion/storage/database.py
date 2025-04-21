"""
Database connection and session management for the data ingestion layer.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager

from ..config.settings import settings

# Create engine
engine = create_engine(
    settings.database.connection_string,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=settings.debug
)

# Create session factory
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

@contextmanager
def get_db_session():
    """Context manager for database sessions.
    
    Yields:
        Session: SQLAlchemy session object
        
    Example:
        with get_db_session() as session:
            result = session.query(Model).all()
    """
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def init_db():
    """Initialize database tables."""
    from .models import Base
    Base.metadata.create_all(bind=engine)

def drop_db():
    """Drop all database tables."""
    from .models import Base
    Base.metadata.drop_all(bind=engine)
