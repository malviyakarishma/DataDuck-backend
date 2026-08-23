from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


from sqlalchemy import text


async def init_db():
    """Create all tables on startup and apply missing column migrations."""
    from app.models import user, database_connection, conversation, query_history, schema_metadata  # noqa
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        try:
            await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS otp_code VARCHAR(6);"))
            await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS otp_expires_at TIMESTAMP WITH TIME ZONE;"))
        except Exception as e:
            logger.warning(f"Note on OTP migration: {e}")
    logger.info("Database tables initialized.")


async def close_db():
    await engine.dispose()
    logger.info("Database connections closed.")
