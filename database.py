from sqlalchemy import Column, Integer, String, Text, create_engine, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import text

Base = declarative_base()

class Transcription(Base):
    __tablename__ = 'transcriptions'

    id = Column(Integer, primary_key=True)
    audio_file_path = Column(String, nullable=False)
    generated_transcription = Column(Text, nullable=True)
    corrected_transcription = Column(Text, nullable=True)
    used_for_training = Column(Boolean, default=False)  # New column

# Database setup
DATABASE_URL = "postgresql+psycopg2://postgres:asd54321@localhost/vtt_telegram_bot"
engine = create_engine(DATABASE_URL)

# Session factory
Session = sessionmaker(bind=engine)

# Test connection
with engine.connect() as connection:
    result = connection.execute(text("SELECT version();"))
    print(f"Connected to: {result.fetchone()}")
