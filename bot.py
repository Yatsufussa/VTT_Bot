import warnings
import logging
import librosa
import pydub
from pydub import AudioSegment
import os
import aiohttp
import aiofiles
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ContentType
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from sqlalchemy.orm import Session
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from database import Session, Transcription  # Database setup

# Suppress all UserWarnings (including the ones for Wav2Vec2 model)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load token from environment variables
TOKEN = "7728775306:AAHUE91quZiTd0X4s2gS5A7VeuJL_bfjeMw"
print(os.path.exists(r'C:\Users\shuhr\Downloads\ffmpeg-7.1-essentials_build\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe'))
AudioSegment.ffmpeg = r'C:\Users\shuhr\Downloads\ffmpeg-7.1-essentials_build\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe'

# Initialize bot and dispatcher
bot = Bot(TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# FSM for corrections
class CorrectionState(StatesGroup):
    awaiting_correction = State()

# Load fine-tuned model and processor
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-finetuned")
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-finetuned")

# Helper function for transcription
def transcribe_audio(file_path):
    """Transcribes the audio file using Wav2Vec2."""
    waveform, sample_rate = librosa.load(file_path, sr=16000)  # Ensure audio is 16kHz
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Command handler for "/start"
@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(f"Hello {message.from_user.first_name}! Send me you audio and i will help you to transcribe")



# Ensure the 'audio' directory exists
if not os.path.exists('audio'):
    os.makedirs('audio')


async def convert_telegram_voice_to_wav(voice_message, output_directory="audio"):
    """
    Converts a Telegram voice message to a .wav file.

    Args:
        voice_message: Telegram voice message object.
        output_directory: Directory to save the converted .wav file.

    Returns:
        Path to the saved .wav file.
    """
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define file paths
    ogg_file_path = os.path.join(output_directory, f"{voice_message.file_id}.ogg")
    wav_file_path = os.path.join(output_directory, f"{voice_message.file_id}.wav")

    # Download the voice message
    file = await bot.get_file(voice_message.file_id)
    file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file.file_path}"

    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as resp:
            if resp.status == 200:
                async with aiofiles.open(ogg_file_path, "wb") as f:
                    await f.write(await resp.read())
            else:
                raise Exception("Failed to download the voice message.")

    # Convert OGG to WAV using pydub
    audio = AudioSegment.from_file(ogg_file_path, format="ogg")
    audio.export(wav_file_path, format="wav")
    os.remove(ogg_file_path)  # Clean up the original .ogg file

    return wav_file_path

@dp.message(F.content_type == ContentType.VOICE)
async def handle_voice_message(message: Message, state: FSMContext):
    try:
        # Convert the voice message to .wav
        wav_file_path = await convert_telegram_voice_to_wav(message.voice)
        await message.reply("Voice message converted to .wav. Starting transcription...")

        # Transcribe the audio
        transcription = transcribe_audio(wav_file_path)

        # Save transcription to the database
        db_session = Session()
        new_transcription = Transcription(audio_file_path=wav_file_path, generated_transcription=transcription)
        db_session.add(new_transcription)
        db_session.commit()

        # Respond with the transcription
        await message.reply(
            f"Here's the transcription:\n\n{transcription}\n\nIs it correct? If not, reply with the correct version."
        )

        # Transition to correction state
        await state.set_state(CorrectionState.awaiting_correction)
    except Exception as e:
        logging.error(f"Error handling voice message: {e}")
        await message.reply("An error occurred while processing your voice message.")


# Handle transcription correction
@dp.message(CorrectionState.awaiting_correction)
async def handle_correction(message: Message, state: FSMContext):
    db_session = Session()
    transcription = db_session.query(Transcription).order_by(Transcription.id.desc()).first()
    transcription.corrected_transcription = message.text
    db_session.commit()
    await message.reply("Thanks for the correction! I'll use this to improve.")
    await state.clear()

# Start polling to get updates
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

