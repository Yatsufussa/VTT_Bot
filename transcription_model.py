from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import text
from datasets import Dataset, DatasetDict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments, DataCollatorWithPadding
import torchaudio
import os
from database import *
import torch
from DataCollator import *
# Wav2Vec2 model setup
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
# Prepare dataset for fine-tuning
def prepare_dataset(batch):
    audio_path = batch["audio_file_path"]
    transcription = batch["corrected_transcription"]

    # Load and resample audio
    waveform, sample_rate = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

    # Ensure the waveform length is between 2 and 7 seconds
    max_length = 16000 * 7  # 7 seconds
    min_length = 16000 * 2  # 2 seconds
    current_length = waveform.shape[1]

    if current_length > max_length:
        waveform = waveform[:, :max_length]
    elif current_length < min_length:
        padding_length = min_length - current_length
        waveform = torch.cat([waveform, torch.zeros(1, padding_length)], dim=1)

    # Process audio and transcription
    input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000).input_values[0]
    labels = processor.tokenizer(transcription).input_ids

    # Return prepared batch
    return {
        "input_values": input_values,
        "labels": labels,
    }



# Load data from the database
def load_data_from_db():
    session = Session()
    try:
        # Query only unused data
        data = session.query(Transcription).filter(
            Transcription.corrected_transcription != None,
            Transcription.used_for_training == False
        ).all()
        dataset = [
            {"audio_file_path": record.audio_file_path, "corrected_transcription": record.corrected_transcription}
            for record in data
        ]
        return dataset
    finally:
        session.close()


def mark_data_as_used(data):
    session = Session()
    try:
        for item in data:
            session.query(Transcription).filter(
                Transcription.audio_file_path == item["audio_file_path"]
            ).update({"used_for_training": True})
        session.commit()
    finally:
        session.close()

# Fine-tune the model
def fine_tune_model():
    print("Loading data from database...")
    data = load_data_from_db()

    if not data:
        print("No corrected transcriptions available in the database. Fine-tuning skipped.")
        return

    print(f"Found {len(data)} records for fine-tuning.")

    # Convert data into a dictionary suitable for Hugging Face Datasets
    data_dict = {
        "audio_file_path": [item["audio_file_path"] for item in data],
        "corrected_transcription": [item["corrected_transcription"] for item in data],
    }

    # Create train and test datasets
    train_dataset = Dataset.from_dict(data_dict)
    test_dataset = Dataset.from_dict({
        "audio_file_path": data_dict["audio_file_path"][:1],  # Use the first record for testing
        "corrected_transcription": data_dict["corrected_transcription"][:1],
    })

    # Combine into a DatasetDict
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Preprocess datasets
    dataset = dataset.map(prepare_dataset, remove_columns=["audio_file_path", "corrected_transcription"], num_proc=4)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./wav2vec2-finetuned",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        fp16=True,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,  # Add this line
        tokenizer=processor.tokenizer,  # Updated to tokenizer, not feature_extractor
    )

    # Start fine-tuning
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning completed.")

    # Mark the data as used after fine-tuning
    mark_data_as_used(data)

    # Save the fine-tuned model
    model.save_pretrained("./wav2vec2-finetuned")
    processor.save_pretrained("./wav2vec2-finetuned")
    print("Fine-tuned model saved.")

# Main entry point
if __name__ == "__main__":
    # Ensure 'audio' directory exists
    if not os.path.exists('audio'):
        os.makedirs('audio')

    # Start fine-tuning
    fine_tune_model()

