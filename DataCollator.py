from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from transformers import Wav2Vec2Processor

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator for CTC (Connectionist Temporal Classification) tasks with improved padding handling.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Pads input values and labels to ensure all tensors in a batch have the same length.
        """
        # Extract input values and labels from features
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input values
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"  # Return PyTorch tensors
        )

        # Pad labels (CTC requires labels to be padded to the longest sequence in the batch)
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )

        # Replace padding token ID in labels with -100, as required for CTC loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )
        batch["labels"] = labels

        # Return the processed batch with padded input values and labels
        return batch
