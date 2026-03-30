"""
IoT-Tokenize Pipeline for the Transformer
Performs Phase 2 string-formatting and BPE tokenization.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import numpy as np

class IoTTokenizer:
    def __init__(self, vocab_size=52000, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            special_tokens=[
                ("<s>", 0),
                ("</s>", 1),
            ],
        )
        
    def _to_text(self, X_seq, feature_names):
        """Convert numerical sequences into formatted text."""
        texts = []
        for seq in X_seq:
            parts = []
            for flow in seq:
                for val, name in zip(flow, feature_names):
                    parts.append(f"{name} {float(val):.4f};")
            texts.append(" ".join(parts))
        return texts

    def fit(self, X_train, feature_names, save_path="iot_tokenizer.json"):
        texts = self._to_text(X_train, feature_names)
        
        special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"] + list(feature_names)
        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=self.vocab_size)
        
        self.tokenizer.train_from_iterator(texts, trainer)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>", length=self.max_length)
        self.tokenizer.enable_truncation(max_length=self.max_length)
        
        if save_path:
            self.tokenizer.save(save_path)
            
    def load(self, save_path="iot_tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(save_path)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>", length=self.max_length)
        self.tokenizer.enable_truncation(max_length=self.max_length)
        
    def transform(self, X_seq, feature_names):
        texts = self._to_text(X_seq, feature_names)
        encoded = self.tokenizer.encode_batch(texts)
        return np.array([e.ids for e in encoded])
