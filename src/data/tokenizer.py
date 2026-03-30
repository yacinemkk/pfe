"""
PHASE 3: Tokenisation pour Transformer (IoT-Tokenize)
Per docs/important.md

Etapes:
3.1: Transformation Structurée
3.2: Création du Vocabulaire
3.3: Encodage BPE
3.4: Padding et Tenseurs
"""

import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import yaml

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print(
        "Warning: tokenizers library not available. Install with: pip install tokenizers"
    )


class IoTTokenizer:
    """
    Tokeniseur BPE pour les flux IoT.

    Vocabulaire: ~52 000 tokens
    Tokens prédéfinis: ptcl, ipv, bi_dur, bi_pkt, etc.
    Tokens réservés: <s>, </s>, <pad>, <unk>, <mask>
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.vocab_size = self.config["tokenizer"]["vocab_size"]
        self.max_length = self.config["tokenizer"]["max_length"]
        self.special_tokens = self.config["tokenizer"]["special_tokens"]

        if TOKENIZERS_AVAILABLE:
            self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            self.tokenizer.pre_tokenizer = Whitespace()
            self.tokenizer.post_processor = TemplateProcessing(
                single="<s> $A </s>",
                special_tokens=[
                    ("<s>", 0),
                    ("</s>", 1),
                ],
            )
        else:
            self.tokenizer = None

    def transform_to_text(
        self,
        X_seq: np.ndarray,
        feature_names: List[str],
        format_type: str = "structured",
    ) -> List[str]:
        """
        Etape 3.1: Transformation Structurée.

        Convertit chaque flux en format:
        nom_feature1 valeur1; nom_feature2 valeur2; ...

        Exemple: ptcl 6; ipv 4; bi_dur 12.5; bi_pkt 150;
        """
        texts = []

        if format_type == "structured":
            for seq in X_seq:
                parts = []
                for flow in seq:
                    for val, name in zip(flow, feature_names):
                        parts.append(f"{name} {float(val):.4f};")
                texts.append(" ".join(parts))

        elif format_type == "compact":
            for seq in X_seq:
                parts = []
                for flow in seq:
                    flow_str = " ".join([f"{v:.4f}" for v in flow])
                    parts.append(flow_str)
                texts.append(" | ".join(parts))

        return texts

    def build_vocabulary(
        self, feature_names: List[str], verbose: bool = True
    ) -> List[str]:
        """
        Etape 3.2: Création du Vocabulaire.

        Tokens prédéfinis: noms des features
        Tokens réservés: <s>, </s>, <pad>, <unk>, <mask>
        """
        vocab = list(self.special_tokens)
        vocab.extend(feature_names)

        if verbose:
            print(f"  Vocabulaire de base: {len(vocab)} tokens")
            print(f"    Tokens réservés: {self.special_tokens}")
            print(f"    Features: {len(feature_names)}")

        return vocab

    def fit(
        self,
        X_train: np.ndarray,
        feature_names: List[str],
        save_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Entraîne le tokeniseur BPE sur les données d'entraînement.

        Args:
            X_train: Séquences d'entraînement
            feature_names: Noms des features
            save_path: Chemin de sauvegarde du tokeniseur
            verbose: Afficher les informations
        """
        if not TOKENIZERS_AVAILABLE:
            raise RuntimeError("tokenizers library not available")

        if verbose:
            print("\n[TOKENIZER] Entraînement BPE")

        texts = self.transform_to_text(X_train, feature_names)
        special_tokens = self.special_tokens + list(feature_names)

        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=self.vocab_size)

        self.tokenizer.train_from_iterator(texts, trainer)

        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(
            pad_id=pad_id, pad_token="<pad>", length=self.max_length
        )
        self.tokenizer.enable_truncation(max_length=self.max_length)

        if verbose:
            print(f"  Vocabulaire: {self.tokenizer.get_vocab_size():,} tokens")
            print(f"  Longueur max: {self.max_length}")

        if save_path:
            self.save(save_path)

    def load(self, load_path: str):
        """Charge un tokeniseur sauvegardé."""
        if not TOKENIZERS_AVAILABLE:
            raise RuntimeError("tokenizers library not available")

        self.tokenizer = Tokenizer.from_file(load_path)
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(
            pad_id=pad_id, pad_token="<pad>", length=self.max_length
        )
        self.tokenizer.enable_truncation(max_length=self.max_length)

    def save(self, save_path: str):
        """Sauvegarde le tokeniseur."""
        if self.tokenizer:
            self.tokenizer.save(save_path)

    def transform(self, X_seq: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Etape 3.4: Convertit les séquences en tenseurs de tokens.

        Returns:
            Array de token IDs (n_sequences, max_length)
        """
        if not TOKENIZERS_AVAILABLE:
            raise RuntimeError("tokenizers library not available")

        texts = self.transform_to_text(X_seq, feature_names)
        encoded = self.tokenizer.encode_batch(texts)

        return np.array([e.ids for e in encoded], dtype=np.int64)

    def decode(self, token_ids: np.ndarray) -> List[str]:
        """Décode les token IDs en texte."""
        if not TOKENIZERS_AVAILABLE:
            raise RuntimeError("tokenizers library not available")

        return [self.tokenizer.decode(ids) for ids in token_ids]

    def encode_single(self, text: str) -> np.ndarray:
        """Encode un texte unique."""
        if not TOKENIZERS_AVAILABLE:
            raise RuntimeError("tokenizers library not available")

        encoded = self.tokenizer.encode(text)
        return np.array(encoded.ids, dtype=np.int64)


class SimpleTokenizer:
    """
    Tokeniseur simple sans BPE (alternative si tokenizers n'est pas disponible).
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.max_length = self.config["tokenizer"]["max_length"]
        self.special_tokens = self.config["tokenizer"]["special_tokens"]
        self.vocab: dict = {}
        self.reverse_vocab: dict = {}

    def fit(self, X_train: np.ndarray, feature_names: List[str], verbose: bool = True):
        """Construit un vocabulaire simple."""
        idx = 0
        for token in self.special_tokens:
            self.vocab[token] = idx
            idx += 1

        for name in feature_names:
            if name not in self.vocab:
                self.vocab[name] = idx
                idx += 1

        for seq in X_train[:100]:
            for flow in seq:
                for val in flow:
                    token = f"{float(val):.4f}"
                    if (
                        token not in self.vocab
                        and idx < self.config["tokenizer"]["vocab_size"]
                    ):
                        self.vocab[token] = idx
                        idx += 1

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"  Vocabulaire: {len(self.vocab):,} tokens")

    def transform(self, X_seq: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Convertit en token IDs."""
        n_seqs = len(X_seq)
        token_ids = np.zeros((n_seqs, self.max_length), dtype=np.int64)

        unk_id = self.vocab.get("<unk>", 0)

        for i, seq in enumerate(X_seq):
            tokens = []
            for flow in seq:
                for val, name in zip(flow, feature_names):
                    tokens.append(self.vocab.get(name, unk_id))
                    token = f"{float(val):.4f}"
                    tokens.append(self.vocab.get(token, unk_id))

            tokens = tokens[: self.max_length]
            token_ids[i, : len(tokens)] = tokens

        return token_ids


def create_tokenizer(config_path: str = "config/config.yaml") -> IoTTokenizer:
    """Factory pour créer le tokeniseur approprié."""
    if TOKENIZERS_AVAILABLE:
        return IoTTokenizer(config_path)
    else:
        return SimpleTokenizer(config_path)


if __name__ == "__main__":
    import numpy as np

    X = np.random.randn(100, 10, 36).astype(np.float32)
    feature_names = [f"feat_{i}" for i in range(36)]

    tokenizer = create_tokenizer()
    tokenizer.fit(X, feature_names)

    tokens = tokenizer.transform(X[:5], feature_names)
    print(f"Tokens shape: {tokens.shape}")
