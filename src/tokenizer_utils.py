"""Tokenizer helpers for the custom T5-style Java pipeline."""
import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from transformers import PreTrainedTokenizerFast
from pathlib import Path


def train_tokenizer(corpus_file: Path, model_prefix: str):
    """Train a SentencePiece unigram tokenizer for code.

    The training corpus is expected to contain one flattened method per line.
    Sentinel tokens are registered up front so the resulting vocabulary can be
    used directly for T5 span-corruption pretraining.
    """
    sentinel_tokens = [f"<extra_id_{i}>" for i in range(100)]
    user_defined_symbols = ",".join(sentinel_tokens)

    spm.SentencePieceTrainer.train(
        input=corpus_file,            # one function per line
        model_prefix=model_prefix,    # output file prefix
        vocab_size=16384,             # target vocabulary size
        hard_vocab_limit=False,       # allow slight deviation
        model_type="unigram",         # Unigram algorithm
        pad_id=0,                     # <pad> = ID 0
        eos_id=1,                     # </s>  = ID 1
        unk_id=2,                     # <unk> = ID 2
        bos_id=-1,                    # T5 doesn't use BOS
        user_defined_symbols=user_defined_symbols,
        character_coverage=1.0,      # keep all code characters available
    )


def load_tokenizer(model_prefix: Path):
    """Reload a trained SentencePiece model as a Hugging Face fast tokenizer."""
    model_path = str(model_prefix.with_suffix(".model"))

    # Load the original SentencePiece model so we can recover pieces and scores.
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    # Rebuild the vocabulary expected by the HF Unigram backend.
    vocab = [(sp.IdToPiece(i), sp.GetScore(i)) for i in range(sp.GetPieceSize())]

    # Mirror SentencePiece tokenization behavior inside Transformers.
    tokenizer_obj = Tokenizer(Unigram(vocab, unk_id=2))
    tokenizer_obj.pre_tokenizer = Metaspace()
    tokenizer_obj.decoder = MetaspaceDecoder()

    sentinel_tokens = [f"<extra_id_{i}>" for i in range(100)]

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=sentinel_tokens,
    )

    return tokenizer
