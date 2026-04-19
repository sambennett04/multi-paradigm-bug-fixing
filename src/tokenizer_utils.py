"""Utilities for tokenizer training and inference."""
import sentencepiece as spm

def train_tokenizer(corpus_file, model_prefix):
    """Trains a SentencePiece tokenizer on the given corpus and writes model and vocab files to location specified by provided model_prefix."""
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
        character_coverage=1.0,
    )

def load_tokenizer():
    pass
