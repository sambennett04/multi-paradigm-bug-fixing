"""Helpers for constructing the custom T5 architecture."""
from transformers import T5Config, T5ForConditionalGeneration


def build_t5_model(tokenizer, embd_hidd_dim, feed_forward_dim, key_val_proj_dim, num_heads, num_encoder_layers, num_decoder_layers):
    """Build a T5 seq2seq model whose vocabulary matches the custom tokenizer."""
    t5_config = T5Config(
        decoder_start_token_id=tokenizer.convert_tokens_to_ids(["<pad>"])[0]
        if "<pad>" in tokenizer.get_vocab() else 0
    )
    t5_config.d_model = embd_hidd_dim
    t5_config.d_ff = feed_forward_dim
    t5_config.d_kv = key_val_proj_dim
    t5_config.num_heads = num_heads
    t5_config.num_layers = num_encoder_layers
    t5_config.num_decoder_layers = num_decoder_layers
    t5_config.vocab_size = len(tokenizer)

    model = T5ForConditionalGeneration(config=t5_config)

    # The base config's embedding table does not know about the custom
    # SentencePiece vocabulary until we resize it here.
    model.resize_token_embeddings(len(tokenizer))

    return model
