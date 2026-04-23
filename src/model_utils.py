"""Utilities for model construction and loading."""
from transformers import T5Config, T5ForConditionalGeneration

def build_t5_model(tokenizer, embd_hidd_dim, feed_forward_dim, key_val_proj_dim, num_heads, num_encoder_layers, num_decoder_layers):
    """Initalize t5 model using given parameters"""
    t5_config = T5Config(
        decoder_start_token_id=tokenizer.convert_tokens_to_ids(["<pad>"])[0]
        if "<pad>" in tokenizer.get_vocab() else 0
    )
    t5_config.d_model = embd_hidd_dim          # embedding and hidden state dimension
    t5_config.d_ff = feed_forward_dim            # feed-forward layer dimension
    t5_config.d_kv = key_val_proj_dim              # key/value projection dimension
    t5_config.num_heads = num_heads          # attention heads
    t5_config.num_layers = num_encoder_layers         # encoder layers
    t5_config.num_decoder_layers = num_decoder_layers # decoder layers
    t5_config.vocab_size = len(tokenizer)

    model = T5ForConditionalGeneration(config=t5_config)

    #resize token embeddings to match tokenizer vocabulary size
    #without, the model crashes when it encounters token IDs beyound the default vocabulary size
    model.resize_token_embeddings(len(tokenizer))

    return model