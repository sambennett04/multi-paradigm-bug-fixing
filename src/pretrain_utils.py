"""Utilities for T5-style span-corruption pretraining."""
from dataclasses import dataclass, field
import random
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

@dataclass
class PretrainConfig:
    """Configuration for unsupervised code pretraining."""
    output_dir : str
    min_tokens: int = 10
    max_tokens: int = 512
    corruption_rate: float = 0.15
    sentinel_ids: list[str] = field(
        default_factory=lambda: [f"<extra_id_{i}>" for i in range(100)]
    )
    max_samples: int | None = None
    num_epochs: int = 3
    batch_size: int = 8
    device : str = "cuda" if torch.cuda.is_available() else "cpu"


def filter_by_length(token_ids : list, pretrain_config : PretrainConfig):
    """Keep only examples within the configured token-length bounds."""
    return pretrain_config.min_tokens <= len(token_ids) <= pretrain_config.max_tokens


def initialize_spans(n_tokens : int, pretrain_config : PretrainConfig):
    """Sample masked token positions and collapse adjacent positions into spans."""
    n_mask = max(1, int(n_tokens * pretrain_config.corruption_rate))
    mask_indices = sorted(random.sample(range(n_tokens), n_mask))

    spans = []
    current_span = [mask_indices[0]]
    for i in range(1, len(mask_indices)):
        if mask_indices[i] == mask_indices[i - 1] + 1:
            current_span.append(mask_indices[i])
        else:
            spans.append(current_span)
            current_span = [mask_indices[i]]
    spans.append(current_span)

    return spans


def build_corrupted_input_and_target(spans : list, token_ids : list, sentinel_token_ids : list):
    """Build one T5 span-corruption training pair from tokenized code."""
    # Flatten span coordinates so membership checks are cheap while rewriting.
    mask_set = {masked_pos for span in spans for masked_pos in span} 

    # Track which sentinel token should replace each masked token position.
    span_map = {}
    for span_idx in range(len(spans)):
        curr_span = spans[span_idx]
        for masked_pos in range(len(spans[span_idx])):
            span_map[curr_span[masked_pos]] = span_idx

    # Input: replace each span with its sentinel token
    input_ids = []
    prev_span_idx = -1
    for pos, tid in enumerate(token_ids):
        if pos in mask_set:
            s_idx = span_map[pos]
            if s_idx != prev_span_idx:
                input_ids.append(sentinel_token_ids[s_idx])
                prev_span_idx = s_idx
        else:
            input_ids.append(tid)
            prev_span_idx = -1

    # The decoder target is the concatenation of removed spans, each prefixed
    # by the same sentinel that marked its location in the encoder input.
    target_ids = []
    for span_idx, span in enumerate(spans):
        target_ids.append(sentinel_token_ids[span_idx])
        for pos in span:
            target_ids.append(token_ids[pos])

    return input_ids, target_ids

def apply_span_corruption(token_ids, sentinel_token_ids, pretrain_config : PretrainConfig):
    """Apply T5 span corruption and return encoder/decoder token sequences."""
    spans = initialize_spans(n_tokens=len(token_ids), pretrain_config=pretrain_config)
    input_ids, target_ids = build_corrupted_input_and_target(spans = spans, token_ids = token_ids, sentinel_token_ids=sentinel_token_ids)
    return input_ids, target_ids


def pretrain_collate_fn(batch, pad_token_id):
    """Pad one pretraining batch and mask padded labels for seq2seq loss."""
    max_input_len = max(len(example["input_ids"]) for example in batch)
    max_label_len = max(len(example["labels"]) for example in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for example in batch:
        input_ids = example["input_ids"]
        labels = example["labels"]

        input_pad_len = max_input_len - len(input_ids)
        label_pad_len = max_label_len - len(labels)

        padded_input_ids = input_ids + [pad_token_id] * input_pad_len
        attention_mask = [1] * len(input_ids) + [0] * input_pad_len
        padded_labels = labels + [pad_token_id] * label_pad_len

        # Hugging Face seq2seq losses ignore label positions set to -100.
        masked_labels = [
            token if token != pad_token_id else -100
            for token in padded_labels
        ]

        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(masked_labels)

    return (
        torch.tensor(batch_input_ids, dtype=torch.long),
        torch.tensor(batch_attention_mask, dtype=torch.long),
        torch.tensor(batch_labels, dtype=torch.long),
    )


def build_pretrain_dataset(raw_pretraining_methods : list[str], tokenizer, pretrain_config : PretrainConfig):
    """Tokenize raw methods and materialize one epoch of corrupted examples."""

    # Resolve all sentinel IDs once to avoid repeated string-to-id lookups.
    sentinel_token_ids = tokenizer.convert_tokens_to_ids(pretrain_config.sentinel_ids)
    
    epoch_examples = []
    for method in raw_pretraining_methods:
        # Corruption should operate on raw method content, not on EOS/pad tokens.
        token_ids = tokenizer.encode(method, add_special_tokens=False)

        if filter_by_length(token_ids=token_ids,pretrain_config=pretrain_config):
            corrupted_input_ids, corrupted_target_ids = apply_span_corruption(
                token_ids=token_ids,
                sentinel_token_ids=sentinel_token_ids,
                pretrain_config=pretrain_config,
            )
            
            input_ids = corrupted_input_ids + [tokenizer.eos_token_id]
            labels = corrupted_target_ids + [tokenizer.eos_token_id]

            epoch_examples.append(
                {
                    "input_ids" : input_ids,
                    "labels" : labels
                }
            )
    
    return epoch_examples


def run_pretraining(raw_pretraining_methods : list[str], model, optimizer, tokenizer, scheduler, pretrain_config : PretrainConfig):
    """Run multi-epoch unsupervised pretraining and save the final checkpoint."""
    if pretrain_config.max_samples is not None:
        raw_pretraining_methods = raw_pretraining_methods[:pretrain_config.max_samples]

    for epoch in tqdm(range(pretrain_config.num_epochs), desc="Pretraining epochs"):
        # Rebuild masking every epoch so the model sees different corruptions of
        # the same code instead of memorizing one fixed mask pattern.
        pretraining_data = build_pretrain_dataset(raw_pretraining_methods=raw_pretraining_methods, tokenizer=tokenizer, pretrain_config=pretrain_config)

        train_dataloader = DataLoader(
            pretraining_data,
            batch_size=pretrain_config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: pretrain_collate_fn(batch, tokenizer.pad_token_id),
        )

        model.train()
        tr_loss, tr_steps = 0.0, 0

        batch_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch in batch_bar:
            input_ids, attention_mask, labels = [x.to(pretrain_config.device) for x in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tr_loss += loss.item()
            tr_steps += 1
            batch_bar.set_postfix(loss=round(loss.item(), 4))

        train_loss = round(tr_loss / tr_steps, 5)
        print(f"Epoch {epoch + 1}: train_loss = {train_loss}")

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(pretrain_config.output_dir)
    # tokenizer.save_pretrained(pretrain_config.output_dir)
        
