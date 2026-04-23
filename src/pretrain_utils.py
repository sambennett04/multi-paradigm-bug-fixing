"""Utilities for model pretraining workflows."""
from dataclasses import dataclass, field
import random
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

@dataclass
class PretrainConfig:
    output_dir : str
    min_tokens: int = 10
    max_tokens: int = 512
    corruption_rate: float = 0.15
    sentinel_ids: list[str] = field(default_factory=lambda: [f"<extra_id_{i}>" for i in range(100)]) #tells data class to call this function each time a new object is constructed, guaranteeing individual lists per PretrainConfig objects
    num_epochs: int = 3
    batch_size: int = 8
    device : str = "cuda" if torch.cuda.is_available() else "cpu"

def filter_by_length(token_ids : list, pretrain_config : PretrainConfig):
    """filters out toknized methods (a set of token ids) with less than pretrain_config.min_tokens tokens and more then pretrain_config.max_tokens tokens"""
    return pretrain_config.min_tokens <= len(token_ids) <= pretrain_config.max_tokens

def initialize_spans(n_tokens : int, pretrain_config : PretrainConfig):
    """Select random positions from 0-n_tokens and grpup consecutreive ones into spans"""
    n_mask = max(1, int(n_tokens * pretrain_config.corruption_rate))  # 15%
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
    """Use randomly generated spans to create pretraing input and output for one tokenized method """
    #fetch all masked positions
    mask_set = {masked_pos for span in spans for masked_pos in span} 

    #map each masked position to is span index
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

    # Target: sentinel followed by the span's original tokens
    target_ids = []
    for span_idx, span in enumerate(spans):
        target_ids.append(sentinel_token_ids[span_idx])
        for pos in span:
            target_ids.append(token_ids[pos])

    return input_ids, target_ids

def apply_span_corruption(token_ids, sentinel_token_ids, pretrain_config : PretrainConfig):
    """Apply span corruption to one tokenized method from raw pretraining data and return pretraining input and target"""
    spans = initialize_spans(n_tokens=len(token_ids), pretrain_config=pretrain_config)
    input_ids, target_ids = build_corrupted_input_and_target(spans = spans, token_ids = token_ids, sentinel_token_ids=sentinel_token_ids)
    return input_ids, target_ids


def pretrain_collate_fn(batch, pad_token_id):
    """
        Apply padding to input_ids and labels and calculate attention mask for one batch of unpadded token_ids (tokenized methods). 
        Facilitates dynamic example padding at batch time.
    """
    #calculate max input and label examples so we can pad all other examples up to their lengths
    max_input_len = max(len(example["input_ids"]) for example in batch)
    max_label_len = max(len(example["labels"]) for example in batch)

    #for one batch: 
    batch_input_ids = [] #padded input_ids
    batch_attention_mask = [] #attention mask 
    batch_labels = [] #padded labels with with -100 for pad tokens so hugging face loss func ignores

    for example in batch:
        #extract input_ids and labels from example object
        input_ids = example["input_ids"]
        labels = example["labels"]

        #decide how much padding each needs
        input_pad_len = max_input_len - len(input_ids)
        label_pad_len = max_label_len - len(labels)

        #apply padding to input_ids
        padded_input_ids = input_ids + [pad_token_id] * input_pad_len

        #compute attention_mask on padded input_ids
        attention_mask = [1] * len(input_ids) + [0] * input_pad_len

        #apply padding to labels
        padded_labels = labels + [pad_token_id] * label_pad_len

        #replace all padded tokens in target with -100, so that loss function ignores missed prediction on pad tokens, ensuring model is not trained to predict padding
        masked_labels = [
            token if token != pad_token_id else -100
            for token in padded_labels
        ]

        #populate batch components
        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(masked_labels)

    #return tensor (of rank 2 -> matrix) of each batch component
    #note each tensor is stacks of individual example components (input_ids, attention_mask, labels)
    return (
        torch.tensor(batch_input_ids, dtype=torch.long),
        torch.tensor(batch_attention_mask, dtype=torch.long),
        torch.tensor(batch_labels, dtype=torch.long),
    )


def build_pretrain_dataset(raw_pretraining_methods : list[str], tokenizer, pretrain_config : PretrainConfig):
    """Build single epoch tokenized, span corrupted dataset"""

    #convert string sentinel tokens to their corresponding tokenizer token ids
    sentinel_token_ids = tokenizer.convert_tokens_to_ids(pretrain_config.sentinel_ids)
    
    epoch_examples = []
    for method in raw_pretraining_methods:
        #tokenize method
        #add special tokens = False ensures no special tokens are added during this tokenization, guaranteeing span corruptions is done on just raw method tokens
        token_ids = tokenizer.encode(method, add_special_tokens=False)

        #include in dataset if num tokens in between 10 and 512
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

    for epoch in tqdm(range(pretrain_config.num_epochs), desc="Pretraining epochs"):
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
    #tokenizer.save_pretrained(pretrain_config.output_dir) [optional save tokenizer again]
        
