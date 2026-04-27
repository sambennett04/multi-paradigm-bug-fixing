"""Utilities for finetuning workflows."""
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

@dataclass
class FinetuneConfig:
    output_dir: str

    #T5 was mostly trained with sequences of 512 input tokens so we replicate the same here soruces (https://discuss.huggingface.co/t/does-t5-truncate-input-longer-than-512-internally/3602, https://discuss.huggingface.co/t/does-t5-truncate-input-longer-than-512-internally/3602)
    max_input_len: int = 512 
    max_target_len: int = 512 

    num_epochs: int = 3
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def finetune_collate_fn(batch, pad_token_id):
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
        masked_labels = [tok if tok != pad_token_id else -100 for tok in padded_labels]

        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(masked_labels)

    return (
        torch.tensor(batch_input_ids, dtype=torch.long),
        torch.tensor(batch_attention_mask, dtype=torch.long),
        torch.tensor(batch_labels, dtype=torch.long),
    )

def build_finetune_dataset(example_pairs, tokenizer, finetune_config):
    dataset = []

    for buggy_code, fixed_code in example_pairs:
        source_text = f"fix bug: {buggy_code}"

        #encode buggy methods --> input_ids and fixed methods --> labels
        #Here we truncate tokenized sequences to match t5s 512 training sequence length
        input_ids = tokenizer.encode(
            source_text,
            add_special_tokens=False,
            truncation=True,
            max_length=finetune_config.max_input_len - 1,
        ) + [tokenizer.eos_token_id]

        labels = tokenizer.encode(
            fixed_code,
            add_special_tokens=False,
            truncation=True,
            max_length=finetune_config.max_target_len - 1,
        ) + [tokenizer.eos_token_id]

        dataset.append({
            "input_ids": input_ids,
            "labels": labels,
        })

    return dataset

def evaluate_model(model, val_dataloader, device):
    model.eval()
    eval_loss, num_batches = 0.0, 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs.loss.item()
            num_batches += 1
    return round(eval_loss / num_batches, 5)

def run_finetuning(train_pairs, val_pairs, model, optimizer, tokenizer, scheduler, finetune_config):
    train_data = build_finetune_dataset(train_pairs, tokenizer, finetune_config)
    val_data = build_finetune_dataset(val_pairs, tokenizer, finetune_config)

    #build training data loader
    train_dataloader = DataLoader(
        train_data,
        batch_size=finetune_config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: finetune_collate_fn(batch, tokenizer.pad_token_id),
    )

    #Error if val_pairs are empty, if empty val_pairs included could cause division by zero in validation
    if not val_pairs:
        raise ValueError("Missing val_pairs")

    #build validation data loader
    val_dataloader = DataLoader(
        val_data,
        batch_size=finetune_config.batch_size,
        collate_fn=lambda batch: finetune_collate_fn(batch, tokenizer.pad_token_id),
    )

    model.to(finetune_config.device)
    best_loss = float("inf")

    for epoch in tqdm(range(finetune_config.num_epochs), desc="Finetuning epochs"):
        model.train()
        tr_loss, tr_steps = 0.0, 0

        batch_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch in batch_bar:
            input_ids, attention_mask, labels = [x.to(finetune_config.device) for x in batch]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            #calculate batch loss
            loss = outputs.loss
            
            #run back propogation
            loss.backward()

            #prevent exploding gradients (normalized betwen 0 - 1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            #update model weights using current gradients
            optimizer.step()

            #update learning rate according to schedule
            scheduler.step()

            #clears accumulated gradients from this step
            optimizer.zero_grad()

            tr_loss += loss.item()
            tr_steps += 1
            batch_bar.set_postfix(loss=round(loss.item(), 4))

        train_loss = round(tr_loss / tr_steps, 5)
        print(f"Epoch {epoch + 1}: train_loss = {train_loss}")
        eval_loss = evaluate_model(model, val_dataloader, finetune_config.device)
        print(f"Epoch {epoch + 1}: validation_loss = {eval_loss}")

        if eval_loss < best_loss:
            best_loss = eval_loss
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(finetune_config.output_dir)
            #tokenizer.save_pretrained(finetune_config.output_dir)
