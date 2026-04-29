"""Utilities for supervised T5 bug-fixing fine-tuning."""
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

@dataclass
class FinetuneConfig:
    """Configuration for seq2seq fine-tuning on buggy/fixed code pairs."""
    output_dir: str

    # Mirror the common T5 512-token training regime for both source and target.
    max_input_len: int = 512 
    max_target_len: int = 512 

    max_train_samples: int | None = None
    max_val_samples: int | None = None
    num_epochs: int = 3
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def finetune_collate_fn(batch, pad_token_id):
    """Pad one supervised batch and mask label padding for loss computation."""
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
    """Convert buggy/fixed code pairs into T5-style source/target examples."""
    dataset = []

    for buggy_code, fixed_code in example_pairs:
        # The task prefix keeps the training format aligned with T5's
        # text-to-text formulation.
        source_text = f"fix bug: {buggy_code}"

        # Reserve one token slot for EOS after truncation.
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
    """Compute average validation loss for one full validation pass."""
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
    """Run supervised fine-tuning and save the best validation checkpoint."""
    if finetune_config.max_train_samples is not None:
        train_pairs = train_pairs[:finetune_config.max_train_samples]
    if finetune_config.max_val_samples is not None:
        val_pairs = val_pairs[:finetune_config.max_val_samples]

    train_data = build_finetune_dataset(train_pairs, tokenizer, finetune_config)
    val_data = build_finetune_dataset(val_pairs, tokenizer, finetune_config)

    #build training data loader
    train_dataloader = DataLoader(
        train_data,
        batch_size=finetune_config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: finetune_collate_fn(batch, tokenizer.pad_token_id),
    )

    if not val_pairs:
        raise ValueError("Missing val_pairs")

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
            
            loss = outputs.loss
            loss.backward()

            # Keep training stable when gradients spike on long code sequences.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
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
            # tokenizer.save_pretrained(finetune_config.output_dir)
