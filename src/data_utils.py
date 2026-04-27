"""Utilities for data loading and preprocessing."""
import json
from pathlib import Path
from datasets import load_dataset

def fetch_pretraining_data_hugging_face(n_samples = 50000, random_seed = 42, audit_sample_count = 3):
    train_total = load_dataset("code_search_net", "java", split = "train")
    train_sample = train_total.shuffle(seed = random_seed).select(range(n_samples))
    method_bodies = [method["whole_func_string"] for method in train_sample]

    print(f"Loaded {len(method_bodies)} Java methods.")
    print("\nAudit Samples:\n")
    for i, method in enumerate(method_bodies[:audit_sample_count], start=1):
        print(f"--- Sample {i} ---")
        print(method[:800])  # preview only
        print()

    return method_bodies

def save_pretraining_data(pretraining_save_path, pretraining_methods):
    save_path = Path(pretraining_save_path)
    flattened_methods = []

    for method in pretraining_methods:
        flattened_method = " ".join(str(method).split())
        flattened_methods.append(flattened_method)

    save_path.write_text("\n".join(flattened_methods), encoding = "utf-8")
    print(f"Pretraining methods saved to: {save_path}")

def load_pretraining_data(pretraining_save_path):
    save_path = Path(pretraining_save_path)
    pretraining_text = save_path.read_text(encoding = "utf-8")

    if not pretraining_text:
        print("failed to load exisitng pretraining data")
        return []

    return pretraining_text.splitlines()

def fetch_finetuning_data_hugging_face(audit_sample_count = 3):
    train_split = load_dataset("google/code_x_glue_cc_code_refinement", name="medium", split="train")
    validation_split = load_dataset("google/code_x_glue_cc_code_refinement", name="medium", split="validation")

    train_pairs = [[example["buggy"], example["fixed"]] for example in train_split]
    validation_pairs = [[example["buggy"], example["fixed"]] for example in validation_split]

    print(f"Loaded {len(train_pairs)} training buggy/fixed pairs.")
    print(f"Loaded {len(validation_pairs)} validation buggy/fixed pairs.")

    print("\nTraining Audit Samples:\n")
    for i, (buggy_method, fixed_method) in enumerate(train_pairs[:audit_sample_count], start=1):
        print(f"--- Train Sample {i} ---")
        print("Buggy:")
        print(buggy_method[:800])
        print("Fixed:")
        print(fixed_method[:800])
        print()

    print("\nValidation Audit Samples:\n")
    for i, (buggy_method, fixed_method) in enumerate(validation_pairs[:audit_sample_count], start=1):
        print(f"--- Validation Sample {i} ---")
        print("Buggy:")
        print(buggy_method[:800])
        print("Fixed:")
        print(fixed_method[:800])
        print()

    return train_pairs, validation_pairs

def save_finetuning_data(finetuning_save_path, finetuning_method_pairs):
    save_path = Path(finetuning_save_path)
    save_path.write_text(json.dumps(finetuning_method_pairs, ensure_ascii=False), encoding="utf-8")
    print(f"Finetuning method pairs saved to: {save_path}")

def load_finetuning_data(finetuning_save_path):
    save_path = Path(finetuning_save_path)
    finetuning_text = save_path.read_text(encoding="utf-8")

    if not finetuning_text:
        print("failed to load exisiting finetuning data")
        return []

    return json.loads(finetuning_text)
