"""Utilities for data loading and preprocessing."""
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
        return []

    return pretraining_text.splitlines()
