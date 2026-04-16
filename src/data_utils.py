"""Utilities for data loading and preprocessing."""
from datasets import load_dataset

def load_pretraining_data(n_samples = 50000, random_seed = 42, audit_sample_count = 3):
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
