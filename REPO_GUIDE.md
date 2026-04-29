# Repository Guide

## Purpose
This project compares multiple bug-fixing pipelines for Java methods:

- custom T5 with unsupervised pretraining, then supervised fine-tuning
- custom T5 trained only with supervised fine-tuning
- Qwen in zero-shot mode
- Qwen with retrieval-augmented generation (RAG)

The main executable workflow lives in [notebooks/multi_paradigm_bug_fixing.ipynb](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/notebooks/multi_paradigm_bug_fixing.ipynb:1).

## File Responsibilities

- [src/data_utils.py](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/src/data_utils.py:1)
  Loads CodeSearchNet and CodeXGLUE splits from Hugging Face and caches them locally for repeatable notebook runs.

- [src/tokenizer_utils.py](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/src/tokenizer_utils.py:1)
  Trains the custom SentencePiece tokenizer and reloads it as a Hugging Face fast tokenizer with T5 sentinel tokens.

- [src/model_utils.py](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/src/model_utils.py:1)
  Builds the custom T5 architecture used for both the pretrained and fresh baselines.

- [src/pretrain_utils.py](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/src/pretrain_utils.py:1)
  Implements T5-style span corruption, dynamic batch padding, and the unsupervised pretraining loop.

- [src/finetune_utils.py](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/src/finetune_utils.py:1)
  Converts buggy/fixed pairs into T5 training examples, runs supervised fine-tuning, and tracks validation loss.

- [src/llm_utils.py](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/src/llm_utils.py:1)
  Implements the structure-aware retrieval pipeline, FAISS knowledge base persistence, Qwen loading, and zero-shot/RAG generation.

- [notebooks/multi_paradigm_bug_fixing.ipynb](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/notebooks/multi_paradigm_bug_fixing.ipynb:1)
  End-to-end experiment driver. It orchestrates setup, tokenizer training, T5 construction, pretraining, fine-tuning, RAG setup, and evaluation.

- [requirements.txt](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/requirements.txt:1)
  Python dependency list for notebooks and source modules.

- [assignment3_report.tex](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/assignment3_report.tex:1)
  Report on implementation and results.

## Practical Notes

- The notebook intentionally caches datasets and outputs so reruns can skip completed phases.
- Qwen evaluation is the slowest part of the pipeline, especially full test-set evaluation with both zero-shot and RAG generation.
- If you are debugging, keep the sample caps in place. If you are running final experiments, remove or raise them.
- The retrieval pipeline assumes Java code snippets and uses Tree-sitter Java parsing.
