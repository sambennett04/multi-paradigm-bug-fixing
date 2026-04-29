# multi_paradigm_bug_fixing

Repository scaffold for the multi-paradigm bug fixing project.

## Dependencies

Install the Python requirements before running the notebooks or `src` utilities:

```bash
pip install -r requirements.txt
```

The retrieval utilities in [src/llm_utils.py](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/src/llm_utils.py:1) now use `tree-sitter-language-pack` for Java parsing instead of the direct `tree-sitter-java` binding. This avoids the `PyCapsule` / `tree_sitter.Language` compatibility issue that can happen when `tree-sitter` and language wheels are out of sync.

The module initializes a shared Java parser at import time:

```python
from tree_sitter_language_pack import get_parser

JAVA_PARSER = get_parser("java")
```

If you are working in a notebook and want the same parser behavior locally, use the same pattern in a cell:

```python
from tree_sitter_language_pack import get_parser

java_parser = get_parser("java")
```
