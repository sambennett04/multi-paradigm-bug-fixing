# multi_paradigm_bug_fixing

Repository scaffold for the multi-paradigm bug fixing project.

## Dependencies

Install the Python requirements before running the notebooks or `src` utilities:

```bash
pip install -r requirements.txt
```

The retrieval utilities in [src/llm_utils.py](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/src/llm_utils.py:1) now use `tree-sitter-languages` for Java parsing instead of the direct `tree-sitter-java` binding. `tree-sitter` is pinned to `0.21.3` in [requirements.txt](/Users/sambennett/Desktop/CSCI555/multi_paradigm_bug_fixing/requirements.txt:1) because newer Python bindings are incompatible with the current `tree_sitter_languages` parser helper. `codebleu` is also pinned to `0.3.0` to avoid source-build failures on Colab's Python 3.12 runtime.

The module initializes a shared Java parser at import time:

```python
from tree_sitter_languages import get_parser

JAVA_PARSER = get_parser("java")
```

If you are working in a notebook and want the same parser behavior locally, use the same pattern in a cell:

```python
from tree_sitter_languages import get_parser

java_parser = get_parser("java")
```
