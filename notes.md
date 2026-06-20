## Use `dotenv` for machine-specific paths

### Files you need

1. `.env` — machine-specific environment file
2. `.env.example` — template to commit
3. .gitignore — ensure `.env` is ignored
4. Optionally `config.py` — central loader for your paths

---

## Example `.env`

Add this to `.env` at repo root:

```env
DATA_ROOT=/media/maindisk/data/mel_harm_CA_all

GJT_TRAIN=gjt_CA_train
GJT_TEST=gjt_CA_test
HOOK_TRAIN=CA_train
HOOK_TEST=CA_test
WIKI_TRAIN=wikifonia_train
WIKI_TEST=wikifonia_test
NOTT_TRAIN=nottingham_train
NOTT_TEST=nottingham_test
```

---

## Example `.env.example`

Commit this template with placeholder values:

```env
DATA_ROOT=/path/to/data/mel_harm_CA_all

GJT_TRAIN=gjt_CA_train
GJT_TEST=gjt_CA_test
HOOK_TRAIN=CA_train
HOOK_TEST=CA_test
WIKI_TRAIN=wikifonia_train
WIKI_TEST=wikifonia_test
NOTT_TRAIN=nottingham_train
NOTT_TEST=nottingham_test
```

---

## Add to .gitignore

Add this line:

```gitignore
.env
```

---

## Install `python-dotenv`

```bash
pip install python-dotenv
```

---

## Load paths in Python

Option A: direct in make_datasets_graph.py

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env from repo root

DATA_ROOT = Path(os.getenv("DATA_ROOT"))
GJT_TRAIN = DATA_ROOT / os.getenv("GJT_TRAIN")
GJT_TEST = DATA_ROOT / os.getenv("GJT_TEST")
```

Option B: use a small `config.py`

`config.py`:
```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = Path(os.getenv("DATA_ROOT", ""))

def data_path(name: str) -> Path:
    return DATA_ROOT / os.getenv(name, "")
```

Then in your script:
```python
from config import data_path

train_path = data_path("GJT_TRAIN")
test_path = data_path("GJT_TEST")
```

---

## Why this is good

- `.env` stays machine-specific and out of git
- `.env.example` documents the required path names
- code stays portable across machines
- path values are easy to update without editing Python files

If you want, I can also give you a concrete patch for make_datasets_graph.py using this pattern.