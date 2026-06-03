from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "Experiments"

patterns = {
    'out_dir = Path("Experiments")': 'out_dir = OUT_DIR',
    'REPORT_DIR = Path("Experiments")': 'REPORT_DIR = OUT_DIR',
    'Path("Experiments")': 'OUT_DIR',
    'out_dir = Path(\\"Experiments\\")': 'out_dir = OUT_DIR',
    'REPORT_DIR = Path(\\"Experiments\\")': 'REPORT_DIR = OUT_DIR',
    'Path(\\"Experiments\\")': 'OUT_DIR',
}

n_changed = 0
for nb in NB_DIR.glob('*.ipynb'):
    text = nb.read_text(encoding='utf-8')
    new_text = text
    for old, new in patterns.items():
        if old in new_text:
            new_text = new_text.replace(old, new)
    if new_text != text:
        bak = nb.with_suffix(nb.suffix + '.bak')
        shutil.copy2(nb, bak)
        nb.write_text(new_text, encoding='utf-8')
        print(f'Patched {nb} (backup: {bak.name})')
        n_changed += 1

print(f'Done. Notebooks changed: {n_changed}')
