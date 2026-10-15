"""Keep the public examples and result table directly executable."""
from pathlib import Path
import numpy as np
from sae_localize import exact_scope_curve

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_same_codes_different_input_scopes():
    x = np.arange(8)
    z = np.column_stack([x & 1, (x >> 1) & 1]).astype(float)
    mixed_ids = ((x & 1) ^ ((x >> 1) & 1)) | (x & ~1)
    np.testing.assert_array_equal(np.sort(mixed_ids), np.arange(8))
    same = exact_scope_curve(z, n_factors=3)
    mixed = exact_scope_curve(z[mixed_ids], n_factors=3)
    assert (same.budget(0), mixed.budget(0)) == (2, 3)
    np.testing.assert_array_equal(np.sort(z, axis=0), np.sort(z[mixed_ids], axis=0))


def test_public_document_links():
    import re
    for page in [ROOT/'README.md', ROOT/'README_cn.md', ROOT/'docs/index.html']:
        html = page.read_text(encoding='utf-8')
        for ref in re.findall(r'(?:href|src)="([^"]+)"', html):
            if ref.startswith(('https:', 'http:', '#', 'mailto:')):
                continue
            path = (page.parent / ref.split('#', 1)[0]).resolve()
            assert path.is_file(), (page.name, ref)


def test_source_release_excludes_manuscript():
    assert not (ROOT / 'paper').exists()
    for path in ROOT.rglob('*'):
        if any(part in {'.git', '.venv', '__pycache__'} for part in path.parts):
            continue
        if path.is_file():
            assert path.suffix.lower() not in {'.tex', '.bib', '.bst', '.sty', '.pdf'}
