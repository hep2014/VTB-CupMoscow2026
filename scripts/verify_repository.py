"""Check portable report links, frozen protocols and relocated model inference.

Uses only repository artifacts; original reviews and dump are not required.
"""
import hashlib
import json
import pathlib
import re
from urllib.parse import unquote

import numpy as np

from context_payment_layer import ContextPaymentLayer

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXCLUDE = {'.local', '.venv', '.git', 'tmp', '__pycache__', '.idea'}


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def public_files():
    # Prune private/large directories before descent.
    import os
    for directory, children, names in os.walk(ROOT):
        children[:] = [n for n in children if n not in EXCLUDE]
        for name in names:
            yield pathlib.Path(directory) / name


def main():
    files = list(public_files())
    checked_links = 0
    for file in files:
        assert file.suffix.lower() not in {'.npz', '.npy', '.dump', '.zip', '.sql', '.pdf', '.pyc', '.log'}, file
        if file.suffix == '.json':
            json.loads(file.read_text(encoding='utf-8'))
        if file.suffix != '.md':
            continue
        for target in re.findall(r'\]\(([^)]+)\)', file.read_text(encoding='utf-8')):
            if re.match(r'(https?://|mailto:|#)', target):
                continue
            assert not re.match(r'([A-Za-z]:|/)', target), f'Nonportable link: {file}: {target}'
            path = (file.parent / unquote(target.split('#')[0])).resolve()
            assert path.is_relative_to(ROOT) and path.exists(), f'Broken link: {file}: {target}'
            checked_links += 1

    folder = ROOT / 'models/context_layer'
    event = read('models/context_layer/example_input.json')
    expected = read('models/context_layer/example_decision.json')
    cases = 0
    for mode, cap in [('low_friction', .001), ('low_friction', .005), ('low_friction', .01), ('high_recall', .005)]:
        layer = ContextPaymentLayer(folder, mode, cap)
        decision = layer.decide(**event)
        held = layer.decide(**{**event, 'mandatory_hold': True})
        assert held['intervene'] and held['reason'] == 'MANDATORY_HOLD'
        if mode == 'low_friction' and cap == .005:
            for key, value in expected.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    assert np.isclose(decision[key], value), (key, decision[key], value)
                else:
                    assert decision[key] == value, (key, decision[key], value)
        try:
            layer.decide(event['features'], event['amount'] + 100)
        except ValueError:
            pass
        else:
            raise AssertionError('Inconsistent amount/features accepted')
        cases += 1

    protocol_hash = hashlib.sha256((folder / 'protocol.json').read_bytes()).hexdigest()
    assert protocol_hash == read('models/context_layer/selection_lock.json')['protocol_sha256']
    assert protocol_hash == read('models/context_layer/run_manifest.json')['protocol_sha256']
    low = read('models/context_layer/low_friction_results.json')['results']['0.005']['policies']
    assert (low['baseline']['tp'], low['baseline']['fp']) == (1423, 1399)
    assert (low['adaptive']['tp'], low['adaptive']['fp']) == (1641, 1354)
    finance = read('research/finance/scenarios.json')
    assert finance['evidence']['policies'] == low
    assert finance['budget']['totalRub'] == 42931250
    assert finance['bankQuarter']['calculated'] is False
    assert finance['bankReturn']['calculated'] is False
    assert 'models/' not in (ROOT / '.gitignore').read_text(encoding='utf-8').splitlines()
    print(json.dumps(dict(status='passed', repository_files=len(files),
                          size_mb=round(sum(p.stat().st_size for p in files) / 1e6, 2),
                          checked_relative_links=checked_links, inference_modes_checked=cases,
                          frozen_protocol_hash_preserved=True,
                          finance_matches_experiment=True), indent=2))


if __name__ == '__main__':
    main()
