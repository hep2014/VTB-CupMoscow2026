"""Verify local source dump and extract selected COPY data without executing SQL."""
import argparse
import hashlib
import pathlib
import shutil
import subprocess
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED = '9ab243e771159d8b5b8df18f12b671afdbc78f7c47849b1f6ea4aaa68b144a3e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['fdh', 'paysim', 'all'], default='all')
    parser.add_argument('--pg-restore', default='C:/Program Files/PostgreSQL/17/bin/pg_restore.exe')
    args = parser.parse_args()
    dump = ROOT / '.local/data/handoff/database.dump'
    if not dump.exists():
        archive = ROOT / '.local/vtb_antifraud_complete.zip'
        if not archive.is_file():
            raise FileNotFoundError('Place the original vtb_antifraud_complete.zip in .local/ first.')
        dump.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            # Fixed member name; no arbitrary archive paths are extracted.
            with source.open('database.dump') as src, dump.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    with dump.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != EXPECTED:
            raise ValueError('Source dump SHA-256 mismatch')
    sources = {
        'fdh': ('fdh_transactions', 'tmp/fdh_context/fdh_data.sql'),
        'paysim': ('paysim_transactions', 'tmp/paysim_audit/paysim_data.sql'),
    }
    for dataset in sources if args.dataset == 'all' else [args.dataset]:
        table, relative = sources[dataset]
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([args.pg_restore, '--data-only', '--no-owner', '--no-privileges',
                        '-t', table, '-f', str(target), str(dump)], check=True)
        print(f'Extracted {table}: {relative}')


if __name__ == '__main__':
    main()
