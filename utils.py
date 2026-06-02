import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path


def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def backup_code(work_dir, verbose=False):
    repo_root = Path(__file__).resolve().parent
    work_dir = Path(work_dir)
    backup_root = work_dir / 'backup'

    include_files = [
        'train.py',
        'val.py',
        'dist_train.sh',
        'utils.py',
        'AGENTS.md',
    ]
    include_dirs = [
        'configs',
        'models',
        'loaders',
    ]

    sources = []
    for rel_path in include_files:
        src = repo_root / rel_path
        if src.is_file():
            sources.append(src)

    for dir_rel in include_dirs:
        root = repo_root / dir_rel
        if not root.exists():
            continue
        for pattern in ('*.py', '*.sh'):
            sources.extend(path for path in root.rglob(pattern) if path.is_file())

    seen = set()
    for src in sorted(sources):
        rel_path = src.relative_to(repo_root)
        if rel_path in seen:
            continue
        seen.add(rel_path)
        dst = backup_root / rel_path

        if verbose:
            logging.info('Copying %s -> %s' % (rel_path, dst.relative_to(work_dir)))

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    snapshot_root = backup_root / '_snapshot'
    snapshot_root.mkdir(parents=True, exist_ok=True)

    def _run_git_command(args):
        try:
            result = subprocess.run(
                ['git', *args],
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            return result.stdout
        except Exception as exc:
            return f'ERROR: {exc}\n'

    (snapshot_root / 'git_head.txt').write_text(
        _run_git_command(['rev-parse', 'HEAD']),
        encoding='utf-8')
    (snapshot_root / 'git_status.txt').write_text(
        _run_git_command(['status', '--short']),
        encoding='utf-8')
    (snapshot_root / 'git_diff.patch').write_text(
        _run_git_command(['diff', '--binary']),
        encoding='utf-8')
