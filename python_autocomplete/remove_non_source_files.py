from pathlib import Path
from typing import Set

from labml import lab


def remove_files(path: Path, keep: Set[str]):
    """
    Remove files
    """

    for p in path.iterdir():
        if p.is_symlink():
            p.unlink()
            continue
        if p.is_dir():
            remove_files(p, keep)
        else:
            if p.suffix not in keep:
                p.unlink()


def main():
    remove_files(lab.get_data_path() / 'source', {'.py'})


if __name__ == '__main__':
    main()
