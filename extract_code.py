#!/usr/bin/env python

"""
Parse all files and write to a single file
"""
import os
import string
from pathlib import Path, PurePath
from typing import List, NamedTuple

from labml import logger, monit, lab

PRINTABLE = set(string.printable)


class _PythonFile(NamedTuple):
    relative_path: str
    project: str
    path: Path


class _GetPythonFiles:
    """
    Get list of python files and their paths inside `data/source` folder
    """

    def __init__(self):
        self.source_path = Path(lab.get_data_path() / 'source')
        self.files: List[_PythonFile] = []
        self.get_python_files(self.source_path)

        logger.inspect([f.path for f in self.files])

    def add_file(self, path: Path):
        """
        Add a file to the list of tiles
        """
        project = path.relative_to(self.source_path).parents
        project = project[len(project) - 2]
        relative_path = path.relative_to(self.source_path / project)

        self.files.append(_PythonFile(relative_path=str(relative_path),
                                      project=str(project),
                                      path=path))

    def get_python_files(self, path: Path):
        """
        Recursively collect files
        """
        for p in path.iterdir():
            if p.is_dir():
                self.get_python_files(p)
            else:
                if p.suffix == '.py':
                    self.add_file(p)


def _read_file(path: Path) -> str:
    """
    Read a file
    """
    with open(str(path)) as f:
        content = f.read()

    content = ''.join(filter(lambda x: x in PRINTABLE, content))

    return content


def _load_code(path: PurePath, source_files: List[_PythonFile]):
    with open(str(path), 'w') as f:
        for i, source in monit.enum(f"Write {path.name}", source_files):
            f.write(f"# PROJECT: {source.project} FILE: {str(source.relative_path)}\n")
            f.write(_read_file(source.path) + "\n")


def main():
    source_files = _GetPythonFiles().files

    logger.inspect(source_files)

    train_valid_split = int(len(source_files) * 0.9)
    _load_code(lab.get_data_path() / 'train.py', source_files[:train_valid_split])
    _load_code(lab.get_data_path() / 'valid.py', source_files[train_valid_split:])


if __name__ == '__main__':
    main()
