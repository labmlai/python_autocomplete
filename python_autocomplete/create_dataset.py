#!/usr/bin/env python

"""
Parse all files and write to a single file
"""
import re
import string
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from pathlib import PurePath
from typing import List, NamedTuple, Set
from typing import Optional

import numpy as np

from labml import lab, monit
from labml import logger
from labml.internal.util import rm_tree

PRINTABLE = set(string.printable)


class PythonFile(NamedTuple):
    relative_path: str
    project: str
    path: Path


def get_python_files():
    """
    Get list of python files and their paths inside `data/source` folder
    """

    source_path = Path(lab.get_data_path() / 'source')
    files: List[PythonFile] = []

    def _add_file(path: Path):
        """
        Add a file to the list of tiles
        """
        project = path.relative_to(source_path).parents
        relative_path = path.relative_to(source_path / project[len(project) - 3])

        files.append(PythonFile(relative_path=str(relative_path),
                                project=str(project[len(project) - 2]),
                                path=path))

    def _collect_python_files(path: Path):
        """
        Recursively collect files
        """
        for p in path.iterdir():
            if p.is_dir():
                _collect_python_files(p)
            else:
                _add_file(p)

    _collect_python_files(source_path)

    logger.inspect([f.path for f in files])

    return files


def _read_file(path: Path) -> str:
    """
    Read a file
    """
    with open(str(path)) as f:
        content = f.read()

    content = ''.join(filter(lambda x: x in PRINTABLE, content))

    return content


def _load_code(path: PurePath, source_files: List[PythonFile]):
    with open(str(path), 'w') as f:
        for i, source in monit.enum(f"Write {path.name}", source_files):
            f.write(f"# PROJECT: {source.project} FILE: {str(source.relative_path)}\n")
            f.write(_read_file(source.path) + "\n")


def get_repos_from_readme(filename: str):
    with open(str(lab.get_data_path() / filename), 'r') as f:
        content = f.read()

    link_pattern = re.compile(r"""
        \[(?P<title>[^\]]*)\] # title
        \((?P<utl>[^\)]*)\) # url
    """, re.VERBOSE)

    res = link_pattern.findall(content)

    github_repos = []
    repo_pattern = re.compile(r'https://github.com/(?P<user>[^/]*)/(?P<repo>[^/#]*)$')
    for title, url in res:
        repos = repo_pattern.findall(url)
        for r in repos:
            github_repos.append((r[0], r[1]))

    return github_repos


def get_awesome_pytorch_readme():
    md = urllib.request.urlopen('https://raw.githubusercontent.com/bharathgs/Awesome-pytorch-list/master/README.md')
    content = md.read()

    with open(str(lab.get_data_path() / 'pytorch_awesome.md'), 'w') as f:
        f.write(str(content))


def download_repo(org: str, repo: str, idx: Optional[int]):
    zip_file = Path(lab.get_data_path() / 'download' / f'{org}_{repo}.zip')

    if zip_file.exists():
        return zip_file

    if idx is not None:
        idx_str = f"{idx:03}: "
    else:
        idx_str = ""

    with monit.section(f"{idx_str} {org}/{repo}") as s:
        try:
            zip = urllib.request.urlopen(f'https://github.com/{org}/{repo}/archive/master.zip')
        except urllib.error.HTTPError as e:
            print(e)
            return
        content = zip.read()

        size = len(content) // 1024
        s.message = f"{size :,}KB"

        with open(str(zip_file), 'wb') as f:
            f.write(content)

    return zip_file


def create_folders():
    path = Path(lab.get_data_path() / 'download')
    if not path.exists():
        path.mkdir(parents=True)
    source = Path(lab.get_data_path() / 'source')

    if not source.exists():
        source.mkdir(parents=True)


def extract_zip(file_path: Path, overwrite: bool = False):
    source = Path(lab.get_data_path() / 'source')

    with monit.section(f"Extract {file_path.stem}"):
        repo_source = source / file_path.stem
        if repo_source.exists():
            if overwrite:
                rm_tree(repo_source)
            else:
                return repo_source
        with zipfile.ZipFile(file_path, 'r') as repo_zip:
            repo_zip.extractall(repo_source)

        return repo_source


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


def batch(overwrite: bool = False):
    with monit.section('Get pytorch_awesome'):
        get_awesome_pytorch_readme()
        repos = get_repos_from_readme('pytorch_awesome.md')

    # Download zips
    for i, r in monit.enum(f"Download {len(repos)} repos", repos):
        download_repo(r[0], r[1], i)

    # Extract downloads
    with monit.section('Extract zips'):
        download = Path(lab.get_data_path() / 'download')

        for repo in download.iterdir():
            extract_zip(repo, overwrite)

    with monit.section('Remove non python files'):
        remove_files(lab.get_data_path() / 'source', {'.py'})


def progressive(overwrite: bool = False):
    # Get repos
    get_awesome_pytorch_readme()
    repos = get_repos_from_readme('pytorch_awesome.md')

    # Download zips
    for i, r in monit.enum(f"Download {len(repos)} repos", repos):
        zip_file = download_repo(r[0], r[1], i)
        extracted = extract_zip(zip_file, overwrite)
        remove_files(extracted, {'.py'})


def main():
    try:
        progressive()
    except KeyboardInterrupt:
        pass

    source_files = get_python_files()

    np.random.shuffle(source_files)

    logger.inspect(source_files)

    train_valid_split = int(len(source_files) * 0.9)
    _load_code(lab.get_data_path() / 'train.py', source_files[:train_valid_split])
    _load_code(lab.get_data_path() / 'valid.py', source_files[train_valid_split:])


if __name__ == '__main__':
    main()
