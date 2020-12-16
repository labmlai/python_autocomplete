import zipfile
from pathlib import Path

from labml.internal.util import rm_tree

from labml import lab, monit


def extract_zips(overwrite: bool = False):
    download = Path(lab.get_data_path() / 'download')
    source = Path(lab.get_data_path() / 'source')

    if not source.exists():
        source.mkdir(parents=True)

    for repo in download.iterdir():
        with monit.section(f"Extract {repo.stem}"):
            repo_source = source / repo.stem
            if repo_source.exists():
                if overwrite:
                    rm_tree(repo_source)
                else:
                    continue
            with zipfile.ZipFile(repo, 'r') as repo_zip:
                repo_zip.extractall(repo_source)


if __name__ == '__main__':
    extract_zips()
