import zipfile
from pathlib import Path

from labml import lab, monit


def main():
    download = Path(lab.get_data_path() / 'download')
    source = Path(lab.get_data_path() / 'source')

    for repo in download.iterdir():
        with monit.section(f"Extract {repo.stem}"):
            repo_source = source / repo.stem
            if repo_source.exists():
                continue
            with zipfile.ZipFile(repo, 'r') as repo_zip:
                repo_zip.extractall(repo_source)

    if not source.exists():
        source.mkdir(parents=True)


if __name__ == '__main__':
    main()
