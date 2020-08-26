#!/usr/bin/env python

import urllib.request
from pathlib import Path

from labml import lab, monit

REPOS = [
    'lab-ml/source_code_modelling',
    ' sherlock-project / sherlock ',
    'Rapptz / discord.py ',
    ' Dod-o / Statistical-Learning-Method_Code '
]


def download_repo(org: str, repo: str):
    if Path(lab.get_data_path() / 'download' / f'{org}_{repo}.zip').exists():
        return

    with monit.section(f"Downloading {org}/{repo}"):
        zip = urllib.request.urlopen(f'https://github.com/{org}/{repo}/archive/master.zip')
        content = zip.read()

        with open(str(lab.get_data_path() / 'download' / f'{org}_{repo}.zip'), 'wb') as f:
            f.write(content)


def download():
    path = Path(lab.get_data_path() / 'download')
    if not path.exists():
        path.mkdir(parents=True)

    for r in REPOS:
        org, repo = [s.strip() for s in r.split('/')]
        download_repo(org, repo)


if __name__ == '__main__':
    download()
