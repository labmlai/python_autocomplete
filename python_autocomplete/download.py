#!/usr/bin/env python
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from labml import lab, monit


def get_repos(filename: str):
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


def get_awesome_pytorch():
    md = urllib.request.urlopen('https://raw.githubusercontent.com/bharathgs/Awesome-pytorch-list/master/README.md')
    content = md.read()

    with open(str(lab.get_data_path() / 'pytorch_awesome.md'), 'w') as f:
        f.write(str(content))


def download_repo(org: str, repo: str, idx: Optional[int]):
    if Path(lab.get_data_path() / 'download' / f'{org}_{repo}.zip').exists():
        return

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

        with open(str(lab.get_data_path() / 'download' / f'{org}_{repo}.zip'), 'wb') as f:
            f.write(content)


def download():
    path = Path(lab.get_data_path() / 'download')
    if not path.exists():
        path.mkdir(parents=True)

    get_awesome_pytorch()
    repos = get_repos('pytorch_awesome.md')

    for i, r in monit.enum(f"Download {len(repos)} repos", repos):
        download_repo(r[0], r[1], i)


if __name__ == '__main__':
    download()
