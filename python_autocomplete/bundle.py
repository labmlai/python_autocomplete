from labml import experiment, lab

if __name__ == '__main__':
    experiment.save_bundle(lab.get_path() / 'bundle.tar.gz', 'a6cff3706ec411ebadd9bf753b33bae6',
                           data_files=['cache/itos.json',
                                       'cache/n_tokens.json',
                                       'cache/stoi.json',
                                       'cache/bpe.json',
                                       ])
