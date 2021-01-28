from labml import experiment, lab

if __name__ == '__main__':
    experiment.save_bundle(lab.get_path() / 'bundle.tar.gz', '39b03a1e454011ebbaff2b26e3148b3d',
                           data_files=['cache/itos.json', 'cache/n_tokens.json', 'cache/stoi.json'])
