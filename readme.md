This is a simpler rewrite of [Python Autocomplete](https://github.com/vpj/python_autocomplete).
It is toy project we started
to see how well a simple LSTM model can autocomplete python code.

## Try it yourself

1. Clone this repo

2. Install requirements from `requirements.txt`

3. Copy data to `./data/source`

4. Run `extract_code.py` to collect all python files, encode and merge them into `all.py`

5. Run `train.py` to train the model

6. Run `evaluate.py` to evaluate the model.

