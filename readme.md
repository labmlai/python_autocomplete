# Source Code Modeling

This repo trains deep learning models on source code.

### Try it yourself

1. Clone this repo
2. Install requirements from `requirements.txt`
3. Download Github repos by running `python_autocomplete/download.py`.
 It downloads all the repos mentioned in
 [PyTorch awesome list](https://github.com/bharathgs/Awesome-pytorch-list).
4. Run `python_autocomplete/extract_downloads.py` to extract the downloaded zip files to `data/source`.
 You can directly copy any python code to `data/source` to train on them.
5. Run `python_autocomplete/remove_non_source_files.py` to all files except `.py` files.
6. Run `create_dataset.py` to collect all python files.
 The collected code will be written to `data/train.py` and, `data/eval.py`.
7. Run `train.py` to train the model.
 *Try changing hyper-parameters like model dimensions and number of layers*.
8. Run `evaluate.py` to evaluate the model.
9. Enjoy!

If you have any questions please open an issue on Github.

Feel free to add interesting repos with lots of Python code to `download.py`.
 Thank you.
 
<p align="center">
  <img src="/python-autocomplete.png?raw=true" width="100%" title="Screenshot">
</p>
