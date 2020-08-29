# Source Code Modeling

This repo trains deep learning models on source code.

### Try it yourself

1. Clone this repo
2. Install requirements from `requirements.txt`
3. Download Github repos by running `download.py`.
 It downloads all the repos mentioned in
 [PyTorch awesome list](https://github.com/bharathgs/Awesome-pytorch-list).
4. Run `extrat_downloads.sh` to extract the downloaded zip files to `data/source`.
 You can directly copy any python code to `data/source` to train on them.
5. Run `create_dataset.py` to collect all python files.
 The collected code will be written to `data/train.py` and, `data/eval.py`.
6. Clone our [transformers repo](https://github.com/lab-ml/transformers).
 And create a symbolic link to `transformers` package inside it (or just copy the entire folder).
6. Run `train.py` to train the model.
 *Try changing hyper-parameters like model dimensions and number of layers*.
7. Run `evaluate.py` to evaluate the model.
8. Enjoy!

If you have any questions please open an issue on Github.

Feel free to add interesting repos with lots of Python code to `download.py`.
 Thank you.
 
<p align="center">
  <img src="/python-autocomplete.png?raw=true" width="100%" title="Screenshot">
</p>
