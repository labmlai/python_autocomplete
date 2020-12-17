# Source Code Modeling

This repo trains deep learning models on source code.

### Try it yourself

1. Clone this repo
2. Install requirements from `requirements.txt`
3. Run `python_autocomplete/create_dataset.py`. 
   * It collects repos mentioned in
 [PyTorch awesome list](https://github.com/bharathgs/Awesome-pytorch-list)
   * Downloads the zip files of the repos
   * Extract the zips
   * Remove non python files
   * Collect all python code to `data/train.py` and, `data/eval.py`
4. Run `python_autocomplete/train.py` to train the model.
 *Try changing hyper-parameters like model dimensions and number of layers*.
5. Run `evaluate.py` to evaluate the model.

<p align="center">
  <img src="/python-autocomplete.png?raw=true" width="100%" title="Screenshot">
</p>
