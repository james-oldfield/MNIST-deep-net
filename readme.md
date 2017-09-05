# MNIST-deep-net

A vanilla Neural Network to classify MNIST dataset images  written in Python / numpy (thus, misses out on many benefits of using a framework like Tensorflow, etc.).

Certainly won't be winning any awards for classification accuracy. More of a PoC vanilla numpy deep neural net to explore NN optimisation concepts.

W.I.P / to come:

- [ ] Optional Adam / momentum / RMSProp optimisation
- [ ] L2 regularisation
- [ ] Dropout
- [ ] Early stopping
- [ ] Xavier weight initialisation

etc.

## install

`pip install -r requirements.txt`

- using python version `3.6.1`
- using flake8 for python linting

## run

`python train.py`

(specify hyper parameters in the python file)

N.B. The dataset itself isn't checked in, you'll have to download that [here](http://yann.lecun.com/exdb/mnist/)
