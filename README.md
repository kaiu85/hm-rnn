hm-rnn

Implementation of a hierarchical multiscale recurrent neural network, following (Chung, Ahn & Bengio, 2016, https://arxiv.org/abs/1609.01704).

This implementation uses plain RNNs (in contrast to the LSTMs in the original paper).
It uses Python 3.6 and was tested with PyTorch 0.41 and scikit-learn 0.19.1.

If you want to learn how to create a new, custom environment to install the required versions of 
Python, PyTorch and scikit-learn, please look at: https://conda.io/docs/user-guide/tasks/manage-environments.html
For a getting started guide to PyTorch look here: https://pytorch.org/get-started/locally/

To start training the model, get some training dataset (in the form of a single .txt file) and put the path in the corresponding line of train.py.
Then you can start training by executing train.py. It will print a segmentation of some training data and some samples from the learned model 
every 100th training step.
