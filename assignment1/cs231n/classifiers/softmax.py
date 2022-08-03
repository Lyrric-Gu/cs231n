from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]

    logits = X @ W   # Interpret the output of output layer as log of probabilities
    data_loss = 0

    for i in range(N):
      logits_i = logits[i]   # pick i_th logits
      logits_i -= logits_i.max()   # for numerical stability, which let the maximum become 0
      
      probs_i = np.exp(logits_i)   # convert back to probabilities
      probs_i /= np.sum(probs_i)   # Normalization

      correct_class_prob_i = probs_i[y[i]]   # pick probability of correct class
      data_loss += -np.log(correct_class_prob_i)   # cross-entropy loss

      for j in range(C):
        if j == y[i]:
          dW[:, j] += (probs_i[j] - 1) * X[i].T
        else:
          dW[:, j] += probs_i[j] * X[i].T
    
    loss = data_loss / N + reg * np.sum(W * W)

    dW = dW / N + 2 * reg * W
      
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]

    logits = X @ W
    logits -= logits.max(axis=1, keepdims=True)   # Numerical stability
    probs = np.exp(logits)  # Convert to (unnormalized) probs
    probs /= np.sum(probs, axis=1, keepdims=True)    # Normalizatioin

    data_loss = np.sum(-np.log(probs[np.arange(N), y])) / N   # sum of -log(prob), where prob = exp(f_yi) / sum(exp(f_j))
    loss = data_loss + reg * np.sum(W * W)

    dlogits = probs
    dlogits[np.arange(N), y] -= 1
    dlogits /= N

    dW = X.T @ dlogits + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
