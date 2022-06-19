"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
        score = np.dot(X[i], W)

        # avoiding numerical instability
        instability = np.max(score)
        score = score - instability

        # probability calculation
        probability = np.exp(score) / np.sum(np.exp(score))
        loss = loss - np.log(probability[y[i]])
        probability[y[i]] -= 1
        
        # gradient
        for j in range(num_class):
            dW[:,j] += X[i,:] * probability[j]

    loss = loss/num_train
    dW = dW/num_train

    # using regularization
    loss = loss + 0.5 * reg * np.sum(W ** 2)
    dW = dW + reg * W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    num_train = X.shape[0]
    num_class = W.shape[1]

    Score = np.dot(X,W)
    Score = Score - np.reshape(Score.max(1), (num_train, 1))
    Probability = np.exp(Score)/np.sum(np.exp(Score), axis = 1, keepdims = True)
    p = np.zeros((num_train, num_class))
    p[range(num_train), y] = 1.0
    loss =- np.sum(p*np.log(Probability))/num_train + 0.5*reg*np.sum(W*W)
    dw = (Probability - p)/num_train
    dW = np.dot(X.T, dw) + reg*W



    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7, 5e-6, 5e-5, 5e-8]
    regularization_strengths = [2.5e4, 5e4, 1000, 10, 6000, 1500, 1000]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################


    for l_rate in learning_rates:
        for r_strength in regularization_strengths:
            softmax = SoftmaxClassifier()
            loss_softmax = softmax.train(X_train, y_train, learning_rate=l_rate, reg=r_strength,
                          num_iters=2000, verbose=False)
            y_train_pred = softmax.predict(X_train)
            y_val_pred = softmax.predict(X_val)
            train_accuracy = np.mean(y_train == y_train_pred)
            val_accuracy = np.mean(y_val == y_val_pred)
            results[(l_rate, r_strength)] = (train_accuracy, val_accuracy)
            if val_accuracy >= best_val:
                best_val = val_accuracy
                best_softmax = softmax
            all_classifiers.append(softmax)


    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
