import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # Ref: https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format

    with gzip.open(f'{image_filesname}', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        x = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
        x = x.reshape((size, nrows * ncols))
        x = x.astype(np.float32)
        x = x / 255.0

    with gzip.open(f'{label_filename}','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
        y = y.reshape((size,)) # (Optional)

    return x, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    ### ANS 1 ###
    # print("### START ###")
    # batch_size = Z.shape[0]
    # batch_loss = list()
    # for row_index, row in enumerate(Z.numpy()):
    #     zy = row[np.argmax(y_one_hot.numpy()[row_index])]
    #     zi = row
    #
    #     loss = np.log(np.sum(np.exp(zi))) - zy
    #     batch_loss.append(loss)
    #
    # print("### END ###")
    # # print(batch_size, batch_loss)
    # result = ndl.Tensor(np.sum(batch_loss) / batch_size)
    # return result
    ### ANS 1 ###

    ### ANS 2 ###
    # batch_size = Z.shape[0]
    #
    #
    # zy = ndl.Tensor(np.argmax(y_one_hot.numpy(), axis=1))
    # loss = ndl.log(ndl.summation(ndl.exp(Z), axes=1)) - zy
    #
    # result = ndl.divide_scalar(ndl.summation(loss), batch_size)
    # print("result:", result)
    # return result
    ### ANS 2 ###

    ### ANS 3 ###
    # Z = Z.numpy()
    # batch_size = Z.shape[0]
    # zy = np.argmax(y_one_hot.numpy(), axis=1)
    # items = Z[np.arange(Z.shape[0]), zy]
    # loss = np.log(np.sum(np.exp(Z), 1)) - items
    # result = ndl.Tensor(np.sum(loss) / batch_size)
    #
    # return result
    ### ANS 3 ###

    ### ANS 4 ###
    batch_size = Z.shape[0]
    loss = ndl.log(ndl.summation(ndl.exp(Z), axes=1)) - \
           ndl.summation(ndl.multiply(y_one_hot, Z),  axes=1)
    result = ndl.divide_scalar(ndl.summation(loss), batch_size)
    print("@@@result:", result)

    return result
    ### ANS 4 ###

    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    batch_size = X.shape[0]
    for i in range(0, batch_size, batch):
        X_batch = X[i : i+batch]
        y_batch = y[i : i+batch]
        X_batch = ndl.Tensor(X_batch)
        #   Z1 = ndl.ops.relu( ndl.ops.matmul( X_batch , W1))
        #   Z = ndl.ops.matmul( Z1 ,W2)
        Z1 = ndl.relu(X_batch @ W1)
        Z = Z1 @ W2
        y_one_hot = np.zeros(Z.shape, dtype="float32")
        y_one_hot[np.arange(Z.shape[0]),y_batch] = 1
        loss = softmax_loss(Z, ndl.Tensor(y_one_hot))
        loss.backward()

        W1 = (W1 - lr * W1.grad).detach()
        W2 = (W2 - lr * W2.grad).detach()
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
