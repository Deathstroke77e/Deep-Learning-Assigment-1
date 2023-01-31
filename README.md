# Deep-Learning-Assigment-1
Implement a neural network and utilize the CIFAR-10 dataset for the analysis.
1. Utilize various activation functions like sigmoid, tanh and critique the performance in
each case.
2. Increase the depth of the given network by adding more Fully-Connected layers till the
point you encounter the vanishing gradient problem. With the help of the results, mention
how to identify it.
3. Suggest and implement methods to overcome the above problem.

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images
Here are the classes in the dataset : airplane , automobile , bird , cat, deer , dog , frog , horse, ship and truck.
Here , Images are 3 channel (3*32*32) , which are converted to grayscale to speed up training (32*32).
For training our model I used the DataLoader() function provided by PyTorch
The loss function used is the cross-entropy.
The model was trained  for 9 epochs and with a learning rate of 0.001 and with Adam optimizer.

Sigmoid Activation Function:

Sigmoid function is known as the logistic function which helps to normalize the output of any input in the range between 0 to 1.  The main purpose of the activation function is to maintain the output or predicted value in the particular range, which makes the good efficiency and accuracy of the model.
Accuracy on test set - 45% 
But, I was encountering the problem of vanishing gradient .

Hyperbolic Tangent Activation Function:

Tanh Activation function is superior then the Sigmoid Activation function because the range of this activation function is higher than the sigmoid activation function. This is the major difference between the Sigmoid and Tanh activation function.
Accuracy on test set - 47%

ReLu (Rectified Linear Unit) Activation Function:

ReLu is the best and most advanced activation function right now compared to the sigmoid and TanH because all the drawbacks like Vanishing Gradient Problem is completely removed in this activation function which makes this activation function more advanced compare to other activation function.
Vanishing gradient issue was resolved and started getting accuracy of >52%.


Implement a neural network on the Gurmukhi dataset and implement the following regularization
techniques from scratch:
1. L-1 regularization
2. L-2 regularization
3. Dropout

Gurmukhi dataset has 1000 images of an indian script gurmukhi consisting of 10 stroke class labels.
Train - test split 4:1
A regression model that uses L1 regularization technique is called Lasso Regression and model which uses L2 is called Ridge Regression.
The key difference between these two is the penalty term.
Ridge regression adds squared magnitude of coefficient as penalty term to the loss function.
With 3 hidden layers accuracy was 88% on test set.
After using L1 regularizer the accuracy improved to 88.5%
After using L2 regularizer the accuracy was highest at 95%.

Drop out:

Dropout is a technique where you remove units in a neural net to simulate training large numbers of architectures simultaneously. Importantly, dropout can drastically reduce the chance of overfitting during training. 

Accuracy after applying drop out (with probability 0.01) was 90%.
