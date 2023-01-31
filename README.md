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

# Sigmoid Activation Function:

Sigmoid function is known as the logistic function which helps to normalize the output of any input in the range between 0 to 1.  The main purpose of the activation function is to maintain the output or predicted value in the particular range, which makes the good efficiency and accuracy of the model.
Accuracy on test set - 45% 
But, I was encountering the problem of vanishing gradient .
```
S=nn.Sigmoid()
class NNet_si(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc2=nn.Linear(1024,128)        # It has 3 hidden layers
    self.fc3=nn.Linear(128,128)
    self.fc4=nn.Linear(128,10)

  def forward(self,input):
    out=input.view(input.size(0),-1)
    out = self.fc2(out)
    out = S(out)
    out = self.fc3(out)
    out = S(out)
    out = self.fc4(out)
    return S(out)     
nets=NNet_si()
```
```
for epoch in range(9):                        
  for x,y in trainset:
       nets.zero_grad()                                  # Here loss value after every epoch is not decreasing.
       output=nets(x)
       loss = loss_fx(output, y) 
       loss.backward() 
       optimizers.step()
  print(loss) 
  ```

# Hyperbolic Tangent Activation Function:

Tanh Activation function is superior then the Sigmoid Activation function because the range of this activation function is higher than the sigmoid activation function. This is the major difference between the Sigmoid and Tanh activation function.
Accuracy on test set - 47%
```
correct=0
with torch.no_grad():
    for data in testset:                              # Accuracy on test set
        X, y = data
        output = nett(X)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct = correct + 1
print("Accuracy: ", round(correct/10000, 3))
Accuracy:  0.355
```

# ReLu (Rectified Linear Unit) Activation Function:

ReLu is the best and most advanced activation function right now compared to the sigmoid and TanH because all the drawbacks like Vanishing Gradient Problem is completely removed in this activation function which makes this activation function more advanced compare to other activation function.
Vanishing gradient issue was resolved and started getting accuracy of >52%.
```
Accuracy:  0.522
```


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
```
for epoch in range(9):                        
  for x,y in trainset:
       neural.zero_grad()
       output=neural(x)
       l1_norm = sum(p.abs().sum() for p in neural.parameters())    # Taking L1 norm of all parameters
       loss = loss_fx(output, y)  + l1_norm*0.001
       loss.backward()
       optimizer.step() 
       
 Accuracy:  0.885
```
After using L2 regularizer the accuracy was highest at 95%.
```
for epoch in range(9):                        
  for x,y in trainset:
       neural.zero_grad()
       output=neural(x)
       l2_norm = sum(torch.linalg.norm(p, 2) for p in neural.parameters())
       loss = loss_fx(output, y)  + l2_norm*0.001
       loss.backward()
       optimizer.step() 
Accuracy:  0.95
```

# Drop out:

Dropout is a technique where you remove units in a neural net to simulate training large numbers of architectures simultaneously. Importantly, dropout can drastically reduce the chance of overfitting during training. 

Accuracy after applying drop out (with probability 0.1) was 30%.
```
class net_Drop(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(1024,64)        
    self.fc2=nn.Linear(64,64)
    self.fc3=nn.Linear(64,10)
    self.dropout=nn.Dropout(p=0.1,inplace=False)

  def forward(self,input):
    out=input.view(input.size(0),-1)
    out = self.fc1(out)
    out = self.dropout(F.relu(out))
    out = self.fc2(out)
    out = self.dropout(F.relu(out))
    out = self.fc3(out)
    return F.log_softmax(out, dim=1)
neural_Drop=net_Drop()

Accuracy:  28.466
```
