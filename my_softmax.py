import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt

#for feature scaling (can be done also in tf right?)
from sklearn.preprocessing import StandardScaler

#Problem Statement
#In this exercise, you will use a neural network to recognize ten handwritten digits, 0-9. 
#This is a multiclass classification task where one of n choices is selected. 
#Automated handwritten digit recognition is widely used today - 
# from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks.

#The data set contains 10000 training examples of handwritten digits.
#Each training example is a 28-pixel x 28-pixel grayscale image of the digit.
#Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
#The 28 by 28 grid of pixels is “unrolled” into a 784-dimensional vector.
#Each training examples becomes a single row in our data matrix X.
#This gives us a 10000 x 784 matrix X where every row is a training example of a handwritten digit image.
#The second part of the training set is a 10000 x 1 dimensional vector y that contains labels for the training set
#y = 0 if the image is of the digit 0, y = 4 if the image is of the digit 4 and so on.

#Load data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
    path='mnist.npz'
)

#Convert the 28x28 matrinc into a 784 array
#Convert 1-D arrays into 2-D because the commands later will require it
#y = np.expand_dims(y, axis=1)
X = x_test.reshape(-1, 784)
y = y_test.reshape(10000, -1)

#get more familiar with the data set

#print some data elements
print ('The first element of X is: ', X[0])
print ('The first element of y is: ', y[0,0])
print ('The last element of y is: ', y[-1,0])

#get the dimensions of data
print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))

#visualize data 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((28,28))
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Display the label above the image
    ax.set_title(y[random_index,0])
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize=14)
plt.show()

#Feature scaling

# Initialize the class
scaler_linear = StandardScaler()

# Compute the mean and standard deviation of the training set then transform it
X_train_scaled = scaler_linear.fit_transform(X)

#print some data elements
print ('The first element of X is: ', X_train_scaled[0])

#Model representation

#The neural network you will use in this assignment has two dense layers with ReLU activations 
#followed by an output layer with a linear activation.
#Recall that our inputs are pixel values of digit images.
#Since the images are of size  28×28, this gives us  784 inputs.
#Softmax is applied separately at the end with the loss function. 
#The parameters have dimensions that are sized for a neural network with 25 units in layer 1,  
# 15 units in layer 2 and  10 output units in layer 3, one for each digit.

tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        tf.keras.Input(shape=(784,)),    #specify input shape
        Dense(25, activation = 'relu' ),
        Dense(15, activation = 'relu'),
        Dense(10, activation = 'linear')

    ], name = "my_model" 
)

model.summary() #summary of the model

[layer1, layer2, layer3] = model.layers

#Examine Weights shapes
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

#compile model with loss function for softmax and Adam optimizer
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

#train the model
history = model.fit(
    X_train_scaled,y, #changed from X to X_train_scaled
    epochs=40
)

#Loss (cost)
#In course 1, we learned to track the progress of gradient descent by monitoring the cost. 
#Ideally, the cost will decrease as the number of iterations of the algorithm increases. 
#Tensorflow refers to the cost as loss. Above, you saw the loss displayed each epoch as model.fit was executing. 
#The .fit method returns a variety of metrics including the loss. This is captured in the history variable above. 
#This can be used to examine the loss in a plot as shown below.

def plot_loss_tf(history):
    fig,ax = plt.subplots(1,1, figsize = (4,3))
    ax.plot(history.history['loss'], label='loss')
    ax.set_ylim([0, 2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()

plot_loss_tf(history)

#make a prediction
#To make a prediction, use Keras predict. Below, X[1015] contains an image of a 4.

print(f'expected value for X[1015]: {y[1015]}')

def display_digit(X):
    """ display a single digit. The input is one digit (784,). """
    fig, ax = plt.subplots(1,1, figsize=(0.5,0.5))
    #widgvis(fig)
    X_reshaped = X.reshape((28,28))
    # Display the image
    ax.imshow(X_reshaped, cmap='gray')
    plt.show()

image_of_four = X[1015]
display_digit(image_of_four)

image_of_four_rescaled= X_train_scaled[1015]
prediction = model.predict(image_of_four_rescaled.reshape(1,784))  # prediction

# The largest output is prediction[4], indicating the predicted digit is a '4'. 
# If the problem only requires a selection, that is sufficient. Use NumPy argmax to select it. 
# If the problem requires a probability, a softmax is required.
print(f" Model output: \n{prediction}")
print(f" Predicted number is: {np.argmax(prediction)}")


#Another example:
test = X[1012]
display_digit(test)
test_rescaled = X_train_scaled[1012]

prediction = model.predict(test_rescaled.reshape(1,784))  # prediction
print(f"Predicted number is: {np.argmax(prediction)}")

#Let's compare the predictions vs the labels for a random sample of 64 digits. This takes a moment to run
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((28,28))
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Predict using the Neural Network
    prediction = model.predict(X_train_scaled[random_index].reshape(1,784))
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)
    
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]},{yhat}",fontsize=10)
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=14)
plt.show()

#Let's look at some of the errors:

def display_errors(model,X,y, X_train_scaled):
    f = model.predict(X_train_scaled, verbose = 0)
    yhat = np.argmax(f, axis=1)
    doo = yhat != y[:,0]
    idxs = np.where(yhat != y[:,0])[0]
    if len(idxs) == 0:
        print("no errors found")
    else:
        cnt = min(8, len(idxs))
        fig, ax = plt.subplots(1,cnt, figsize=(5,1.2))
        fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.80]) #[left, bottom, right, top]

        for i in range(cnt):
            j = idxs[i]
            X_reshaped = X[j].reshape((28,28))

            # Display the image
            ax[i].imshow(X_reshaped, cmap='gray')

            # Predict using the Neural Network
            prediction = model.predict(X_train_scaled[j].reshape(1,784), verbose=0)
            prediction_p = tf.nn.softmax(prediction)
            yhat = np.argmax(prediction_p)

            # Display the label above the image
            ax[i].set_title(f"{y[j,0]},{yhat}",fontsize=10)
            ax[i].set_axis_off()
            fig.suptitle("Label, yhat", fontsize=12)
    return(len(idxs))

tot_errors = display_errors(model,X,y,X_train_scaled)
print( f"{tot_errors} errors out of {len(X)} images")
print(f'Error ratio: {tot_errors/len(X)}')
