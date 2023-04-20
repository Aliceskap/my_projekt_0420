import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
#%matplotlib widget
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')

# UNQ_C1
# GRADED CELL: my_softmax

#build my softmax with numpy
def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    
    a = np.zeros_like(z)
    N = z.shape[0]
    for i in range(N):
        a[i] = np.exp(z[i])
    a = a/sum(a)

    return a

z = np.array([1., 2., 3., 4.])
a = my_softmax(z)
atf = tf.nn.softmax(z) #computes softmax activations
print(f"my_softmax(z):         {a}")
print(f"tensorflow softmax(z): {atf}")

my_softmax(z)

#Problem Statement
#In this exercise, you will use a neural network to recognize ten handwritten digits, 0-9. 
#This is a multiclass classification task where one of n choices is selected. 
#Automated handwritten digit recognition is widely used today - 
# from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks.

# def load_data():
#     X = np.load("data/X.npy")
#     y = np.load("data/y.npy")
#     return X, y
# #capire come si fa a prendere dati da MINST

# # load dataset
# X, y = load_data()

# #The data set contains 5000 training examples of handwritten digits.
# #Each training example is a 20-pixel x 20-pixel grayscale image of the digit.
# #Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
# #The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector.
# #Each training examples becomes a single row in our data matrix X.
# #This gives us a 5000 x 400 matrix X where every row is a training example of a handwritten digit image.
# #The second part of the training set is a 5000 x 1 dimensional vector y that contains labels for the training set
# #y = 0 if the image is of the digit 0, y = 4 if the image is of the digit 4 and so on.

#I dowload the data my way
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
    path='mnist.npz'
)

X = x_test.reshape(-1, 784)
y = y_test.reshape(10000, -1)

# #get more familiar with the data set

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

#fig.tight_layout(pad=0.5)
#widgvis(fig)
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
#Model representation

#The neural network you will use in this assignment has two dense layers with ReLU activations 
#followed by an output layer with a linear activation.
#Recall that our inputs are pixel values of digit images.
#Since the images are of size  20×20, this gives us  400 inputs.
#Softmax is applied separately at the end with the loss function. 
#The parameters have dimensions that are sized for a neural network with 25 units in layer 1,  
# 15 units in layer 2 and  10 output units in layer 3, one for each digit.

# UNQ_C2
# GRADED CELL: Sequential model
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

#### Examine Weights shapes
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
    X,y,
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
    #widgvis(fig)
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
    """ display a single digit. The input is one digit (400,). """
    fig, ax = plt.subplots(1,1, figsize=(0.5,0.5))
    #widgvis(fig)
    X_reshaped = X.reshape((28,28))
    # Display the image
    ax.imshow(X_reshaped, cmap='gray')
    plt.show()

image_of_four = X[1015]
display_digit(image_of_four)

prediction = model.predict(image_of_four.reshape(1,784))  # prediction

print(f" predicting a four: \n{prediction}")
print(f" Largest Prediction index: {np.argmax(prediction)}")

# The largest output is prediction[2], indicating the predicted digit is a '2'. 
# If the problem only requires a selection, that is sufficient. Use NumPy argmax to select it. 
# If the problem requires a probability, a softmax is required:

prediction_p = tf.nn.softmax(prediction)

print(f" predicting a four. Probability vector: \n{prediction_p}")
print(f"Total of predictions: {np.sum(prediction_p):0.3f}")

#np.argmax returns the index of the largest value, i.e. the predicted value (the one with highest probability)
yhat = np.argmax(prediction_p)

print(f"np.argmax(prediction_p): {yhat}")

#Another example:
test = X[1012]
display_digit(test)

prediction = model.predict(test.reshape(1,784))  # prediction
print(f"Predicted number is: {np.argmax(prediction)}")

#Let's compare the predictions vs the labels for a random sample of 64 digits. This takes a moment to run
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
#widgvis(fig)
for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((28,28))
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Predict using the Neural Network
    prediction = model.predict(X[random_index].reshape(1,784))
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)
    
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]},{yhat}",fontsize=10)
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=14)
plt.show()

#Let's look at some of the errors:

def display_errors(model,X,y):
    f = model.predict(X)
    yhat = np.argmax(f, axis=1)
    doo = yhat != y[:,0]
    idxs = np.where(yhat != y[:,0])[0]
    if len(idxs) == 0:
        print("no errors found")
    else:
        cnt = min(8, len(idxs))
        fig, ax = plt.subplots(1,cnt, figsize=(5,1.2))
        fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.80]) #[left, bottom, right, top]
        #widgvis(fig)

        for i in range(cnt):
            j = idxs[i]
            X_reshaped = X[j].reshape((28,28))

            # Display the image
            ax[i].imshow(X_reshaped, cmap='gray')

            # Predict using the Neural Network
            prediction = model.predict(X[j].reshape(1,784))
            prediction_p = tf.nn.softmax(prediction)
            yhat = np.argmax(prediction_p)

            # Display the label above the image
            ax[i].set_title(f"{y[j,0]},{yhat}",fontsize=10)
            ax[i].set_axis_off()
            fig.suptitle("Label, yhat", fontsize=12)
    return(len(idxs))


print( f"{display_errors(model,X,y)} errors out of {len(X)} images")

print('Isn\'t this too many errors?')