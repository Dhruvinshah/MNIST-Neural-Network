
import numpy as np
import time
import pandas as pd
import sys
import csv
start_time = time.time()
df = pd.read_csv(str(sys.argv[1]), header = None)

print(len(df))
x_train = np.array(df)
x_train = (x_train/255).astype('float32')


df1 = pd.read_csv(str(sys.argv[2]), header= None)
final_res = []
temp = np.array(df1)
for i in range(len(temp)):
    res=[0 for x in range(10)]
    temp1 = int(temp[i])
    res[temp1]=1
    final_res.append(res)

y_train = np.array(final_res).astype('float32')


df2 = pd.read_csv(str(sys.argv[3]), header = None)
x_test = np.array(df2)
x_test = (x_test/255).astype('float32')




class NeuralNetwork():
    # def __init__(self, sizes, epochs=1, l_rate=0.02):
    def __init__(self):
        self.sizes = [784, 128, 64, 10]
        self.epochs = 400
        self.l_rate = 0.02

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def sigmoid(self, x):
        # if derivative:
        #     return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        # if derivative:
        #     return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]

        params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'].T)
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):

        params = self.params
        change_w = {}
        
        error = (output - y_train.T) 
        change_w['W3'] = (1./30)*np.matmul(error, params['A2'].T)
        
        error = np.dot(params['W3'].T, error)* self.sigmoid_derivative(params['Z2'])
        change_w['W2'] = (1./30)*np.matmul(error, params['A1'].T)

        
        error = np.dot(params['W2'].T, error)  * self.sigmoid_derivative(params['Z1'])
        change_w['W1'] = (1./30)*np.matmul(error, params['A0'])


        return change_w

    def update_network_parameters(self, changes_to_w):
        
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    

    def predict_output(self,x_test):
        predictions = []
        for x in x_test:
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append([pred])
        
        self.write_output(predictions)

    def write_output(self,predictions): 
        file = open('test_predictions.csv', 'w+', newline ='') 
        with file:     
            write = csv.writer(file) 
            write.writerows(predictions)
        # return predictions

    def train(self, x_train, y_train, x_test):
        batch = 30
        for iteration in range(self.epochs):
            
            for i in range(0,len(x_train),batch):
                output = self.forward_pass(x_train[i:i+batch])
                changes_to_w = self.backward_pass(y_train[i:i+batch], output)
                self.update_network_parameters(changes_to_w)
            
        
       
        output = self.predict_output(x_test)
        
            
neural_net = NeuralNetwork()

neural_net.train(x_train, y_train, x_test)
print(time.time() - start_time)

# This is how you run it!
# python NeuralNetwork3.py train_image.csv train_label.csv test_image.cv