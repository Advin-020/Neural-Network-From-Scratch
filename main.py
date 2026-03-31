import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

#Constructing Layer
class Layer_Dense: 
    #randomly assigning weights and biases
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        #weights are shaped (n_inputs, n_neurons) to avoid transpose later
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        
        #set regulation/lambda values
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    #calculating output :- 
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    #backward pass 
    def backward(self, dvalues):
        #gradient on parameters
        self.dweights = np.dot(self.inputs.T , dvalues)
        self.dbiases = np.sum(dvalues , axis=0, keepdims=True)
        
        #gradients on regularization
        #L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        #L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        #L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        #L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        #gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

#Input Layer
class Layer_Input:
    #forward pass
    def forward(self,inputs,training):
        self.output = inputs

#Dropout Layer
class Layer_Dropout:
    def __init__(self, rate):
        #assiging value of rate
        #rate(input) : failiing/dropping rate
        #self.rate : sucess rate(inverted of rate)
        self.rate = 1 - rate
    
    #forward pass
    def forward(self, inputs, training):
        
        self.inputs = inputs
        
        if not training:
            self.output = inputs.copy()
            return self.output
        
        #genrate scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        
        #apply mask to output
        self.output =  inputs * self.binary_mask
    
    #backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
    
#Activation Function :- ReLU
class Activation_ReLU:
    #forward pass
    def forward(self, inputs,training):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    
    #backward pass
    def backward(self, dvalues):
        #copying to modify the original variable
        self.dinputs = dvalues.copy()
        #zero gradient for negative values
        self.dinputs[self.inputs <=0] = 0
    
    #Calculate predictions for output
    def predictions(self, outputs):
        return outputs

#Activation Function :- SoftMax -> For output layer
class Activation_SoftMax:
    #SoftMax => Expotential + Normilization
    #forward pass
    def forward(self, input, training):
        #Subtracted max value to prevent from overflow
        exp_values = np.exp(input - np.max(input , axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values , axis=1, keepdims=True)
        self.output = probabilities
    
    #backward pass
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for index ,(single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            
            single_output = single_output.reshape(-1,1)
            #calculate jacobian matrix of output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output , single_output.T)
            #calculate sampel-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix , single_dvalues)
    
    #Calculate predictions for output
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
          
#Activation Function :- Sigmoid -> For output Layer(Binary Logistic Regression)
class Activation_Sigmoid:
    #Forward Pass
    def forward(self,inputs, training):
        self.inputs = inputs
        self.output = 1/(1 + np.exp(-inputs))
        
    #backward pass
    def backward(self,dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    #Calculate predictions for output
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

#Activation Function :- Linear Actication
class Activation_Linear:
    #fordward pass
    def forward(self,inputs, training):
        self.inputs = inputs
        self.output = inputs
    
    #backward pass
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        
    #Calculate predictions for output
    def predictions(self, outputs):
        return outputs
    
#Parent Loss Class
class Loss:
    
    #set/remember trainable data
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    
    #calculate data and regularization loss
    def calculate(self, output, y, *, include_regularization=False):
        
        #calculate sample losses
        sampel_loses = self.forward(output,y)
        
        #calculate mean loss
        data_loss = np.mean(sampel_loses)
        
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    #regularization loss calculation
    def regularization_loss(self):
        
        #default value
        regularization_loss = 0
        
        #calculate regularization loss for all trainable layers
        for layer in self.trainable_layers:
            
            #L1 regularization -> weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
                
            #L2 regularization -> weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            #L1 regularization -> biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            
            #L2 regularization -> biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        
        return regularization_loss
    
    #delete this :- test onlyy
    def regularization_loss1(self,layer):
        
        #default value
        regularization_loss = 0
        
        #L1 regularization -> weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
                
        #L2 regularization -> weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        #L1 regularization -> biases
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            
        #L2 regularization -> biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        
        return regularization_loss
       
#Loss Function :- Categorical Cross Entropy Loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        sampel = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        #Scaler value -> [0,1,1]
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sampel), y_true]
        
        #One-Hot encoded -> [[1,0], [0,1], [0,1]]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_log_liklihood = -np.log(correct_confidence)
        return negative_log_liklihood
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        #if sparse than turn it int one-hot encoded
        if len(y_true.shape) ==1:
            y_true = np.eyes(labels)[y_true]
        
        #calculating gradient
        self.dinputs = - y_true / dvalues
        #normilizing gradient
        self.dinputs /= samples    

#Loss Function :- Binary Cross Entropy Loss
class Loss_BinaryCrossentropy(Loss):
    #forward pass
    def forward(self,y_pred,y_true):
        #clip data(from both side) to prevent division by 0
        y_pred_clipped = np.clip(y_pred , 1e-7 , 1 - 1e-7)
        
        #Sample wise loss calculation
        sample_loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_loss = np.mean(sample_loss, axis=-1)
        
        return sample_loss
    
    #backward pass
    def backward(self,dvalues,y_true):
        #number of samples
        samples = len(dvalues)
        #number of outputs
        outputs = len(dvalues[0])
        
        #clip data
        clipped_dvalues = np.clip(dvalues, 1e-7 , 1 - 1e-7)
        
        #calcutale graident
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        
        #normalize graident
        self.dinputs = self.dinputs / samples
        
#Loss Function :- Mean Squared Loss
class Loss_MeanSquaredError(Loss):
    #forward pass
    def forward(self,y_pred,y_true):
        sample_losses = np.mean((y_true - y_pred)**2,axis=-1)
        return sample_losses
    
    #backward pass
    def backward(self,dvalues,y_true):
        #number of samples
        sample = len(dvalues)
        #number of outputs
        output = len(dvalues[0])
        
        #gradients
        self.dinputs = -2 * (y_true - dvalues) / output
        #normalize 
        self.dinputs =  self.dinputs / sample
        
#Loss FUnction :- Mean Absolute Loss
class Loss_MeanAbsoluteLoss(Loss):
    #fordward pass
    def forward(self,y_pred,y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred),axis=-1)
        return sample_losses
    
    #backward pass
    def backward(self,dvalues,y_true):
        #number of samples
        samples = len(dvalues)
        #number of outputs
        outputs = len(dvalues[0])
        
        #graident
        self.dinputs = np.sign(y_true - dvalues)/outputs
        #normalizee
        self.dinputs = self.dinputs / samples
    
#combined softmax and loss
class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    #backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        #if labels are one-hot encoded
        # trun them into scaler
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true ,axis=1)
        
        self.dinputs = dvalues.copy()
        #calculate graidents
        self.dinputs[range(samples), y_true] -= 1
        
        #normalize the gradients
        self.dinputs /= samples
        
#optimizer:- SGD
class Optimizer_SGD:
    #initialize optimizer
    #default learning rate = 1
    def __init__(self, learning_rate=1. , decay=0., momentum=0.):
         self.learning_rate = learning_rate
         self.current_learning_rate = learning_rate
         self.decay = decay
         self.iterations = 0
         self.momentum = momentum
    
    #before updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    #updating parameters
    def update_params(self, layer):
        #if we are using momentum
        if self.momentum:
            #if layer dont have momentum arrays
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            weight_update = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_update
            
            bias_update = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_update
        #without momentum
        else:
            layer.weights += -self.learning_rate * layer.dweights
            layer.biases += -self.learning_rate * layer.dbiases
        
        layer.weights += weight_update
        layer.biases += bias_update

    #after update
    def post_update_params(self):
        self.iterations += 1

#optimizer:- AdaGrad
class Optimizer_Adagrad:
    #initialize optimizer
    def __init__(self, learning_rate=1. , decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
    
    #before updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    #updating parameters
    def update_params(self, layer):
        #if layer dont have cache arrays
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
                
        #update cache
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        #update params
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    #after update
    def post_update_params(self):
        self.iterations += 1
        
#optimizer:- RMSprop
class Optimizer_RMSprop:
    #initialize optimizer
    def __init__(self, learning_rate=0.001 , decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    #before updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    #updating parameters
    def update_params(self, layer):
        #if layer dont have cache arrays
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        #update cache    
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        
        #update params
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    #after update
    def post_update_params(self):
        self.iterations += 1
               
#optimizer:- Adam
class Optimizer_Adam:
    #initialize optimizer
    def __init__(self, learning_rate=0.001 , decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    #before updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    #updating parameters
    def update_params(self, layer):
        #if layer dont have cache arrays
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentums = np.zeros_like(layer.weights)
            
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentums = np.zeros_like(layer.biases)
            
        #update momentum
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        #get corrected momentum
        #iteration starts from 0 , hence added 1
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        #update cache
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        
        #get correctedd cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        #update params
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    #after update
    def post_update_params(self):
        self.iterations += 1
    
#model class    
class Model:
    #initializing the model
    def __init__(self):
        #list of network objects(containing all layers and next & prev values of those layers)
        self.layers = []
        self.softmax_classifier_output = None
        
    #add objects(layers & activation functions)
    def add(self, layer):
        self.layers.append(layer)
        
    #set loss and optimizer
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
        
    #finalize model
    def finalize(self):
        #creating and setting input layer
        self.input_layer = Layer_Input()
        
        #number of objects
        layer_count = len(self.layers)
        
        #initialazing list of trainable data
        self.trainable_layers = []
        
        for i in range(layer_count):
            #first layers -> prev is input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
                
            #all layers except first and last 
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            
            #last layer -> next is loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
        
            #if layer have string "weights"
            #than it is trainable
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        
            #update loss object with trainable layers
            self.loss.remember_trainable_layers(self.trainable_layers)
        
        #if output activation is : Softmax
        #and loss is: Categorical
        #than create an combined object
        if isinstance(self.layers[-1], Activation_SoftMax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            #Combined object of activation & loss function
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

                
    #train the model
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        
        #initialize accuracy object
        self.accuracy.init(y)
        
        #traing loop
        for epoch in range(1, epochs+1):
            
            #perform forward pass
            output = self.forward(X,training=True)
            
            #calculate loss
            data_loss, regularization_loss = self.loss.calculate(output,y,include_regularization=True)
            loss = data_loss + regularization_loss
            
            #get predictions and
            #calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions,y)

            #backward pass
            self.backward(output,y)
            
            #optimize : update parameters ->
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            
            #print info ->
            if not epoch % print_every:
                print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate}')
                
        if validation_data is not None:
            
            X_val , y_val = validation_data
            
            #forward pass
            output = self.forward(X_val,training=False)
            
            #calculate loss
            loss = self.loss.calculate(output,y_val)
            
            #get predictions
            predictions = self.output_layer_activation.predictions(output)
            #calculate accuracy
            accuracy = self.accuracy.calculate(predictions,y_val)
            
            #print info
            print(f'validation, acc:{accuracy:.3f}, loss:{loss:.3f}')
            
   
    #forward pass[through all layers]
    def forward(self, X, training):
        
        self.input_layer.forward(X,training)
        
        #forward pass for all layers
        #prev layer output as current input
        for layer in self.layers:
            layer.forward(layer.prev.output,training)
            
        return layer.output
    
    #backward pass
    def backward(self,output,y):
        
        if self.softmax_classifier_output is not None:
            
            self.softmax_classifier_output.backward(output,y)
            
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            
            #backward pass on all but the last one
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
                
            return
                
        
        self.loss.backward(output,y)
        
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
        
#Accuracy class
class Accuracy:
    def calculate(self, predictions, y):
        #get comparison results
        comparisons = self.compare(predictions, y)
        
        #calculate an accuracy
        accuracy = np.mean(comparisons)
        
        #return accuracy
        return accuracy
       
#Accuracy :- Regression
class Accuracy_Regression(Accuracy):
    
    #initilizing precision
    def __init__(self):
        self.precision = None
    
    #Calculate precision value
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
            
    #compare predictions to ground truth
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
        
#Accuracy :- Categorical
class Accuracy_Categorical(Accuracy):
    
    #no initialization needed
    def init(self,y):
        pass
    
    #compare predictions to ground truth
    def compare(self, predictions, y):
        '''
        
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
            
        '''
        return (predictions == y)


X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = Model()

model.add(Layer_Dense(2, 32, weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(32, 3))
model.add(Activation_SoftMax())

model.set(loss=Loss_CategoricalCrossentropy(),optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),accuracy=Accuracy_Categorical())

model.finalize()

model.train(X, y, validation_data=(X_test, y_test),epochs=10000, print_every=100)