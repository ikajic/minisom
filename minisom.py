from __future__ import division
from numpy import sqrt,sqrt,array,unravel_index,nditer,linalg,random,subtract,power,exp,pi,zeros,arange,outer,meshgrid
from collections import defaultdict
 
# TODO
import pylab as plt
from copy import copy
import plot_som as psom
from numpy import apply_along_axis, newaxis


"""
    Minimalistic implementation of the Self Organizing Maps (SOM).

    Giuseppe Vettigli 2013.
"""

class MiniSom:
    def __init__(self,x,y,input_len,sigma=1.0,learning_rate=0.5):
        """
            Initializes a Self Organizing Maps.
            x,y - dimensions of the SOM
            input_len - number of the elements of the vectors in input
            sigma - spread of the neighborhood function (Gaussian)
            (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
            learning_rate - initial learning rate
            (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
        """
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.weights = random.rand(x,y,input_len)*2-1 # random initialization
        self.weights = array([v/linalg.norm(v) for v in self.weights]) # normalization
        self.weights_init = copy(self.weights)
        self.activation_map = zeros((x,y))
        self.neigx = arange(x)
        self.neigy = arange(y) # used to evaluate the neighborhood function
        self.neighborhood = self.gaussian

    def _activate(self,x):
        """ Updates matrix activation_map, in this matrix the element i,j is the response of the neuron i,j to x """
        s = subtract(x,self.weights) # x - w
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.activation_map[it.multi_index] = linalg.norm(s[it.multi_index]) # || x - w ||
            it.iternext()

    def activate(self,x):
        """ Returns the activation map to x """
        self._activate(x)
        return self.activation_map

    def gaussian(self,c,sigma):
        """ Returns a Gaussian centered in c """
        d = 2*pi*sigma*sigma
        ax = exp(-power(self.neigx-c[0],2)/d)
        ay = exp(-power(self.neigy-c[1],2)/d)
        return outer(ax,ay) # the external product gives a matrix

    def diff_gaussian(self,c,sigma):
        """ Mexican hat centered in c (unused) """
        xx,yy = meshgrid(self.neigx,self.neigy)
        p = power(xx-c[0],2) + power(yy-c[1],2)
        d = 2*pi*sigma*sigma
        return exp(-(p)/d)*(1-2/d*p)

    def winner(self,x):
        """ Computes the coordinates of the winning neuron for the sample x """
        self._activate(x)
        return unravel_index(self.activation_map.argmin(),self.activation_map.shape)

    def update(self,x,win,t):
        """
            Updates the weights of the neurons.
            x - current pattern to learn
            win - position of the winning neuron for x (array or tuple).
            eta - learning rate
            t - iteration index
        """
        # eta(t) = eta(0) / (1 + t/T) 
        # keeps the learning rate nearly constant for the first T iterations and then adjusts it
        eta = self.learning_rate/(1+t/self.T)
        sig = self.sigma/(1+t/self.T) # sigma and learning rate decrease with the same rule
        g = self.neighborhood(win,sig)*eta # improves the performances
        it = nditer(g, flags=['multi_index'])
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            self.weights[it.multi_index] += g[it.multi_index]*(x-self.weights[it.multi_index])            
            # normalization
            self.weights[it.multi_index] = self.weights[it.multi_index] / linalg.norm(self.weights[it.multi_index])
            it.iternext()

    def quantization(self,data):
        """ Assigns a code book (weights vector of the winning neuron) to each sample in data. """
        q = zeros(data.shape)
        for i,x in enumerate(data):
            q[i] = self.weights[self.winner(x)]
        return q


    def random_weights_init(self,data):
        """ Initializes the weights of the SOM picking random samples from data """
                	
	# ranges of values for each dimension
        self.mins = data.min(axis=0)
        self.ranges = data.max(axis=0)-data.min(axis=0)
        
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.weights[it.multi_index] = data[int(random.rand()*len(data)-1)]
            self.weights[it.multi_index] = self.weights[it.multi_index]/linalg.norm(self.weights[it.multi_index])
            it.iternext()
            
	self.weights_init = copy(self.weights)

    def train_random(self,data,num_iteration):        
        """ Trains the SOM picking samples at random from data """
        self._init_T(num_iteration)        
       
        
        # store weights before the training
        self.p_weights_init = (copy(self.weights)*self.ranges+self.mins)
        
        # start training on normalized weights
        for iteration in range(num_iteration):
            rand_i = int(round(random.rand()*len(data)-1)) # pick a random sample          
            self.update(data[rand_i],self.winner(data[rand_i]),iteration)          
      
        # project weights into data space
        self.p_weights=self.weights*self.ranges
        self.p_weights+=self.mins

    def train_batch(self,data,num_iteration):
        """ Trains using all the vectors in data sequentially """
        self._init_T(len(data)*num_iteration)
        iteration = 0
        while iteration < num_iteration:
            idx = iteration % (len(data)-1)
            self.update(data[idx],self.winner(data[idx]),iteration)
            iteration += 1

    def _init_T(self,num_iteration):
        """ Initializes the parameter T needed to adjust the learning rate """
        self.T = num_iteration/2 # keeps the learning rate nearly constant for the first half of the iterations

    def distance_map(self):
        """ Returns the average distance map of the weights.
            (Each mean is normalized in order to sum up to 1) """
        um = zeros((self.weights.shape[0],self.weights.shape[1]))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1,it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1,it.multi_index[1]+2):
                    if ii >= 0 and ii < self.weights.shape[0] and jj >= 0 and jj < self.weights.shape[1]:
                        um[it.multi_index] += linalg.norm(self.weights[ii,jj,:]-self.weights[it.multi_index])
            it.iternext()
        um = um/um.max()
        return um

    def activation_response(self,data):
        """ 
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        a = zeros((self.weights.shape[0],self.weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def quantization_error(self,data):
        """ 
            Returns the quantization error computed as the average distance between
            each input sample and its best matching unit.            
        """
        error = 0
        for x in data:
            error += linalg.norm(x-self.weights[self.winner(x)])
        return error/len(data)

    def win_map(self,data):
    	"""
    	    Returns a dictionary wm where wm[(i,j)] is a list with all the patterns
    	    that have been mapped in the position i,j.
    	"""
    	winmap = defaultdict(list)
    	for x in data:
    		winmap[self.winner(x)].append(x)
    	return winmap
    	
    def get_weights(self):
    	"""
    		Flattens the two first dimensions of weights and returns 2D initial
    		and trained weights
    	"""
    	d1, d2, d3 = self.weights.shape
    	return self.weights_init.reshape(d1*d2, d3), self.weights.reshape(d1*d2, d3)
    	 

# helper functions used for normalization

def get_ranges(data):
	mins = data.min(axis=0)
	ranges = data.max(axis=0)-data.min(axis=0)
	
	return mins, ranges
	
if __name__ == '__main__':
	x = 10
	d = 3

	nrm = lambda d: apply_along_axis(lambda x: linalg.norm(x), 1, d)
	
	data = random.rand(1000,d)
	mins, ranges = get_ranges(data)
        
	#data = apply_along_axis(lambda x: x/linalg.norm(x),1, data) # data normalization
		
	som = MiniSom(x, x, d, sigma=0.6, learning_rate=0.5)
	som.random_weights_init(data)
	
	n = nrm(som.weights_init.reshape(x**2, d)) # norms

	#w = som.weights_init.reshape(x**2, d)*n[:, newaxis]*som.ranges+som.mins
	
	som.train_random(data, 500)
	
	wi, w = som.get_weights()
	psom.plot_data3d(final_som=w, data=data, init_som=wi, nr_nodes=x**2)#, w_init)
	plt.show()
