from __future__ import division
from copy import copy
from numpy import sqrt,sqrt,array,unravel_index,nditer,linalg,random,subtract,power,exp,pi,zeros,arange,outer,meshgrid, apply_along_axis, newaxis, genfromtxt
from collections import defaultdict
 
# TODO
import pylab as plt
import plot_som as psom
import pdb

"""
    Minimalistic implementation of the Self Organizing Maps (SOM).

    Giuseppe Vettigli 2013.
"""
class Normalizer:

	def __init__(self, data):
		self.data = data
		self.n, self.d = data.shape
		self.mins = data.min(axis=0)
		self.ranges = data.max(axis=0)-data.min(axis=0)
		self.means = data.mean(axis=0)
		self.stds = data.std(axis=0)
		self.forbnorms = apply_along_axis(lambda x: linalg.norm(x),1,data) 
	
	def none(self):
		return self.data
		
	def zscores(self):
		return (self.data-self.means)/self.stds
		
	def minmax(self):
		return (self.data-self.mins)/self.ranges
		
	def forbnorm(self):
		return self.data/self.forbnorms[:, newaxis]
		

class MiniSom:
    def __init__(self,x,y,input_len, data,sigma=1.0,learning_rate=0.5, norm='none'):
        """
            Initializes a Self Organizing Maps.
            x,y - dimensions of the SOM
            input_len - number of the elements of the vectors in input
            data - values in a matrix form
            sigma - spread of the neighborhood function (Gaussian)
            (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
            learning_rate - initial learning rate
            (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
            norm - use normalization of data and weights
        """
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.data = data
        self.activation_map = zeros((x,y))
        self.neigx = arange(x)
        self.neigy = arange(y) # used to evaluate the neighborhood function
        self.neighborhood = self.gaussian
        
        # weights are initialized in data ranges 
        self.norm = Normalizer(data)
        self.data = getattr(self.norm, norm)()
        self.init_weights(x,y, input_len)
       
        #self.weights = array([v/linalg.norm(v) for v in self.weights])
        self.weights_init = copy(self.weights)

    def _activate(self,x):
        """ Updates matrix activation_map, in this matrix the element i,j is the response of the neuron i,j to x """
        s = subtract(x,self.weights) # x - w
        #pdb.set_trace()
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.activation_map[it.multi_index] = linalg.norm(s[it.multi_index]) # || x - w ||
            #self.activation_map[it.multi_index] = s[it.multi_index]
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
       
    def init_weights(self, x, y, input_len):
		"""
		Initialize weights values in the range of data
		"""
		rw_min = self.data.min(axis=0)
		rw_max = self.data.max(axis=0)
		self.weights = random.rand(x, y, input_len)*(rw_max-rw_min)+rw_min
        
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
            it.iternext()

    def quantization(self,data):
        """ Assigns a code book (weights vector of the winning neuron) to each sample in data. """
        q = zeros(data.shape)
        for i,x in enumerate(data):
            q[i] = self.weights[self.winner(x)]
        return q


    def random_weights_init(self):
        """ Initializes the weights of the SOM picking random samples from data """
        
        data = self.data
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.weights[it.multi_index] = data[int(random.rand()*len(data)-1)]
            it.iternext()
            
	self.weights_init = copy(self.weights)

    def train_random(self,num_iteration):        
        """ Trains the SOM picking samples at random from data """
        self._init_T(num_iteration)        
        data = self.data
             
        # start training on normalized weights
        for iteration in range(num_iteration):
            rand_i = int(round(random.rand()*len(data)-1)) # pick a random sample          
            self.update(data[rand_i],self.winner(data[rand_i]),iteration)          
      

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
    
    # TODO: refactor this method
    def get_weights(self):
    	"""
    		Flattens the two first dimensions of weights and returns 2D initial
    		and trained weights
    	"""
    	d1, d2, d3 = self.weights.shape
    	winit = self.weights_init.reshape(d1*d2, d3)
    	w = self.weights.reshape(d1*d2, d3)
    	
    	return winit, w
	
