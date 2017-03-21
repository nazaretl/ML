import numpy

# -----------------------------------------------------------------------------
# modules.Module
#    Abstract class that defines a neural network module (e.g. layer).
# -----------------------------------------------------------------------------
class Module:
    
    def __init__(self):    pass
    def update(self,lr):   pass
    def forward(self,X):   pass
    def backward(self,DY): pass

# -----------------------------------------------------------------------------
# modules.Linear
#    Linear layer mapping neuron activations at a given layer to the
#    preactivations of the next layer.
# -----------------------------------------------------------------------------
class Linear(Module):
    
    def __init__(self,m,n,seed):
        self.m = m
        self.n = n
        self.W = numpy.random.mtrand.RandomState(seed).normal(0,m**(-.5),[m,n])
        self.B = numpy.zeros([n])
    
    def forward(self,X):
        self.X = X
        return numpy.dot(X,self.W)+self.B
    
    def backward(self,DY):
        self.DW = numpy.dot(self.X.T,DY)
        self.DB = DY.sum(axis=0)
        return numpy.dot(DY,self.W.T)
    
    def update(self,lr):
        self.W -= lr*self.DW
        self.B -= lr*self.DB

# -----------------------------------------------------------------------------
# modules.Identity
#    Layer that does nothing except propagating the input and backpropagating
#    the gradient. It can also represent a linear neuron activation function.
# -----------------------------------------------------------------------------
class Identity(Module):

    def forward(self,X): return X

    def backward(self,DY): return DY

# -----------------------------------------------------------------------------
# modules.Sequential
#    Container module that propagates the input and backpropagates the gradient
#    in a sequence of neural network layers.
# ----------------------------------------------------------------------------- 
class Sequential(Module):
    
    def __init__(self,modules):
        self.modules = modules
    
    def forward(self,X):
        for m in self.modules: X = m.forward(X)
        return X
    
    def backward(self,DY):
        for m in self.modules[::-1]: DY = m.backward(DY)
	return DY
    
    def update(self,lr):
        for m in self.modules: m.update(lr)

