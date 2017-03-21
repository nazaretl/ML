import numpy

# -----------------------------------------------------------------------------
# datasets.toy()
#    Returns a simple dataset that consists of two clusters of data points, one
#    for each class. Clusters are linearly nonseparable and nonlinearly
#    separable.
# -----------------------------------------------------------------------------
def toy(test=False):

	N = 200

	rstate = numpy.random.mtrand.RandomState(12345 if test else 23456)

	U1 = rstate.uniform(-0.3,numpy.pi+1.1,[N,1])
	U2 = rstate.uniform(-0.3,numpy.pi+1.1,[N,1])+numpy.pi

	X1 = rstate.normal(0,0.1,[N,2]) + numpy.concatenate([-0.5+numpy.cos(U1),numpy.sin(U1)],axis=1)
	X2 = rstate.normal(0,0.1,[N,2]) + numpy.concatenate([ 0.5+numpy.cos(U2),numpy.sin(U2)],axis=1)

	Y1 =  numpy.ones([N])
	Y2 = -numpy.ones([N])

	X = numpy.concatenate([X1,X2])
	Y = numpy.concatenate([Y1,Y2])

	return X,Y

