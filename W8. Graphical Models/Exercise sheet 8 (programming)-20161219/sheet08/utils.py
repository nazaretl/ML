import scipy,scipy.io

# -------------------------------------------------------------------
# getData:
#
# Arguments:
#   None
#
# Return:
#   The training data (inputs X, labels T, patch subclasses Z)
# -------------------------------------------------------------------

def getData():
	data = scipy.io.loadmat('mnist.mat')
	X = data['X'][:34000].astype('float32')*1.0   # MNIST digits
	T = data['T'][:34000,0].astype('float32')*1.0 # labels
	Z = data['Z'][:34000].astype('float32')*1.0   # patch subclasses
	return X,T,Z


# -------------------------------------------------------------------
# getAccuracy:
#
# Measure the accuracy of the predictor function given as argument
#
# Arguments:
# - predict: pointer to a function that return the class from an input
# - debug:   test on only 100 samples
#
# Return:
# - the classification accuracy
# -------------------------------------------------------------------
def getAccuracy(predict,debug=False):

	# get test data
	data = scipy.io.loadmat('mnist.mat')
	X = data['X'][34000:].astype('float32')*1.0   # MNIST digits
	T = data['T'][34000:,0].astype('float32')*1.0 # labels

	# iterate over test data
	nbcorrect = 0.0
	nbiter = 100 if debug else len(X)

	for k in range(nbiter):
		nbcorrect += (T[k]==predict(X[k]))
		if k%25==0: print('it: %03d  acc: %.3f'%(k,nbcorrect/(k+1)))

	# return accuracy
	return nbcorrect / nbiter

