import sklearn,sklearn.svm,numpy
import matplotlib
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# utils.LinearSVC
#    Wrapper for Scikit-learn LinearSVC with useful methods for the programming
#    assignment
# -----------------------------------------------------------------------------
class LinearSVC:

	def __init__(self,C=0):
		self.C = C

	def optimize(self,X,Y):
		self.svc = sklearn.svm.LinearSVC(C=self.C,dual=False)
		self.svc.fit(X,Y)
		self.w  = self.svc.coef_.flatten()
		self.b  = self.svc.intercept_
		self.s = numpy.dot(X,self.w)+self.b
		self.xi = numpy.maximum(0,1-self.s*Y)
		self.Y  = Y

		return (self.w**2).sum() / self.C + (self.xi**2).sum()

	def predictions(self):
		return numpy.sign(self.s)

	def gradient(self):
		return -2*numpy.outer(self.xi*self.Y,self.w)

# -----------------------------------------------------------------------------
# utils.visualize()
#    Plots several visualizations of what the neural network has learned
# -----------------------------------------------------------------------------
def visualize(X,DX,F,Y):

	fig = plt.figure(figsize=(15,5))

	# Plot the labeled dataset
	p = fig.add_subplot(131)
	p.set_title('labels')
	for x,y in zip(X,Y):
		color = 'red' if y==-1 else 'blue'
		p.plot(x[0],x[1],'.',color=color,ms=5)
		p.axis([-2,2,-2,2])

	# Plot the labeled dataset as predicted by the SVC
	p = fig.add_subplot(132)
	p.set_title('predictions')
	for x,f in zip(X,F):
		color = 'red' if f==-1 else 'blue'
		p.plot(x[0],x[1],'.',color=color,ms=5)
		p.axis([-2,2,-2,2])

	# Plot the gradient of the SVC objective in the input space
	p = fig.add_subplot(133)
	p.set_title('gradients')
	for x,dx,f in zip(X,DX,F):
		color = 'black'
		p.plot(x[0],x[1],'.',color=color,ms=2)
		p.axes.arrow(x[0], x[1], 0.1*dx[0], 0.1*dx[1], color=color, head_width=0.05, head_length=0.1)
		p.axis([-2,2,-2,2])

