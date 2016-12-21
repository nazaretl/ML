import numpy as np
import itertools
import scipy
from scipy.misc import logsumexp

''' 
    Factor graph inference implementation
    Able to do inference with and without evidence
    
'''

class GraphicalModelNode(object):
    ''' 
        Graphical model base node
    '''
    
    def __init__(self,name):
        '''
            Initialise a graphical model node
        '''
        self.name = name
        self.neighbours=[]
        self.responses = dict()
        
    def passMessageTo(self,node):
        '''
            Pass a message from the current node to the node: node
            
            This message must be in the log domain!
        '''
        raise NotImplementedError
        
    def receiveMessageFrom(self,node,message):
        '''
            Receive a message from node
            and store it
            
            This expects a message in the log domain
            
            Raises an exception when a message from a non-neighbour is received
        '''
        if node not in self.neighbours:
            raise Exception("Sender is not a neighbour")
        self.responses[node]=message
	if np.isnan(message).sum() > 0:
		raise Exception("I do not like receiving nans")    
            
    def initiateMessagePassing(self,parent):
        '''
            Initiate message passing in the subtree.
            The messages will be passed from the leaves to the root.
            
            To start it, select a root node and set parent=None
            
            It processes the tree by depth first search.
            
            After receiving messages from all children, 
            the node will pass a message to its parent.
            
            If the root node has received all messages from its children,
            it initiates a DFS algorithm to send the messages to all the leaves
            
            After finishing the method, it is possible to
            a) when there is no evidence, to compute the marginal of each variable
            b) when there is evidence in other nodes, to compute the marginal conditional distribution based on the evidence from those nodes.
            
            c) when there is evidence in the current node, the distribution will be wrong :)
        '''
        # Clear the responses if you are the root
        if parent == None:
            self.clearResponses(None)
        # Step one recursion in the three
        # This ensures that you receive all messages
        for n in self.neighbours:
            if n is not parent:
                n.initiateMessagePassing(self)
        
        # Step 2 after receiving the messages, we can send back to the parent and to all our children
        if parent != None:
            #print "Node: ", self.name, " sending message to node ", parent.name 
            self.passMessageTo(parent)
        else:
            #print "Starting downsteam message passing"
            self.initiateMessagePassingDownstream(None)
            
            
    def initiateMessagePassingDownstream(self,parent):
        for n in self.neighbours:
            if n is not parent:
                #print "Node: ", self.name, " sending message downstream to node ", n.name 
                self.passMessageTo(n)
                n.initiateMessagePassingDownstream(self)
                
    def clearResponses(self,parent):
        self.responses.clear()
        for n in self.neighbours:
            if n is not parent:
                #print "Node: ", self.name, " sending message downstream to node ", n.name 
                n.clearResponses(self)
class VariableNode(GraphicalModelNode):
    '''
        A node for a variable
    '''
    def __init__(self,name,ndim,evidence=None,softEvidence=False):
        super(VariableNode, self ).__init__(name)
        self.ndim = ndim
        self.evidence = evidence
        
    def passMessageTo(self,node):
        '''
        
            Bishop Eq. 8.69
            The message from a variable node to a factor node is simply:
            the product of the message it received from the other factor nodes
            
            
            MESSAGE MUST BE IN LOG DOMAIN!
        '''
        if node not in self.neighbours:
            raise Exception("We can't send something to a non-neighbour")
        
        for n in self.neighbours:
            if n is not node and not self.responses.has_key(n):
                raise Exception("We can't send something because we did not receive all the required messages")
        
        
        if self.evidence != None:
                message = np.log(0.000001+np.zeros(self.ndim))
                message[self.evidence]=0.999999
        else:
            message = np.zeros(self.ndim)
            for n in self.neighbours:
                if n is not node:
                    message = message+self.responses[n]
        # send the message
        #print "Message from Var", self.name, " -> ", node.name, ": ", np.exp(message)
        node.receiveMessageFrom(self,message)
        
    def computeMarginal(self):
        for n in self.neighbours:
            if not self.responses.has_key(n):
                raise Exception("We can't scompute the marginal because we did not receive all the required messages")
        all_messages = np.vstack(self.responses.values())
        
        marginal = np.sum(all_messages,axis=0)
        return marginal
        
class FactorNode(GraphicalModelNode):
    ''' 
        A node for a vertex
    '''
    def __init__(self,name,factor,neighbours):
        '''
           - factor resides in the LOG DOMAIN! is the value of the factor
             in case of a root node with N neighbours:
             N-D numpy array of dim M_1,M_2,..,M_N
             
            
        '''
        super(FactorNode, self ).__init__(name)
        self.factor = factor
	if np.isnan(self.factor).sum() > 0:
		raise Exception("Do not give me a factor with nan's in")
        self.neighbours = neighbours
        
        # Check the dimensions of the factors
        if len(factor.shape) != len( self.neighbours):
            raise Exception("There is a problem with the factor dimensions, there are not as many neighbours as dimensions in the factor")
        factor_shape = factor.shape
        for i in range(len( self.neighbours)):
            if factor_shape[i] != self.neighbours[i].ndim:
                raise Exception("There is a problem with the factor dimensions")
        
        ## Add to the neighbours of variable node
        for n in neighbours:
            n.neighbours.append(self)
        
        
        
    def passMessageTo(self,node):
        '''
        
            Bishop Eq. 8.69
            The message from a variable node to a factor node is simply:
            the product of the message it received from the other factor nodes
            
            MESSAGE MUST BE IN LOG DOMAIN!
        '''
        if node not in self.neighbours:
            raise Exception("We can't send something to a non-neighbour")
        
        for n in self.neighbours:
            if (n is not node) and (not self.responses.has_key(n)):
                raise Exception("We can't send something because we did not receive all the required messages")
               
        # Find position in neighbour list of the node to send it to and roll its axis forward in the factor
        target_position = np.nan
        for i in range(len(self.neighbours)):
            if self.neighbours[i] is node:
                target_position = i
                break
        
        ## Based on the target position, roll the axis
        cur_factor  = np.rollaxis(self.factor,target_position,0)
        self.cur_factor = cur_factor
        # Collect all the messages
        needed_messages = []
        for n in self.neighbours:
            if (n is not node) and (self.responses.has_key(n)):
                needed_messages.append(self.responses[n])
        
        # Compute the ranges to sum over
        needed_ranges = [range(needed_message.shape[0]) for needed_message in needed_messages]
        ranges = ranges=[r for r in itertools.product(*needed_ranges)]
        
        # Collect all the submessages we have to sum over
        # This has a shape of (ndim for the target node) X (Variable values to sum over) 
        submessages = np.nan+np.zeros((node.ndim,len(ranges)))
        
        self.ranges = ranges
        # Fill in all dimensions of the message
        for d_m in range(submessages.shape[0]):
            for cur_range_index in range(len(ranges)):
                # The submessage is the factor multiplied with the parts of the submessages
                cur_range = ranges[cur_range_index]
                submessages[d_m][cur_range_index]=cur_factor[d_m][cur_range]
                # Loop over the values in the current range and add them to the submessage (jn the log domain)
                for i in range(len(cur_range)):
                    submessages[d_m,cur_range_index]+=needed_messages[i][cur_range[i]]
        self.submessages = submessages
                
                
        message = np.array([logsumexp(submessages[d_m,:]) for d_m in range(submessages.shape[0])])
        
        #print "Message from factor", self.name, " -> ", node.name, ": ", np.exp(message)
        
        #send the message
        node.receiveMessageFrom(self,message)
    








