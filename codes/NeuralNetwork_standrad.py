import numpy
import scipy.special
class neuralNetwork:    #定义一个类
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):   #初始化神经网络
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learningrate
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))   #正态分布中心、正态分布宽度、输出值维度
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function=lambda x:scipy.special.expit(x)
        
            
    def train(self,inputs_list,targets_list):    #训练神经网络
        inputs=numpy.array(inputs_list,ndmin=2).T  
        targets=numpy.array(targets_list,ndmin=2).T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outpus=self.activation_function(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outpus)
        final_outputs = self.activation_function(final_inputs) 
        output_errors=targets-final_outputs      #此行开始调整权重
        hidden_errors=numpy.dot(self.who.T,output_errors)
        self.who+=self.lr*numpy.dot((output_errors *final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outpus))
        self.wih+=self.lr*numpy.dot((hidden_errors *hidden_outpus*(1.0-hidden_outpus)),numpy.transpose(inputs))
        

    def query(self,inputs_list):    #查询神经网络
        inputs=numpy.array(inputs_list,ndmin=2).T   
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outpus=self.activation_function(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outpus)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

input_nodes=3
hidden_nodes=3
output_nodes=3
learning_rate=0.3

n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)  #创建一个对象
