import matplotlib.pyplot
import pylab
import numpy
import scipy.special
import imageio

class neuralNetwork:  # 定义一个类
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):  # 初始化神经网络
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
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

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes,output_nodes, learning_rate)  # 创建一个对象

training_data_file = open("D:\study\MINIST\mnist_train.csv", 'r')  # 创建此文件的句柄
training_data_list = training_data_file.readlines()  # 创建一个列表，列表的一项表示文件中的一行字符串
training_data_file.close()

for i in training_data_list:
    all_values = i.split(',')
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99+0.01)
    targets = numpy.zeros(output_nodes)+0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)


img_array = imageio.imread("D:\study\Set14\手写数字28^2\Handwritten3.png", as_gray=True)
img_data = 255.0-img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99)+0.01
outputs = n.query(img_data)
answer = numpy.argmax(outputs)
print(answer)