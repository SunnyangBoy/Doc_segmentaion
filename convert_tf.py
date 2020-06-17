from torch.autograd import Variable
import torch
from torch import nn
import numpy as np
import onnx
from onnx_tf.backend import prepare
from detectron2 import model_zoo
import torchvision.transforms as transforms

# Load the trained model from file
my_model = model_zoo.get("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml", trained=False)
#my_model = torch.load('/home/ubuntu/cs/publaynet/output/model_0021999.pth')

if torch.cuda.is_available():
    my_model.cuda()
else:
    my_model.cpu()

my_model.eval()

'''
for key in my_model['model'].keys():
    #print(my_model['model'].keys())# ['backbone.bottom_up.stem.conv1.weight'].size())
    print(key)
    print(my_model['model'][key].size())
'''

# Export the trained model to ONNX
input_list = []
toTensor = transforms.Compose([transforms.ToTensor()])
#dummy_input = Variable(torch.randn(1, 1, 1)) # nchw one black and white 28 x 28 picture will be the input to the model
dummy_input = np.ones((1, 1, 1), dtype=np.uint8)
dummy_input = toTensor(dummy_input)
input_dict = {}
input_dict['image'] = dummy_input
input_list.append(input_dict)

torch.onnx.export(my_model, input_list, "output/pulaynet.onnx")

# Load the ONNX file
tmp_model = onnx.load('output/publaynet.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(tmp_model)

# Input nodes to the model
print('inputs:', tf_rep.inputs)

# Output nodes from the model
print('outputs:', tf_rep.outputs)

# All nodes in the model
print('tensor_dict:')
print(tf_rep.tensor_dict)

tf_rep.export_graph('output/publaynet.pb')
