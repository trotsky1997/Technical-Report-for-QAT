#!/usr/bin/env python3.7
# coding: utf-8

# # Installation

# In[1]:


import sys
sys.path.append('/home/qq/.local/lib/python3.7/site-packages/aimet_common/x86_64-linux-gnu')
sys.path.append('/home/qq/.local/lib/python3.7/site-packages/aimet_common/x86_64-linux-gnu/aimet_tensor_quantizer-0.0.0-py3.7-linux-x86_64.egg/')

import os
os.environ['LD_LIBRARY_PATH'] +=':/home/qq/.local/lib/python3.7/site-packages/aimet_common/x86_64-linux-gnu'


# In[2]:


import torch
from torchvision import models
from aimet_torch.quantsim import QuantizationSimModel
# m = models.resnet18().cuda()
# sim = QuantizationSimModel(m, dummy_input=torch.rand(1, 3, 224, 224).cuda())
# print(sim)


# In[3]:


import torch
from torchvision import models
from aimet_torch import batch_norm_fold
from aimet_torch import cross_layer_equalization
from aimet_torch import utils
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics as tm
import numpy as np
from efficientnet_pytorch import EfficientNet
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from aimet_torch import bias_correction
from aimet_torch.quantsim import QuantParams,QuantScheme
from aimet_torch.examples.mobilenet import MobileNetV2
from aimet_torch.utils import create_fake_data_loader

pl.seed_everything(42)

# In[104]:

data = sys.argv[1] #["cifar","mnist","kmnist","fashionmnist"]
types = sys.argv[2] #"shufflenet","efficientnet","mobilenet"
byte_wid = 8
names = None

def config_name():
    global names,data,types,byte_wid
    names = f"{data}-{types}-{byte_wid}"


print("training",config_name())

# # train a resnet-18 on cifar-10 with pytorch-lightning

# In[105]:


def gen_data(data):
    
    if data == "cifar":
        dataset = torchvision.datasets.CIFAR10(root='./dataset/',download=True,transform=transforms.ToTensor())
        batch_size = 64*4
        train_data,test_data = torch.utils.data.random_split(dataset,[45000,5000])
        train_data_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size)
        test_data_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size)
    elif data == "kmnist":
        dataset = torchvision.datasets.KMNIST(root='./dataset/',download=True,transform=transforms.ToTensor())
        batch_size = 64*4
        train_data,test_data = torch.utils.data.random_split(dataset,[55000,5000])
        train_data_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size)
        test_data_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size)
    elif data == "mnist":
        dataset = torchvision.datasets.MNIST(root='./dataset/',download=True,transform=transforms.ToTensor())
        batch_size = 64*4
        train_data,test_data = torch.utils.data.random_split(dataset,[55000,5000])
        train_data_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size)
        test_data_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size)
    elif data == "fashionmnist":
        dataset = torchvision.datasets.FashionMNIST(root='./dataset/',download=True,transform=transforms.ToTensor())
        batch_size = 64*4
        train_data,test_data = torch.utils.data.random_split(dataset,[55000,5000])
        train_data_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size)
        test_data_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size)
    return batch_size,train_data_loader,test_data_loader
batch_size,train_data_loader,test_data_loader = gen_data(data)


# In[106]:


class Net(pl.LightningModule):
    def __init__(self,types,pretrained=False,learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.shim = None
        if data in ["mnist","kmnist","fashionmnist"]:
            self.shim = nn.Conv2d(1, 3, 1,bias=False)
            self.shim.weight.data.copy_(torch.ones_like(self.shim.weight).clone().detach())
        if types == "shufflenet":
            self.model = models.shufflenet_v2_x0_5(num_classes=10,pretrained=pretrained)
        elif types == "efficientnet":
            self.model = EfficientNet.from_name('efficientnet-b1',num_classes=10)
        elif types == "mobilenet":
            self.model = models.mobilenet_v2(num_classes=10,pretrained=pretrained)
        elif types == "resnet":
            self.model = models.resnet18(num_classes=10,pretrained=pretrained)
    def forward(self,x):
        if self.shim != None:
            x = self.shim(x)
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        with torch.no_grad():
            acc = accuracy(y_hat,y)
            self.log("train_loss",loss)
            self.log("train_acc",acc)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        with torch.no_grad():
            loss = F.cross_entropy(y_hat, y)
            acc = accuracy(y_hat,y)
            self.log("val_loss",loss)
            self.log("val_acc",acc)


# In[107]:


model = Net(types)
early_stop_callback = EarlyStopping(monitor="train_acc", min_delta=0.001, patience=150, verbose=False, mode="max")
logger = TensorBoardLogger(save_dir=os.getcwd(), version=names+"-train", name="lightning_logs")
trainer = pl.Trainer(precision=16,max_epochs=2,gpus=[0],log_every_n_steps=10,val_check_interval=0.01,logger=logger,auto_lr_find=True,gradient_clip_val=0.5)
try:
    trainer.tune(model,train_data_loader, test_data_loader)
except:
    pass
trainer.fit(model,train_data_loader, test_data_loader)


# In[108]:


# get_ipython().run_line_magic('load_ext', 'tensorboard')
# get_ipython().run_line_magic('reload_ext', 'tensorboard')
# get_ipython().run_line_magic('tensorboard', '--logdir ./lightning_logs')


# # quantize the model

# In[109]:

a = ["pre_quant","tricked","quantized","finetuned"]
a_i = 0
def acc_record(item,acc):
    with open("./acc.txt","a") as f:
        f.write(f"\n{names}\t{item}\t{acc}")       
        
@torch.no_grad()
def evaluate_model(model: torch.nn.Module, eval_iterations: int, use_cuda: bool = False) -> float:
    """
    This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's eval function does not
    match this signature, please create a simple wrapper.

    Note: Honoring the number of iterations is not absolutely necessary.
    However if all evaluations run over an entire epoch of validation data,
    the runtime for AIMET compression will obviously be higher.

    :param model: Model to evaluate
    :param eval_iterations: Number of iterations to use for evaluation.
            None for entire epoch.
    :param use_cuda: If true, evaluate using gpu acceleration
    :return: single float number (accuracy) representing model's performance
    """
    global a_i
    metric = tm.Accuracy().cuda()
    for x,y in test_data_loader:
        y_hat = model(x.cuda())
        acc = metric(y_hat,y.cuda())
    acc = metric.compute()
    acc_record(a[a_i % len(a)],acc.item())
    a_i += 1
    # model.log("quant_acc",acc)
    return acc.item()



def quantize_model(model):

    model = model.eval()
    if data in ["mnist","kmnist","fashionmnist"]:
        input_shape = (1, 1, 32, 32)
    else:
        input_shape = (1, 3, 32, 32)
    # Fold batchnorm layers
    folded_pairs = batch_norm_fold.fold_all_batch_norms(model, input_shape)
    bn_dict = {}
    for conv_bn in folded_pairs:
        bn_dict[conv_bn[0]] = conv_bn[1]

    # Replace any ReLU6 layers with ReLU
    utils.replace_modules_of_type1_with_type2(model, torch.nn.ReLU6, torch.nn.ReLU)

    # Perform cross-layer scaling on applicable layer sets
    cls_set_info_list = cross_layer_equalization.CrossLayerScaling.scale_model(model, input_shape)

    # Perform high-bias fold
    cross_layer_equalization.HighBiasFold.bias_fold(cls_set_info_list, bn_dict)


    # In[110]:


    # Bias Correction related imports
    params = QuantParams(weight_bw=byte_wid, act_bw=byte_wid, round_mode="nearest", quant_scheme=QuantScheme.post_training_tf_enhanced)

    # Perform Bias Correction
    bias_correction.correct_bias(model.to(device="cuda"), params, num_quant_samples=2000,
                                    data_loader=train_data_loader, num_bias_correct_samples=2000)
    evaluate_model(model.cuda(),1)


    # In[114]:


    sim = QuantizationSimModel(model, default_output_bw=byte_wid, default_param_bw=byte_wid, dummy_input=torch.rand(*input_shape).cuda(),config_file='/home/qq/.local/lib/python3.7/site-packages/aimet_common/quantsim_config/default_config.json')
    sim.compute_encodings(forward_pass_callback=evaluate_model, forward_pass_callback_args=5)
    evaluate_model(model.cuda(),1)

    # In[115]:


    logger = TensorBoardLogger(save_dir=os.getcwd(), version=names+"-quant", name="lightning_logs")
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.001, patience=50, verbose=False, mode="max")
    trainer = pl.Trainer(precision=16,max_epochs=1,gpus=[0],log_every_n_steps=10,val_check_interval=0.01,logger=logger,auto_lr_find=True)
    try:
        trainer.tune(model,train_data_loader, test_data_loader)
    except:
        pass
    trainer.fit(model,train_data_loader, test_data_loader)


    # In[49]:


    evaluate_model(sim.model.cuda(),1)


    # # Export to onnx and import to  SNPE

    # In[63]:


    if types == "efficientnet":
        model.model.set_swish(memory_efficient=False)
        model.eval()
    sim.export(path='./export', filename_prefix=names, dummy_input=torch.rand(*input_shape))


def save_model():
    torch.save(model.state_dict(), f"./export/{data}-{types}.pth")
def load_model():
    model = Net(types)
    model.load_state_dict(torch.load(f"./export/{data}-{types}.pth"))
    model.eval()
    return model

save_model()
for byte_wid in [8,6,4]:
    config_name()
    quantize_model(load_model())