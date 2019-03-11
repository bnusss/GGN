# 导入必要的包
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import argparse
import time
import copy
from utils.model import *
# from models import *
from tools import *
import pickle
import os
import datetime
torch.set_default_tensor_type('torch.DoubleTensor')






# Training settings
parser = argparse.ArgumentParser(description='Boolean Network')
parser.add_argument('--epoch_num', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--experiments', type=int, default=10,
                    help='number of experiments (default: 10)')
parser.add_argument('--dynamics-steps', type=int, default=20,
                    help='number of steps for dynamics learning (default: 20)')
parser.add_argument('--reconstruct-steps', type=int, default=10,
                    help='number of steps for reconstruction (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=2050,
                    help='random seed (default: 2050)')
parser.add_argument('--prediction-steps', type=int, default=1,
                    help='prediction steps in data (default: 10)')
parser.add_argument('--dyn-type', type=str, default='table',
                    help='different kind of dynamics(table and prob)(default:table)')

args = parser.parse_args()





# start training
Epoch_Num = args.epoch_num
Batch_Size = args.batch_size
# times to train dyn learner and network generator in one epoch
Dyn_Steps = args.dynamics_steps
Net_Steps = args.reconstruct_steps
# type of dynsmics
Dyn_Type = args.dyn_type

print("Data Loading...")
# get data
train_data_loader,valid_data_loader,test_data_loader,edges_train = load_bn_ggn(batch_size=Batch_Size, dyn_type = Dyn_Type)
print('train set batch num : '+str(len(train_data_loader)))
print('val set batch num : '+str(len(valid_data_loader)))
print('test set batch num : '+str(len(test_data_loader)))
print('batch_size:'+str(Batch_Size))



# train dyn trainer
def train_batch_dyn(optimizer,dyn_learner,adj,data_train,data_target,loss_fn):
    loss = 0
    #optimizer
    optimizer.zero_grad()
    
    num_nodes = adj.size(0)
    #adj mat
    adj = adj.unsqueeze(0)
    adj = adj.repeat(data_train.size()[0],1,1)
    adj = adj.cuda() if use_cuda else adj
    

    # get result caculated by neural network
    output = dyn_learner(data_train,adj)
    output = output.permute(0,2,1)
    
    # caculate the difference
    data_target = data_target.long()
    accus = cacu_accu(output,data_target)
    loss = loss_fn(output,data_target)
    loss.backward()
    
    # optimizer for dyn learner 
    optimizer.step()
    
    return loss,accus


# data format:batch_size, num_nodes, time_steps(10), dimension(4)
def train_batch_generator(optimizer_network,gumbel_generator,dyn_learner,data_train,data_target,loss_fn):
    optimizer_network.zero_grad()

    out_matrix = gumbel_generator.sample()
    out_matrix = out_matrix.unsqueeze(0)
    out_matrix = out_matrix.repeat(data_train.size()[0], 1, 1)
    gumbel_generator.drop_temperature()
    
   

    # get result caculated by neural network
    output = dyn_learner(data_train,out_matrix)
    output = output.permute(0,2,1)
    
    # caculate the difference
    data_target = data_target.long()
    loss = loss_fn(output,data_target)
    loss.backward()
    optimizer_network.step()
    return loss,out_matrix



# ------------------------------------------
# dyn learner and gumbel generator
num_nodes = edges_train.shape[0]
#dynamics learner and optimizer
dyn_learner = GumbelGraphNetworkClf(2)
dyn_learner = dyn_learner.double()
if use_cuda:
    dyn_learner = dyn_learner.cuda()
optimizer_dyn = optim.Adam(dyn_learner.parameters(),lr = 0.001)

#generate network structure
gumbel_generator = Gumbel_Generator(sz = num_nodes,temp = 10,temp_drop_frac = 0.9999)
gumbel_generator = gumbel_generator.double()
if use_cuda:
    gumbel_generator = gumbel_generator.cuda()
optimizer_network = optim.Adam(gumbel_generator.parameters(),lr = 0.1)


#loss function
loss_fn = torch.nn.NLLLoss()

#standard adj mat
standard_adj = torch.Tensor(edges_train)


# ------------------------------------------
# save log and model
# create save folder
now = datetime.datetime.now()
timestamp = now.isoformat()
save_folder = '{}/exp{}/'.format('./logs', timestamp)
os.mkdir(save_folder)

# saving path
generator_file = os.path.join(save_folder, 'generator.pt')
dyn_file = os.path.join(save_folder, 'dyn_learner.pt')
test_data_loader_address = os.path.join(save_folder, 'test_loader.pickle')
standard_adj_address = os.path.join(save_folder, 'standard_adj.pickle')

# saving test loader
with open(test_data_loader_address,'wb') as f:
    pickle.dump(test_data_loader,f)
# saving standard adj
with open(standard_adj_address,'wb') as f:
    pickle.dump(standard_adj,f)


# recorder
loss = 0
dyn_losses = []
net_losses = []
accu_record = []
err_nets = []


# start training
for epoch in range(Epoch_Num):
    print('epoch running:'+str(epoch)+' / '+str(Epoch_Num))

    print('use gumbel')
    adj = gumbel_generator.sample(hard=True)

    # 先训练dynamics
    losses = []
    accuracies = []
    losses_in_gumbel = []
    # dyn_learner.train()
    print("\n***************Dyn Training******************")
    for i in range(Dyn_Steps):
        step_accu = []
        for batch_idx, (data_train,data_target) in enumerate(train_data_loader):
            if use_cuda:
                data_train = data_train.cuda()
                data_target = data_target.cuda()
            loss,accu = train_batch_dyn(optimizer_dyn,dyn_learner,adj,data_train,data_target,loss_fn)
            record_loss = loss.data.tolist()
            losses.append(record_loss)
            accuracies.append(1-accu)
        print("\n")
        print('epoch: '+str(epoch)+' dyn training '+str(i))
        print('loss: '+str(record_loss))
        print('accuracy：'+str(accu))
    print("\n***************Gumbel Training******************")
    for i in range(Net_Steps):
        step_loss = 0
        step_accu = []
        for batch_idx, (data_train,data_target) in enumerate(train_data_loader):
            if use_cuda:
                data_train = data_train.double().cuda()
                data_target = data_target.cuda()
            loss,out_matrix = train_batch_generator(optimizer_network,gumbel_generator,dyn_learner,data_train,data_target,loss_fn)
            record_loss = loss.data.tolist()
            losses_in_gumbel.append(record_loss)
        err_net = constructor_evaluator_withdiag(gumbel_generator, 500, standard_adj)
        print("\n")
        print('epoch: '+str(epoch)+' net training '+str(i))
        print('loss: '+str(record_loss))
        print("err_net:"+str(err_net))
        err_nets.append(err_net)
    # validate
    # dyn_learner.eval()
    val_losses = []
    print("validating")
    for batch_idx, (data_val,data_target) in enumerate(valid_data_loader):
        if use_cuda:
            data_val = data_val.double().cuda()
            data_target = data_target.cuda()
        val_loss = get_valid_loss(gumbel_generator,dyn_learner,data_val,data_target,loss_fn)
        val_losses.append(val_loss)
    if epoch == 0:
        bset_val_loss = np.mean(np.array(val_losses))
        print("best model so far , saving...")
        torch.save(gumbel_generator.state_dict(), generator_file)
        torch.save(dyn_learner.state_dict(), dyn_file)
    else:
        if np.mean(np.array(val_losses)) < bset_val_loss:
            print("best model so far , saving...")
            torch.save(gumbel_generator.state_dict(), generator_file)
            torch.save(dyn_learner.state_dict(), dyn_file)

# draw the train loss, val loss and err_net



# start testing
print("testing")
# load best model
gumbel_generator.load_state_dict(torch.load(generator_file))
dyn_learner.load_state_dict(torch.load(dyn_file))
# run best model on test set
accu_all = []
for batch_idx, (data_test,data_target) in enumerate(test_data_loader):
    if use_cuda:
        data_test = data_test.double().cuda()
        data_target = data_target.cuda()
    accu = get_test_accu(gumbel_generator,dyn_learner,data_test,data_target)
    accu_all.append(accu)
print('accuracy:'+str(np.mean(np.array(accu_all))))


err_net = constructor_evaluator_withdiag(gumbel_generator, 500, standard_adj)
print('err_net:'+str(err_net))

out_matrix = gumbel_generator.sample(hard = True)
(tpr,fpr) = tpr_fpr(out_matrix.cpu(),standard_adj)
print('tpr:'+str(tpr))
print('fpr:'+str(fpr))








