import torch
import numpy as np
import pickle
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

# if use cuda
use_cuda = torch.cuda.is_available()

def cacu_accu_new_loss(output,target):
    position_num = output.size(0) * output.size(1)
    right_num = 0
    for i in range(output.size(0)):
        for j in range(output.size(1)):
            pre_pos_val = output[i][j][0].tolist()
            real_pos_val = target[i][j][0].tolist()
            if pre_pos_val < 0.5 and real_pos_val == 0:
                right_num += 1
            elif pre_pos_val >= 0.5 and real_pos_val == 1:
                right_num += 1
    return right_num / position_num




# caculate diff between gumbel adj and standard adj
def cacu_mat(adj,standard_adj):
    cuda0 = torch.device('cuda:0')
    adj = adj.to(cuda0,torch.int32)
    stan = standard_adj.to(cuda0,torch.int32)
    return torch.sum(torch.abs(adj-stan)).tolist()

def get_offdiag(sz):
    ## 返回一个大小为sz的下对角线矩阵
    offdiag = torch.ones(sz, sz)
    for i in range(sz):
        offdiag[i, i] = 0
    if use_cuda:
        offdiag = offdiag.cuda()
    return offdiag   


def constructor_evaluator(gumbel_generator, tests, obj_matrix, sz):
    obj_matrix = obj_matrix.cuda()
    errs= []
    for t in range(tests):
        out_matrix = gumbel_generator.sample()
        out_matrix_c = 1.0*(torch.sign(out_matrix-1/2)+1)/2
        err = torch.sum(torch.abs(out_matrix_c * get_offdiag(sz) - obj_matrix * get_offdiag(sz)))
        err = err.cpu() if use_cuda else err
        errs.append(err.data.numpy())
        
    err_net = np.mean(errs)
    return err_net

def constructor_evaluator_withdiag(gumbel_generator, tests, obj_matrix):
    obj_matrix = obj_matrix.cuda()
    sz = obj_matrix.size(0)
    errs= []
    for t in range(tests):
        out_matrix = gumbel_generator.sample()
        err = torch.sum(torch.abs(out_matrix - obj_matrix))
        err = err.cpu() if use_cuda else err
        errs.append(err.data.numpy())
        
    err_net = np.mean(errs)
    return err_net

def tpr_fpr(out,adj):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(out.size(0)):
        for j in range(out.size(0)):
            if adj[i][j] == 1:
                # positive
                if out[i][j] == 1:
                    # true positive
                    tp += 1
                else:
                    # false positive
                    fn += 1
            else:
                # negative
                if out[i][j] == 1:
                    # true negative
                    fp += 1
                else:
                    # false negative
                    tn += 1
    # tpr = tp /  (tp + fp)

    tpr = float(tp) / (tp + fn)
    fpr = float(fp) / (fp + tn)
    return(tpr , fpr)



def crop_data(data):
    data_crop_arr = []
    INPUT_STEPS = 1
    PREDICT_STEPS = 9
    SEQUENCE_LEN = INPUT_STEPS + PREDICT_STEPS
    sequence_num_per_frame = data.shape[2] // (INPUT_STEPS+PREDICT_STEPS)

    for i in range(sequence_num_per_frame):
        data_crop = data[:,:,SEQUENCE_LEN*i:SEQUENCE_LEN*(i+1),:]
        data_crop_arr.append(data_crop)

    data_crop_arr = np.concatenate(data_crop_arr, axis=0)
    return data_crop_arr

def weighted(data, v_x_ratio):
    a = data.shape
    b = (a[0],a[1],a[2],1)
    c = np.ones(b)
    d = v_x_ratio * np.ones(b)
    e = np.concatenate((c,d), axis=3)
    weighted_data = e * data
    return weighted_data



def load_bn_ggn(batch_size = 128,dyn_type='table'):

    # address
    series_address = './data/bn/mark-14771-adjmat.pickle'
    adj_address = './data/bn/mark-14771-series.pickle'

    # 5/7 for training, 1/7 for validation and 1/7 for test
    use_state = 1024


    # adj mat
    with open(series_address,'rb') as f:
        edges = pickle.load(f,encoding='latin1')
    # time series data
    with open(adj_address,'rb') as f:
        info_train = pickle.load(f,encoding='latin1')

    # if too large...
    if info_train.shape[0] > 100000:
        info_train = info_train[:100000]
    info_train_list = info_train.tolist()
    has_loaded = []
    i = 0
    while len(has_loaded) < use_state:
        # if dyn type == table then we have to make sure that each state we load is different
        if dyn_type == 'table':
            # print(i)
            if info_train_list[i] not in has_loaded:
                has_loaded.append(info_train_list[i])
            i = i+2
        elif dyn_type == 'prob':
            # then we dont require they are different
            has_loaded.append(info_train_list[i])
            i = i+2
        else:
            print('Error in loading')
            debug()
    info_train = info_train[:i+2]

    # 即将用到的数据，先填充为全0
    data_x = np.zeros((int(info_train.shape[0]/2),info_train.shape[1],2))
    data_y = np.zeros((int(info_train.shape[0]/2),info_train.shape[1]))


    # random permutation
    indices = np.random.permutation(data_x.shape[0])
    data_x_temp = [data_x[i] for i in indices]
    data_y_temp = [data_y[i] for i in indices]
    data_x = np.array(data_x_temp)
    data_y = np.array(data_y_temp)


    # 预处理成分类任务常用的数据格式
    for i in range(int(info_train.shape[0] / 2)):
        for j in range(info_train.shape[1]):
            if info_train[2*i][j][0] == 0.:
                data_x[i][j] = [1,0]
            else:
                data_x[i][j] = [0,1]
            if info_train[2*i+1][j][0] == 0.:
                data_y[i][j] = 0
            else:
                data_y[i][j] = 1

    # random permutation
    indices = np.random.permutation(data_x.shape[0])
    data_x_temp = [data_x[i] for i in indices]
    data_y_temp = [data_y[i] for i in indices]
    data_x = np.array(data_x_temp)
    data_y = np.array(data_y_temp)

    # seperate train set,val set and test set
    # train / val / test == 5 / 1 / 1 
    train_len = int(data_x.shape[0] * 5 / 7)
    val_len = int(data_x.shape[0] * 6 / 7)
    # seperate
    feat_train = data_x[:train_len]
    target_train = data_y[:train_len]
    feat_val = data_x[train_len:val_len]
    target_val = data_y[train_len:val_len]
    feat_test = data_x[val_len:]
    target_test = data_y[val_len:]

    # change to torch.tensor
    feat_train = torch.DoubleTensor(feat_train)
    feat_val = torch.DoubleTensor(feat_val)
    feat_test = torch.DoubleTensor(feat_test)
    target_train = torch.LongTensor(target_train)
    target_val = torch.LongTensor(target_val)
    target_test = torch.LongTensor(target_test)

    # put into tensor dataset
    train_data = TensorDataset(feat_train, target_train)
    val_data = TensorDataset(feat_val, target_val)
    test_data = TensorDataset(feat_test,target_test)

    # put into dataloader
    train_data_loader = DataLoader(train_data, batch_size=batch_size,drop_last=True)
    valid_data_loader = DataLoader(val_data, batch_size=batch_size,drop_last=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size,drop_last=True)


    return train_data_loader,valid_data_loader,test_data_loader,edges

def load_cml_ggn(batch_size = 128):
    data_path = './data/cml/data_lambd3.6_coupl0.2_node10.pickle'
    with open(data_path, 'rb') as f:
        object_matrix, train_data, val_data, test_data = pickle.load(f) # (samples, nodes, timesteps, 1)
    print('\nMatrix dimension: %s Train data size: %s Val data size: %s Test data size: %s'
          % (object_matrix.shape, train_data.shape, val_data.shape, test_data.shape))


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    return train_loader,val_loader,test_loader,object_matrix



def load_kuramoto_ggn(batch_size = 128):
    k_over_kc = 1.1

    train_fp = './data/kuramoto/ERtrain-5000sample-1.1kc10node-100timestep-2vec.npy'
    val_fp = './data/kuramoto/ERval-1000sample-1.1kc10node-100timestep-2vec.npy'
    test_fp = './data/kuramoto/ERtest-1000sample-1.1kc10node-100timestep-2vec.npy'
    adj_fp = './data/kuramoto/ERadj-10sample-1.1kc10node-100timestep-2vec.npy'

    object_matrix = np.load(adj_fp)
    train_data = weighted(np.load(train_fp)[:5000,:,:,:], 0.5/k_over_kc)
    val_data = weighted(np.load(val_fp)[:1000,:,:,:], 0.5/k_over_kc)
    test_data = weighted(np.load(test_fp)[:1000,:,:,:], 0.5/k_over_kc)
    num_nodes = object_matrix.shape[0]
    data_max = train_data.max()

    train_dataset = crop_data(train_data)
    val_dataset = crop_data(val_data)
    test_dataset = crop_data(test_data)

    train_dataset = np.asarray(train_dataset, dtype=np.float32)
    val_dataset = np.asarray(val_dataset, dtype=np.float32)
    test_dataset = np.asarray(test_dataset, dtype=np.float32)



    print('\nMatrix dimension: %s Train data size: %s Val data size: %s Test data size: %s'
          % (object_matrix.shape, train_dataset.shape, val_dataset.shape, test_dataset.shape))
    if use_cuda:
        object_matrix = torch.from_numpy(object_matrix).float().cuda()
        train_dataset = torch.from_numpy(train_dataset).cuda()
        val_dataset = torch.from_numpy(val_dataset).cuda()
        test_dataset = torch.from_numpy(test_dataset).cuda()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, val_loader, test_loader, object_matrix


def train_batch_dyn_bn(train_data_loader,optimizer_dyn,dyn_learner,adj,loss_fn,simulation_type,prediction_steps):
    step_accu = []
    for batch_idx, (data_train,data_target) in enumerate(train_data_loader):
        if use_cuda:
            data_train = data_train.cuda()
            data_target = data_target.cuda()
        loss,accu = train_dyn_learner_bn(optimizer_dyn,dyn_learner,adj,data_train,data_target,loss_fn)
        #loss = train_one_batch_new_loss(optimizer_dyn,dyn_learner,adj,data_train,data_target,loss_fn1)
        record_loss = loss.data.tolist()
        losses.append(record_loss)
        if args.simulation_type == 'bn':
            accuracies.append(1-accu)
    print("\n")
    print('epoch: '+str(epoch)+' dyn training '+str(i))
    print('loss: '+str(record_loss))
    print('accuracy：'+str(accu))


def tran_batch_dyn(train_data_loader,optimizer_network,gumbel_generator,simulation_type,prediction_steps,dyn_learner,loss_fn):
    loss_records = []
    for step in range(1, args.dynamics_steps + 1):
        loss_record = []
        mse_record = []
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            loss, mse = train_dynamics_learner(optimizer, dynamics_learner,
                                               matrix, data, args.nodes, args.prediction_steps)
            loss_record.append(loss.item())
            mse_record.append(mse.item())
        print('loss: '+str(np.mean(loss_record)))
        if simulation_type == 'bn':
            print('accuracy：'+str(accu))
        else:
            print('mse:'+str(np.mean(mse_record)))
        # print('\nDynamics learning step: %d, loss: %f, MSE: %f' % (step, np.mean(loss_record), np.mean(mse_record)))

def train_batch_net(train_data_loader,optimizer_network,gumbel_generator,dyn_learner,loss_fn):
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

def train_batch_dyn_cml(args, dynamics_learner, gumbel_generator, optimizer,
                   device, train_loader, epoch, experiment, writer):
    matrix = gumbel_generator.sample(hard=True)  # Sample from gumbel generator

    fig = plt.figure()
    plt.imshow(matrix.to('cpu').numpy(), cmap='gray')
    writer.add_figure('Gumbel-Sample' + '/experiment'+str(experiment), fig, epoch)
    plt.close()

    loss_records = []
    mse_records = []
    for step in range(1, args.dynamics_steps + 1):
        loss_record = []
        mse_record = []
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            loss, mse = train_dynamics_learner(optimizer, dynamics_learner,
                                               matrix, data, args.nodes, args.prediction_steps)
            loss_record.append(loss.item())
            mse_record.append(mse.item())
        loss_records.append(np.mean(loss_record))
        mse_records.append(np.mean(mse_record))
        print('\nDynamics learning step: %d, loss: %f, MSE: %f' % (step, np.mean(loss_record), np.mean(mse_record)))


# dyn trainer
def train_dyn_learner_bn(optimizer,dyn_learner,adj,data_train,data_target,loss_fn,positional=False,pos_enc=None,optimizer_pos=None):
    loss = 0
    #optimizer
    optimizer.zero_grad()
    
    num_nodes = adj.size(0)
    #adj mat
    adj = adj.unsqueeze(0)
    adj = adj.repeat(data_train.size()[0],1,1)
    adj = adj.cuda() if use_cuda else adj
    
    
    # positional encoding give up
    if positional:
        pos_arr = torch.tensor(np.arange(0.,float(num_nodes))).repeat(data_train.shape[0],1)
        if use_cuda:
            pos_arr = pos_arr.cuda()
        pos_res = pos_enc(pos_arr)
        pos_res = torch.unsqueeze(pos_res,2).repeat(1,1,2) # (batch_size,node_num) => (batch_size,node_num,2)

        # add poitional encoding result to the 
        data_train = data_train+pos_res

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
    if positional:
        optimizer_pos.step()
    
    return loss,accus



# 动力学学习器dynamics_learner的一步训练
# relations的格式为num_nodes, num_nodes
# data格式为：batch_size, num_nodes, time_steps(10), dimension(4)
def train_dyn_learner_cml(optimizer, dynamics_learner, relations, data, sz, steps):
    # dynamics_learner.train()
    optimizer.zero_grad()
    
    adjs = relations.unsqueeze(0)
    adjs = adjs.repeat(data.size()[0],1,1)
    adjs = adjs.cuda() if use_cuda else adjs
    
    input = data[:, :, 0, :]
    target = data[:, :, 1 : steps, :]
    output = input
    
    outputs = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3))
    outputs = outputs.cuda() if use_cuda else outputs
    # 完成steps-1步预测，output格式为：batchsize, num_nodes, time_steps, dimension
    for t in range(steps - 1):
        output = dynamics_learner(output, adjs)
        outputs[:,:,t,:] = output
    
    loss = torch.mean(torch.abs(outputs - target))
    loss.backward()
    optimizer.step()
    mse = F.mse_loss(outputs, target)
    if use_cuda:
        loss = loss.cpu()
        mse = mse.cpu()
    return loss, mse


# data格式为：batch_size, num_nodes, time_steps(10), dimension(4)
# gumbel_generator为生成器，dynamics_learner为动力学预测器
def train_batch_generator(optimizer_network,gumbel_generator,dyn_learner,data_train,data_target,loss_fn):
    optimizer_network.zero_grad()

    out_matrix = gumbel_generator.sample()
    out_matrix = out_matrix.unsqueeze(0)
    out_matrix = out_matrix.repeat(data_train.size()[0], 1, 1)
    gumbel_generator.drop_temperature()
    
    
    # positional encoding
    # pos_arr = torch.tensor(np.arange(0.,float(num_nodes))).repeat(data_train.shape[0],1)
    # if use_cuda:
    #     pos_arr = pos_arr.cuda()
    # pos_res = pos_enc(pos_arr)
    # pos_res = torch.unsqueeze(pos_res,2).repeat(1,1,2) 
   

    # get result caculated by neural network
    output = dyn_learner(data_train,out_matrix)
    output = output.permute(0,2,1)
    
    # caculate the difference
    data_target = data_target.long()
    loss = loss_fn(output,data_target)
    loss.backward()
    optimizer_network.step()
    return loss,out_matrix

def get_valid_loss(gumbel_generator,dyn_learner,data_train,data_target,loss_fn):
    out_matrix = gumbel_generator.sample()
    out_matrix = out_matrix.unsqueeze(0)
    out_matrix = out_matrix.repeat(data_train.size()[0], 1, 1)
    gumbel_generator.drop_temperature()

    output = dyn_learner(data_train,out_matrix)
    output = output.permute(0,2,1)

    data_target = data_target.long()
    loss = loss_fn(output,data_target)
    return loss.data.cpu().tolist()

# output:[batch_num,node_num,2]
# target[batch_num,node_num]
def cacu_accu(output,target):
    if output.size(1) == 2:
        output=  output.permute(0,2,1)
    output = output.cpu()
    target = target.cpu()
    right = 0.
    accu_all_list = []
    for i in range(output.size(0)):
        accu_batch = []
        for j in range(output.size(1)):
            if output[i][j][0] >= output[i][j][1]:
                if target[i][j] == 0:
                    right += 1
                elif target[i][j] == 1:
                    continue
                else:
                    print('error pos 1')
                    debug()
            elif output[i][j][0] < output[i][j][1]:
                if target[i][j] == 1:
                    right += 1
                elif target[i][j] == 0:
                    continue
                else:
                    print('error pos 2')
                    debug()
            else:
                print('error pos 0')
                debug()
    return right / target.size(0) /target.size(1)

def get_test_accu(gumbel_generator,dyn_learner,data_train,data_target):
    out_matrix = gumbel_generator.sample()
    out_matrix = out_matrix.unsqueeze(0)
    out_matrix = out_matrix.repeat(data_train.size()[0], 1, 1)
    gumbel_generator.drop_temperature()

    output = dyn_learner(data_train,out_matrix)
    
    # caculate the difference
    data_target = data_target.long()
    accus = cacu_accu(output,data_target)
    return accus

def test(simulation_type,prediction_steps,gumbel_generator,dyn_learner,standard_adj,test_data_loader):
    if simulation_type == 'bn':
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
