import sys
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import Metrics
import Utils
from Mynet import *
from Param import *
from Param_Mynet import *
from Dataset_Setting import *
from Mynet_Setting import *
from torch.utils.data import Dataset

class TensorDatasetIndex(Dataset):
    def __init__(self, *tensors) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return index, tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

def get_key(query, holder=-1, mem_start=0):
    keys = []
    for i in range(1, len_c+1):
        key = query - i*day_interval + mem_start
        key[key<0] = holder
        keys.append(key)

        key = query - i * day_interval + TIMESTEP_IN + mem_start
        key[key < 0] = holder
        keys.append(key)
    return keys

def get_value(keys, memory):
    values = []
    for key in keys:
        value = memory[key]
        values.append(value)
    return values

def update_value(idx, new_value, memory, mem_start=0):
    memory[idx+mem_start] = new_value
    return memory

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    XS_C, YS_C = [], []
    XS_C2, YS_C2 = [], []
    if mode == 'TRAIN':
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    # XS = np.concatenate((XS, XS_C, YS_C, XS_C2, YS_C2), axis=-1)
    # XS = np.concatenate((XS, YS_C), axis=-1)
    XS = XS.transpose(0, 3, 2, 1)
    return XS,  YS

class NTN(nn.Module):
    def __init__(self, input_dim, feature_map_dim):
        super(NTN, self).__init__()
        self.interaction_dim = feature_map_dim
        self.V = nn.Parameter(torch.randn(feature_map_dim, input_dim * 2, 1), requires_grad=True)
        nn.init.xavier_normal_(self.V)
        self.W1 = nn.Parameter(torch.randn(feature_map_dim, input_dim, input_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W1)
        self.W2 = nn.Parameter(torch.randn(feature_map_dim, input_dim, input_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W2)
        self.b = nn.Parameter(torch.zeros(feature_map_dim), requires_grad=True)

    def forward(self, x_1, x_2):
        feature_map = []
        for i in range(self.interaction_dim):
            x_1_t = torch.matmul(x_1, self.W1[i])
            x_2_t = torch.matmul(x_2, self.W2[i])
            part1 = torch.cosine_similarity(x_1_t, x_2_t, dim=-1).unsqueeze(dim=-1)
            a = torch.cat([x_1, x_2], dim=-1)
            part2 = torch.matmul(torch.cat([x_1, x_2], dim=-1), self.V[i])
            fea = part1 + part2 + self.b[i]
            feature_map.append(fea)
        feature_map = torch.cat(feature_map, dim=-1)
        return torch.relu(feature_map)

class mynet_(nn.Module):
    def __init__(self, device, num_nodes=320, in_dim=1, supports=None):
        super(mynet_, self).__init__()
        self.encoder = gwnet_head(device, num_nodes=num_nodes, in_dim=in_dim, supports=supports)

        self.memory = nn.Parameter(torch.randn(20, 32), requires_grad=True)
        nn.init.xavier_normal_(self.memory)
        self.mem_proj1 = nn.Linear(in_features=256, out_features=32)
        self.mem_proj2 = nn.Linear(in_features=32, out_features=256)

        self.cts_proj1 = nn.Linear(in_features=256, out_features=32)
        self.cts_proj2 = nn.Linear(in_features=25, out_features=256)

        self.NTN1 = NTN(input_dim=32, feature_map_dim=5)
        self.NTN2 = NTN(input_dim=32, feature_map_dim=5)
        self.NTN3 = NTN(input_dim=32, feature_map_dim=5)
        self.NTN4 = NTN(input_dim=32, feature_map_dim=5)
        self.NTN5 = NTN(input_dim=64, feature_map_dim=5)


        self.meta_W = nn.Linear(256, 10)
        self.meta_b = nn.Linear(256, 1)

        self.decoder = Decoder(in_dim=256, hidden_dim=512, out_dim=12)
        self.merge = nn.Linear(in_features=10, out_features=1)
        # self.merge = nn.Linear(in_features=2, out_features=1)
    def forward(self, x, embedding):
        hidden_x = self.encoder(x)
        hidden_merge = torch.cat([hidden_x]+embedding, dim=-1)

        hidden_merge_re = hidden_merge.permute(0, 2, 3, 1)
        query = self.mem_proj1(hidden_merge_re)
        att = torch.softmax(torch.matmul(query, self.memory.t()), dim=-1)
        res_mem = torch.matmul(att, self.memory)
        res_mem = self.mem_proj2(res_mem)
        res_mem = res_mem.permute(0, 3, 1, 2)

        hidden_cts = self.cts_proj1(hidden_merge_re)
        px_1 = self.NTN1(hidden_cts[:, :, 0, :], hidden_cts[:, :, 1, :])
        px_2 = self.NTN2(hidden_cts[:, :, 1, :], hidden_cts[:, :, 3, :])
        px_3 = self.NTN3(hidden_cts[:, :, 0, :], hidden_cts[:, :, 3, :])
        py_1 = self.NTN4(hidden_cts[:, :, 2, :], hidden_cts[:, :, 4, :])
        xy_1 = hidden_cts[:, :, 1:3, :].reshape(-1, N_NODE, 2 * 32)
        xy_2 = hidden_cts[:, :, 3:5, :].reshape(-1, N_NODE, 2 * 32)
        pxy_1 = self.NTN5(xy_1, xy_2)

        sim_mtx = torch.cat([px_1, px_2, px_3, py_1, pxy_1], dim=-1)
        sim_mtx = self.cts_proj2(sim_mtx)

        W = self.meta_W(sim_mtx)
        b = self.meta_b(sim_mtx)
        W = torch.reshape(W, (-1, N_NODE, 10, 1))
        b = b.view(-1, 1, N_NODE, 1)
        W = torch.softmax(W, dim=-2)
        hidden = torch.cat((hidden_merge, res_mem), dim=-1)
        hidden = torch.einsum("bdnt,bntj->bdnj", [hidden, W]) + b

        output = self.decoder(hidden)
        return output, hidden_x.detach()

def getModel(name):
    if ADJPATH:
        adj_mx = load_adj(ADJPATH, ADJTYPE)
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        model = mynet_(device, num_nodes=N_NODE, in_dim=CHANNEL, supports=supports).to(device)
    else:
        model = mynet_(device, num_nodes=N_NODE, in_dim=CHANNEL, supports=None).to(device)
    return model

def evaluateModel(model, criterion, data_iter, memory, holder=0, mem_start=0, update=False):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for idx, (x, y) in data_iter:
            keys = get_key(idx, holder=holder, mem_start=mem_start)
            values = get_value(keys, memory)
            y_pred, new_values = model(x, values)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            memory = update_value(idx, new_values, memory, mem_start=mem_start) if update else memory
        return l_sum / n

def predictModel(model, data_iter, memory, holder=0, mem_start=0, update=False):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for idx, (x, y) in data_iter:
            keys = get_key(idx, holder=holder, mem_start=mem_start)
            values = get_value(keys, memory)
            YS_pred_batch, new_values = model(x, values)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
            memory = update_value(idx, new_values, memory, mem_start=mem_start) if update else memory
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def trainModel(name, mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name)
    summary(model, [(BATCHSIZE, CHANNEL,N_NODE,TIMESTEP_IN)].extend([(BATCHSIZE, 256, N_NODE, CHANNEL) for i in range(4)]), batch_dim=None, device=device)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = TensorDatasetIndex(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    print('XS_torch.shape:  ', XS_torch.shape)
    print('YS_torch.shape:  ', YS_torch.shape)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, 12, shuffle=False)

    memory = torch.randn((XS.shape[0], 256, N_NODE, CHANNEL), dtype=torch.float).to(device)
    nn.init.xavier_normal_(memory)
    placeholder = torch.ones((1, 256, N_NODE, CHANNEL), dtype=torch.float).to(device)
    memory = torch.cat([memory, placeholder], dim=0)
    
    min_val_loss = np.inf
    wait = 0

    print('LOSS is :',LOSS)
    if LOSS == "MaskMAE":
        criterion = Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    
    for epoch in range(EPOCH):
        memory_tmp = memory.clone()
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for idx, (x, y) in train_iter:
            keys = get_key(idx, holder=len(memory)-1, mem_start=0)
            values = get_value(keys, memory_tmp)
            optimizer.zero_grad()
            y_pred, new_values = model(x, values)
            memory_tmp = update_value(idx, new_values, memory_tmp, mem_start=0)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter, memory_tmp, mem_start=0, update=True)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + name + '.pt')
            memory = memory_tmp
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
            
    torch_score = evaluateModel(model, criterion, train_iter, memory, holder=len(memory)-1, mem_start=0, update=True)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, 12, shuffle=False), memory, holder=len(memory)-1, mem_start=0, update=True)
    np.save(PATH + '/' + MODELNAME + '_memory.npy', memory[-len_c*day_interval-1:-1].cpu())
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())
        
def testModel(name, mode, XS, YS):
    if LOSS == "MaskMAE":
        criterion = Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = TensorDatasetIndex(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, 12, shuffle=False)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH+ '/' + name + '.pt'))

    his_memory = torch.Tensor(np.load(PATH + '/' + MODELNAME + '_memory.npy'))
    memory = torch.randn((XS.shape[0], 256, N_NODE, CHANNEL), dtype=torch.float)
    memory = torch.cat([his_memory, memory], dim=0).to(device)

    torch_score = evaluateModel(model, criterion, test_iter, memory, mem_start=len(his_memory), update=True)
    YS_pred = predictModel(model, test_iter, memory, mem_start=len(his_memory), update=True)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('Model Testing Ended ...', time.ctime())

DATA_SET = sys.argv[1]
GPU = sys.argv[2] if len(sys.argv) == 3 else '0'
DATANAME = DATA_SET
N_NODE = SETTING[DATA_SET]['n_node']
CHANNEL = SETTING[DATA_SET]['fea']
FLOWPATH = SETTING[DATA_SET]['data_file']
ADJPATH = SETTING[DATA_SET]['adj_file']
ADJTYPE = 'doubletransition'
TRAINRATIO = Utils.get_train_ratio(SETTING[DATA_SET])
data_type = FLOWPATH.split('.')[-1]
if data_type=='csv':
    data = pd.read_csv(FLOWPATH, index_col=0).values
elif data_type == 'npy':
    if DATANAME == 'COVID-US':
        data = np.load(FLOWPATH).sum(axis=-1)
    elif DATANAME == 'COVID-CHI':
        data = np.load(FLOWPATH)[:, :, :, 0].reshape((6600, -1))
        # data = np.load(FLOWPATH).sum(axis=-1).reshape((6600, -1))
    elif DATANAME=='Japan-OD':
        data = np.load(FLOWPATH).sum(axis=-1)

day_interval = MODEL_SETTING[DATA_SET]['day_interval']
# len_c = 1
# start = int(len_c * day_interval)
################# Parameter Setting #######################
MODELNAME = 'Mynet'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save_dataset_tmp/save_{}/'.format(DATANAME) + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
# torch.backends.cudnn.deterministic = True
########################################################### 
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
###########################################################
scaler = StandardScaler()
data = scaler.fit_transform(data)
print('data.shape', data.shape)
###########################################################
def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Mynet.py', PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_Mynet.py', PATH)
    shutil.copy2('Dataset_Setting.py', PATH)
    shutil.copy2('Mynet_Setting.py', PATH)
    
    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    print('TRAIN XS.shape YS,shape', trainXS[0].shape, trainYS.shape)
    trainModel(MODELNAME, 'train', trainXS, trainYS)
    
    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS(data, 'TEST')
    print('TEST XS.shape, YS.shape', testXS[0].shape, testYS.shape)
    testModel(MODELNAME, 'test', testXS, testYS)

    
if __name__ == '__main__':
    main()

    
