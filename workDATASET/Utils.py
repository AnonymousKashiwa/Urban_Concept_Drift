import pickle
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp


def masked_mae(preds, labels, null_val=0.0):
    preds[preds < 1e-5] = 0
    labels[labels < 1e-5] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_adj(pkl_filename, adjtype, n_node=0):
    if adjtype == "identity":
        adj = [np.diag(np.ones(n_node)).astype(np.float32)]
        return adj
    elif adjtype == "doubletransition":
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
        return adj
    elif adjtype == "raw":
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
        return adj_mx
    else:
        error = 0
        assert error, "adj type not defined"


def get_train_ratio(dataset_setting):
    start = dataset_setting['train_start']
    end = dataset_setting['test_end']
    test_start = dataset_setting['test_start']
    freq = dataset_setting['freq']
    time_line = pd.date_range(start, end + '235959', freq=freq)
    test_start_time = pd.Timestamp(test_start)
    test_time_line = time_line[time_line.slice_indexer(test_start_time, )]
    train_ratio = 1 - len(test_time_line) / len(time_line)
    return train_ratio

def get_timeline(dataset_setting):
    start = dataset_setting['train_start']
    end = dataset_setting['test_end']
    freq = dataset_setting['freq']
    time_line = pd.DataFrame({'date':pd.date_range(start, end + '235959', freq=freq)})
    return time_line
