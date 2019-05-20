import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os, sys
from scipy.sparse import csr_matrix
import torch.nn.functional as F

from utils import *
from model import *

def trans(matrix):
    (coords, values, shape) = matrix
    res = torch.sparse.DoubleTensor(torch.LongTensor(coords.transpose()), torch.DoubleTensor(values), torch.Size(shape))
    return res

def load_ti(dataset):
    fn = open('./data/ind.{}.test.index'.format(dataset)).readlines()
    fn = [ int(l.split()[0]) for l in fn]
    return np.array(fn)

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.double
    torch.set_default_dtype(dtype)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    adj, features, y_train, y_val, y_test, train_mask,val_mask,test_mask=load_data(args.dataset)    
    tindex = load_ti(args.dataset)
    print(tindex.shape)
    print(np.sum(test_mask))
    test_len = np.sum(test_mask)
    dim_ot = y_train.shape[1]
    
    y_train, y_val, y_test = torch.LongTensor(y_train).to(device),torch.Tensor(y_val).to(device), \
            torch.Tensor(y_test).to(device)
    train_mask, val_mask, test_mask = 1*train_mask.reshape(-1), 1*val_mask.reshape(-1), 1*test_mask.reshape(-1)
    train_mask2 = torch.ByteTensor(train_mask)
    train_mask3 = torch.LongTensor(train_mask*np.arange(train_mask.shape[0]))
    train_mask4 = torch.masked_select(train_mask3, train_mask2)
    val_mask2 = torch.ByteTensor(val_mask)
    val_mask3 = torch.LongTensor(val_mask*np.arange(val_mask.shape[0]))
    val_mask4 = torch.masked_select(val_mask3, val_mask2)
    test_mask2 = torch.ByteTensor(test_mask)
    test_mask3 = torch.LongTensor(test_mask*np.arange(test_mask.shape[0]))
    test_mask4 = torch.masked_select(test_mask3, test_mask2)

    train_mask, val_mask, test_mask = torch.Tensor(train_mask).to(device),torch.Tensor(val_mask).to(device), \
            torch.Tensor(test_mask).to(device)
    #adj += sp.diags(np.ones(adj.shape[0]))
    adj, features = preprocess_adj(adj), preprocess_features(features)
    dim_nd, dim_ft, dim_hd = features[2][0], features[2][1], 16 #features[2][1]
    adj, features =  trans(adj).to(device), trans(features).to(device)
    d_crt = nn.CrossEntropyLoss()
    model = GCN2(dim_nd, dim_ft, dim_hd, dim_ot).to(device)
    opt_m = optim.Adam(model.parameters(), lr = args.lr)

    def l2(model, loss, l2_reg=5e-4):
        for param in model.parameters():
            loss += param.pow(2).sum() * l2_reg
        return loss

    def evaluate(output_v, target_v):
        loss = d_crt(output_v, target_v)
        #predlv = torch.max(output_v, 1)[1]
        #print(torch.sum(predlv == target_v).data.cpu().numpy()/1000 )
        print("evaluate: ep: {}, loss: {}".format(ep, loss.data.cpu().numpy()))

        
    def test():
        output = model(adj, features.to_dense())
        predls = torch.max(output, 1)[1]
        labels = torch.max(y_test, 1)[1]
        #predls, labels = predls[tindex], labels[tindex]
        predls, labels = predls[test_mask4], labels[test_mask4]
        print(predls)
        #print(labels)
        print(torch.sum(predls == labels).data.cpu().numpy()/test_len )


    def Print():
        for param in model.parameters():
            print(param)

    y_train_s = torch.index_select(y_train, 0, train_mask4)
    target_s = torch.max(y_train_s, 1)[1]
    y_train_v = torch.index_select(y_train, 0, val_mask4)
    target_v = torch.max(y_train_v, 1)[1]
    y_train_t = torch.index_select(y_train, 0, test_mask4)
    target_t = torch.max(y_train_t, 1)[1]

    for ep in range(args.epoch):
        
        output = model(adj, features.to_dense())
        output_s = torch.index_select(output, 0, train_mask4)
        loss = d_crt(output_s, target_s)
        loss = l2(model, loss)
        #print("ep: {}, loss: {}".format(ep, loss.data.cpu().numpy()))
        opt_m.zero_grad()
        loss.backward()
        opt_m.step()
        output_v = torch.index_select(output, 0, val_mask4)
        evaluate(output_v, target_v)
    
    test()
    #Print()
if __name__ == '__main__':
    dataset = 'cora' # cora, citeseer, pubmed
    seed = 123
    lr = 0.02
    use_cuda = False
    epoch = 10
    parser = argparse.ArgumentParser(description='parameters ...')
    parser.add_argument('--dataset', default=dataset, help='dataset ')
    parser.add_argument('--seed', default=seed, type=int, help='random seed ')
    parser.add_argument('--lr', default=lr, type=float, help='learning rate ')
    parser.add_argument('--use_cuda', default=use_cuda, type=bool, help='cuda ')
    parser.add_argument('--epoch', default=epoch, type=int, help='epoches ')
    
    args = parser.parse_args()
    print(args)
    train(args)


