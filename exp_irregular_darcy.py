import os
import argparse
import numpy as np
import scipy.io as sio
import torch
from tqdm import *
from lapy import TriaMesh
from lapy import Solver
from utils.testloss import TestLoss
from model.HPM_Irregular_Mesh import Model
from utils.normalizer import UnitGaussianNormalizer

parser = argparse.ArgumentParser('HPM')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--n-hidden', type=int, default=128, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=4, help='layers')
parser.add_argument('--n-heads', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--freq_num', type=int, default=32)
parser.add_argument('--spectral_pos_embedding', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='HPM')
parser.add_argument('--data_path', type=str, default='/data/hpm/Darcy.mat')
args = parser.parse_args()
eval = args.eval
save_name = args.save_name
print(f"Save Name: {save_name}")


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():

    ntrain = args.ntrain
    ntest = 200
    s = 2290

    data = sio.loadmat(args.data_path)

    # Compute spectral embedding from mesh Laplacian (LBO eigenvectors)
    k = args.freq_num + 5
    Points = np.vstack((data['MeshNodes'], np.zeros(s).reshape(1, -1)))
    mesh = TriaMesh(Points.T, data['MeshElements'].T - 1)
    fem = Solver(mesh)
    evals, LBO_MATRIX = fem.eigs(k=k)  # N, K

    y_dataIn = torch.Tensor(data['u_field'])
    x_dataIn = torch.Tensor(data['c_field'])

    x_data = x_dataIn
    y_data = y_dataIn

    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    print('x_train:', x_train.shape, 'y_train:', y_train.shape)
    print('x_test:', x_test.shape, 'y_test:', y_test.shape)

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    x_train = x_train.reshape(ntrain, -1, 1)
    x_test = x_test.reshape(ntest, -1, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, x_train, y_train),
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, y_test),
                                              batch_size=args.batch_size,
                                              shuffle=False)

    print("Dataloading is over.")

    model = Model(space_dim=0,
                  n_layers=args.n_layers,
                  n_hidden=args.n_hidden,
                  dropout=args.dropout,
                  n_head=args.n_heads,
                  Time_Input=False,
                  mlp_ratio=args.mlp_ratio,
                  fun_dim=1,
                  out_dim=1,
                  freq_num=args.freq_num,
                  ref=args.ref,
                  unified_pos=args.unified_pos,
                  spectral_pos_embedding=args.spectral_pos_embedding,
                  spectral_embedding=LBO_MATRIX).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(args)
    count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)
    y_normalizer.cuda()

    for ep in tqdm(range(args.epochs)):

        model.train()
        train_loss = 0

        for pos, fx, y in train_loader:

            x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x, None).squeeze(-1)

            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            loss = myloss(out, y)
            loss.backward()

            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += loss.item()
            scheduler.step()

        train_loss = train_loss / ntrain
        print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))

        model.eval()
        rel_err = 0.0
        with torch.no_grad():
            for pos, fx, y in test_loader:
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                out = model(x, None).squeeze(-1)
                out = y_normalizer.decode(out)

                tl = myloss(out, y).item()
                rel_err += tl

        rel_err /= ntest
        print("rel_err:{}".format(rel_err))

        if ep % 100 == 0:
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            print('save model')
            torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    print('save model')
    torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))


if __name__ == "__main__":
    main()
