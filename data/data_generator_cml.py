import networkx as nx
import scipy.sparse
import argparse
import torch
import numpy as np
import pickle

use_cuda = torch.cuda.is_available()
np.random.seed(2050)
torch.manual_seed(2050)

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=10, help='Number of nodes, default=10')
parser.add_argument('--samples', type=int, default=7000, help='Number of samples in simulation, default=7000')
parser.add_argument('--prediction-steps', type=int, default=10, help='prediction steps, default=10')
parser.add_argument('--evolving-steps', type=int, default=100, help='evolving steps, default=100')
parser.add_argument('--lambd', type=float, default=3.6, help='lambda in logistic map, default=3.6')
parser.add_argument('--coupling', type=float, default=0.2, help='coupling coefficent, default=0.2')
parser.add_argument('--data-name', type=str, default='./cml/data_lambd3.6_coupl0.2_node10.pickle', help='data name to save')
args = parser.parse_args()


def logistic_map(x, lambd=args.lambd):
    # return 1 - lambd * x ** 2
    return lambd * x * (1 - x)


class CMLDynamicSimulator():
    def __init__(self, batch_size, sz, s):
        self.s = s
        self.thetas = torch.rand(batch_size, sz)
        # random 4-regular graph
        self.G = nx.random_regular_graph(4, sz, seed=2050)
        # self.G = nx.cycle_graph(sz)
        A = nx.to_scipy_sparse_matrix(self.G, format='csr')
        n, m = A.shape
        diags = A.sum(axis=1)
        D = scipy.sparse.spdiags(diags.flatten(), [0], m, n, format='csr')

        self.obj_matrix = torch.FloatTensor(A.toarray())
        self.inv_degree_matrix = torch.FloatTensor(np.linalg.inv(D.toarray()))

        if use_cuda:
            self.thetas = self.thetas.cuda()
            self.obj_matrix = self.obj_matrix.cuda()
            self.inv_degree_matrix = self.inv_degree_matrix.cuda()

    def SetMatrix(self, matrix):
        self.obj_matrix = matrix
        if use_cuda:
            self.obj_matrix = self.obj_matrix.cuda()

    def SetThetas(self, thetas):
        self.thetas = thetas
        if use_cuda:
            self.thetas = self.thetas.cuda()

    def OneStepDiffusionDynamics(self):
        self.thetas = (1 - self.s) * logistic_map(self.thetas) + self.s * \
                      torch.matmul(torch.matmul(logistic_map(self.thetas), self.obj_matrix), self.inv_degree_matrix)
        return self.thetas


if __name__ == '__main__':
    # 传入参数
    num_nodes = args.nodes
    num_samples = args.samples
    prediction_steps = args.prediction_steps
    evolve_steps = args.evolving_steps
    lambd = args.lambd
    coupling = args.coupling

    # 生成数据
    simulator = CMLDynamicSimulator(batch_size=num_samples, sz=num_nodes, s=coupling)
    simulates = np.zeros([num_samples, num_nodes, evolve_steps, 1])
    sample_freq = 1
    for t in range((evolve_steps + 1) * sample_freq):
        locs = simulator.OneStepDiffusionDynamics()
        if t % sample_freq == 0:
            locs = locs.cpu().data.numpy() if use_cuda else locs.data.numpy()
            simulates[:, :, t // sample_freq - 1, 0] = locs
    data = torch.Tensor(simulates)
    print('原始数据维度：', simulates.shape)

    # 数据切割
    prediction_num = data.size()[2] // prediction_steps
    for i in range(prediction_num):
        last = min((i + 1) * prediction_steps, data.size()[2])
        feat = data[:, :, i * prediction_steps: last, :]
        if i == 0:
            features = feat
        else:
            features = torch.cat((features, feat), dim=0)
    # features数据格式：sample, nodes, timesteps, dimension=1)
    print('切割后的数据维度：', features.shape)

    # shuffle
    features_perm = features[torch.randperm(features.shape[0])]

    # 划分train, val, test
    train_data = features_perm[: features.shape[0] // 7 * 5, :, :, :]
    val_data = features_perm[features.shape[0] // 7 * 5: features.shape[0] // 7 * 6, :, :, :]
    test_data = features_perm[features.shape[0] // 7 * 6:, :, :, :]

    print(train_data.shape, val_data.shape, test_data.shape)

    results = [simulator.obj_matrix, train_data, val_data, test_data]

    with open(args.data_name, 'wb') as f:
        pickle.dump(results, f)
