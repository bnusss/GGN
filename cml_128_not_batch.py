import matplotlib
matplotlib.use('Agg')
import pickle
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from tools import *
from utils.model import *
from utils.process_128 import *
# from tensorboardX import SummaryWriter


def train_dynamics(args, dynamics_learner, gumbel_generator, optimizer,
                   device, train_loader, epoch, experiment,skip_conn,object_matrix):
    matrix = gumbel_generator.sample(hard=True)  # Sample from gumbel generator
    matrix = torch.tensor(object_matrix).cuda()


    fig = plt.figure()
    plt.imshow(matrix.to('cpu').numpy(), cmap='gray')
    plt.close()

    loss_records = []
    mse_records = []
    mses = 0
    for step in range(1, args.dynamics_steps + 1):
        loss_record = []
        mse_record = []
        mses = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            mse = train_dynamics_learner(optimizer, dynamics_learner,
                                               matrix, data, args.nodes, args.prediction_steps,skip_conn)
            mses += mse
            # loss_record.append(loss.item())
            mse_record.append(mse.item())
            # print(batch_idx/)
            if batch_idx % 128 == 0:
                mses.backward()
                optimizer.step()
                print(np.mean(mse_record))
                mse_list = []
                loss = 0
                mses = 0
                optimizer.zero_grad()
        # loss_records.append(np.mean(loss_record))
        # mse_records.append(np.mean(mse_record))
        # print('\nDynamics learning step: %d, loss: %f, MSE: %f' % (step, np.mean(loss_record), np.mean(mse_record)))


def val_dynamics(args, dynamics_learner, gumbel_generator,
                 device, val_loader, epoch, experiment, best_val_loss,skip_conn):
    matrix = gumbel_generator.sample(hard=True)  # Sample from gumbel generator

    loss_record = []
    mse_record = []
    for batch_idx, data in enumerate(val_loader):
        data = data.to(device)
        loss, mse = val_dynamics_learner(dynamics_learner, matrix, args.nodes, data, args.prediction_steps,skip_conn)
        loss_record.append(loss.item())
        mse_record.append(mse.item())

    print('\nDynamics validation: loss: %f, MSE: %f' % (np.mean(loss_record), np.mean(mse_record)))

    if best_val_loss > np.mean(loss_record):
        torch.save(dynamics_learner.state_dict(), args.dynamics_path)
        torch.save(gumbel_generator.state_dict(), args.gumbel_path)

    return np.mean(loss_record)


def train_gumbel(args, dynamics_learner, gumbel_generator, optimizer_network,
                 device, train_loader, object_matrix, epoch, experiment,skip_conn):
    object_matrix.to(device)

    loss_records = []
    net_error_records = []
    tpr_records = []
    fpr_records = []
    for step in range(1, args.reconstruct_steps + 1):
        loss_record = []
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            loss, _ = train_net_reconstructor(optimizer_network, gumbel_generator, dynamics_learner,
                                              args.nodes, data, args.prediction_steps,skip_conn)
            loss_record.append(loss.item())
        loss_records.append(np.mean(loss_record))
        if step % 1 == 0:
            net_error, tpr, fpr = constructor_evaluator(gumbel_generator, 500, object_matrix, args.nodes)
            net_error_records.append(net_error)
            tpr_records.append(tpr)
            fpr_records.append(fpr)
            print('\nGumbel training step: %d, loss: %f' % (step, np.mean(loss_record)))
            print('Net error: %f, TPR: %f, FPR: %f' % (net_error, tpr, fpr))



def test(args, dynamics_learner, gumbel_generator, device, test_loader, object_matrix, experiment,skip_conn):
    # load model
    dynamics_learner.load_state_dict(torch.load(args.dynamics_path))
    gumbel_generator.load_state_dict(torch.load(args.gumbel_path))

    # evaluate network
    net_error, tpr, fpr = constructor_evaluator(gumbel_generator, 500, object_matrix, args.nodes)

    # evaluate dynamics
    matrix = gumbel_generator.sample(hard=True)  # Sample from gumbel generator

    dynamics_learner.eval()
    loss_record = []
    mse_record = []
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        loss, mse = val_dynamics_learner(dynamics_learner, matrix, args.nodes, data, args.prediction_steps)
        loss_record.append(loss.item())
        mse_record.append(mse.item())

    print('\nTest: Net error: %f, TPR: %f, FPR: %f' % (net_error, tpr, fpr))
    print('loss: %f, mse: %f' % (np.mean(loss_record), np.mean(mse_record)))
    

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Coupled Map Lattice and Kuramoto')
    parser.add_argument('--simulation-type', type=str, default='cml',
                        help='simulation type to choose(cml or kuramoto)')
    parser.add_argument('--nodes', type=int, default=10,
                        help='number of nodes in data')
    parser.add_argument('--dims', type=int ,default=1,
                        help='information dimension in data(1 for CML and 2 for kuramoto)')
    parser.add_argument('--skip', type=int ,default=1,
                        help='weather to use skip connection(recommend to use in kuramoto)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs, default: 15)')
    parser.add_argument('--experiments', type=int, default=10,
                        help='number of experiments (default: 10)')
    parser.add_argument('--dynamics-steps', type=int, default=30,
                        help='number of steps for dynamics learning (default: 30)')
    parser.add_argument('--reconstruct-steps', type=int, default=0,
                        help='number of steps for reconstruction (default: 5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=2050,
                        help='random seed (default: 2050)')
    parser.add_argument('--prediction-steps', type=int, default=10,
                        help='prediction steps in data (default: 10)')
    parser.add_argument('--dynamics-path', type=str, default='./saved/dynamics.pickle',
                        help='path to save dynamics learner (default: ./saved/dynamics.pickle)')
    parser.add_argument('--gumbel-path', type=str, default='./saved/gumbel.pickle',
                        help='path to save gumbel generator (default: ./saved/gumbel.pickle)')
    parser.add_argument('--data-path', type=str, default='./data/test.pickle',
                        help='path to load data (default: ./data/test.pickle)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print('Device:', device)

    # Loading data
    train_loader,val_loader,test_loader,object_matrix = load_cml_ggn(batch_size=1)

    for experiment in range(1, args.experiments + 1):
        print('\n---------- Experiment %d ----------' % experiment)

        best_val_loss = np.inf
        best_epoch = 0

        gumbel_generator = Gumbel_Generator(sz=args.nodes, temp=10, temp_drop_frac=0.9999).to(device)
        optimizer_network = optim.Adam(gumbel_generator.parameters(), lr=0.1)

        dynamics_learner = GumbelGraphNetwork(args.dims).to(device)
        optimizer = optim.Adam(dynamics_learner.parameters(), lr=0.0001)

        for epoch in range(1, args.epochs + 1):
            print('\n---------- Experiment %d  Epoch %d ----------' % (experiment, epoch))
            train_dynamics(args, dynamics_learner, gumbel_generator, optimizer,device, train_loader, epoch, experiment,args.skip,object_matrix)
            val_loss = val_dynamics(args, dynamics_learner, gumbel_generator,
                                    device, val_loader, epoch, experiment, best_val_loss,args.skip)
            train_gumbel(args, dynamics_learner, gumbel_generator, optimizer_network,
                         device, train_loader, object_matrix, epoch, experiment,args.skip)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
            print('\nCurrent best epoch: %d, best val loss: %f' % (best_epoch, best_val_loss))

        print('\nBest epoch: %d' % best_epoch)
        test(args, dynamics_learner, gumbel_generator, device, test_loader, object_matrix, experiment,args.skip)


if __name__ == '__main__':
    main()

