#!/usr/bin/python
import argparse
import torch
import time
import torch.nn.functional as F
from torch import distributed, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count

    def __str__(self):
        return '{:.6f}'.format(self.average)


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output, label):
        predictions = output.data.argmax(dim=1)
        correct = predictions.eq(label.data).sum().item()

        self.correct += correct
        self.count += output.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)


class Trainer(object):

    def __init__(self, net, optimizer, train_loader, test_loader):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()

            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc))

    def train(self):
        train_loss = Average()
        train_acc = Accuracy()

        self.net.train()
        for data, label in self.train_loader:
            data = data.cuda()
            label = label.cuda()

            output = self.net(data)
            loss = F.nll_loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            # average the gradients
            # self.average_gradients()
            self.optimizer.step()

            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, label)
        return train_loss, train_acc

    def evaluate(self):
        test_loss = Average()
        test_acc = Accuracy()

        self.net.eval()

        with torch.no_grad():
            for data, label in self.test_loader:
                data = data.cuda()
                label = label.cuda()

                output = self.net(data)
                loss = F.cross_entropy(output, label)

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, label)

        return test_loss, test_acc

    def average_gradients(self):
        world_size = distributed.get_world_size()
        for p in self.net.parameters():
            if p.requires_grad:
                distributed.all_reduce(p.grad.data, op=distributed.ReduceOp.SUM)
                p.grad.data /= float(world_size)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_dataloader(root, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST(
        root, train=True, transform=transform, download=False)
    sampler = DistributedSampler(train_set)

    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler)

    test_loader = data.DataLoader(
        datasets.MNIST(root, train=False, transform=transform, download=False),
        batch_size=batch_size,
        shuffle=False)

    return train_loader, test_loader


def run(args):
    net = Net().cuda()
    for p in net.parameters():
        distributed.all_reduce(p.data)
        p.data /= distributed.get_world_size()
    # net = DistributedDataParallel(net, device_ids=[args.gpu], output_device=args.gpu, use_dgc=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.001)
    optimizer.ratio = 0.2

    train_loader, test_loader = get_dataloader(args.root, args.batch_size)

    trainer = Trainer(net, optimizer, train_loader, test_loader)
    trainer.fit(args.epochs)


def init_process(args):
    torch.cuda.set_device(args.gpu)
    distributed.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        rank=args.rank,
        world_size=args.world_size)


def main():
    global lr
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help='Name of the backend to use.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23458',
        help='URL specifying how to initialize the package.')
    parser.add_argument(
        '-r', '--rank', type=int, help='Rank of the current process.')
    parser.add_argument(
        '-s',
        '--world-size',
        type=int,
        default=1,
        help='Number of processes participating in the job.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.005)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    lr = args.learning_rate
    print(args)

    init_process(args)
    run(args)


if __name__ == '__main__':
    main()

