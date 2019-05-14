import torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
import sys, os, argparse, logging, datetime
import util
from util import info
from collections import deque

class Trainer(object):
	def __init__(self, trainloader, testloader, net, **kwargs):
		self.trainloader = train
		self.testloader = test
		self.net = net
		self.loss_fn = nn.MSELoss()
		self.test_every = kwargs.get('test_every', 100)
		self.print_every = kwargs.get('print_every', 100)
		self.save_model_fn = kwargs.get('save_model', save_model)
		self._tics = []

	def run(self, n_epochs):
		self.net.train()
		self.iter_i       = 0
		self.best_test    = 2*10
		self.test_losses  = []
		self.train_losses = []
		self.batch_losses = deque(maxlen=1000)
		for self.epoch_i in range(n_epochs):
			self.train_epoch()
		
	def train_epoch(self):
		self.tic()
		for self.batch_i, batch in enumerator:
			self.train_batch(batch)
			if self.iter_i % self.test_every:
				self.test()
			self.iter_i += 1
		self.toc()
		self.log_epoch_loss()
		self.epoch_i += 1

	def train_batch(self, batch):
		self.tic()
		self.optim.zero_grad()
		loss = self._loss(batch)
		loss.backward()
		optim.step()
		self.batch_losses.append(loss.item())
		tictoc = self.toc()
		if self.iter_i % self.print_every:
			self.print('TRAIN', loss.item(), tictoc)

	def _loss(self, batch):
		X = batch['X']
		y = batch['y']
		hyp = self.net(X)
		return self.loss_fn(hyp, y)

	def _loss_for_dataloader(self, dataloader, mode):
		self.tic()
		self.net.eval()
		self.net.zero_grad()
		sizes  = np.array([len(batch) for batch in dataloader])
		losses = np.array([self._loss(batch).item() for batch in dataloader])
		loss   = np.mean(losses / sizes)
		self.net.train()
		tictoc = self.toc()
		self.print(mode, loss, tictoc)
		return loss

	def test(self):
		loss = self._loss_for_dataloader(self.testloader, 'TEST')
		if loss < self.best_test:
			self.best_test = loss
			self.save_model_fn()
		self.test_losses.append(loss)

	def log_epoch_loss(self):
		loss = self._loss_for_dataloader(self.trainloader, 'EPOCH')
		self.train_losses.append(loss)

	def tic(self):
		self._tics.append(datetime.datetime.now())
	
	def toc(self):
		return (datetime.datetime.now() - self._tics.pop())

	def print(self, mode, loss, tictoc):
		info('%6s  %4d:%-7d %e %s' % (mode, self.epoch_i, self.iter_i, loss, str(tictoc)))

def save_model():
	torch.save({
		'state_dict': master_net.state_dict,
		'optim_dict': trainer.optim.state_dict,
		'best_test': trainer.best_test,
		'iter_i': trainer.iter_i,
		'epoch_i': trainer.epoch_i,
		'batch_i': trainer.batch_i,
		}, args.save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train')
	parser.add_argument('--test')
	parser.add_argument('-l', '--log_path', '--l')
	parser.add_argument('-m', '--load_path', '--m')
	parser.add_argument('-s', '--save_path', '--s')
	parser.add_argument('-a', '--arch')
	parser.add_argument('--print_every', type=int, default=100)
	parser.add_argument('--test_every', type=int, default=100)
	parser.add_argument('--print_every', type=int, default=100)
	parser.add_argument('--train_bs', type=int, default=64)
	parser.add_argument('--test_bs', type=int, default=64)
	parser.add_argument('-n', '--n_saes', type=int)
	parser.add_argument('--ep', type=int, help='Max epochs to train')
	args = parser.parse_args()
	assert args.save_path is not None

	util.init_logger(args.log_path)
	logger.info(args)

	# Make train dataloader
	if args.train is not None:
		trainset = dataset.XDataset( args.train )
		trainloader = DataLoader( trainset, batch_size=args.train_bs, shuffle=True )
	# Make test dataloader
	if args.test is not None:
		testset = dataset.XDataset( args.test )
		testloader = DataLoader( testset, batch_size=args.test_bs )
	# Build model
	master_net = network.Net(arch=args.arch)
	if os.path.exists(args.model_path):
		dump = torch.load(args.model_path)
		master_net.load_state_dict( dump['state_dict'] )
	net = master_net.subnet(n_saes)
	# Build optimizer
	optim   = torch.optim.SGD( net, lr=args.lr )
	if os.path.exists(args.model_path):
		optim.load_state_dict( dump['optim_state_dict'] )
	# Build trainer
	trainer = Trainer(trainloader, testloader, net,
		test_every=args.test_every, print_every=args.print_every)
	# Train
	trainer.run(args.ep)
