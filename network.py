import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import util
from util import info

# Ref basic operations
# https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/
# Ref KLDivLoss
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html

USE_CUDA = torch.cuda.is_available()

def assert_shape(t, shape):
	assert len(t.shape) == len(shape), (t is not None, t.shape)
	for i, v in enumerate(shape):
		assert t.shape[i] == v, t.shape

class Net(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.arch = None
		if 'state_dict' in kwargs:
			self.load_state_dict(kwargs['state_dict'])
		if 'arch' in kwargs:
			self._init_arch(kwargs['arch'])
	def _init_arch(self, arch):
		if self.arch is None:
			self.arch = arch
			dims = np.array(arch)
			dims = np.hstack((dims, dims[-2::-1]))
			self.dims = tuple(zip(dims[:-1], dims[1:]))
			self.layers = nn.Sequential(
				*[nn.Linear(*pair) for pair in self.dims]
				)
		elif not np.array_equal(self.arch, arch):
			raise Exception('Trying to set arch to a new shape after arch has already been set')
	def _build_block(self, index):
		n = len(self.dims)
		seq = nn.Sequential()
		seq.add_module('dropout', nn.Dropout(0.2))
		seq.add_module('linear', self.layers[index])
		if index not in [n-1, n//2-1]:
			seq.add_module('relu', nn.ReLU())
		return seq
	def subnet(self, n_saes):
		n = len(self.dims)
		print('Requested %d' % n_saes)
		if n_saes > n // 2:
			raise Exception('Requested %d SAEs, but only %d are present' % (n_saes, len(self.arch)))
		seq = nn.Sequential()
		for i in range(n_saes):
			seq.add_module(str(i), self._build_block(i))
		for i in range(n-n_saes, n):
			seq.add_module(str(i), self._build_block(i))
		return seq

if __name__ == '__main__':
	x = Net('tmp.pth', arch=[16, 8, 4])
	print(x.arch)
	print(x.dims)
	print(x.subnet(1))
	print(x.subnet(2))
	print(x.subnet(3))
