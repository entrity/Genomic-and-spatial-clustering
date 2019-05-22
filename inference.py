import torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
import sys, os, argparse, logging, datetime
import util
from util import info
import network, dataset
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Location of dataset')
parser.add_argument('-m', '--load_path', '--m', help='Location of model')
parser.add_argument('-s', '--save_path', '--s', help='Where to save embeddings')
parser.add_argument('-a', '--arch')
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('-n', '--n_saes', type=int)
args = parser.parse_args()
assert args.save_path is not None

# Make dataloader
dataset = dataset.XDataset( args.data )
dataloader = DataLoader( dataset, batch_size=args.bs, shuffle=False )

# Build model
master_net = network.Net(arch=[int(x) for x in args.arch.split()])
dump = torch.load(args.load_path)
master_net.load_state_dict( dump['state_dict'] )
info('Loaded from %s' % args.load_path)

# Get encoder only
encoder = master_net.get_encoder()

# Get output embeddings
embeddings = torch.cat([ encoder(batch) for batch in dataloader ])

# Save embeddings
np.save( args.save_path, embeddings.detach().numpy() )
