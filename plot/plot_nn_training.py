import sys, os
import numpy as np
import matplotlib.pyplot as plt

FIN = sys.argv[1]

ep_data = np.fromregex(FIN, 'EPOCH\s+(\d+):(\d+)\s+(\S+)\s+.*', [
	('ep', np.int32), ('iter', np.int64), ('loss', np.double)])
test_data = np.fromregex(FIN, 'TEST\s+(\d+):(\d+)\s+(\S+)\s+.*', [
	('ep', np.int32), ('iter', np.int64), ('loss', np.double)])

S = np.where( ep_data['ep'] == 0 )[-1].item()

fig = plt.figure()
plt.plot(ep_data['ep'][S:], ep_data['loss'][S:], c='m')
plt.plot(test_data['ep'][S:], test_data['loss'][S:], c='b')
plt.show()
