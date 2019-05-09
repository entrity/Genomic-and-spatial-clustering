from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

COL_PCA = 0
COL_DIM = 1
COL_KM  = 2
COL_KNN = 3
COL_LAP = 4 # 0 for n, 1 for rw
COL_KLD = 5
COL_POS = 6
COL_NEG = 7
COL_SCR = 8
COL_BNZ = 9
COL_PNZ = 10
COL_NNZ = 11
COL_SPA = 12
labels  = [
	'PCA', 'DIM', 'KM', 'KNN', 'LAP',
	'KLD',
	'POS', 'NEG', 'SCR',
	'BNZ', 'PNZ', 'NNZ',
	'SPA',
]

res = np.loadtxt('results.tsv')

R_NORM = res[:,COL_LAP] == 1.
R_RWLK = res[:,COL_LAP] == 0.
res_norm = res[R_NORM]
res_rwlk = res[R_RWLK]

def p3d(x, y, z):
	if isinstance(x, int) and isinstance(y, int) and isinstance(z, int):
		x = res[:,x]
		y = res[:,y]
		z = res[:,z]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x, y, z)
	for angle in range(0, 360):
		ax.view_init(30, angle)
		plt.draw()
		plt.pause(.001)

def p2d(x, y, no_show=False):
	if isinstance(x, int) and isinstance(y, int):
		x = res[:,x]
		y = res[:,y]
	fig = plt.figure()
	plt.scatter(x, y)
	if not no_show:
		plt.show()

def cmp_scrs(row_a, row_b, col_a, col_b, xlbl, ylbl, legend, subplot=111):
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	plt.xlabel(xlbl)
	plt.ylabel(ylbl)
	ax.scatter(res[row_a,col_a], res[row_a,col_b], c='b', label=legend[0])
	ax.scatter(res[row_b,col_a], res[row_b,col_b], c='g', label=legend[1])
	ax.legend()

def good_plots():
	cmp_scrs(res[:,COL_KM]==12, res[:,COL_KM]==24, COL_KLD, COL_SCR, 'kl div', '"score"', ['12 clusters', '24 clusters'], 121)
	plt.show()
	p3d(COL_POS, COL_NEG, COL_SCR)

# good_plots()

def disp(arr):
	for row in arr:
		lap = 'N' if row[4] else 'RW'
		spa = '' if row[12] else '+'
		print('%f\t%f\t%d\t%d\t%d\t%d\t%s\t%s' % (row[5], row[8], row[0], row[1], row[2], row[3], lap, spa))

# python plot/plot_results.py | sed -e 's/\t/ \& /g' -e 's/$/ \\/'
def topnbottom(arr, sort_col):
	order = np.argsort(arr[:,sort_col])
	# print(arr[order,sort_col])
	# return
	inorder = arr[order,:]
	print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % ('KL div', 'ACC', 'PCA', 'DIM', 'KM', 'KNN', 'LAP', 'SPATIAL'))
	disp(inorder[:5,:])
	print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % ('KL div', 'ACC', 'PCA', 'DIM', 'KM', 'KNN', 'LAP', 'SPATIAL'))	
	disp(inorder[-5:,:])

# print('Top and bottom by KL Divergence')
# topnbottom(res, COL_KLD)
# print('Top and bottom by ACC')
# topnbottom(res, COL_SCR)

def latex_topnbottom():
	cols = [5, 8, 0, 1, 2, 3, 4, 12]
	getters = {
		COL_SPA: lambda v : '' if v else '+',
		COL_LAP: lambda v : 'N' if v else 'RW',
		COL_KM : int,
		COL_SCR: int,
		COL_PCA: lambda v : '' if v else '+',
		COL_KNN: int,
		COL_DIM: lambda v : int(v) if v else '-',
	}
	def disp_rows(rows):
		g = lambda col : getters[col](row[col]) if col in getters else row[col]
		for row in rows:
			print(' & '.join([str(g(c)) for c in cols]) + r'\\')
	def disp_topnbottom(sort_col):
		print(' & '.join([ labels[i] for i in cols ]) + r' \\\hline')
		order = np.argsort(res[:,sort_col])
		inorder = res[order,:]
		disp_rows(inorder[:5,:])
		print(' & '.join(['...' for i in cols]) + r'\\')
		disp_rows(inorder[-5:,:])

	print(r'\multicolumn{%d}{c}{%s} \\\hline' % (len(cols), 'Best and Worst by ACC'))
	disp_topnbottom(COL_SCR)
	print(r'\multicolumn{%d}{c}{%s} \\\hline' % (len(cols), 'Best and Worst by KLD'))
	disp_topnbottom(COL_KLD)
latex_topnbottom()

def plot_kldiv_vs_acc():
	sel = res
	sel = sel[sel[:,COL_KNN] == 4]
	scr = lambda col, val : sel[ sel[:,col] == val, COL_SCR ]
	kld = lambda col, val : sel[ sel[:,col] == val, COL_KLD ]
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	ax.scatter(scr(COL_KM,12), kld(COL_KM,12), c='b', label='k=12')
	ax.scatter(scr(COL_KM,24), kld(COL_KM,24), c='g', label='k=24')
	plt.xlabel('ACC')
	plt.ylabel('KL divergence')
	ax.legend()
	plt.show()
# cmp_scrs(R_NORM, R_RWLK, COL_KLD, COL_SCR, 'kl div', '"score"', ['norm lap', 'rwalk lap'])
# cmp_scrs(res[:,COL_PCA]==0, res[:,COL_PCA]==50, COL_KLD, COL_SCR, 'kl div', '"score"', ['no PCA', 'PCA dim = 50'], 121)


# p2d(res[:,COL_SCR], res[:,COL_KLD])

# print(sel.shape)
# sel = sel[sel[:,COL_LAP] == 0.]
# print(sel.shape)
# sel = sel[sel[:,COL_KM] == 12]
# sel = sel[sel[:,COL_PCA] == 0]
# import IPython; IPython.embed(); # to do: delete
# sel = sel[sel[:,COL_KNN] == 16]



# p3d(COL_PCA, COL_DIM, COL_KM)
# p3d(COL_KNN, COL_DIM, COL_KM)