from histimator.fcn import loglikelihood
from matplotlib import pyplot as plt

nll = [-loglikelihood(5,i*.1) for i in range(1,100)]
minnll = 100
for l in range(len(nll)):
    if nll[l] <= min(nll): 
        minnll = l+1
plt.figure(figsize=(8,6))
plt.plot([i*.1 for i in range(1,100)],nll,lw=2,alpha=0.5,linestyle='--',label=r'$-\ln\mathcal{L}(\theta=5)$')
plt.text(8.2, 10.5, r'minimum at {}'.format(minnll*.1))
plt.title('negative log likelihood distribution')
plt.xlabel(r'possible values for theta $\theta$')
plt.ylabel(r'$-\ln\mathcal{L}(\theta)$')
plt.legend()
plt.show()


from histimator.fcn import binnedloglikelihood


def get_histogram(template,n_events):
    n_bins = len(template)
    return [template[i]*n_events/sum(template) for i in range(n_bins)]

template = [2.,5.,3.]
data_template = get_histogram(template,30)
model_template = get_histogram(template,30)
print data_template, model_template

template = [1.,1.,1.]
data_template = get_histogram(template,30)
nll = [-binnedloglikelihood(data_template,get_histogram(template,i*.1)) for i in range(1,1000)]
minnll = 100000
for l in range(len(nll)):
    if nll[l] <= min(nll): 
        minnll = l+1
plt.figure()
plt.plot([i*.1 for i in range(1,1000)],nll,lw=2,alpha=0.5,linestyle='--',label=r'$-\ln\mathcal{L}(\theta$'+'={})'.format(sum(data_template)))
plt.title('negative log likelihood distribution')
plt.xlabel(r'possible values for theta $\theta$')
plt.ylabel(r'$-\ln\mathcal{L}(\theta)$')
plt.legend()
plt.text(70, 80, r'minimum at {}'.format(minnll*.1))
plt.show()


def get_histogram(a,n_bins,n_events):
    if a!=0:
        shape = [(i*a) for i in range(n_bins)]
    else:
        shape = [1 for i in range(n_bins)]
    return [s+(n_events/n_bins)-(sum(shape)/n_bins) for s in shape]

data_template = get_histogram(0.5,4,50)
nll = [-binnedloglikelihood(data_template,get_histogram(i*0.001-1,4,50)) for i in range(2000)]
minnll = 100.
for l in range(len(nll)):
    if nll[l] <= min(nll): 
        minnll = l
#print minnll*0.01
plt.figure()
plt.plot([i*0.001-1 for i in range(2000)],nll,lw=2,alpha=0.5,linestyle='--',label=r'$-\ln\mathcal{L}(\theta$)')
plt.title('negative log likelihood distribution')
plt.xlabel(r'possible values for theta $\theta$')
plt.ylabel(r'$-\ln\mathcal{L}(\theta)$')
plt.legend()
plt.show()

import numpy as np
new_array = np.empty((500, 20))
minnll = 10000
mini = 0
minj = 0
for i in range(500):
    for j in range(20):
        nll = -binnedloglikelihood(data_template,get_histogram(i*0.01-3,4,40+j))
        new_array[i,j] = nll
        if nll < minnll:
            minnll = nll
            mini = i
            minj = j

from matplotlib import cm
X,Y = np.meshgrid([i*0.01-3 for i in range(500)], [40+i for i in range(20)])
fig, ax = plt.subplots()
cs = ax.contourf(X, Y, new_array.transpose(), cmap=cm.PuBu_r)
cbar = fig.colorbar(cs)
plt.text(.5, 50, 'x')
plt.show()
