from histimator.benchmark import Benchmark

b = Benchmark()
x, y1, y2 = b.Time()

from matplotlib import pyplot as plt

plt.errorbar(x, y1[:,0].tolist(),y1[:,1].tolist(), color='b',linestyle='None', marker='s',label= 'scipy bfgs')
plt.errorbar(x, y2[:,0].tolist(),y2[:,1].tolist(), color='r',linestyle='None', marker='^',label= 'iMinuit Migrad')
plt.legend()
plt.xlabel('number of samples')
plt.ylabel('time to fit in seconds')

plt.show()
 
