from histimator.Models import HistiModel
from matplotlib import pyplot as plt

testModel = HistiModel("test model")
data = [11.,12.,14.,17.,14.,12.,11.]
flat_background = [10,10,10,10,10,10,10]
peak = [2.,4.,8.,14.,8.,4.,2.]
testModel.SetData(data)
bkg = testModel.Sample("background")
bkg.SetHisto(flat_background)
testModel.AddSample(bkg)
sig = testModel.Sample("sig")
sig.SetHisto(peak)
sig.AddNormFactor("some norm",1,0,3)
testModel.AddSample(sig)

m = testModel.getFitter()
m.migrad()

print m.values

x,y = m.profile('x',subtract_min=True)
plt.plot(x,y,alpha=0.5,linestyle='--',label=r'$-\ln\mathcal{L}(\theta$)')
plt.title('negative log likelihood distribution')
plt.xlabel(r'possible values for theta $\theta$')
plt.ylabel(r'FCN')
plt.legend()
plt.show()
