A solution for performing maximum likelihood estimation on models built from histogram templates. 

the histimator core directory has a file called Models containing the core HistiModel class.

the model is initialised 

```
from histimator.Model import HistiModel
model = HistiModel("model name")
```
Each channel is defined as:
```
from histimodel.Channel import HistiChannel
SR = HistiChannel("SignalRegion")
```

data can be added to the channels as
```
SR.SetData([list of bin contents])
```

any number of samples are defined as:
```
from histimator.Sample import HistiSample
sig = HistiSample("Signal")
bkg = HistiSample("Background")
```

each of which needs a histogram:
```
sig.SetHisto([list of bin contents])
bkg.SetHisto([list of bin contents])
```

currently the only parameters available are an overal normalisation on these templates.
this is given with a name an initial value (default 1) and a range (default [0.1,10]). Currently no implementation is actually in place to tell Minuit about this range...
```
sig.AddNorm("some norm",1,0,3)
```

Finally, the samples must be added to the channel and this added to the model.
```
SR.AddSample(sig)
SR.AddSample(bkg)
model.AddChannel(SR)
```

The negative log likelihood can be evaluated for any point in parameter space:
nll = model.NLL(2.)

now a Minuit test result can be recovered.
```
from iminuit import Minuit
m = Minuit(model.NLL)
m.migrad
```

this has various built in plotting functionality. file test_HF_struct.py makes the following plot using matplotlib

![normfit](https://github.com/Histimator/Histimator/blob/master/examples/fitnorm.png "norm fit")

## UML
![uml](https://github.com/Histimator/Histimator/blob/master/examples/HistimatorUML.png "uml")
