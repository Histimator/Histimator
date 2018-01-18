A solution for performing maximum likelihood estimation on models built from histogram templates. 

the histimator core directory has a file called Models containing the core HistiModel class.

the model is initialised 

```
model = HistiModel("model name")
```

data is added to the model as
```
model.SetData([list of bin contents])
```

any number of samples are defined as:
```
sig = model.Sample("Signal")
bkg = model.Sample("Background")
```
each of which needs a histogram:
```
sig.SetHisto([list of bin contents])
bkg.SetHisto([list of bin contents])
```

currently the only parameters available are an overal normalisation on these templates.
this is given with a name an initial value (default 1) and a range (default [0.1,10]). Currently no implementation is actually in place to tell Minuit about this range...
```
sig.AddNormFactor("some norm",1,0,3)
```

Finally, the samples must be added to the model.
```
model.AddSample(sig)
model.AddSample(bkg)
```

now a Minuit test result can be recovered.
```
m = model.getFitter()
```

this has various built in plotting functionality. file histimodel_test.py makes the following plot using matplotlib

![normfit](https://github.com/Histimator/Histimator/blob/master/examples/fitnorm.png "norm fit")