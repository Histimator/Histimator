def test_simple():
    from histimator import HistiModel, HistiSample, HistiChannel, BinnedLH
    from iminuit import Minuit
    import numpy as np

    m = HistiModel('model')
    hist_sig = np.asarray([1,2,3,4])
    hist_bkg = np.asarray([5,5,5,5])
    signal = HistiSample('signal')
    binedges = [0,1,2,3,4]
    signal.SetHisto((hist_sig,binedges))
    signal.AddNorm("mu",1.0,0,3)
    background = HistiSample('background')
    background.SetHisto((hist_bkg, binedges))
    chan = HistiChannel("simple")
    chan.AddSample(signal)
    chan.AddSample(background)
    data = hist_sig*.5 +hist_bkg

    chan.SetData(data)
    m.AddChannel(chan)
    blh = BinnedLH(m)
    params = m.Parameters()
    minimiser = Minuit(blh, **params)
    minimiser.migrad()
    minimiser.hesse()
    minimiser.minos()

    assert abs(.5 - minimiser.values['mu']) < .5
    print ("Simple Model Success!")
    
def test_med():
    from histimator import HistiModel, HistiSample, HistiChannel, BinnedLH
    from iminuit import Minuit
    import numpy as np

    m = HistiModel('model')
    hist_sig = np.linspace(1,50,50)

    signal = HistiSample('signal')
    binedges = np.linspace(0,50,51)
    signal.SetHisto((hist_sig,binedges))
    signal.AddNorm("mu",1.0,0,3)
    background1 = HistiSample('background1')
    background1.SetHisto((np.ones(50)*3, binedges))
    background1.AddOverallSys("NormSyst",0.9,1.1)
    background2 = HistiSample('background2')
    background2.SetHisto((np.ones(50)*1, binedges))
    var = hist_sig *.01
    background2.AddHistoSys("HistoSyst",np.ones(50)-var,np.ones(50)+var)
    chan = HistiChannel("simple")
    chan.AddSample(signal)
    chan.AddSample(background1)
    chan.AddSample(background2)
    data = hist_sig*.5 +np.ones(50)*4

    chan.SetData(data)
    m.AddChannel(chan)
    blh = BinnedLH(m)
    params = m.Parameters()
    minimiser = Minuit(blh, **params)
    minimiser.migrad()
    minimiser.hesse()
    minimiser.minos()

    assert abs(.5 - minimiser.values['mu']) < .5
    print ("Simple Model Success!")
    
