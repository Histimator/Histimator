

class HistoPdf:
    def __init__(self, pars, channel):
        assert len(samples) > 0 and samples[0].Histo
        modelhist = [0 for i in range(len(self.Samples[0].Histo))]
        for sample in channel.Samples:
            sample_hist = channel.Samples[sample].Evaluate(pars)
            for bin in range(len(modelhist)):
                modelhist[bin] = modelhist[bin]+sample_hist[bin]
        self.pdf = modelhist
    

