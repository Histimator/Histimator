import math

def loglikelihood(observed, expected):
    return observed * math.log(expected) - expected

def binnedloglikelihood(observed, expected):
    assert len(observed) == len(expected)
    return sum([observed[bin_i]*math.log(expected[bin_i]) for bin_i in range(len(observed))]) - sum(expected)

def binnedChi2(observed, expected):
    assert len(observed) == len(expected)
    return sum([((observed[bin_i] - expected[bin_i])**2)/expected[bin_i] for bin_i in range(len(observed))])

def get_histogram(a,n_bins,n_events):
    if a!=0:
        shape = [(i*a) for i in range(n_bins)]
    else:
        shape = [1 for i in range(n_bins)]
    return [s+(n_events/n_bins)-(sum(shape)/n_bins) for s in shape]

def llhfcn(x):
    data_template = get_histogram(0.5,4,50)
    variation = get_histogram(0.5,4,x)
    return -binnedloglikelihood(data_template, variation)

def llhfcn2(x,y):
    data_template = get_histogram(0.5,4,50)
    variation = get_histogram(y,4,x)
    return -binnedloglikelihood(data_template, variation)
