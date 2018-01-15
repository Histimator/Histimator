import math

def loglikelihood(observed, expected):
    return observed * math.log(expected) - expected

def binnedloglikelihood(observed, expected):
    assert len(observed) == len(expected)
    return sum([observed[bin_i]*math.log(expected[bin_i]) for bin_i in range(len(observed))]) - sum(expected)

def binnedChi2(observed, expected):
    assert len(observed) == len(expected)
    return sum([((observed[bin_i] - expected[bin_i])**2)/expected[bin_i] for bin_i in range(len(observed))])
