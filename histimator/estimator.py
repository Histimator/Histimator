# -*- coding: utf-8 -*-
import models
import numpy as np
from iminuit.util import describe


def xlogyx(x, y):
    return np.where(
        z < y,
        +x*np.log((y-x)/x),
        -x*np.log((x-y)/y)
    )

class FakeFuncCode:

    def __init__(self, f, prmt=None, dock=0, append=None):
        # f can either be tuple or function object
        self.co_varnames = describe(f)
        self.co_argcount = len(self.co_varnames)
        self.co_argcount -= dock
        self.co_varnames = self.co_varnames[dock:]

        if prmt is not None:
            for i, p in enumerate(prmt):
                self.co_varnames[i] = p

        if isinstance(append, str):
            append = [append]

        if append is not None:
            old_count = self.co_argcount
            self.co_argcount += len(append)
            self.co_varnames = tuple(
                list(self.co_varnames[:old_count]) +
                append +
                list(self.co_varnames[old_count:]))


class BinnedLH(object):

    def __init__(self, model, data=None, bins=40, weights=None,
                 weighterrors=None, bound=None,
                 badvalue=1000000, extended=False,
                 use_w2=False, nint_subdiv=1):
        """
        Create a Poisson Binned Likelihood object from given PDF **f** and
        **data** (raw points not histogram). Constant term and expected minimum
        are subtracted off (aka. log likelihood ratio). The exact calculation
        will depend on **extended** and **use_w2** keyword parameters.
        .. math::
            \\textrm{BinnedLH} = -\sum_{i \in bins} s_i \\times  \left(  h_i \\times \log (\\frac{E_i}{h_i}) + (h_i-E_i) \\right)
        where
            - :math:`h_i` is sum of weight of data in ith bin.
            - :math:`b_i` is the width of ith bin.
            - :math:`N` is total number of data. :math:`N = \sum_i h_i`.
            - :math:`E_i` is expected number of occupancy in ith bin from PDF
              calculated using average of pdf value at both sides of the bin
              :math:`l_i, r_i`. The definition for :math:`E_i` depends whether
              extended likelihood is requested.
              If extended likelihood is requested (``extended=True``):
              .. math::
                    E_i = \\frac{f(l_i, arg\ldots )+f(r_i, arg \ldots )}{2} \\times b_i
              If extended likelihood is NOT requested (``extended=False``):
              .. math::
                    E_i = \\frac{f(l_i, arg \ldots )+f(r_i, arg \ldots )}{2} \\times b_i \\times N
              .. note::
                    You are welcome to patch this with a the using real area.
                    So that, it's less sensitive to bin size. Last time I check
                    ROOFIT used **f** evaluated at midpoint.
            - :math:`s_i` is a scaled factor. It's 1 if ``sum_w2=False``.
              It's :math:`s_i = \\frac{h_i}{\sum_{j \in \\textrm{bin }i} w_j^2}`
              if ``sum_w2=True``. The factor will scale the statistics to the
              unweighted data.
            .. note::
                You may wonder why there is :math:`h_i-E_i` added at the end
                for each term of the sum. They sum up to zero anyway.
                The reason is the precision near the minimum. If we taylor
                expand the logarithmic term near :math:`h_i\\approx E_i` then
                the first order term will be :math:`h_i-E_i`. Subtracting this
                term at the end gets us the nice pure parabolic behavior for
                each term at the minimum.
        **Arguments**
            - **f** callable object. PDF that describe the data. The parameters
              are parsed using iminuit's ``describe``. The first positional
              arguement is assumed to be independent parameter. For example:
              ::
                    def gauss(x, mu, sigma):#good
                        pass
                    def bad_gauss(mu, sigma, x):#bad
                        pass
            - **data** 1D array of data.
            - **bins** number of bins data should be histogrammed. Default 40.
            - **weights** Optional 1D array of weights. Default ``None``
              (all 1's).
            - **weighterrors** Optional 1D array of weight errors. Default ``None``
              This is usually used for binned datasets where you want to manipulate
              each bin's error. It doesn't make sense if the data is unbinned data.
            - **bound** tuple(min,max). Histogram bound. If ``None`` is given,
              bound is automatically determined from data. Default None.
            - **badvalue** Optional number. The value that will be used to
              represent log(lh) (notice no minus sign). When the likelihood
              is <= 0. This usually indicate your PDF is faraway from minimum
              or your PDF parameter has gone into unphysical region and return
              negative probability density. This should be a large POSITIVE
              number so that iminuit will avoid those points. Default 100000.
            - **extended** Boolean whether this likelihood should be extended
              likelihood or not. Default False.
            - **use_w2** Scale -log likelihood so that to the original
              unweighted statistics. Default False.
            - **nint_subdiv** controls how BinnedLH do the integral to find
              expect number of event in each bin. The number represent the
              number of subdivisions in each bin to do simpson3/8 rule.
              Default 1.
        """
        if isinstance(model, models.HistiModel):
            self.f = model.pdf
            self.binedges = model.binedges
            print "[debug] bin edges : ", model.binedges
            self.func_code = FakeFuncCode(self.f, dock=True)
        else:
            print "ERROR model should be an instance of HistiModels"

        # This is the part where the binning is done, now should be replaced
        # by feeding directly the model and make sure the model has data
        # h, self.edges = np.histogram(data, bins, range=bound, weights=weights)

        print model.data
        print model.binedges

        if hasattr(model, "data"):
            self.data = model.data
            if self.data is None:
                print "error: data is None, please feed the model with data"
            else:
                # use binned data only
                self.h = np.asarray(data)
                self.N = self.h.sum()
        else:
            print "error: model has no attribute data"

        self.use_w2 = use_w2
        self.extended = extended
        if bound is None:
            bound = min(data), max(data)
        self.mymin, self.mymax = bound

        # this is part is commented for the moment and need some thoughs
        # if weights is not None:
        #         if weighterrors is None:
        #             self.w2, _ = np.histogram(data, bins, range=bound,
        #                                       weights=weights * weights)
        #         else:
        #             self.w2, _ = np.histogram(data, bins, range=bound,
        #                                       weights=weighterrors * weighterrors)
        #     else:
        #         self.w2, _ = np.histogram(data, bins, range=bound, weights=None)
        #     self.w2 = float2double(self.w2)
        #     self.midpoints = mid(self.edges)
        #     self.binwidth = np.diff(self.edges)

        #     self.bins = bins
        #     self.badvalue = badvalue
        #     self.ndof = self.bins - (self.func_code.co_argcount - 1)
        #     self.nint_subdiv = nint_subdiv

    def __call__(self, *arg):
        """
        Calculate sum -log(poisson binned likelihood) given positional
        arguments
        """
        self.last_arg = arg

        bw = np.diff(self.binedges)
        th = self.h
        # tm = np.array(
        #     [self.f(self.binedges[i]+bw[i]) for i in range(bw.shape[0])]
        # )
        tm = np.array(
            [ self.binedges[i]+(bw[i]*0.5) for i in range(bw.shape[0])]
        )
        print tm
        if not self.extended:
            return (xlogyx(th, tm*self.N) + (th-tm*self.N)).sum()
        else:
            return 0.0

    # def draw(self, minuit=None, ax = None,
    #          parmloc=(0.05, 0.95), nfbins=200, print_par=True,
    #          args=None, errors=None, parts=False, no_plot=False):
    #     """
    #     Draw comparison between histogram of data and pdf.
    #     **Arguments**
    #         - **minuit** Optional but recommended ``iminuit.Minuit`` object.
    #           If minuit is not ``None``, the pdf will be drawn using minimum
    #           value from minuit and parameters and error will be shown.
    #           If minuit is ``None``, then pdf will be drawn using argument from
    #           the last call to ``__call__``. Default ``None``
    #         - **ax** matplotlib axes. If not given it will be drawn on current
    #           axes ``gca()``.
    #         - **parmloc** location of parameter print out. This is passed
    #           directy to legend loc named parameter. Default (0.05,0.95).
    #         - **nfbins** how many point pdf should be evaluated. Default 200.
    #         - **print_par** print parameters and error on the plot.
    #           Default True.
    #         - **no_plot** Set this to True if you only want the return value
    #     **Returns**
    #     ((data_edges, data_y), (errorp,errorm), (total_pdf_x, total_pdf_y), parts)
    #     """
    #     return plotting.draw_blh(self, minuit=minuit,
    #                              ax=ax, parmloc=parmloc, nfbins=nfbins, print_par=print_par,
    #                              args=args, errors=errors, parts=parts, no_plot=no_plot)

    # def draw_residual(self, minuit=None, ax = None, parmloc=(0.05,0.95),
    #                   print_par=False, args=None, errors=None, norm=False,
    #                   **kwargs):
    #     """
    #     Draw difference between data and pdf.
    #     **Arguments**
    #         - **minuit** Optional but recommended ``iminuit.Minuit`` object.
    #           If minuit is not ``None``, the pdf will be drawn using minimum
    #           value from minuit and parameters and error will be shown.
    #           If minuit is ``None``, then pdf will be drawn using argument from
    #           the last call to ``__call__``. Default ``None``
    #         - **ax** matplotlib axes. If not given it will be drawn on current
    #           axes ``gca()``.
    #         - **parmloc** location of parameter print out. This is passed
    #           directy to legend loc named parameter. Default (0.05,0.95).
    #         - **print_par** print parameters and error on the plot.
    #           Default True.
    #         - **norm** If True, draw difference normalized by error
    #           Default False.
    #         - **kwargs** Passed to :meth:`probfit.plotting.draw_residual`
    #     """
    #     return plotting.draw_residual_blh(self, minuit=minuit,
    #                                       ax=ax, parmloc=parmloc, print_par=print_par,
    #                                       args=args, errors=errors, norm=norm,
    #                                       **kwargs)

    # def default_errordef(self):
    #     return 0.5

    # def show(self, *arg, **kwd):
    #     """
    #     Same thing as :meth:`draw`. But show the figure immediately.
    #     .. seealso::
    #         :meth:`draw` for arguments.
    #     """
    #     ret = self.draw(*arg, **kwd)
    #     plt.show()
    #     return ret
