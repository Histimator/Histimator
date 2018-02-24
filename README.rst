==========
Histimator
==========


.. image:: https://img.shields.io/pypi/v/histimator.svg
        :target: https://pypi.python.org/pypi/histimator

.. image:: https://img.shields.io/travis/yhaddad/histimator.svg
        :target: https://travis-ci.org/yhaddad/histimator

.. image:: https://readthedocs.org/projects/histimator/badge/?version=latest
        :target: https://histimator.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/yhaddad/histimator/shield.svg
     :target: https://pyup.io/repos/github/yhaddad/histimator/
     :alt: Updates



A solution for performing maximum likelihood estimation on models built from histogram templates.


* Free software: GNU General Public License v3
* Documentation: https://histimator.readthedocs.io.


Features
--------

* TODO

Usage
-----
the histimator core directory has a file called Models containing the core HistiModel class.

the model is initialised::

    from histimator.models import HistiModel
    model = HistiModel("model name")

Each channel is defined as::

     from histimodel.Channel import HistiChannel
     SR = HistiChannel("SignalRegion")

data can be added to the channels as::

     SR.SetData([list of data points])

any number of samples are defined as::

    from histimator.models import HistiSample
    sig = HistiSample("Signal")
    bkg = HistiSample("Background")

each of which needs a histogram::

     sig.SetHisto(numpy.histogram)
     bkg.SetHisto(numpy.histogram)

currently the only parameters available are an overal normalisation on these templates.
this is given with a name an initial value (default 1) and a range (default [0.1,10]). Currently no implementation is actually in place to tell Minuit about this range...::

     sig.AddNorm("some_norm",1,0,3)

Finally, the samples must be added to the channel and this added to the model.::

	 SR.AddSample(sig)
	 SR.AddSample(bkg)
	 model.AddChannel(SR)

This model can now be evaluated using probfit Binned Likelihood function::

     from iminuit import Minuit
     from probfit import BinnedLH
     blh = BinnedLH(model.pdf, data, bins=10, bound=bound, extended=True)
     m = Minuit(blh, some_norm=0.5, error_some_norm=1.5)
     m.migrad()

this has various built in plotting functionality. 

.. image:: https://github.com/Histimator/Histimator/blob/master/examples/fitnorm.png


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
