StatisticalClearSky
===================

|PyPI release| |Anaconda Cloud release| |Build Status| |codecov|

*Statistical estimation of a clear sky signal from PV system power data*

This project implements an algorithm based on `Generalized Low Rank
Models <https://stanford.edu/~boyd/papers/glrm.html>`__ for estimating
the output of a solar PV system under clear sky or "cloudless"
conditions, given only measured power as an input. Noteably, no system
configuration information, modeling parameters, or correlated
environmental data are required. You can read more about this work in
these two papers [`1 <https://arxiv.org/abs/1907.08279>`__,
`2 <https://ieeexplore.ieee.org/abstract/document/8939335>`__].

We actually recommend that users generally not invoke this software
directly. Instead, we recommend using the API provided by `Solar Data
Tools <https://github.com/slacgismo/solar-data-tools>`__.

Contents
^^^^^^^^

.. toctree::
   :maxdepth: 1

   setup
   usage
   jupyter_notebook
   tests
   contributing
   versioning
   authors
   license
   references
   acknowledgments




Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


.. |PyPI release| image:: https://img.shields.io/pypi/v/statistical-clear-sky.svg
   :target: https://pypi.org/project/statistical-clear-sky/
.. |Anaconda Cloud release| image:: https://anaconda.org/slacgismo/statistical-clear-sky/badges/version.svg
   :target: https://anaconda.org/slacgismo/statistical-clear-sky
.. |Build Status| image:: https://travis-ci.com/tadatoshi/StatisticalClearSky.svg?branch=development
   :target: https://travis-ci.com/tadatoshi/StatisticalClearSky
.. |codecov| image:: https://codecov.io/gh/tadatoshi/StatisticalClearSky/branch/development/graph/badge.svg
   :target: https://codecov.io/gh/tadatoshi/StatisticalClearSky
