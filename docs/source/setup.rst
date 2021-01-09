Getting Started
---------------

You can install pip package or Anaconda package for this project.

Recommended: Set up ``conda`` environment with provided ``.yml`` file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Updated September 2020*

We recommend seting up a fresh Python virutal environment in which to
use ``solar-data-tools``. We recommend using the
`Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`__
package management system, and creating an environment with the
environment configuration file named ``pvi-user.yml``, provided in the
top level of this repository. This will install the ``solar-data-tools``
package as well.

Please see the Conda documentation page, "`Creating an environment from
an environment.yml
file <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`__\ "
for more information.

Installation
~~~~~~~~~~~~

If you are using pip:

.. code:: sh

    $ pip install statistical-clear-sky

As of February 11, 2019, it fails because scs package installed as a
dependency of cxvpy expects numpy to be already installed. `scs issue
85 <https://github.com/cvxgrp/scs/issues/85>`__ says, it is fixed.
However, it doesn't seem to be reflected in its pip package. Also, cvxpy
doesn't work with numpy version less than 1.16. As a work around,
install numpy separatly first and then install this package. i.e.

.. code:: sh

    $ pip install 'numpy>=1.16'
    $ pip install statistical-clear-sky

If you are using Anaconda, the problem described above doesn't occur
since numpy is already installed. And during statistical-clear-sky
installation, numpy is upgraded above 1.16:

.. code:: sh

    $ conda install -c slacgismo statistical-clear-sky

Solvers
^^^^^^^

The default convex solver included with cvxpy is ECOS, which is open
source. However this solver tends to fail on problems with >1000
variables, as it does not work for this algorithm.

So, the default behavior of the code is to use the commercial Mosek
solver. Thus, we encourage you to install it separately as below and
obtain the license on your own.

-  `mosek <https://www.mosek.com/resources/getting-started/>`__ - For
   using MOSEK solver.

   If you are using pip:

   .. code:: sh

       $ pip install -f https://download.mosek.com/stable/wheel/index.html Mosek

   If you are using Anaconda:

   .. code:: sh

       $ conda install -c mosek mosek==8.1.43

Academic licenses are available for free here:
https://www.mosek.com/products/academic-licenses/
