Jupyter Notebook Examples
-------------------------

Alternatively, you can clone this repository (GIT) and execute the
example codes under notebooks folder.

Simplest way to install dependencies if you are using pip is by

.. code:: sh

    $ pip install -r requirements.txt

As mentioned in the section, "Getting Started" above, as of February 11,
2019, it fails because scs package installed as a dependency of cxvpy
expects numpy to be already installed. `scs issue
85 <https://github.com/cvxgrp/scs/issues/85>`__ says, it is fixed.
However, it doesn't seem to be reflected in its pip package. Also, cvxpy
doesn't work with numpy version less than 1.16. As a work around,
install numpy separatly first and install the other packages using
requirements.txt. i.e.

.. code:: sh

    $ pip install 'numpy>=1.16'
    $ pip install -r requirements.txt
