# StatisticalClearSky

[![Build Status](https://travis-ci.com/tadatoshi/StatisticalClearSky.svg?branch=development)](https://travis-ci.com/tadatoshi/StatisticalClearSky)
[![codecov](https://codecov.io/gh/tadatoshi/StatisticalClearSky/branch/development/graph/badge.svg)](https://codecov.io/gh/tadatoshi/StatisticalClearSky)

Statistical estimation of a clear sky signal from PV system power data

## Getting Started

You can install pip package or Anaconda package for this project.
Alternatively, you can clone this repository (GIT) and execute the example codes under notebooks folder.

### Prerequisites

When you install this project as PIP package, dependencies are automatically installed.

As of February 11, 2019, it fails because scs package installed as a dependency of cxvpy expects numpy to be already installed.
[scs issue 85](https://github.com/cvxgrp/scs/issues/85) says, it is fixed.
However, it doesn't seem to be reflected in its pip package.
Also, cvxpy doesn't work with numpy version less than 1.16.
As a work around, install numpy separatly first and then install this package.
i.e.
```sh
$ pip install numpy>=1.16
```

When you use this project in any other ways, the following instruction can be useful.

Simplest way to install dependencies if you are using pip is by

```sh
$ pip install -r requirements.txt
```

As of February 11, 2019, it fails because scs package installed as a dependency of cxvpy expects numpy to be already installed.
[scs issue 85](https://github.com/cvxgrp/scs/issues/85) says, it is fixed.
However, it doesn't seem to be reflected in its pip package.
Also, cvxpy doesn't work with numpy version less than 1.16.
As a work around, install numpy separatly first and install the other packages using requirements.txt. i.e.
```sh
$ pip install numpy>=1.16
$ pip install -r requirements.txt
```

In case, you run example codes under notebooks folder in Jupyter notebook, especially run in Anaconda environment, you may need to take care of the following.   

* [cvxpy](https://www.cvxpy.org/) - For Convex optimization.

    If you are using pip:
    ```sh
    $ pip install cvxpy
    ```

    If you are using Anaconda:
    ```sh
    $ conda install -c conda-forge lapack
    $ conda install -c cvxgrp cvxpy
    ```

* [mosek](https://www.mosek.com/resources/getting-started/) - For using MOSEK solver.

    An example code in Jupyter notebook depends on it.
    Thus, this package is necessary only when running the example code.

    If you are using pip:
    ```sh
    $ pip install -f https://download.mosek.com/stable/wheel/index.html Mosek
    ```

    If you are using Anaconda:
    ```sh
    $ conda install -c mosek mosek
    ```

* [cassandra-driver](http://datastax.github.io/python-driver/index.html) - [Optional] - For accessing Cassandra database.

    An example code in Jupyter notebook depends on it.
    Thus, this package is necessary only when running the example code.

    If you are using pip:
    ```sh
    $ pip install cassandra-driver
    ```

    If you are using Anaconda:
    ```sh
    $ conda install -c conda-forge cassandra-driver
    ```

#### Solvers

By default, ECOS solver is used, which is supported by cvxpy because it is Open Source.

However, it is found that Mosek solver is more stable. Thus, we encourage you to install it separately as below and obtain the license on your own.

* [mosek](https://www.mosek.com/resources/getting-started/) - For using MOSEK solver.

    If you are using pip:
    ```sh
    $ pip install -f https://download.mosek.com/stable/wheel/index.html Mosek
    ```

    If you are using Anaconda:
    ```sh
    $ conda install -c mosek mosek
    ```

### Installing

If you are using pip:

```
$ pip install statistical-clear-sky
```

As of February 11, 2019, it fails because scs package installed as a dependency of cxvpy expects numpy to be already installed.
[scs issue 85](https://github.com/cvxgrp/scs/issues/85) says, it is fixed.
However, it doesn't seem to be reflected in its pip package.
Also, cvxpy doesn't work with numpy version less than 1.16.
As a work around, install numpy separatly first and then install this package.
i.e.
```
$ pip install numpy>=1.16
$ pip install statistical-clear-sky
```

If you are using Anaconda:

```
$ conda install -c tadatoshi statistical-clear-sky
```

## Usage

#### Example 1: Simplest example with the fewest number of input parameters.

Using default solver (Open Source solver: ECOS)

```
import numpy as np
from statistical_clear_sky.algorithm.iterative_clear_sky\
 import IterativeClearSky

# Usually read from a CSV file or a database with more data,
# covering 1 day (column) and a few years (row):
power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                            [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                            [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                            [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

iterative_clear_sky = IterativeClearSky(power_signals_d)

iterative_clear_sky.minimize_objective()

clear_sky_signals = iterative_clear_sky.clear_sky_signals()
degradation_rate = iterative_clear_sky.degradation_rate()
```

#### Example 2: Using a different solver.

The default solver ECOS is not stable with large set of input data.
The following example shows how to specify to use Mosek solver.

```
import numpy as np
from statistical_clear_sky.algorithm.iterative_clear_sky\
 import IterativeClearSky
from statistical_clear_sky.solver_type import SolverType

# Usually read from a CSV file or a database with more data,
# covering 1 day (column) and a few years (row):
power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                            [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                            [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                            [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

iterative_clear_sky = IterativeClearSky(power_signals_d,
                                        solver_type=SolverType.mosek)

iterative_clear_sky.minimize_objective()

clear_sky_signals = iterative_clear_sky.clear_sky_signals()
degradation_rate = iterative_clear_sky.degradation_rate()
```

#### Example 3: Setting rank for Generalized Low Rank Modeling.

By default, rank of low rank matrices is specified to be 4.
You can change it by specifying rank_k keyword argument (in constructor).

```
import numpy as np
from statistical_clear_sky.algorithm.iterative_clear_sky\
 import IterativeClearSky

# Usually read from a CSV file or a database with more data,
# covering 1 day (column) and a few years (row):
power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                            [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                            [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                            [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

iterative_clear_sky = IterativeClearSky(power_signals_d, rank_k=6)

iterative_clear_sky.minimize_objective()

# Get the resulting left low rank matrix and right low rank matrix for evaluation.
left_low_rank_matrix = iterative_clear_sky.left_low_rank_matrix()
# The above can be also obtained as l_cs_value:
l_cs_value = iterative_clear_sky.l_cs_value()

# Get the resulting right low rank matrix for evaluation.
right_low_rank_matrix = iterative_clear_sky.right_low_rank_matrix()
# The above can be also obtained as r_cs_value:
r_cs_value = iterative_clear_sky.r_cs_value()

clear_sky_signals = iterative_clear_sky.clear_sky_signals()

degradation_rate = iterative_clear_sky.degradation_rate()
# The above can be also obtained as beta_value:
beta_value = iterative_clear_sky.beta_value()
```

#### Example 4: Setting different hyper-parameters for minimization of objective function of Generalized Low Rank Modeling.

There are three hyper-parameters in the objective function of Generalized Low Rank Modeling, i.e. mu_l, mu_r, and tau.
By default, mu_l is set to 1.0, mu_r is set to 20.0, and tau is set to 0.8.
You can change it by specifying mu_l, mu_r, and tau keyword arguments in minimize_objective method.

```
import numpy as np
from statistical_clear_sky.algorithm.iterative_clear_sky\
 import IterativeClearSky

# Usually read from a CSV file or a database with more data,
# covering 1 day (column) and a few years (row):
power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                            [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                            [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                            [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

iterative_clear_sky = IterativeClearSky(power_signals_d)

iterative_clear_sky.minimize_objective(mu_l=5e2, mu_r=1e3, tau=0.9)

clear_sky_signals = iterative_clear_sky.clear_sky_signals()
degradation_rate = iterative_clear_sky.degradation_rate()
```

#### Example 5: Setting different control parameters for minimization of objective function of Generalized Low Rank Modeling.

There are three control parameters in the objective function of Generalized Low Rank Modeling, i.e. termination criterion - eps, and maximum number of iteration - max_iter.
By default, eps is set to 1e-3, max_iter is set to 100.
You can change it by specifying eps and max_iter keyword arguments in minimize_objective method.

```
import numpy as np
from statistical_clear_sky.algorithm.iterative_clear_sky\
 import IterativeClearSky

# Usually read from a CSV file or a database with more data,
# covering 1 day (column) and a few years (row):
power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                            [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                            [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                            [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

iterative_clear_sky = IterativeClearSky(power_signals_d)

iterative_clear_sky.minimize_objective(eps=1e-6, max_iter=10)

clear_sky_signals = iterative_clear_sky.clear_sky_signals()
degradation_rate = iterative_clear_sky.degradation_rate()
```

#### Example 6: Setting limit on degradation rate.

You can specify the maximum degradation and minimum degradation by setting max_degradation and min_degradation keyword arguments in minimize_objective method.
By default, they are set not to be used.

```
import numpy as np
from statistical_clear_sky.algorithm.iterative_clear_sky\
 import IterativeClearSky

# Usually read from a CSV file or a database with more data,
# covering 1 day (column) and a few years (row):
power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                            [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                            [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                            [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

iterative_clear_sky = IterativeClearSky(power_signals_d)

iterative_clear_sky.minimize_objective(max_degradation=0.5, min_degradation=0.0)

clear_sky_signals = iterative_clear_sky.clear_sky_signals()
degradation_rate = iterative_clear_sky.degradation_rate()
```

#### Example 7: Estimating clear sky signals without degradation.

You can estimate clear sky signals based on the assumption that there is no year-to-year degradation.
In this case, you can set is_degradation_calculated keyword argument to False in minimize_objective method.
By default, it's set to True.

```
import numpy as np
from statistical_clear_sky.algorithm.iterative_clear_sky\
 import IterativeClearSky

# Usually read from a CSV file or a database with more data,
# covering 1 day (column) and a few years (row):
power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                            [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                            [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                            [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

iterative_clear_sky = IterativeClearSky(power_signals_d)

iterative_clear_sky.minimize_objective(is_degradation_calculated=False)

clear_sky_signals = iterative_clear_sky.clear_sky_signals()
```

## Running the tests

So far, there are unit tests. Integration tests may be added if found necessary.

### Unit tests (developer tests)

1. GIT clone this project.

2. In the project directory in terminal,

    ```
    $ python -m unittest
    ```

    This runs all the tests under tests folder.

All the tests are placed under "tests" directory directly under the project directory.
It is using "unittest" that is a part of Python Standard Library by default.
There may be a better unit testing framework.
But the reason is to invite as many contributors as possible with variety of background.

### Coding style tests

[pylint](https://www.pylint.org/) is used to check if coding style is conforming to "PEP 8 -- Style Guide for Python Code"

Note: We are open to use [LGTM](https://lgtm.com/).
However, since we decided to use another code coverage tool [codecov](https://codecov.io/) based on a comment by project's Technical Advisory Council, we decided not to use another tool that does code coverage.
We are also open to use other coding style tools.

Example of using pylint:
```
$ pylint statistical_clear_sky
```

## Contributing

Please read [CONTRIBUTING.md](https://github.com/bmeyers/StatisticalClearSky/contributing) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/bmeyers/StatisticalClearSky/tags).

## Authors

* **Bennet Meyers** - *Initial work and Main research work* - [Bennet Meyers GitHub](https://github.com/bmeyers)

* **Tadatoshi Takahashi** - *Refactoring and Packaging work and Research support work* - [Tadatoshi Takahashi GitHub](https://github.com/tadatoshi)

See also the list of [contributors](https://github.com/bmeyers/StatisticalClearSky/contributors) who participated in this project.

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details

## References

[1] B. Meyers, M. Tabone, and E. C. Kara, "Statistical Clear Sky Fitting Algorithm," IEEE Photovoltaic Specialists Conference, 2018.

## Acknowledgments

* The authors would like to thank Professor Stephen Boyd from Stanford University for his input and guidance and Chris Deline, Mike Deceglie, and Dirk Jordan from NREL for collaboration. 
