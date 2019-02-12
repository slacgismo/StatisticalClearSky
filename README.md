# StatisticalClearSky
Statistical estimation of a clear sky signal from PV system power data

## Getting Started

You can install pip package or Anaconda package for this project.
Alternatively, you can clone this repository (GIT) and execute the example codes under notebooks folder.

### Prerequisites

When you install this project as PIP package, dependencies are automatically installed.
When you use this project in any other ways, the following instruction can be useful.

Simplest way to install dependencies if you are using pip is by

```sh
$ pip install -r requirements.txt
```

As of February 11, 2019, it fails because scs package installed as a dependency of cxvpy expects numpy to be already installed.
[scs issue 85](https://github.com/cvxgrp/scs/issues/85) says, it is fixed.
However, it doesn't seem to be reflected in its pip package.
As a work around, install numpy separatly first and install the other packages using requirements.txt. i.e.
```sh
$ pip install numpy
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

* [mosek](https://www.mosek.com/resources/getting-started/) - [Optional] - For using MOSEK solver.

    This package is necessary only when running an example code.

    If you are using pip:
    ```sh
    $ pip install -f https://download.mosek.com/stable/wheel/index.html Mosek
    ```

    If you are using Anaconda:
    ```sh
    $ conda install -c mosek mosek
    ```

### Installing

[To be added]

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

[To be added]

Explain how to run the automated tests for this system

### Break down into end to end tests

[To be added]

Explain what these tests test and why

```
Give an example
```

### And coding style tests

[To be added]

Explain what these tests test and why

```
Give an example
```

## Contributing

Please read [CONTRIBUTING.md](https://github.com/bmeyers/StatisticalClearSky/contributing) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/bmeyers/StatisticalClearSky/tags).

## Authors

* **Bennet Meyers** - *Initial work* - [Bennet Meyers GitHub](https://github.com/bmeyers)

* **Tadatoshi Takahashi** - *Packaging work* - [Tadatoshi Takahashi GitHub](https://github.com/tadatoshi)

See also the list of [contributors](https://github.com/bmeyers/StatisticalClearSky/contributors) who participated in this project.

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details

## References

[1] D. Sera, R. Teodorescu, and P. Rodriguez, "PV panel model based on datasheet values," IEEE Photovoltaic Specialists Conference, 2018.

## Acknowledgments

* [To be added.]
