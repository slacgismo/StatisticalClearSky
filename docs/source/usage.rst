Usage
-----

As a part of Python code or inside Jupyter notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example 1: Simplest example with the fewest number of input parameters.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using default solver (Open Source solver: ECOS)

.. code:: python

    import numpy as np
    from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting

    # Usually read from a CSV file or a database with more data,
    # covering 1 day (column) and a few years (row):
    power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                                [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                                [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                                [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

    iterative_fitting = IterativeFitting(power_signals_d)

    iterative_fitting.execute()

    clear_sky_signals = iterative_fitting.clear_sky_signals()
    degradation_rate = iterative_fitting.degradation_rate()

Example 2: Estimating clear sky signals without degradation.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can estimate clear sky signals based on the assumption that there is
no year-to-year degradation. In this case, you can set
is\_degradation\_calculated keyword argument to False in execute method.
By default, it's set to True.

.. code:: python

    import numpy as np
    from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting

    # Usually read from a CSV file or a database with more data,
    # covering 1 day (column) and a few years (row):
    power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                                [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                                [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                                [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

    iterative_fitting = IterativeFitting(power_signals_d)

    iterative_fitting.execute(is_degradation_calculated=False)

    clear_sky_signals = iterative_fitting.clear_sky_signals()

Example 3: Using a different solver.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default solver ECOS is not stable with large set of input data. The
following example shows how to specify to use Mosek solver by passing
solver\_type keyword argument (to the constructor).

.. code:: python

    import numpy as np
    from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting

    # Usually read from a CSV file or a database with more data,
    # covering 1 day (column) and a few years (row):
    power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                                [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                                [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                                [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

    iterative_fitting = IterativeFitting(power_signals_d,
                                         solver_type='MOSEK')

    iterative_fitting.execute()

    clear_sky_signals = iterative_fitting.clear_sky_signals()
    degradation_rate = iterative_fitting.degradation_rate()

Example 4: Setting rank for Generalized Low Rank Modeling.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, rank of low rank matrices is specified to be 6. You can
change it by specifying rank\_k keyword argument (in the constructor).

.. code:: python

    import numpy as np
    from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting

    # Usually read from a CSV file or a database with more data,
    # covering 1 day (column) and a few years (row):
    power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                                [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                                [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                                [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

    iterative_fitting = IterativeFitting(power_signals_d, rank_k=6)

    iterative_fitting.execute()

    # Get the resulting left low rank matrix and right low rank matrix for evaluation.
    left_low_rank_matrix = iterative_fitting.left_low_rank_matrix()
    # The above can be also obtained as l_cs_value:
    l_cs_value = iterative_fitting.l_cs_value

    # Get the resulting right low rank matrix for evaluation.
    right_low_rank_matrix = iterative_fitting.right_low_rank_matrix()
    # The above can be also obtained as r_cs_value:
    r_cs_value = iterative_fitting.r_cs_value

    clear_sky_signals = iterative_fitting.clear_sky_signals()

    degradation_rate = iterative_fitting.degradation_rate()
    # The above can be also obtained as beta_value:
    beta_value = iterative_fitting.beta_value

Example 5: Setting different hyper-parameters for minimization of objective function of Generalized Low Rank Modeling.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are three hyper-parameters in the objective function of
Generalized Low Rank Modeling, i.e. mu\_l, mu\_r, and tau. By default,
mu\_l is set to 1.0, mu\_r is set to 20.0, and tau is set to 0.8. You
can change it by specifying mu\_l, mu\_r, and tau keyword arguments in
execute method.

.. code:: python

    import numpy as np
    from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting

    # Usually read from a CSV file or a database with more data,
    # covering 1 day (column) and a few years (row):
    power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                                [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                                [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                                [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

    iterative_fitting = IterativeFitting(power_signals_d)

    iterative_fitting.execute(mu_l=5e2, mu_r=1e3, tau=0.9)

    clear_sky_signals = iterative_fitting.clear_sky_signals()
    degradation_rate = iterative_fitting.degradation_rate()

Example 6: Setting different control parameters for minimization of objective function of Generalized Low Rank Modeling.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are three control parameters in the objective function of
Generalized Low Rank Modeling, i.e. exit criteria -
exit\_criterion\_epsilon, and maximum number of iteration -
max\_iteration. By default, exit\_criterion\_epsilon is set to 1e-3,
max\_iteration is set to 100. You can change it by specifying eps and
max\_iteration keyword arguments in execute method.

.. code:: python

    import numpy as np
    from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting

    # Usually read from a CSV file or a database with more data,
    # covering 1 day (column) and a few years (row):
    power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                                [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                                [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                                [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

    iterative_fitting = IterativeFitting(power_signals_d)

    iterative_fitting.execute(exit_criterion_epsilon=1e-6, max_iteration=10)

    clear_sky_signals = iterative_fitting.clear_sky_signals()
    degradation_rate = iterative_fitting.degradation_rate()

Example 7: Setting limit on degradation rate.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can specify the maximum degradation and minimum degradation by
setting max\_degradation and min\_degradation keyword arguments in
execute method. By default, they are set not to be used.

.. code:: python

    import numpy as np
    from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting

    # Usually read from a CSV file or a database with more data,
    # covering 1 day (column) and a few years (row):
    power_signals_d = np.array([[0.0, 0.0, 0.0, 0.0],
                                [1.33389997, 1.40310001, 0.67150003, 0.77249998],
                                [1.42349994, 1.51800001, 1.43809998, 1.20449996],
                                [1.52020001, 1.45150006, 1.84809995, 0.99949998]])

    iterative_fitting = IterativeFitting(power_signals_d)

    iterative_fitting.execute(max_degradation=0.0, min_degradation=-0.5)

    clear_sky_signals = iterative_fitting.clear_sky_signals()
    degradation_rate = iterative_fitting.degradation_rate()
