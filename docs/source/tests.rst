Running the tests
-----------------

Unit tests (developer tests)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. GIT clone this project.

2. In the project directory in terminal,

   ::

       $ python -m unittest

   This runs all the tests under tests folder.

All the tests are placed under "tests" directory directly under the
project directory. It is using "unittest" that is a part of Python
Standard Library by default. There may be a better unit testing
framework. But the reason is to invite as many contributors as possible
with variety of background.

Coding style tests
~~~~~~~~~~~~~~~~~~

`pylint <https://www.pylint.org/>`__ is used to check if coding style is
conforming to "PEP 8 -- Style Guide for Python Code"

Note: We are open to use `LGTM <https://lgtm.com/>`__. However, since we
decided to use another code coverage tool
`codecov <https://codecov.io/>`__ based on a comment by project's
Technical Advisory Council, we decided not to use another tool that does
code coverage. We are also open to use other coding style tools.

Example of using pylint:

In the project directory in terminal,

::

    $ pylint statistical_clear_sky
