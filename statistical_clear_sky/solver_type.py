from enum import Enum

class SolverType(Enum):
    ecos = 'ECOS'
    # osqp = 'OSQP' # cvxpy supports this solver but in the context of this
                    # project, raises an error:
                    # cvxpy.error.SolverError:
                    # Problem could not be reduced to a QP, and no conic
                    # solvers exist among candidate solvers (['OSQP']).
    # scs = 'SCS' # cvxpy supports this solver but in the context of this
                  # project, raises an error:
                  # cvxpy.error.SolverError:
                  # Either candidate conic solvers (['SCS']) do not support
                  # the cones output by the problem (), or there are not
                  # enough constraints in the problem.
    mosek = 'MOSEK'