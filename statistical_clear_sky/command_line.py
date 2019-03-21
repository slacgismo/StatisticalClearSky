'''
This module contains the code for command line interface for
 statistical_clear_sky
'''

import argparse
import distutils
import numpy as np
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting
from statistical_clear_sky.solver_type import SolverType

def main():

    '''
    Examples
    -----------

    Iterative Fitting:
    $ statistical_clear_sky execute power_signals.csv
    --rank 4 --solver_type mosek --mu_l 5e2 --mu_r 1e3 --tau 0.9
    --max_iteration 10
    '''

    parser = argparse.ArgumentParser(prog='statistical_clear_sky')

    subparsers = parser.add_subparsers(
        help="Executes iterative fitting algorithm")

    parser_parameter_extraction = subparsers.add_parser('execute')
    parser_parameter_extraction.add_argument('power_signals_file', nargs='?',
        help=("Path and name of the CSV file containing input power signals, "
        "representing a matrix with row for dates and colum for time of day"))
    parser_parameter_extraction.add_argument('--rank', nargs='?', type=int,
        required=False, help="Rank for target low-rank matrices")
    parser_parameter_extraction.add_argument('--solver_type', nargs='?',
        type=lambda x:SolverType[x], choices=['ecos', 'mosek'], required=False,
        help="Solver type")
    parser_parameter_extraction.add_argument('--reserve_test_data', nargs='?',
        type=lambda x:bool(distutils.util.strtobool(x)), required=False,
        help="Ratio of test data that is reserved (> 0.0 and <= 1.0)")
    parser_parameter_extraction.add_argument('--auto_fix_time_shifts',
        nargs='?', type=lambda x:bool(distutils.util.strtobool(x)),
        required=False, help="Specifies if time shift is fixed automatically")
    parser_parameter_extraction.add_argument('--mu_l', nargs='?', type=float,
        required=False, help="Hyper-parameter for the second term")
    parser_parameter_extraction.add_argument('--mu_r', nargs='?', type=float,
        required=False, help="Hyper-parameter for the thrid term")
    parser_parameter_extraction.add_argument('--tau', nargs='?', type=float,
        required=False, help="Hyper-parameter for the fourth term")
    parser_parameter_extraction.add_argument('--exit_criterion_epsilon',
        nargs='?', type=float, required=False,
        help="Exit criterion for iteration")
    parser_parameter_extraction.add_argument('--max_iteration',
        nargs='?', type=int, required=False,
        help="Maximum number of iterations")
    parser_parameter_extraction.add_argument('--is_degradation_calculated',
        nargs='?', type=lambda x:bool(distutils.util.strtobool(x)),
        required=False, help=("Specifies if estimating clear sky signals is " "based on the assumption that there is year-to-year degradation "
        "or not"))
    parser_parameter_extraction.add_argument('--max_degradation',
        nargs='?', type=float, required=False,
        help=("Upper limit of degradation. "
        "(Note; degradation rate is zero or a negative value"))
    parser_parameter_extraction.add_argument('--min_degradation',
        nargs='?', type=float, required=False,
        help=("Lower limit of degradation. "
        "(Note; degradation rate is zero or a negative value"))
    parser_parameter_extraction.add_argument('--verbose',
        nargs='?', type=lambda x:bool(distutils.util.strtobool(x)),
        required=False, help="If True, verbose message is printed out")

    # Note: Calls execute_iterative_fitting function with the arguments:
    parser_parameter_extraction.set_defaults(func=execute_iterative_fitting)

    args = parser.parse_args()
    # Calls the function specified in "set_defaults" method:
    args.func(args)

def execute_iterative_fitting(args):

    with open(args.power_signals_file) as file:
        power_signals_d = np.loadtxt(file, delimiter=',')

    constructor_keyword_arguments = {}
    if args.rank is not None:
        constructor_keyword_arguments['rank_k'] = args.rank
    if args.solver_type is not None:
        constructor_keyword_arguments['solver_type'] = args.solver_type
    if args.solver_type is not None:
        constructor_keyword_arguments[
            'reserve_test_data'] = args.reserve_test_data
    if args.solver_type is not None:
        constructor_keyword_arguments[
            'auto_fix_time_shifts'] = args.auto_fix_time_shifts

    iterative_fitting = IterativeFitting(power_signals_d,
                                         **constructor_keyword_arguments)

    method_keyword_arguments = {}
    if args.mu_l is not None:
        method_keyword_arguments['mu_l'] = args.mu_l
    if args.mu_r is not None:
        method_keyword_arguments['mu_r'] = args.mu_r
    if args.tau is not None:
        method_keyword_arguments['tau'] = args.tau
    if args.exit_criterion_epsilon is not None:
        method_keyword_arguments[
            'exit_criterion_epsilon'] = args.exit_criterion_epsilon
    if args.max_iteration is not None:
        method_keyword_arguments['max_iteration'] = args.max_iteration
    if args.is_degradation_calculated is not None:
        method_keyword_arguments[
            'is_degradation_calculated'] = args.is_degradation_calculated
    if args.max_degradation is not None:
        method_keyword_arguments['max_degradation'] = args.max_degradation
    if args.min_degradation is not None:
        method_keyword_arguments['min_degradation'] = args.min_degradation
    if args.verbose is not None:
        method_keyword_arguments['verbose'] = args.verbose

    iterative_fitting.execute(**method_keyword_arguments)

    np.savetxt('clear_sky_signals.csv', iterative_fitting.clear_sky_signals(),
        delimiter=',')
    print('degradation_rate=', iterative_fitting.degradation_rate())

if __name__ == "__main__":
    main()
