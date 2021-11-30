"""Contains primary class types for optimization analyses.

These are:

* :class:`Data` which holds the tables giving objective/constraint scores for each potential decision.
* :class:`Problem` which holds a single optimization problem.
* :class:`Analysis` and subclasses, which hold a collection of problems that make up a trade-off curve or other analysis.
    - :class:`Single`
    - :class:`ConfigList`
    - :class:`NDimFrontier`
    - :class:`WeightTableIterator`
"""

import os
import copy
import glob
import json
import warnings
from collections import namedtuple
import cvxpy as cvx
import pandas as pd
import numpy as np

from natcap.root import arith_parser as ap


"""
Implemented analysis types are:
atype_to_class = {
    'single': Single,
    'frontier': Frontier,
    'n_dim_frontier': NDimFrontier,
    'weight_table_iter': WeightTableIterator,
    'n_dim_outline': NDimFrontierOutline
}
"""


class NPFixedSeed:
    """
    For use as a context manager for controllable rng. Ensures that numpys's rng is returned to
    original state outside of the calling context.
    """
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.state)


class Data(object):
    """
    Holds data needed for optimization

    After initialization refer to ``data.factor_names`` and ``data.option_names``
    for the labels attached to each category. These are taken from file or column names depending
    on the files_by_factor argument.

    Specific data tables can be referenced by ``data[factor_name]``, similar to dictionary reference
    Columns for particular options can be referenced by ``data[factor_name, option_name]``

    """

    def __init__(self, data_dir, sdu_id_col, data_cols=None,
                 baseline_file=None, file_list=None, files_by_factor=False,
                 file_glob_str=None, sample=None, sample_seed=None):
        """
        Args:
        * data_dir: directory to load data from.
        * sdu_id_col: name of SDU key column

        Keyword Args:
        * data_cols: names of columns to pull from csv files, default (None) pulls all
        * baseline_file: optional name of baseline file to move to first position in option list
        * file_list: list of files (only the basename) to load from data_dir.
        * files_by_factor: if True, treats separate files as corresponding to factors with one column per option.
            Otherwise treats files as corresponding to options with one column per factor
        * file_glob_str: if None, loads files in data_dir matching '*.csv', otherwise will match given pattern

        """

        # save the calling arguments
        self.data_dir = data_dir
        self.sdu_id_col = sdu_id_col
        self.data_cols = data_cols
        self.baseline_file = baseline_file
        self.file_list = file_list
        self.files_by_factor = files_by_factor
        self.file_glob_str = file_glob_str
        self.sample = sample
        self.sample_seed = sample_seed

        self.option_names = None
        self.factor_names = None

        # get the names of the files to load
        if file_list is None:
            if file_glob_str is None:
                data_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
            else:
                data_files = sorted(glob.glob(os.path.join(data_dir, file_glob_str)))
        else:
            data_files = sorted([os.path.join(data_dir, f) for f in file_list])

        if baseline_file is not None and baseline_file in data_files:
            data_files.remove(baseline_file)
            data_files.insert(0, baseline_file)

        self.data_files = data_files

        # load the data files and ensure they're sorted by SDU_ID
        dfs = [pd.read_csv(f) for f in data_files]

        # determine which columns to keep (data columns + SDU_ID column)
        if self.data_cols is None:
            non_sdu_cols = list(dfs[0].columns)
            non_sdu_cols.remove(self.sdu_id_col)
            cols_to_keep = [self.sdu_id_col] + non_sdu_cols
        else:
            cols_to_keep = [self.sdu_id_col] + self.data_cols
            non_sdu_cols = self.data_cols
        for df, data_file in zip(dfs, data_files):
            for col in cols_to_keep:
                if col not in df.columns:
                    print('WARNING: column {} not found in {}. Filled with 0 values.'.format(
                        col, os.path.basename(data_file)
                    ))
                    df[col] = 0
        dfs = [df[cols_to_keep] for df in dfs]

        # (optional) sample the data (smaller datasets for testing)
        if self.sample:
            # n_rows = len(dfs[0])
            # with FixedSeed(self.sample_seed):
            #     sample_rows = np.random.choice(n_rows, self.sample, replace=False)
            dfs = [df.sample(n=self.sample, random_state=self.sample_seed) for df in dfs]

        # sort dfs by sdu_id_col
        for df in dfs:
            df.sort_values(by=sdu_id_col, inplace=True)

        self.sdu_ids = dfs[0][sdu_id_col].copy()

        # transfer the data from data frames to numpy arrays
        # the arrays are aligned with one per factor, with rows for SDUs, cols for each option
        self.tables = dict()
        if self.files_by_factor:
            # this case assumes one file per factor, with non-SDU columns corresponding to each option
            self.option_names = copy.copy(non_sdu_cols)
            self.factor_names = [os.path.splitext(os.path.basename(p))[0]
                                 for p in self.data_files]
            for df, factor in zip(dfs, self.factor_names):
                mat = np.array(df[non_sdu_cols])
                mat[np.isnan(mat)] = 0
                self.tables[factor] = mat
            self.n_parcels, self.n_opts = self.tables[self.factor_names[0]].shape
        else:
            # this case assumes one file per option, with non-SDU columns corresponding to each factor
            self.option_names = [os.path.splitext(os.path.basename(p))[0]
                                 for p in self.data_files]
            self.factor_names = copy.copy(non_sdu_cols)
            matshape = (len(dfs[0]), len(data_files))
            for s in non_sdu_cols:
                mat = np.zeros(matshape)
                for i, df in enumerate(dfs):
                    mat[:, i] = np.array(df[s])
                    mat[np.isnan(mat)] = 0
                self.tables[s] = mat
            self.n_parcels, self.n_opts = matshape

        self.option_index = {opt: i for i, opt in enumerate(self.option_names)}

    def __getitem__(self, item):
        if isinstance(item, tuple):
            # the case where we have two strings (factor, option)
            return self.tables[item[0]][:, self.option_index[item[1]]]
        else:
            return self.tables[item]

    def __contains__(self, item):
        return item in self.tables

    def __repr__(self):
        pass

    def __str__(self):
        return '{} files from {}'.format(len(self.data_files), self.data_dir)

    def factor_values_for_solution(self, sol, varnames=None):
        """
        Returns a dictionary containing the summed values for each factor
        in data. keys are factornames, values are the sums.

        Caller can subset which cols to return with varnames list.

        :param sol: real [0 - 1] valued array of size (n_parcels, n_opts)
        :param varnames: optional list of factors to evaluate
        :return:
        """

        results = dict()
        if varnames is None:
            factors = self.data_cols
        else:
            factors = varnames

        for factor in factors:
            d = self.tables[factor]
            results[factor] = np.sum(d * sol)

        return results


Constraint = namedtuple('Constraint', ['factor', 'type', 'value'])


class Problem(object):
    # TODO: fixed_vals and excluded_vals have inconsistent approaches to x (integer vs multi-binary)
    # TODO: how to distinguish exclusive sets as == 1 vs <= 1 (or to change the 1 value!)

    """
    The problem class. Represents a single optimization evaluation.

    problem_def should be a dict structured as follows

    * 'weights': {factorname: weight} for each factor <- i.e. another dict
    * 'constraints': [ (factorname, type, value), ...] list of single-factor constraints
    * 'factor_by_option_constraints': [ (factorname, option, type, value), ... ] applies the constraint
        only to the listed option.

        deprecated:
            'targets': {factorname: target} for each factor with a target
            'target_types': {factorname: '<', '>', or '=' } (inequals will allow ==)


    optional problem args:
        'fixed_vals': {index: value}: x_index will be constrained to value
        'excluded_vals' {index: list}: x_index_j will be set to zero for each
                                      j in list
        'exclusive_choices': None, list, or list of lists. The values in the list should be the j-indices
            of the options to be made mutually-exclusive
            * None: default, assumes that all choices are mutually exclusive
            * list: only the options in the list are exclusive with each other. All options not in the
                list can be selected regardless of selection in the listed options, or the other
                non-listed options
            * list of lists: creates several sets of mutually exclusive options, but can choose one option
                from each set. Options not included in a list can be selected or not independent of
                selection in the mutually-exlusive sets.
        'normalize_objectives': True or False (assumed False if arg not present)
            this will rescale all arguments to be between 0 and 1
            'flatten_objectives': True or False (assumed False if arg not present)
            this replaces objective values with their relative rank
            NOTE: I think this might produce solutions that aren't actually
            on the frontier.


    """

    def __init__(self, data, problem_def, solver=None):
        """
        Problem creation function

        after the problem has been run, the results are attainable as:

        * :attr:`p.solution`: nsdus x nopts numpy array with values between 0 and 1
        * :attr:`p.objective_values`: {factor: value} summed scores for all factors

        :param data: a Data object
        :param problem_def: a dictionary describing the problem to solve
        :param solver: optional argument to specify the solver
        """
        self.data = data
        self.problem_def = copy.deepcopy(problem_def)
        self.solved = False
        self.solver = solver
        self.solution = None
        self.objective_values = None
        self.variable_type = None
        self.constraints = []
        self.weights = {}

        # determine variable type
        if 'use_linear_vars' in self.problem_def and \
                        self.problem_def['use_linear_vars'] is True:
            self.variable_type = 'Linear'
        elif 'target_types' in self.problem_def and '=' in self.problem_def['target_types'].values():
            self.variable_type = 'Linear'
        else:
            self.variable_type = 'Boolean'

        # construct list of Constraints
        if 'constraints' in self.problem_def:
            for c in self.problem_def['constraints']:
                if not len(c) == 3:
                    raise ValueError('{} is not a valid constraint'.format(c))
                self.constraints.append(Constraint(*c))
            print('self.constraints: {}'.format(self.constraints))

        # construct weights
        for f, w in self.problem_def['weights'].items():
            self.weights[f] = w

    def __repr__(self):
        return('Problem({}, {}, solver={}'.format('data', 'problem_def', 'solver'))

    def _str__(self):
        pass

    def solve(self):
        """Builds the MILP problem and calls a solver.

        Returns:
            solution (np.array): Optimized decision vector (matrix)
        """
        factors = self.data.factor_names
        nparcels = self.data.n_parcels
        nopts = self.data.n_opts
        ndvs = nparcels * nopts

        def xij_index(i, j):
            # returns column index in G and A for given xij, where i is the parcel
            # unit, and j is the option.
            return i * nopts + j

        # Declare variables
        if self.variable_type == 'Linear':
            x = cvx.Variable(ndvs, nonneg=True)
        elif self.variable_type == 'Boolean':
            x = cvx.Variable(ndvs, boolean=True)
        else:
            x = cvx.Variable(ndvs, boolean=True)

        # CONSTRUCT OBJECTIVE FUNCTION
        # the objective is just a weighted sum of the different factors
        v = np.zeros(ndvs)
        print('Problem factors: {}'.format(factors))
        for f, w in self.weights.items():
            fv = self.data[f].ravel()
            if 'normalize_objectives' in self.problem_def and self.problem_def['normalize_objectives'] is True:
                print('normalizing factor {}'.format(f))
                fv = self.normalize_factor_data(fv)
            print('applying weight {}'.format(w))
            fv = fv * w
            v = v + fv

        # x has size (ndvs, 1), v has size (ndvs, ) - i.e. is 1-dimensional
        # doing v * x with these sizes will compute dot product.
        objective = cvx.Maximize(v @ x)

        # CONSTRUCT CHOICE CONSTRAINTS

        # DEFINE EXCLUSIVE OPTION SETS
        # the default case is that all options are exclusive (first clause)
        if 'exclusive_choices' not in self.problem_def or self.problem_def['exclusive_choices'] is None:
            # default case: all choices are mutually exclusive
            # 'exclusive_choices' not in self.problem_def or self.problem_def['exclusive_choices'] is None
            cm, cv = self.single_choice_constraint_mats(nparcels, nopts)
            parcel_choice_constraint = [cm @ x == cv]
        elif self.problem_def['exclusive_choices'] == []:
            # case with no exclusive options
            parcel_choice_constraint = []
        elif not isinstance(self.problem_def['exclusive_choices'][0], list):
            # case with just one exclusive set
            cm, cv = self.single_choice_constraint_mats(nparcels, nopts,
                                                        opt_set=self.problem_def['exclusive_choices'])
            parcel_choice_constraint = [cm @ x <= cv]
        elif isinstance(self.problem_def['exclusive_choices'][0], list):
            # case with multiple exclusive sets
            parcel_choice_constraint = []
            for opt_set in self.problem_def['exclusive_choices']:
                cm, cv = self.single_choice_constraint_mats(nparcels, nopts, opt_set=opt_set)
                parcel_choice_constraint.append(cm @ x <= cv)
        else:
            raise Exception("invalid options for problem_def['exclusive_choices']")

        # FIXED VALUES CONSTRAINTS
        # these set x_ij = 1 so that choices can be locked in
        # 'fixed_vals': {index: value}: x_index will be constrained to value
        if 'fixed_vals' in self.problem_def.keys():
            fixed_vals = self.problem_def['fixed_vals']
        else:
            fixed_vals = dict()
        fixed_val_constraints = [x[xij_index(i, j)] == 1 for i, j in fixed_vals.items()]

        # EXCLUDED VALUES CONSTRAINTS
        # these set values to zero so that they won't be chosen
        # 'excluded_vals' {index: list}: x_index_j will be set to zero for each
        #                               j in list

        if 'excluded_vals' in self.problem_def.keys():
            excluded_vals = self.problem_def['excluded_vals']
        else:
            excluded_vals = dict()
        excluded_val_constraints = []
        for i, l in excluded_vals.items():
            excluded_val_constraints.extend(
                [x[xij_index(i, j)] == 0 for j in l])

        # ADD EXCLUDE CONSTRAINTS FOR 0 AREA ACTIVITIES
        # These are based on the 'exclude' column that is automatically created by the
        # preprocessing step.
        for i in range(nparcels):
            for j in range(nopts):
                if self.data['exclude'][i, j] == 1:
                    excluded_val_constraints.append(
                        x[xij_index(i, j)] == 0)

        # CONSTRUCT FACTOR CONSTRAINTS
        factor_constraints = []

        for c in self.constraints:
            print('applying {}'.format(c))
            factor_value = ap.apply(self.data, c.factor).ravel() @ x
            if c.type in ('<', '<='):
                factor_constraints.append(factor_value <= c.value)
            elif c.type in ('>', '>='):
                factor_constraints.append(factor_value >= c.value)
            else:
                factor_constraints.append(factor_value == c.value)

        if 'targets' in self.problem_def:
            warnings.showwarning('replace "targets and "target_types" with "constraints"',
                                 DeprecationWarning, 'optim_core.py', 314)
            for f in self.problem_def['targets'].keys():
                factor_value = self.data[f].ravel() @ x
                factor_target = self.problem_def['targets'][f]
                if self.problem_def['target_types'][f] in ('<', '<='):
                    factor_constraints.append(factor_value <= factor_target)
                elif self.problem_def['target_types'][f] in ('>', '>='):
                    factor_constraints.append(factor_value >= factor_target)

        # COMPLEX CONSTRAINTS
        # TODO: figure out the right place to do different parts of this
        fbo_constraints = []
        if 'factor_by_option_constraints' in self.problem_def:
            # fbo constraints should have the form (factor, option, type, value)
            # they only apply the constraint sum(f*x) >/< value to xs for given option
            for fboc in self.problem_def['factor_by_option_constraints']:
                factor, option, type, value = fboc
                print('adding FBO constraint: {} for {} {} {}'.format(*fboc))
                opt_index = self.data.option_index[option]
                a = self.data[factor, option]
                lhs = sum(a * x[opt_index::nopts])
                if type in ('<', '<='):
                    fbo_constraints.append(lhs <= value)
                elif type in ('>', '>='):
                    fbo_constraints.append(lhs >=  value)
                else:
                    fbo_constraints.append(lhs ==  value)


        # BUILD OPTIMIZATION PROBLEM
        constraints = parcel_choice_constraint + fixed_val_constraints + \
                      excluded_val_constraints + factor_constraints + \
                      fbo_constraints
        constraints = [c for c in constraints if c is not None]

        prob = cvx.Problem(objective, constraints)
        prob.solve(verbose=False, solver=self.solver)  # solver=cvx.GLPK_MI
        print("status:", prob.status)
        print("optimal value", prob.value)

        self.solution = np.array([x.value for x in prob.variables()]).reshape(nparcels, nopts)
        self.objective_values = self.data.factor_values_for_solution(self.solution)
        self.solved = True

        print("solution total: {}".format(np.sum(self.solution)))
        return self.solution

    @staticmethod
    def normalize_factor_data(fv):
        if (np.max(fv) - np.min(fv)) == 0:
            # this will be the case for the endpoints in the frontier runs
            fv = 0 * fv
        else:
            fv = ((fv - np.min(fv)) / (np.max(fv) - np.min(fv)))
        return fv

    @staticmethod
    def single_choice_constraint_mats(nunits, nopts, opt_set=None):
        """
        converts nunits integer variables to nunits*nopts binary variables.
        use case here is for choosing 1 out of nopts options for each unit

        :param nunits:
        :param nopts:
        :param opt_set:
        """
        nvars = nunits * nopts

        if opt_set is None:
            exclusive_choice = np.ones(nopts)
        else:
            exclusive_choice = np.zeros(nopts)
            for i in opt_set:
                exclusive_choice[i] = 1

        m = np.zeros((nunits, nvars), dtype='int')
        for i in range(nunits):
            m[i, nopts * i:nopts * (i + 1)] = exclusive_choice

        # c = np.ones((nunits, 1), dtype='int')
        c = np.ones(nunits, dtype='int')

        return m, c

    def add_weight(self, factorname, weight):
        self.weights[factorname] = weight

    def remove_weight(self, factorname):
        self.weights.pop(factorname)

    def add_constraint(self, factorname, value, type):
        # TODO should this function just take list/tuples?
        self.constraints.append(Constraint(factorname, value, type))

    def remove_constraint(self, factorname, value, type):
        # note for this, namedtuples and tuples will evaluate == if they have the same values in each position
        self.constraints = [c for c in self.constraints if c != (factorname, value, type)]

    def solution_to_csv(self, filepath):
        if self.solved:
            d = {
                self.data.sdu_id_col: list(self.data.sdu_ids),
            }
            table_cols = [self.data.sdu_id_col]
            for i, f in enumerate(self.data.option_names):
                d[f] = self.solution[:, i]
                table_cols.append(f)
            df = pd.DataFrame(data=d)
            df = df[table_cols]
            df.to_csv(filepath, index=False)


class Analysis(object):
    """Parent class for all analysis types.


    What are the functions/capabilities that all analyses need to have?

    #. run
    #. get item: analysis[i] returns ith point
    #. iterate through stored points
    #. save to reports (summary, full details, maps)

    This class assumes that all problems will be added to a list (self.problems), and that
    the problems have a problem.solve() method. Calling self.run() on the analysis will
    call solve() on all problems.

    """

    def __init__(self, data, config):
        """
        Creates empty :attr:`self.problems` list.

        :type data: :class:`.Data`
        :param data: contains problem data

        :type config: dict
        :param config: values to configure specific analysis type
        """
        self.data = data
        self.config = copy.deepcopy(config)
        self.problems = []

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, position):
        return self.problems[position]

    def __iter__(self):
        return iter(self.problems)

    def run(self):
        """
        calls :meth:`~.Problem.solve()` on all problems in `self.problems`

        :return: None
        """
        for p in self.problems:
            p.solve()

    def report(self, report_file, report_type=None):
        pass

    def problem_dict(self):
        problem = dict()
        return problem

    def save_decision_tables(self, folder):
        """
        Calls :meth:`~.Problem.solution_to_csv` on all problems in :attr:`self.problems`. Files will be
        generated at :file:`folder/solution_id.csv`.

        :type folder: str
        :param folder: output folder (will be created if it does not exist)

        :return: None
        """
        if not os.path.isdir(folder):
            os.makedirs(folder)
        for i, p in enumerate(self.problems):
            filename = os.path.join(folder, 'solution_{}.csv'.format(i))
            p.solution_to_csv(filename)

    def save_summary_table(self, folder, add_cols=None):
        """
        Creates a table with summary stats (summed results) for each solution in the :class:`.Analysis`.

        :type folder: str
        :param folder: output folder (will be created if it does not exist)

        :type add_cols: dict
        :param add_cols: column names and data to add to output table

        :return:
        """

        if not os.path.isdir(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, 'summary_table.csv')
        cols = ['point'] + self.data.factor_names
        records = []
        for i, p in enumerate(self.problems):
            result = self.data.factor_values_for_solution(p.solution)
            result['point'] = i
            record = [result[k] for k in cols]
            records.append(record)

        df = pd.DataFrame.from_records(records, columns=cols)
        if add_cols is not None:
            for col, vals in add_cols.items():
                df[col] = vals
        df.to_csv(filename, index=False)

    def summed_choices(self):
        """
        Sums :math:`x_{pc}` for each parcel and choice across all solutions.

        :return: np.array (n_parcels, n_opts)
        """
        n_parcels = self.data.n_parcels
        n_opts = self.data.n_opts
        choice_sum = np.zeros((n_parcels, n_opts))
        for p in self.problems:
            choice_sum += p.solution
        return choice_sum

    def save_summed_choices(self, folder):
        """
        Saves results of :meth:`~Analysis.summed_choices` to a csv with option names as headers.

        The file will be saved to :file:`folder/summed_choices.csv`.

        :type folder: str
        :param folder: output folder (will be created if it does not exist)
        :return:
        """
        if not os.path.isdir(folder):
            os.makedirs(folder)

        summed_choices = self.summed_choices()
        d = {
            self.data.sdu_id_col: list(self.data.sdu_ids),
        }
        table_cols = [self.data.sdu_id_col]
        for i, f in enumerate(self.data.option_names):
            d[f] = summed_choices[:, i]
            table_cols.append(f)
        df = pd.DataFrame(data=d)
        df = df[table_cols]
        df.to_csv(os.path.join(folder, 'summed_choices.csv'), index=False)

    def factor_scores(self, factor):
        return [p.objective_values[factor] for p in self.problems]

    @staticmethod
    def _get_point_def_from_frontier_with_weight_mult(frontier_def, weight_mult_dict):
        point_def = copy.deepcopy(frontier_def)  # need deepcopy to copy the weights dicts
        # just need to overwrite weights
        for obj, w in weight_mult_dict.items():
            point_def['weights'][obj] *= w
        return point_def


class Single(Analysis):
    """Analysis container for a single optimization run.
    """
    def __init__(self, data, config, weights, solver=None):
        """

        Args:
            data: instance of :class:`.Data`
            config:
            solver:
        """
        super(Single, self).__init__(data, config)  # copies config into self.config
        self.weights = weights
        self.solver = solver
        print('self.config: {}'.format(self.config))

    def problem_dict(self):
        problem = super(Single, self).problem_dict()
        problem['weights'] = self.weights
        return problem


class ConfigList(Analysis):
    """Helper analysis for script-driven batching.

    Allows creation of analyses from a list of configuration dictionaries.

    e.g. iterating through constraint values.

    """
    def __init__(self, data, config_list, solver=None):
        super(ConfigList, self).__init__(data, None)
        self.solver = solver
        self.config_list = copy.deepcopy(config_list)
        for c in self.config_list:
            self.problems.append(Problem(self.data, c, solver=solver))


class NDimFrontier(Analysis):
    """Monte Carlo sampling of N-dimensional frontier.

    This class generates random uniformly distributed weight vectors and
    runs the optimization for each of these.

    config should be a dict structured as follows:
        'npts': number of points to build in frontier
        'objectives': [factornames]
        'weights': {factornames}: +/- 1.0 for each factor to determine max or min goals
        'constraints': [ (factorname, type, value), ... ]
        'fixedvals': {index: value}: x_index will be constrained to value
        'excludedvals' {index: list}: x_index_j will be set to zero for each
                                      j in list
    deprecated:
        'targets': {factornames}: target for each factor with a target
        'targettypes': {factornames}: '<', '>', or '=' (inequals will allow ==)


    example of frontier_def coming from root_ui:
    {
        'analysis_type': u'n_dim_frontier',
        'flatten_objectives': False,
        'normalize_objectives': True,
        'npts': 15,
        'objectives': ['sed_hyd', 'phos_wetl', 'n_export'],
        'targets': {'maskpixha': 20000.0},
        'targettypes': {'maskpixha': '='},
        'use_mp': False,
        'weights': {'n_export': -1.0, 'phos_wetl': -1.0, 'sed_hyd': -1.0}
    }

    """

    def __init__(self, data, config, solver=None):
        super(NDimFrontier, self).__init__(data, config)
        self.data = data
        self.config = config
        self.solver = solver
        self.npts = self.config['npts']
        self.nobjs = len(self.config['weights'])
        self.weight_vectors = self._make_weight_vectors(self.npts, self.nobjs)
        for wv in self.weight_vectors:
            weight_dict = {f: v for f, v in zip(self.config['weights'].keys(), wv)}
            point_def = self._get_point_def_from_frontier_with_weight_mult(self.config, weight_dict)
            print('adding point with weights: {}'.format(wv))
            self.problems.append(Problem(self.data, point_def, solver=self.solver))

    @staticmethod
    def _make_weight_vectors(npts, ndims):
        """

        :param npts:
        :param ndims:
        :return:
        """
        pts = np.random.multivariate_normal(np.zeros(ndims), np.identity(ndims), npts)
        norm_pts = np.abs(np.array([p / np.linalg.norm(p) for p in pts]))
        return norm_pts


class WeightTableIterator(Analysis):
    """Analysis with user-specifed weights for each contained problem.

    config should be a dict structured as follows:
    .. code:
    {
        'npts': number of points to build in frontier
        'objectives': [factornames]
        'weights': {factornames}: +/- 1.0 for each factor to determine max or min goals
        'constraints': [ (factorname, type, value), ... ]
        'fixedvals': {index: value}: x_index will be constrained to value
        'excludedvals' {index: list}: x_index_j will be set to zero for each
                                      j in list
    deprecated:
        'targets': {factornames}: target for each factor with a target
        'targettypes': {factornames}: '<', '>', or '=' (inequals will allow ==)
    }


    example of frontier_def coming from root_ui:
    {
        'analysis_type': u'n_dim_frontier',
        'flatten_objectives': False,
        'normalize_objectives': True,
        'npts': 15,
        'objectives': ['sed_hyd', 'phos_wetl', 'n_export'],
        'targets': {'maskpixha': 20000.0},
        'targettypes': {'maskpixha': '='},
        'use_mp': False,
        'weights': {'n_export': -1.0, 'phos_wetl': -1.0, 'sed_hyd': -1.0}
    }

    """
    def __init__(self, data, config, solver=None):
        """
        :class:`WeightTableIterator` requires that :attr:`config['weights']` contain the path
        to a .csv file with the appropriate sub-objective weights for each run to be
        performed.

        Args:
            data: instance of :class:`.Data`
            config (dict):
            solver:
        """
        super(WeightTableIterator, self).__init__(data, config)
        self.data = data
        self.config = config
        self.solver = solver
        weight_table = pd.read_csv(self.config['weights'])
        weight_cols = list(weight_table.columns)
        self.problems = []
        self.weights = []
        for row in weight_table.iterrows():
            weight_dict = {f: row[1][f] for f in weight_cols}
            print('adding point with weights: {}'.format(weight_dict))
            point_def = copy.deepcopy(config)
            point_def['weights'] = weight_dict
            self.weights.append(weight_dict)
            self.problems.append(Problem(self.data, point_def, solver=self.solver))
        self.npts = len(self.problems)


# class NDimFrontierOutline(Analysis):
#     """
#     This class should internally hold n Frontier class objects, and be able
#     to iterate through all n.
#     """
#
#     def __init__(self, config):
#         super(NDimFrontierOutline, self).__init__(config)
#
#
# class SimpleSortFrontier(Analysis):
#     """
#     The simple sort frontier just ranks options by benefit/cost, and returns the desired number of points
#     along the curve constructed by selecting in decreasing benefit/cost order.
#     """
#
#     def __init__(self, data, config):
#         super(SimpleSortFrontier, self).__init__(config)


atype_to_class = {
    'single': Single,
    'n_dim_frontier': NDimFrontier,
    'weight_table': WeightTableIterator
    # 'n_dim_outline': NDimFrontierOutline
}


def available_analysis_types():
    return sorted(list(atype_to_class.keys()))


def run(config_file):
    config = ensure_args_dict(config_file)
    data = Data(config['data_dir'], config['sdu_id_col'])

    atype = config['analysis_type']
    analysis = atype_to_class[atype](data, config, solver=cvx.GUROBI)

    analysis.run()


def ensure_args_dict(args):
    """Helper to accept args as a json file or dict.

    Args:
        args (dict or path to json file):

    Returns: args as a dict

    """
    if isinstance(args, dict):
        return args
    else:
        return json.load(open(args))


if __name__ == '__main__':

    test_data = False
    test_problem = False
    test_manual_frontier = False
    test_ndimfrontier = False
    test_weight_table_iterator = True
    test_files_by_factor = False

    test_choice_constraint_mat = False
    if test_choice_constraint_mat:
        print(Problem.single_choice_constraint_mats(3, 4))
        print(Problem.single_choice_constraint_mats(3, 4, [0, 1]))
        print(Problem.single_choice_constraint_mats(3, 4, [0, 2, 3]))

    problem_def = {
        'analysis_type': 'single',
        # 'objectives': ['N_LOADING_REL', 'S_LOADING_REL'],
        'weights': {'N_LOADING_REL': -1.0, 'S_LOADING_REL': -1.0},
        'constraints': [('AG_PROFIT_REL', '>=', -1000000),
                        ('N_LOADING_REL', '<=', -10)],
        'npts': 10,
        'use_linear_vars': True,
        'normalize_objectives': True
    }

    if any([test_data, test_problem, test_ndimfrontier,
            test_manual_frontier, test_weight_table_iterator]):
        data_folder = '/Users/hawt0010/Projects/ROOT/root_source/sample_data/multi_bmp'
        data = Data(data_folder, 'HRU',
                    data_cols=['N_LOADING_REL',
                               'P_LOADING_REL',
                               'S_LOADING_REL',
                               'AG_PROFIT_REL'])
        n_parcels = data.n_parcels
        n_opts = data.n_opts

    if test_data:
        sample_sol = np.random.randint(0, 2, (n_parcels, n_opts))
        print(data.factor_values_for_solution(sample_sol))

    if test_problem:
        prob = Problem(data, problem_def)
        prob.solver = cvx.GUROBI
        opt_sol = prob.solve()
        print(prob.objective_values)
        prob.solution_to_csv('test_sol.csv')

    if test_manual_frontier:
        manual_problem_def = {
            'analysis_type': 'single',
            'objectives': ['N_LOADING_REL', 'S_LOADING_REL'],
            'weights': {'N_LOADING_REL': -1.0, 'S_LOADING_REL': -1.0},
            'use_linear_vars': True,
            'normalize_objectives': True
        }
        prof_vals = [-3000000.0, -2000000.0, -1000000.0]
        probs = [Problem(data, manual_problem_def) for _ in prof_vals]
        for p, c in zip(probs, prof_vals):
            p.add_constraint('AG_PROFIT_REL', '>=', c)
            p.solve()
            print(p.objective_values)

    if test_ndimfrontier:
        output_folder = "/Users/hawt0010/Projects/ROOT/root_source/sample_output/ndf"
        ndf = NDimFrontier(data, problem_def, solver=cvx.GUROBI)
        ndf.run()
        print(len(ndf))
        for p in ndf:
            print(p.objective_values)
        ndf.save_decision_tables(output_folder)
        ndf.save_summary_table(output_folder)
        ndf.save_summed_choices(output_folder)

    if test_weight_table_iterator:
        output_folder = "/Users/hawt0010/Projects/ROOT/root_source/sample_output/wti"
        pdef = copy.deepcopy(problem_def)
        pdef['weights'] = "/Users/hawt0010/Projects/ROOT/root_source/sample_data/multi_bmp_weight_table.csv"
        wti = WeightTableIterator(data, pdef)
        wti.run()
        wti.save_summary_table(output_folder)
        wti.save_summed_choices(output_folder)
        wti.save_decision_tables(output_folder)


    if test_files_by_factor:
        data_folder = "/Users/hawt0010/Projects/ROOT/root_source/sample_data/files_by_factor"
        data = Data(data_folder, 'SDU', files_by_factor=True)
        print('data.n_parcels: {}'.format(data.n_parcels))
        print('data.n_opts: {}'.format(data.n_opts))
        print('data.option_names: {}'.format(data.option_names))
        print('data.factor_names: {}'.format(data.factor_names))
