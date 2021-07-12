"""Helper module to parse expressions from the combined factors table.
"""

import operator
import numpy as np


FUNCTIONS = ['abs', 'log', 'sqrt', 'sum', 'min', 'max']
FUN_FUNCS = [
    np.abs,
    np.log,
    np.sqrt,
    np.sum,
    np.min,
    np.max
]
FUN_FUNC_DICT = {fun: func for fun, func in zip(FUNCTIONS, FUN_FUNCS)}

OPERATORS = ['^', '*', '/', '+', '-']
OP_FUNCS = [
    operator.pow,
    operator.mul,
    operator.truediv,
    operator.add,
    operator.sub
]
ASSOC = ['R', 'L', 'L', 'L', 'L']
PREC = [4, 3, 3, 2, 2]
PREC_DICT = {op: score for score, op in zip(PREC, OPERATORS)}
ASSOC_DICT = {op: assoc for assoc, op in zip(ASSOC, OPERATORS)}
OP_FUNC_DICT = {op: func for op, func in zip(OPERATORS, OP_FUNCS)}
PARENS = ['(', ')']
ARITH_TOKENS = OPERATORS + PARENS

ALL_AP_TOKENS = FUNCTIONS + OPERATORS + PARENS


def apply(data_obj, arith_string):
    """Applies a given expression to the values in data container.

    Assumes a dict-style lookup, i.e. :code:`data_obj['varname']`. In particular,
    this works with column names in DataFrames, and factor names in :class:`.optim_core.Data`.

    An expression like "factor1 + 2 * factor2" returns a np.array containing the results of executing
    :code:`data_obj['factor1'] + 2 * data_obj['factor2']`.

    Args:
        data_obj (DataFrame or optim_core.Data): data container
        arith_string (string): mathematical expression

    Returns:
        np.array containing the result
    """
    tokens = _tokenize(arith_string)
    op_queue = _token_list_to_op_queue(tokens)
    result = _apply_op_queue_to_df(data_obj, op_queue)
    return result


def _tokenize(arith_string):
    """Split an expression into tokens

    Tokens will be the operator characters or parens, or complete words/numbers.

    Args:
        arith_string (string): expression to parse

    Returns: list of tokens

    """
    tokens = []
    cur = ''
    for c in arith_string:
        if c == ' ':
            continue
        if c in ARITH_TOKENS:
            # we found an operator
            if len(cur) > 0:
                tokens.append(cur)
            tokens.append(c)
            cur = ''
        else:
            cur += c
    if len(cur) > 0:
        tokens.append(cur)

    return tokens


def _token_list_to_op_queue(token_list):
    """Implementation of Dijkstra's shunting-yard algorithm

    Converts standard infix ordering to a queue of operations to perform

    Args:
        token_list (list of strings):

    Returns: re-ordered list

    """

    out_queue = []
    op_stack = []

    for tok in token_list:
        if tok not in ARITH_TOKENS and tok not in FUNCTIONS:
            out_queue.append(tok)
        elif tok in FUNCTIONS:
            op_stack.append(tok)
        elif tok in OPERATORS:
            while len(op_stack) > 0 and (
                op_stack[-1] in FUNCTIONS or
                (op_stack[-1] in OPERATORS and PREC_DICT[op_stack[-1]] > PREC_DICT[tok]) or
                (op_stack[-1] in OPERATORS and PREC_DICT[op_stack[-1]] == PREC_DICT[tok] and
                    ASSOC_DICT[op_stack[-1]] == 'L')
            ) and op_stack[-1] != '(':
                out_queue.append(op_stack.pop())
            op_stack.append(tok)
        elif tok == '(':
            op_stack.append(tok)
        elif tok == ')':
            while len(op_stack) > 0 and op_stack[-1] != '(':
                out_queue.append(op_stack.pop())
            op_stack.pop()  # discard '('

    while len(op_stack) > 0:
        out_queue.append(op_stack.pop())

    return out_queue


def _apply_op_queue_to_df(data_obj, op_queue):
    """Executes an operation queue as applied to a data object

    Strings in the queue that do not match an arith operator or function are used to look up
    the corresponding entries in the data object. Assumes a dict-style lookup,
    i.e. :code:`data_obj['varname']`.

    functions abs, log, and sqrt are applied element-wise
    functions sum, min, and max extract summary statistics for the named column

    Args:
        data_obj: object holding data to process
        op_queue: list of operation tokens in queued order

    Returns: result of calculations

    """
    stack = []
    for tok in op_queue:
        if tok not in ARITH_TOKENS and tok not in FUNCTIONS:
            # could be a data frame column or a number
            if tok in data_obj:
                stack.append(np.array(data_obj[tok]))
            else:
                stack.append(float(tok))
        elif tok in FUNCTIONS:
            # only single argument functions
            a = stack.pop()
            f = FUN_FUNC_DICT[tok]
            stack.append(f(a))
        elif tok in OPERATORS:
            b = stack.pop()
            a = stack.pop()
            f = OP_FUNC_DICT[tok]
            stack.append(f(a, b))
    return stack.pop()
