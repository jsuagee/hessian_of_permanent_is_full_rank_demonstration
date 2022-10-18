from permanent import permanent
import numpy as np
from multiprocessing import Pool
from itertools import product, permutations

from matrix_operation_recorders import RowOperation, ColumnOperation, \
ZeroOutRowRecord, ZeroOutColumnRecord

def index_matrix(n):
    size = n ** 2
    A = product(range(n), repeat=4)
    A = list(A)
    B = np.reshape(A, (size, size, 4))
    return B

def removeRow_Col(aMatrix, row1, row2, col1, col2):
    # remove row i and j
    # remove col k and l
    A = np.delete(aMatrix, [row1, row2], axis=0)
    B = np.delete(A, [col1, col2], axis=1)
    return B

def compute_perm_(aMatrix, row1, row2, col1, col2):
    if row1 == row2 or col1 == col2:
        return 0
    else:
        A = removeRow_Col(aMatrix, row1, row2, col1, col2)
        B = np.array(A, dtype=complex)
        return permanent.permanent(B)

def compute_perm(A, B, ij):
    '''
    This is a helper function to calculate the i,j'th entry of the Hessian matrix.
    :param A: A square numpy matrix
    :param B: 3 dimensional numpy array. B[r][s] is a 4 length array [i,j,k,l] where
              the row r corresponds to the variable X_ij, and column s corresponds to the variable
              X_kl. This is created by the index_mat() function.
    :param ij: tuple of indexes each from [1,...,n^2]
    :return:  Tuple of the coordinates (i,j), and the entry of the Hessian matrix
              corresponding to the ith column and jth row.
    '''
    i, j = ij
    row1 = int(B[i][j][0])
    row2 = int(B[i][j][2])
    col1 = int(B[i][j][1])
    col2 = int(B[i][j][3])
    return ((i, j), compute_perm_(A, row1, row2, col1, col2))

def Hessian(A, pool_size=10):
    '''
    Calculates the Hessian of the Permanent polynomial at a matrix A.
    :param A: A square matrix
    :param pool_size: number of processer pools to use (makes a difference if A is large)
    :return: H, the Hessian of the permanent polynomial evaluated at the matrix A.
    '''
    n = len(A)
    B = index_matrix(n)
    size = n ** 2
    # place holder for future calculated entries of the Hessian. Every entry will be computed
    # but to be safe we could have initialized all of these values to nan's.
    H = [[0] * (size) for _ in range(size)]

    args_list = ((A, B, ij) for ij in product(range(size), range(size)))
    p = Pool(pool_size)
    with p:
        result = p.starmap(compute_perm, args_list)
    for (i, j), v in result:
        H[i][j] = v
    return H


def makeitReal(aMatrix):
    n = len(aMatrix)
    A = [[0] * (n) for i in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = int(aMatrix[i][j].real)
    return A

def remainRow_Col(n, aMatrix):
    a = [x for x in range(n ** 2)]
    b = [x for x in range(0, n ** 2, n + 1)]
    row_idx = np.array([x for x in a if x not in b])
    return np.array(aMatrix)[row_idx[:, None], row_idx]

def initial_matrix(m):
    # returns initial matrix of dimension m
    A = np.eye(m, dtype=int)
    A[0,:] = 1
    A[:,0] = 1
    A[0,0] = 1 - m
    return A
    # n = m - 1
    # t = np.eye(n, dtype=int)
    # arr = [1 for i in range(n)]
    # # add arr as a column at position 0
    # t = np.insert(t, 0, arr, axis=1)
    # # add arr as a row at position 0
    # arr = [1 - m] + arr
    # t = np.insert(t, 0, arr, axis=0)
    # return np.array(t)

class IndexMap:
    def __init__(self, original_dimension, standard_order=True):
        self.original_dim = original_dimension
        self.index_dict = {}
        self.inverse_dict = {}

        if standard_order:
            ordering = permutations(range(self.original_dim), r=2)
        else:
            # TODO:
            assert standard_order
        c = 0
        for i, j in ordering:
            self.index_dict[(i, j)] = c
            self.inverse_dict[c] = (i, j)
            c += 1

    def __call__(self, i, j):
        return self.index_dict[(i, j)]

    def inverse(self, i):
        return self.inverse_dict[i]


def row_column_reduce(A, row_index_map,
               column_index_map=None,
               inplace=False,
               verbose=True):
    '''
    Performs all of the row and column reductions on A.
    :param A: A truncated Hessian matrix
    :param row_index_map: An IndexMap object
    :param column_index_map: An IndexMap object
    :param inplace: Whether to modify A in place or not
    :param verbose: Bool. If True then store full reduction history for later display
    :return: (a reduced hessian matrix, List of operations history)
    '''
    # remember base 0 indices
    if inplace:
        A_ = A
    else:
        A_ = np.copy(A)
    if column_index_map == None:
        column_index_map = row_index_map
    if verbose:
        operation_sequence_list = []

    d = row_index_map.original_dim

    # All column operations first:
    for k in range(1, d):
        for l in range(1, d):
            if l == k:
                continue
            k1 = column_index_map(k, 0)
            kl = column_index_map(k, l)
            # Column operation:
            A_[:, kl] = A_[:, kl] - A_[:, k1]
            if verbose:
                operations_record = ColumnOperation(kl, k1, A_, column_index_map)
                operation_sequence_list.append(operations_record)

    # Now all row operations:
    for k in range(1,d):
        for l in range(1,d):
            if l == k:
                continue
            k1 = row_index_map(k, 0)
            kl = row_index_map(k, l)
            # Row operation:
            A_[kl, :] = A_[kl, :] - A_[k1, :]
            if verbose:
                operations_record = RowOperation(kl, k1, A_, row_index_map)
                operation_sequence_list.append(operations_record)
    if verbose:
        return A_, operation_sequence_list
    else:
        return A_

def final_reduce(A, row_index_map, col_index_map=None,
                 inplace=False,
                 verbose=False):
    '''
    Performs all final row and column zeroing out operations.
    :param A: Numpy matrix
    :param row_index_map: IndexMap
    :param col_index_map: IndexMap (defaults to what was passed to the row_index_map)
    :param inplace: If True then modify A, if False then preserve A and output a new matrix.
    :param verbose: If True, produce a list of matrices that can be used to track all of the
                    operations performed in sequence.
    :return: A matrix if verbose=False,
             A tuple (matrix, sequence of matrices) if verbose=True
    '''
    #
    if col_index_map == None:
        col_index_map = row_index_map
    if inplace:
        A_ = A
    else:
        A_ = np.copy(A)
    if verbose:
        operation_sequence_list = []

    d = row_index_map.original_dim
    I = [row_index_map(i, 0) for i in range(1, d)]
    J = [col_index_map(j, 0) for j in range(1, d)]

    # This will pick out the first d columns which should be each with a single 1
    # entry. The i will be the column index and the row_index value will be the
    # row in that column that contains the single 1 value.
    for i, row_index in enumerate(I):
        # Following makes sure that the i'th column is indeed what we claim it is:
        #   all zeros, except for a 1 entry at the row_index.
        nonzero_index = np.nonzero(A_[:, i])
        assert A_[row_index, i] == 1
        assert nonzero_index[0].shape[0] == 1
        assert A_[nonzero_index, i] == 1

        # Now set all the other entries in that row equal to 0
        # (using elementary column operations):
        other_indices = np.arange(A_.shape[1], dtype=np.int32)
        A_[row_index, (other_indices != i)] = 0
        if verbose:
            operations_record = ZeroOutRowRecord(i, row_index, A_,
                                                 row_index_map, col_index_map)
            operation_sequence_list.append(operations_record)

    # Does the same thing with the first d rows:
    for j, column_index in enumerate(J):
        # Following makes sure that the j'th row is indeed what we claim it is:
        #   all zeros, except for a 1 entry at the column_index.
        nonzero_index = np.nonzero(A_[j, :])
        assert nonzero_index[0].shape[0] == 1
        assert A_[j, nonzero_index] == 1

        # Now set all the other entries in that column equal to 0
        # (using elementary row operations):
        other_indices = np.arange(A_.shape[1], dtype=np.int32)
        A_[(other_indices != j), column_index] = 0
        if verbose:
            operations_record = ZeroOutColumnRecord(j, column_index, A_,
                                                    row_index_map, col_index_map)
            operation_sequence_list.append(operations_record)
    if verbose:
        return A_, operation_sequence_list
    else:
        return A_


