import pandas as pd


class MatrixOperation:
    def __init__(self):
        self.result = None

    def wrap_result_in_a_dataframe(self, row_index_map, col_index_map=None):
        if col_index_map == None:
            col_index_map = row_index_map
        return wrap_in_a_dataframe(self.result, row_index_map, col_index_map)

    def return_final_summary_form(self, row_index_map, col_index_map=None):
        A = self.wrap_result_in_a_dataframe(row_index_map, col_index_map)
        return (self.output_index_info(), A)


class RowOperation(MatrixOperation):
    def __init__(self, i, j, result,
                 row_index_map):
        '''
        Records result of elementary operation that subtracts row R' from row R.
        :param i: index of row R
        :param j: index of row R'
        :param result: resulting matrix
        '''
        self.type = "CombinedRowOperation"
        self.index_of_row_to_change = i
        self.index_of_row_to_use_in_subtraction = j
        self.result = result.copy()

        self.row_index_map = row_index_map

    def output_index_info(self):
        return (self.type, self.row_index_map.inverse(self.index_of_row_to_change),
                self.row_index_map.inverse(self.index_of_row_to_use_in_subtraction))


class ColumnOperation(MatrixOperation):
    def __init__(self, i, j, result,
                 column_index_map):
        '''
        Records result of elementary operation that subtracts column C' from column C.
        :param i: index of column C
        :param j: index of column C'
        :param result: resulting matrix
        '''
        self.type = "CombinedColumnOperation"
        self.index_of_column_to_change = i
        self.index_of_column_to_use_in_subtraction = j
        self.result = result.copy()

        self.column_index_map = column_index_map

    def output_index_info(self):
        return (self.type, self.column_index_map.inverse(self.index_of_column_to_change),
                self.column_index_map.inverse(self.index_of_column_to_use_in_subtraction))


class ZeroOutRowRecord(MatrixOperation):
    def __init__(self, column_index, row_index, result,
                 row_index_map, column_index_map):
        '''
        Supposing that the matrix contains a column with a single nonzero entry equal to 1,
        where this entry appears in the (i,j) position in the matrix.
        This object will record the result of using the j'th column to zero out all other entries
        in the matrix that appear in the i'th row.

        :param column_index: index of the column (j in the description above).
        :param row_index: i in the description above.
        :param result: The resulting matrix after zeroing out the other i'th row entries.
        '''
        self.type = "ZeroOutRowRecord"
        self.row_index = row_index
        self.column_index = column_index
        self.result = result.copy()

        self.row_index_map = row_index_map
        self.column_index_map = column_index_map

    def output_index_info(self):
        return (self.type, self.row_index_map.inverse(self.row_index),
                self.column_index_map.inverse(self.column_index))

class ZeroOutColumnRecord(MatrixOperation):
    def __init__(self, column_index, row_index, result,
                 row_index_map, column_index_map):
        '''
        Supposing that the matrix contains a row with a single nonzero entry equal to 1,
        where this entry appears in the (i,j) position in the matrix.
        This object will record the result of using the i'th row to zero out all other entries
        in the matrix that appear in the j'th column.

        :param column_index: index of the column (j in the description above).
        :param row_index: i in the description above.
        :param result: The resulting matrix after zeroing out the other i'th row entries.
        '''
        self.type = "ZeroOutColumnRecord"
        self.row_index = row_index
        self.column_index = column_index
        self.result = result.copy()

        self.row_index_map = row_index_map
        self.column_index_map = column_index_map

    def output_index_info(self):
        return (self.type, self.row_index_map.inverse(self.row_index),
                self.column_index_map.inverse(self.column_index))



def wrap_in_a_dataframe(A, row_index_map, col_index_map=None):
    '''
    Utility function that wraps a matrix in a dataframe for easier display and formatting.
    :param A: A matrix
    :param row_index_map: An IndexMap object
    :param col_index_map: An IndexMap object
    :return: A dataframe wrapping A
    '''
    if col_index_map == None:
        col_index_map = row_index_map
    nrows, ncols = A.shape
    assert nrows == ncols
    row_indices = [row_index_map.inverse(i) for i in range(nrows)]
    col_indices = [col_index_map.inverse(j) for j in range(nrows)]
    df = pd.DataFrame(A, columns=col_indices, index=row_indices)
    return df







