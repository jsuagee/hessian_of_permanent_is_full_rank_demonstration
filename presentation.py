from bs4 import BeautifulSoup as bs
import html
import argparse

import hessian
from html_templates import original_html, script
from matrix_operation_recorders import wrap_in_a_dataframe


#########################################################################
# Functions to generate html tables for each kind of matrix operation
# used to reduce the original matrix to final form.

def get_html_for_row_operation(df, row_to_modify, row_to_use):
    html = df.to_html()
    soup = bs(html, 'html.parser')
    table = soup.table

    tr_list = table.findAll("tr")
    i = list(df.index).index(row_to_modify)
    row = tr_list[i + 1]
    row["style"] = "background-color: red"

    i = list(df.index).index(row_to_use)
    row = tr_list[i + 1]
    row["style"] = "background-color: pink"
    return soup

def get_html_for_column_operation(df, col_to_modify, col_to_use):
    html = df.to_html()
    soup = bs(html, 'html.parser')

    indices = list(df.columns)
    col_to_modify_index = indices.index(col_to_modify) + 1
    col_to_use_index = indices.index(col_to_use) + 1

    spans, styles = [], []
    if col_to_use_index < col_to_modify_index:
        spans.append(col_to_use_index)
        styles.append("")
        spans.append(1)
        styles.append("background-color: pink")
        if col_to_modify_index >= col_to_use_index + 2:
            spans.append(col_to_modify_index - col_to_use_index - 1)
            styles.append("")
        spans.append(1)
        styles.append("background-color: red")
        if col_to_modify_index < len(indices) - 1:
            spans.append(len(indices) - col_to_modify_index - 1)
            styles.append("")
    else:
        spans.append(col_to_modify_index)
        styles.append("")
        spans.append(1)
        styles.append("background-color: red")
        if col_to_use_index >= col_to_modify_index + 2:
            spans.append(col_to_use_index - col_to_modify_index - 1)
            styles.append("")
        spans.append(1)
        styles.append("background-color: pink")
        if col_to_use_index < len(indices) - 1:
            spans.append(len(indices) - col_to_use_index - 1)
            styles.append("")

    table = soup.table
    col_group = soup.new_tag("colgroup")
    for span, style in zip(spans, styles):
        col = soup.new_tag("col")
        col["span"] = str(span)
        if style != "":
            col["style"] = style
        col_group.append(col)
    table.insert(0, col_group)

    return soup

def get_html_for_zero_out_row_operation(df, shared_index):
    html = df.to_html()
    soup = bs(html, 'html.parser')

    reverse_shared_index = (shared_index[1], shared_index[0])
    column_indices = list(df.columns)
    col_index = column_indices.index(reverse_shared_index) + 1

    spans, styles = [], []
    spans.append(col_index)
    styles.append("")
    spans.append(1)
    styles.append("background-color: pink")
    if col_index < len(column_indices):
        spans.append(len(column_indices) - col_index)
        styles.append("")

    table = soup.table
    col_group = soup.new_tag("colgroup")
    for span, style in zip(spans, styles):
        col = soup.new_tag("col")
        col["span"] = str(span)
        if style != "":
            col["style"] = style
        col_group.append(col)
    table.insert(0, col_group)

    tr_list = soup.findAll("tr")
    i = list(df.index).index(shared_index)
    row = tr_list[i + 1]
    row["style"] = "background-color: pink"

    return soup

def get_html_for_zero_out_column_operation(df, shared_index):
    # is actually the same as get_html_for_zero_out_row_operation() above.
    return get_html_for_zero_out_row_operation(df, shared_index)

# End of functions for constructing html tables for matrix operations.
#########################################################################


def get_tables(operations_sequence):
    '''
    Constructs all of the tables and messages that will be displayed in the html output file.
    :param operations_sequence: 2nd return value from a call of hessian.final_reduce()
    :return: (List of pandas dataframes,
              List of messages corresponding to each of these dataframes)
    '''
    # Append Initial dataframe:
    _, df = operations_sequence[0]
    df.replace(0, '.', inplace=True)
    df_html = df.to_html()
    soup = bs(df_html, 'html.parser')
    tables_html, messages = [], []
    tables_html.append(soup)
    messages.append("Initial State.")

    for (operation_type, (i, j), (k, l)), df in operations_sequence[1:-1]:
        df.replace(0, '.', inplace=True)
        if operation_type == "ZeroOutColumnRecord":
            msg = \
                "The " + str((j, i)) + "'th row has a single nonzero entry."
            msg += " Use this to zero out the whole column " + str((i, j)) + "."
            # f = lambda row: style_row_or_column_func(row, (i,j), (k,l))
            soup = get_html_for_zero_out_column_operation(df, (j, i))
        elif operation_type == "ZeroOutRowRecord":
            msg = \
                "The " + str((j, i)) + "'th column has a single nonzero entry."
            msg += " Use this to zero out the whole row " + str((i, j)) + "."
            soup = get_html_for_zero_out_row_operation(df, (i, j))
        elif operation_type == "CombinedColumnOperation":
            msg = "Subract column indexed by " + str((k, l)) + " from column indexed by " + str((i, j)) + "."
            soup = get_html_for_column_operation(df, (k, l), (i, j))
        elif operation_type == "CombinedRowOperation":
            msg = "Subract row indexed by " + str((k, l)) + " from row indexed by " + str((i, j)) + "."
            soup = get_html_for_row_operation(df, (k, l), (i, j))

        messages.append(msg)
        tables_html.append(soup)

    # Append final dataframe:
    _, df = operations_sequence[-1]
    df.replace(0, '.', inplace=True)
    df_html = df.to_html()
    soup = bs(df_html, 'html.parser')

    tables_html.append(soup)
    messages.append("Final State.")
    return tables_html, messages

def build_html(tables, messages):
    soup = bs(original_html, 'html.parser')

    message_container = soup.body.find("div", class_="message-container")
    for m in messages:
        p = soup.new_tag("p")
        p["class"] = "message"
        p["style"] = "display: none;"
        p.string = m
        message_container.append(p)

    slideshow_container = soup.find("div", class_="slideshow-container")
    for t in tables:
        div = soup.new_tag("div")
        div["style"] = "display: none;"
        div["class"] = "slide"
        div.append(t)
        slideshow_container.append(div)

    controls_container = soup.body.find("div", class_="controls-container")

    b = soup.new_tag("button")
    b["onclick"] = "plusSlides(-1)"
    b.string = "Backward"
    controls_container.append(b)
    b = soup.new_tag("button")
    b["onclick"] = "plusSlides(1)"
    b.string = "Forward"
    controls_container.append(b)

    script_tag = soup.new_tag("script")
    script_tag.append(script)
    soup.body.append(script_tag)
    return soup

def write_html_to_file(operations_sequence, filename):
    tables, messages = get_tables(operations_sequence)
    soup = build_html(tables, messages)
    with open(filename, 'w') as f:
        f.write(str(soup))

def main(args):
    d = args.d
    verbose_history = args.verbose_history

    index_map = hessian.IndexMap(d)
    Matrix_E = hessian.initial_matrix(d)
    H = hessian.Hessian(Matrix_E, pool_size=args.pool_size)

    # calculation of the permanent using the chosen implementation requires
    # complex dtypes, and returns complex dtypes. Our matrices are all real though
    # so we throw the imaginary parts away.
    H = hessian.makeitReal(H)

    # Truncation step:
    # Throw away all columns and rows corresponding to variables X_ij, where i = j.
    A = hessian.remainRow_Col(d, H)

    if verbose_history:
        A_reduced, operations_sequence = hessian.row_column_reduce(A, index_map,
                                                            verbose=verbose_history)
        A_final, final_operations_sequence = hessian.final_reduce(A_reduced,
                                                                  index_map,
                                                                  verbose=verbose_history)
        operations_sequence.extend(final_operations_sequence)
    else:
        # TODO: (temporarily assert verbose history so we throw an error.)
        assert verbose_history

    # For final summary:
    A_df = wrap_in_a_dataframe(A, index_map)

    if verbose_history:
        output_to_save = [(None, A_df)]
        remaining_output = [r.return_final_summary_form(index_map) for
                            r in operations_sequence]
        output_to_save.extend(remaining_output)
        output_reordered = [(op2, df1) for (op1, df1), (op2, df2)
                            in zip(output_to_save[:-1], output_to_save[1:])]

        output_reordered = [("initial state", A_df)] + output_reordered
        output_reordered += [("final state", output_to_save[-1][1])]

    write_html_to_file(output_reordered, args.output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-history", "-v", default=True,
                        action='store_true',
                        dest="verbose_history",
                        help="Record full history of all matrix operations used to \
                        reduce matrix to final form. (Defaults to True, and False \
                        option is not yet written.)")
    parser.add_argument("--dimension", "-d", type=int, default=4, dest="d",
                        help="Dimension of the original matrix. (Default is 4.)")
    parser.add_argument("--pool-size", "-p", type=int, default=4, dest="pool_size",
                        help="Number of processes to use when computing all of the \
                             permanents of the submatrices. (Default is 4.)")
    parser.add_argument("--output", "-o", default="", dest="output_filename",
                        help="File name for output. Should include .html extension. \
                        Defaults to output-d, where d is the dimension argument.")

    args = parser.parse_args()
    if args.output_filename == "":
        args.output_filename = "output-{}.html".format(args.d)

    main(args)






