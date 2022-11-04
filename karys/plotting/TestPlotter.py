from typing import List
from karys.data.configs.CsvDataConfig import CsvDataConfig
from karys.data.wrappers.CsvDataWrapper import CsvDataWrapper
from matplotlib import pyplot as plt


def get_x(d: List):
    return list(range(len(d)))

# def plot_test_result(filename: str, model_test_result: CsvModelOutput, column_names: List[str]):
#     test_result_df = model_test_result.to_dataframe(column_names).dropna()
#     cols = test_result_df.columns
#     input_cols = [x for x in cols if "input" in x]
#     label_cols = [x for x in cols if "label" in x]
#     output_cols = [x for x in cols if "output" in x]
#     size = len(input_cols)
#     fig,axes = plt.subplots(1,size,figsize=(24*size,24))
#     if size > 1:
#         for j in range(size):
#             col_out = test_result_df[output_cols[j]]
#             col_label = test_result_df[label_cols[j]]
#             col_input = test_result_df[input_cols[j]]
#             axes[j].plot(get_x(col_label),col_label,color="black")
#             axes[j].plot(get_x(col_input),col_input,color="red")
#             axes[j].plot(get_x(col_out),col_out,color="blue")
#             axes[j].title.set_text(output_cols[j])
#     else:
#         col_out = test_result_df[output_cols[0]]
#         col_label = test_result_df[label_cols[0]]
#         col_input = test_result_df[input_cols[0]]
#         axes.plot(get_x(col_label),col_label,color="black")
#         axes.plot(get_x(col_input),col_input,color="red")
#         axes.plot(get_x(col_out),col_out,color="blue")
#         axes.title.set_text(output_cols[0])
    
#     fig.savefig(filename)
#     plt.close()

def plot_propogated_data(filename: str, propogated_datawrapper: CsvDataWrapper, data_ref: CsvDataConfig):
    size = len(data_ref.data_columns)
    fig,axes = plt.subplots(1,size,figsize=(24*size,24))
    for i,col_name in enumerate(data_ref.data_columns):
        axes[i].plot(propogated_datawrapper.data.index,propogated_datawrapper[col_name],color='black')
        axes[i].plot(propogated_datawrapper.data.index,propogated_datawrapper[col_name],color='blue')

    plt.savefig(filename)
    plt.close()