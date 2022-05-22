from typing import List
from data.CsvDataConfig import CsvDataConfig
from data.CsvDataWrapper import CsvDataWrapper
from data.CsvModelTestResult import CsvModelTestResult
from matplotlib import pyplot as plt

from models.CsvModel import CsvModel

def get_x(d: List):
    return list(range(len(d)))

def plot_test_result(filename: str, model_test_result: CsvModelTestResult):
    test_result_df = model_test_result.to_dataframe()
    
    any_dc_in = lambda x: len([d for d in model_test_result.data_ref.data_columns if d in x]) > 0
    cols = [x for x in test_result_df.columns[1:] if any_dc_in(x)]

    pred_cols = [x for x in cols if "pred" in x]
    label_cols = [str(x).replace("_pred","_label") for x in pred_cols]
    input_cols = [str(x).replace("_pred","_input") for x in pred_cols]
    size = len(pred_cols)
    fig,axes = plt.subplots(1,size,figsize=(24*size,24))
    if size > 1:
        for j in range(size):
            col_pred = test_result_df[pred_cols[j]]
            col_label = test_result_df[label_cols[j]]
            col_input = test_result_df[input_cols[j]]
            axes[j].plot(get_x(col_label),col_label,color="black")
            axes[j].plot(get_x(col_input),col_input,color="red")
            axes[j].plot(get_x(col_pred),col_pred,color="blue")
            axes[j].title.set_text(pred_cols[j])
    else:
        col_pred = test_result_df[pred_cols[0]]
        col_label = test_result_df[label_cols[0]]
        col_input = test_result_df[input_cols[0]]
        axes.plot(get_x(col_label),col_label,color="black")
        axes.plot(get_x(col_input),col_input,color="red")
        axes.plot(get_x(col_pred),col_pred,color="blue")
        axes.title.set_text(pred_cols[0])
    
    fig.savefig(filename)
    plt.close()

def propgate_and_plot(filename: str, model: CsvModel, data_to_propgate: CsvDataWrapper, data_ref: CsvDataConfig, steps_forward: int):
    propogated_data = model.propagate_data(data_to_propgate, steps_forward)

    size = len(data_ref.data_columns)
    fig,axes = plt.subplots(1,size,figsize=(24*size,24))
    for i,col_name in enumerate(data_ref.data_columns):
        axes[i].plot(data_to_propgate.data.index,data_to_propgate[col_name],color='black')
        axes[i].plot(propogated_data.data.index,propogated_data[col_name],color='blue')

    plt.savefig(filename)
    plt.close()