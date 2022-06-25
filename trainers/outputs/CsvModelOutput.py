
##dto class representing the output of a csv training/testing
##output_dict[index_key] -> {loss: np.float32, input: np.array, output: np.array, labels: np.array}
import numpy as np
import pandas as pd

class CsvModelOutput:
    def __init__(self, time_delta):
        self.output_dict = {}
        self.time_delta = time_delta
    
    def add(self,key, loss, input, output, label):
        self.output_dict[key] = {"loss": loss, "input": input, "output": output, "label": label}
    
    @property
    def total_loss(self):
        return np.sum([v["loss"] for v in self.output_dict.values()])
    
    def get_column_dataframe(self, column_id, column_names):
        new_col_names = [column_id + "_" + c for c in column_names]
        inputs = {k:v[column_id] for k, v in self.output_dict.items()}
        output_df = {}
        for timestamp, input in inputs.items():
            ## input will be shape (Horizon, columns)
            H,C = input.shape
            assert C == len(column_names)
            tx, td = timestamp, self.time_delta
            for i in range(H):
                ti = tx - (H-i)*td
                if ti not in output_df:
                    output_df[ti] = input[i]
        return pd.DataFrame.from_dict(output_df,orient='index',columns=new_col_names).sort_index()


    # ok bear with me this is gonna get weird 
    def to_dataframe(self, column_names):
        input_df = self.get_column_dataframe("input",column_names)
        output_df = self.get_column_dataframe("output",column_names)
        labels_df = self.get_column_dataframe("label",column_names)
        df = pd.concat([input_df, output_df, labels_df],axis=1)
        df.to_csv("test_output/test.csv")
        return df