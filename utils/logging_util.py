import os
import pandas as pd

def make_results_dir(results_path):
    os.makedirs(results_path, exist_ok=True)
    
def logging(PROJECT_NAME, fold, test_output, results_path="results"):
    loss = test_output[0]['test_loss']
    cindex = test_output[0]['test_c_index']
    results = [fold, loss, cindex]
    
    result_df = pd.DataFrame([results], 
                            columns=["Fold", "Loss", "C_index"])

    if os.path.exists(f"{results_path}/{PROJECT_NAME}.csv"):
        result_df_orig = pd.read_csv(f"{results_path}/{PROJECT_NAME}.csv")
        result_df = pd.concat([result_df_orig, result_df], axis=0)
        result_df = result_df.sort_values('Fold')
        result_df.to_csv(f"{results_path}/{PROJECT_NAME}.csv", index=False)
    else:
        result_df.to_csv(f"{results_path}/{PROJECT_NAME}.csv", index=False)