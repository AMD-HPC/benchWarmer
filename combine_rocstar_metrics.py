import pandas as pd
import os
import glob

# Specify the directory
directory = '/home/khoffmey/work/PubBench/rocstar_metrics'

# Iterate through the files in the directory
runtimes_df = pd.read_csv('/home/khoffmey/work/PubBench/runtime_data/pubbench_results_all_gpus.csv')
print(runtimes_df.columns)
if 'Power' not in runtimes_df.columns:
    runtimes_df['Power'] = None
print(runtimes_df.columns)
for subdir in glob.glob(os.path.join(directory, '*/')):
    gpu = subdir.split('/')[-2]
    for file in glob.glob(os.path.join(subdir, '*')):
        filename = file.split('/')[-1]
        if len(filename.split('_')) == 5:
            if os.stat(file).st_size > 0:  # Check if the file is not empty
                power_df = pd.read_csv(file)
                power_df.columns = power_df.columns.str.strip()
                # print(file)
                # print(file.split('_'))
                nop, optype, datatype = filename.split('_')[2], filename.split('_')[3].upper(), filename.split('_')[-1][:-4].upper()
                # print(filename.split('_')[3])
                # print(nop, optype, datatype)
                if optype == 'RSQRT':
                    optype = 'RSQ'
                ai_col = [col for col in runtimes_df.columns if optype.upper() in col and datatype.upper() in col]
                if not ai_col:
                    print('ai not found!')
                    exit(1)
                
                max_power = power_df['Power[W]'].max()
                # print(max_power)
                runtimes_df.loc[
                    (runtimes_df['GPU'] == gpu) & 
                    (runtimes_df['Iterations'] == int(nop)) & 
                    (runtimes_df[ai_col[0]] != 0), 
                    'Power'
                ] = max_power


runtimes_df.to_csv('/home/khoffmey/work/PubBench/runtime_data/pubbench_results_all_gpus.csv', index=False)

# runtimes_df.query("GPU == @gpu and Iterations == @nop and @ai_col != 0")['Power'] == 
