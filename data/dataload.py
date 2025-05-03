import pandas as pd

#loading the first set of 10 files
parquet_files = [f"output_part_{i}.parquet" for i in range(11)]
dfs1 = [pd.read_parquet(file) for file in parquet_files]
df1 = pd.concat(dfs, ignore_index=True)

#loading the second set of 10 files
parquet_files1 = [f"output_part_{i}.parquet" for i in range(11,21)]
dfs2 = [pd.read_parquet(file) for file in parquet_files1]
df2 = pd.concat(dfs1, ignore_index=True)

#creating a full dataset using the Pandas concat function.
dfs=[df1,df2]
dff=pd.concat(dfs)

df=dff

