import pandas as pd

#loading the first set of 10 files
parquet_files = [f"../data/output_part_{i}.parquet" for i in range(11)]
dfs1 = [pd.read_parquet(file) for file in parquet_files]
df1 = pd.concat(dfs, ignore_index=True)
#loading the second set of 10 files
parquet_files1 = [f"../data/output_part_{i}.parquet" for i in range(11,21)]
dfs2 = [pd.read_parquet(file) for file in parquet_files1]
df2 = pd.concat(dfs1, ignore_index=True)
#creating a full dataset using the Pandas concat function.
dfs=[df1,df2]
dff=pd.concat(dfs)
dff ['Payload_filled'] = dff['PayloadStrings'].fillna("<EMPTY>").astype(str)
dff_OP = dff[dff['Payload_filled'] != "<EMPTY>"].copy()

df=dff_OP

#replace 'Benign' with 0 and 'Malicious' by 1
df['Label'] = df['Label'].replace({'Benign': 0, 'Malicious': 1})
df1 = df[df['Payload_filled'].apply(lambda x: len(x) <= 10000)]
df=df1

entropy_features = df.iloc[:, :10].values  # Next 10 columns (Entropy_1 to Entropy_10)
tcp_features = df.iloc[:, 10:154].values  # First 144 columns (binary TCP header)
payloads = df['Payload_filled'].astype(str)  # Text payload
labels = df['Label'].values  # Binary classification (0 = benign, 1 = malicious)

print(entropy_features)
print(tcp_features)
print(payloads)
print(labels)

# Convert and export each as CSV
pd.DataFrame(entropy_features).to_csv(f"{export_folder}/entropy_features.csv", index=False, header=False)
pd.DataFrame(tcp_features).to_csv(f"{export_folder}/tcp_features.csv", index=False, header=False)
payloads.to_csv(f"{export_folder}/payloads.csv", index=False, header=True)
pd.DataFrame(labels, columns=["Label"]).to_csv(f"{export_folder}/labels.csv", index=False)

print("Export completed.")