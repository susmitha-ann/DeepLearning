import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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

#Tokenization and Padding of Payloads (Text Data)
tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")
tokenizer.fit_on_texts(payloads)
vocab_size = len(tokenizer.word_index) + 1
max_len = 100  # Max sequence length (adjust as needed)
payload_sequences = tokenizer.texts_to_sequences(payloads)
payload_padded = pad_sequences(payload_sequences, maxlen=max_len, padding='post')

X_tcp, X_tcp_test, X_entropy, X_entropy_test, X_payload, X_payload_test, y_train, y_test = train_test_split(
    tcp_features, entropy_features, payload_padded, labels, test_size=0.2, random_state=42
)

model1 = load_model("model_10PF_K.keras")

#model evaluation
y_pred = model1.predict([X_tcp_test, X_entropy_test, X_payload_test])
y_pred_label = (y_pred > 0.5).astype(int)

print(confusion_matrix(y_test, y_pred_label))
print(classification_report(y_test, y_pred_label))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_label)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred_label))

# Plot
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Malicious', 'Malicious'],
            yticklabels=['Not Malicious', 'Malicious'],
            annot_kws={"color": "black"})  # ensures visibility
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()