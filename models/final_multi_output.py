import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Input Layer for TCP Header Features
tcp_input = Input(shape=(144,))
tcp_dense = Dense(16, activation='relu')(tcp_input)
tcp_dense = Dropout(0.3)(tcp_dense)
tcp_dense = Dense(8, activation='relu')(tcp_dense)
tcp_dense = Dropout(0.3)(tcp_dense)

# Input Layer for Entropy Features
entropy_input = Input(shape=(10,))
entropy_dense = Dense(16, activation='relu')(entropy_input)
entropy_dense = Dropout(0.3)(entropy_dense)
entropy_dense = Dense(8, activation='relu')(entropy_dense)

#Input Layer for Payload (Text)
payload_input = Input(shape=(max_len,))
embedding_layer = Embedding(vocab_size, 64, input_length=max_len)(payload_input)
lstm_layer = LSTM(64, dropout=0.3, recurrent_dropout=0.2)(embedding_layer)

#Merge all features
merged = Concatenate()([tcp_dense, entropy_dense, lstm_layer])
merged_dense = Dense(16, activation='relu')(merged)
merged_dense = Dropout(0.4)(merged_dense)
merged_dense = Dense(8, activation='relu')(merged_dense)
output = Dense(1, activation='sigmoid')(merged_dense)  # Sigmoid for binary classification

#Define and Compile Model
model = Model(inputs=[tcp_input, entropy_input, payload_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Model Summary
model.summary()

# Train Model
history = model.fit(
    [X_tcp, X_entropy, X_payload], y_train,
    validation_data=([X_tcp_test, X_entropy_test, X_payload_test], y_test),
    epochs=1, batch_size=32
)