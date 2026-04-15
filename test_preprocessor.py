import numpy as np

def create_sequences_with_categorical(X, y, seq_length=10, stride=10):
    X_seq, y_seq = [], []
    unique_labels = np.unique(y)

    for label in unique_labels:
        mask = y == label
        X_group = X[mask]
        y_group = y[mask]

        n = len(X_group) - seq_length + 1
        if n <= 0:
            continue
        for i in range(0, n, stride):
            if i + seq_length <= len(X_group):
                X_seq.append(X_group[i : i + seq_length])
                y_seq.append(y_group[i + seq_length - 1])

    return np.array(X_seq), np.array(y_seq)

# Mock 1.1M rows, 16 features
X_mock = np.random.rand(200, 16)
y_mock = np.zeros(200)

X_seq, y_seq = create_sequences_with_categorical(X_mock, y_mock, 10, 10)
print("X_seq shape:", X_seq.shape)
