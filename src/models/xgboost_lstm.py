import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from torch.utils.data import DataLoader

class XGBoostLSTMClassifier(nn.Module):
    """
    Hybrid XGBoost-LSTM Model.
    - LSTM is used for temporal feature extraction.
    - A PyTorch linear surrogate is used to train LSTM end-to-end and generate adversarial attacks.
    - An actual XGBoost classifier is fitted on the extracted features for the final hybrid decision.
    """
    def __init__(self, input_size, num_classes, lstm_config):
        super().__init__()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_config.get("hidden_size", 128),
            num_layers=lstm_config.get("num_layers", 2),
            batch_first=True,
            bidirectional=lstm_config.get("bidirectional", True),
            dropout=lstm_config.get("dropout", 0.3) if lstm_config.get("num_layers", 2) > 1 else 0,
        )
        
        lstm_output_size = lstm_config.get("hidden_size", 128) * (2 if lstm_config.get("bidirectional", True) else 1)
        
        # Surrogate PyTorch classifier for gradient-based training and attacks
        self.surrogate_classifier = nn.Sequential(
            nn.Dropout(lstm_config.get("dropout", 0.3)),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(lstm_config.get("dropout", 0.3)),
            nn.Linear(lstm_output_size // 2, num_classes)
        )
        
        # Actual XGBoost model
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=6, 
            eval_metric='mlogloss', 
            use_label_encoder=False
        )
        self.xgb_fitted = False

    def extract_features(self, x):
        _, (hidden, _) = self.lstm(x)
        if self.lstm.bidirectional:
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            features = hidden[-1]
        return features

    def forward(self, x):
        features = self.extract_features(x)
        
        # If XGBoost is fitted and gradients are NOT required (inference/evaluation),
        # use the robust XGBoost predictions.
        if self.xgb_fitted and not x.requires_grad:
            features_np = features.detach().cpu().numpy()
            probs = self.xgb_model.predict_proba(features_np)
            return torch.tensor(probs, device=x.device, dtype=torch.float32)
            
        # Otherwise (during training or PGD attack generation), use surrogate classifier.
        return self.surrogate_classifier(features)
        
    def fit_xgboost(self, dataloader: DataLoader, device: torch.device):
        """
        Extract features from the trained LSTM and fit the XGBoost classifier.
        """
        self.eval()
        X_features = []
        Y_labels = []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(device)
                features = self.extract_features(X_batch)
                X_features.append(features.cpu().numpy())
                Y_labels.append(y_batch.numpy())
                
        X_train = np.vstack(X_features)
        Y_train = np.concatenate(Y_labels)
        
        print(f"Fitting XGBoost on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.xgb_model.fit(X_train, Y_train)
        self.xgb_fitted = True
        print("XGBoost trained successfully.")
