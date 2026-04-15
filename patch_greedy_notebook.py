#!/usr/bin/env python3
import re

with open("build_greedy_notebook.py", "r") as f:
    content = f.read()

# 1. IMPORTS
content = content.replace(
    "from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerClassifier",
    "from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerClassifier\nfrom src.models.transformer import NLPTransformerClassifier\nfrom src.data.tokenizer import create_tokenizer"
)

# 2. CREATE_MODEL
create_model_old = """    elif model_type == 'cnn_bilstm_transformer':
        return CNNBiLSTMTransformerClassifier(input_size, num_classes, seq_length=SEQ_LENGTH)
    else:"""
create_model_new = """    elif model_type == 'cnn_bilstm_transformer':
        return CNNBiLSTMTransformerClassifier(input_size, num_classes, seq_length=SEQ_LENGTH)
    elif model_type == 'nlp_transformer':
        return NLPTransformerClassifier(vocab_size=52000, num_classes=num_classes, max_seq_length=576, pad_token_id=2)
    else:"""
content = content.replace(create_model_old, create_model_new)

# 3. TRAIN_GREEDY_PHASE DEF
content = content.replace(
    "    lr=5e-4, batch_size=64, save_path=None,\n):",
    "    lr=5e-4, batch_size=64, save_path=None,\n    is_nlp=False, tokenizer=None, features=None,\n):"
)

# 4. TRAIN_GREEDY_PHASE INNER LOOP
content = content.replace(
    """                    if p_drop > 0:
                        mask_c = (torch.rand(X_clean_t.shape[0], 1, X_clean_t.shape[2], device=device) > p_drop).float()
                        X_clean_t = X_clean_t * mask_c / (1.0 - p_drop)
                        mask_a = (torch.rand(X_adv_t.shape[0], 1, X_adv_t.shape[2], device=device) > p_drop).float()
                        X_adv_t = X_adv_t * mask_a / (1.0 - p_drop)""",
    """                    if is_nlp:
                        X_clean_t = torch.LongTensor(tokenizer.transform(X_batch.numpy(), features)).to(device)
                        X_adv_t = torch.LongTensor(tokenizer.transform(X_adv_mixed, features)).to(device)
                    # p_drop removed for simplicity on NLP in this branch or kept for non-nlp"""
)

# Replace the non-AFD mixed branch
mixed_branch_old = """                    X_input = torch.FloatTensor(X_mixed).to(device)
                    if p_drop > 0:"""
mixed_branch_new = """                    if is_nlp:
                        X_input = torch.LongTensor(tokenizer.transform(X_mixed, features)).to(device)
                    else:
                        X_input = torch.FloatTensor(X_mixed).to(device)
                    if p_drop > 0 and not is_nlp:"""
content = content.replace(mixed_branch_old, mixed_branch_new)

# Replace the non-mixed branch
clean_branch_old = """                X_input = X_batch.to(device)
                if p_drop > 0:"""
clean_branch_new = """                if is_nlp:
                    X_input = torch.LongTensor(tokenizer.transform(X_np, features)).to(device)
                else:
                    X_input = X_batch.to(device)
                if p_drop > 0 and not is_nlp:"""
content = content.replace(clean_branch_old, clean_branch_new)

# 5. CRASH_TEST_GREEDY
content = content.replace(
    "                      k_values=None, label=''):",
    "                      k_values=None, label='', is_nlp=False, tokenizer=None, features=None):"
)
content = content.replace(
    "    X_t = torch.FloatTensor(X_eval).to(device)",
    "    X_t = torch.LongTensor(tokenizer.transform(X_eval, features)).to(device) if is_nlp else torch.FloatTensor(X_eval).to(device)"
)
content = content.replace(
    "            X_adv_t = torch.FloatTensor(X_adv).to(device)",
    "            X_adv_t = torch.LongTensor(tokenizer.transform(X_adv, features)).to(device) if is_nlp else torch.FloatTensor(X_adv).to(device)"
)

# 6. DISCRIMINATOR and IOT ROUTER
content = content.replace(
    "    def __init__(self, normal_model, adversarial_model, discriminator, threshold=0.5):",
    "    def __init__(self, normal_model, adversarial_model, discriminator, threshold=0.5, is_nlp=False, tokenizer=None, features=None):\n        self.is_nlp = is_nlp\n        self.tokenizer = tokenizer\n        self.features = features"
)
content = content.replace(
    "        logits_adv = self.adversarial(X)",
    "        if hasattr(self, 'is_nlp') and self.is_nlp and self.tokenizer is not None:\n            # Need X as numpy for tokenizer\n            X_np = X.cpu().numpy()\n            X_ids = self.tokenizer.transform(X_np, self.features)\n            X_adj = torch.LongTensor(X_ids).to(X.device)\n            logits_adv = self.adversarial(X_adj)\n        else:\n            logits_adv = self.adversarial(X)",
    1
)

# 7. TRAIN_MODEL_GREEDY - Add Tokenizer fitting and pass params
train_func_old = """    input_size = X_train.shape[2]
    num_classes = len(np.unique(y_train))"""
train_func_new = """    input_size = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    is_nlp = (model_type == 'nlp_transformer')
    tokenizer = None
    if is_nlp:
        tokenizer = create_tokenizer()
        print(f"\\n  [TOKENIZER] Fitting BPE tokenizer on training data...")
        tokenizer.fit(X_train, features, verbose=False)"""
content = content.replace(train_func_old, train_func_new)

# Patch Phase A parameters
content = content.replace(
    "batch_size=batch_size, save_path=phase_a_path,",
    "batch_size=batch_size, save_path=phase_a_path, is_nlp=is_nlp, tokenizer=tokenizer, features=features"
)
content = content.replace(
    "label='Phase A')",
    "label='Phase A', is_nlp=is_nlp, tokenizer=tokenizer, features=features)"
)

# Patch Phase B parameters
content = content.replace(
    "batch_size=batch_size, save_path=phase_b_path,",
    "batch_size=batch_size, save_path=phase_b_path, is_nlp=is_nlp, tokenizer=tokenizer, features=features"
)
content = content.replace(
    "label='Phase B')",
    "label='Phase B', is_nlp=is_nlp, tokenizer=tokenizer, features=features)"
)

# Patch Phase C parameters
content = content.replace(
    "batch_size=batch_size, save_path=phase_c_path,",
    "batch_size=batch_size, save_path=phase_c_path, is_nlp=is_nlp, tokenizer=tokenizer, features=features"
)
content = content.replace(
    "label='Phase C')",
    "label='Phase C', is_nlp=is_nlp, tokenizer=tokenizer, features=features)"
)

# Patch Phase D parameters
content = content.replace(
    "batch_size=batch_size, save_path=phase_d_path,",
    "batch_size=batch_size, save_path=phase_d_path, is_nlp=is_nlp, tokenizer=tokenizer, features=features"
)
content = content.replace(
    "label='Phase D')",
    "label='Phase D', is_nlp=is_nlp, tokenizer=tokenizer, features=features)"
)

# Router instanciation patch
content = content.replace(
    "router = IoTRouter(normal_model, model, disc, threshold=0.5)",
    "router = IoTRouter(normal_model, model, disc, threshold=0.5, is_nlp=is_nlp, tokenizer=tokenizer, features=features)"
)

# Test evaluation loop patch for ADV k4
# need to adapt: X_adv_t = torch.FloatTensor(X_adv).to(device) to the is_nlp condition.
test_k4_old = """X_adv_t = torch.FloatTensor(X_adv).to(device)"""
test_k4_new = """X_adv_t = torch.LongTensor(tokenizer.transform(X_adv, features)).to(device) if is_nlp else torch.FloatTensor(X_adv).to(device)"""
# We only want to replace it in the test loop block, we can just replace all of them since it's functionally correct!
content = content.replace(test_k4_old, test_k4_new)

# Add nlp_transformer to the list of models for CSV and JSON
models_list_old = """models_csv = [
    ("lstm", "LSTM"),
    ("bilstm", "BiLSTM"),
    ("cnn_lstm", "CNN-LSTM"),
    ("xgboost_lstm", "XGBoost-LSTM"),
    ("transformer", "Transformer"),
    ("cnn_bilstm_transformer", "CNN-BiLSTM-Transformer"),
]"""
models_list_new = """models_csv = [
    ("lstm", "LSTM"),
    ("bilstm", "BiLSTM"),
    ("cnn_lstm", "CNN-LSTM"),
    ("xgboost_lstm", "XGBoost-LSTM"),
    ("transformer", "Transformer"),
    ("cnn_bilstm_transformer", "CNN-BiLSTM-Transformer"),
    ("nlp_transformer", "NLP-Transformer"),
]"""
content = content.replace(models_list_old, models_list_new)

with open("build_greedy_notebook.py", "w") as f:
    f.write(content)
print("build_greedy_notebook.py successfully patched.")
