import re

with open('/home/pc/Desktop/pfe/greedy_new_optimized.py', 'r') as f:
    source = f.read()

# 1. Update Configuration
if "CSV_USE_BALANCED_PREPROCESSED = True" in source and "JSON_USE_BALANCED_PREPROCESSED" not in source:
    source = source.replace(
        "CSV_USE_BALANCED_PREPROCESSED = True\n",
        "CSV_USE_BALANCED_PREPROCESSED = True\n"
        "JSON_USE_BALANCED_PREPROCESSED = True\n"
        "JSON_SMOTE_FORCE_REBUILD = False\n"
        "JSON_SMOTE_CACHE_VERSION = 'v2-stronger-balance'\n"
        "JSON_SMOTE_TARGET_QUANTILE = 0.65\n"
        "JSON_SMOTE_MAX_MULTIPLIER = 128.0\n"
        "JSON_SMOTE_MAX_NEW_SAMPLES = 500000\n"
        "JSON_SMOTE_CONTEXT_MULTIPLIER = 1.25\n"
        "JSON_SMOTE_K_NEIGHBORS = 5\n"
        "JSON_SMOTE_RANDOM_STATE = 42\n"
    )
    
# 2. Update smote config getter
if "def get_csv_smote_config():" in source and "def get_json_smote_config():" not in source:
    new_func = """
def get_json_smote_config():
    return {
        'cache_version': JSON_SMOTE_CACHE_VERSION,
        'target_quantile': float(JSON_SMOTE_TARGET_QUANTILE),
        'max_multiplier': float(JSON_SMOTE_MAX_MULTIPLIER),
        'max_new_samples': int(JSON_SMOTE_MAX_NEW_SAMPLES),
        'context_multiplier': float(JSON_SMOTE_CONTEXT_MULTIPLIER),
        'k_neighbors': int(JSON_SMOTE_K_NEIGHBORS),
        'random_state': int(JSON_SMOTE_RANDOM_STATE),
    }

"""
    source = source.replace("def get_csv_smote_config():", new_func + "def get_csv_smote_config():")
    
# 3. Apply smote function generic
if "def apply_smote_to_preprocessed_csv" in source and "def apply_smote_to_preprocessed_dataset" not in source:
    source = source.replace("def apply_smote_to_preprocessed_csv(data, save_dir=None, force_rebuild=False):", 
                            "def apply_smote_to_preprocessed_dataset(data, dataset_type, save_dir=None, force_rebuild=False):")
    
    source = source.replace("if not CSV_USE_BALANCED_PREPROCESSED:", 
                            "use_balanced = CSV_USE_BALANCED_PREPROCESSED if dataset_type == 'csv' else JSON_USE_BALANCED_PREPROCESSED\n    if not use_balanced:")
    
    source = source.replace("smote_config = get_csv_smote_config()", 
                            "smote_config = get_csv_smote_config() if dataset_type == 'csv' else get_json_smote_config()")
                            
    source = source.replace("_load_preprocessed_dataset(save_dir, 'csv', expected_smote_config=smote_config)", 
                            "_load_preprocessed_dataset(save_dir, dataset_type, expected_smote_config=smote_config)")
    
    source = source.replace("print('  imbalanced-learn is not installed; returning the original preprocessed CSV.')", 
                            "print(f'  imbalanced-learn is not installed; returning the original preprocessed {dataset_type.upper()}.')")
                            
    source = source.replace("print('  Building a capped class-wise SMOTE cache for the CSV training split...')", 
                            "print(f'  Building a capped class-wise SMOTE cache for the {dataset_type.upper()} training split...')")
                            
    source = source.replace("CSV_SMOTE_MAX_NEW_SAMPLES", "smote_config['max_new_samples']")
    source = source.replace("CSV_SMOTE_TARGET_QUANTILE", "smote_config['target_quantile']")
    source = source.replace("CSV_SMOTE_MAX_MULTIPLIER", "smote_config['max_multiplier']")
    source = source.replace("CSV_SMOTE_CONTEXT_MULTIPLIER", "smote_config['context_multiplier']")
    source = source.replace("CSV_SMOTE_K_NEIGHBORS", "smote_config['k_neighbors']")
    source = source.replace("CSV_SMOTE_RANDOM_STATE", "smote_config['random_state']")
    
    # Need to fix build_smote_augmentation_plan and _generate_classwise_smote_samples
    source = source.replace("def build_smote_augmentation_plan(y_train):", 
                            "def build_smote_augmentation_plan(y_train, smote_config):")
    source = source.replace("def _generate_classwise_smote_samples(X_train, y_train, class_id, target_count, rng):", 
                            "def _generate_classwise_smote_samples(X_train, y_train, class_id, target_count, rng, smote_config):")
                            
    source = source.replace("build_smote_augmentation_plan(y_train)", "build_smote_augmentation_plan(y_train, smote_config)")
    source = source.replace("_generate_classwise_smote_samples(X_train, y_train, class_id, target_count, rng)", "_generate_classwise_smote_samples(X_train, y_train, class_id, target_count, rng, smote_config)")
    
    source = source.replace("_save_preprocessed_dataset(save_dir, 'csv', balanced, smote_config=smote_config)", 
                            "_save_preprocessed_dataset(save_dir, dataset_type, balanced, smote_config=smote_config)")
    source = source.replace("print(f'  Balanced CSV cache saved to {save_dir}')", 
                            "print(f'  Balanced {dataset_type.upper()} cache saved to {save_dir}')")
                            
# 4. Fix calls in load_and_display_csv_dataset
if "apply_smote_to_preprocessed_csv(" in source:
    source = source.replace("apply_smote_to_preprocessed_csv(", "apply_smote_to_preprocessed_dataset(")
    source = source.replace("data,", "data,\n        'csv',")
    
# 5. Add SMOTE call in load_and_display_json_dataset
if "def load_and_display_json_dataset" in source and "apply_smote_to_preprocessed_dataset" not in source:
    source = source.replace("return data", 
                            "return apply_smote_to_preprocessed_dataset(\n        data,\n        'json',\n        save_dir=JSON_SMOTE_PREPROCESSED_DIR if JSON_USE_BALANCED_PREPROCESSED else None,\n        force_rebuild=JSON_SMOTE_FORCE_REBUILD,\n    )")
                            
# 6. Update load_dataset_from_drive
if "def load_dataset_from_drive(dataset_type):" in source and "JSON_SMOTE_PREPROCESSED_DIR" not in source.split("def load_dataset_from_drive(dataset_type):")[1]:
    # This is complex, better to rewrite the function
    new_func = """def load_dataset_from_drive(dataset_type):
    if dataset_type == 'csv' and CSV_USE_BALANCED_PREPROCESSED:
        cached = _load_preprocessed_dataset(
            CSV_SMOTE_PREPROCESSED_DIR,
            'csv',
            expected_smote_config=get_csv_smote_config(),
        )
        if cached is not None and not CSV_SMOTE_FORCE_REBUILD:
            print('  Loading cached balanced CSV from Drive...')
            return cached
            
    if dataset_type == 'json' and JSON_USE_BALANCED_PREPROCESSED:
        cached = _load_preprocessed_dataset(
            JSON_SMOTE_PREPROCESSED_DIR,
            'json',
            expected_smote_config=get_json_smote_config(),
        )
        if cached is not None and not JSON_SMOTE_FORCE_REBUILD:
            print('  Loading cached balanced JSON from Drive...')
            return cached

    preprocessed_dir = f'{DRIVE_RESULTS_DIR}/preprocessed/{dataset_type}'
    data = _load_preprocessed_dataset(preprocessed_dir, dataset_type)
    if data is not None:
        if dataset_type == 'csv':
            return apply_smote_to_preprocessed_dataset(
                data,
                'csv',
                save_dir=CSV_SMOTE_PREPROCESSED_DIR if CSV_USE_BALANCED_PREPROCESSED else None,
                force_rebuild=CSV_SMOTE_FORCE_REBUILD,
            )
        else:
            return apply_smote_to_preprocessed_dataset(
                data,
                'json',
                save_dir=JSON_SMOTE_PREPROCESSED_DIR if JSON_USE_BALANCED_PREPROCESSED else None,
                force_rebuild=JSON_SMOTE_FORCE_REBUILD,
            )

    print(f'  No preprocessed data found. Loading {dataset_type.upper()} fresh...')
    if dataset_type == 'csv':
        return load_and_display_csv_dataset(
            CSV_DATA_DIR, seq_length=SEQ_LENGTH, stride=STRIDE,
            save_dir=CSV_PREPROCESSED_DIR
        )
    return load_and_display_json_dataset(
        JSON_DATA_DIR, seq_length=SEQ_LENGTH, stride=STRIDE,
        max_records=MAX_RECORDS, save_dir=JSON_PREPROCESSED_DIR
    )
"""
    # Replace the old function
    pattern = r"def load_dataset_from_drive\(dataset_type\):.*?(?=CSV_PREPROCESSED_DIR =)"
    source = re.sub(pattern, new_func + "\n", source, flags=re.DOTALL)
    
if "CSV_SMOTE_PREPROCESSED_DIR =" in source and "JSON_SMOTE_PREPROCESSED_DIR =" not in source:
    source = source.replace("CSV_SMOTE_PREPROCESSED_DIR = f'{DRIVE_RESULTS_DIR}/preprocessed/csv_smote'",
                            "CSV_SMOTE_PREPROCESSED_DIR = f'{DRIVE_RESULTS_DIR}/preprocessed/csv_smote'\nJSON_SMOTE_PREPROCESSED_DIR = f'{DRIVE_RESULTS_DIR}/preprocessed/json_smote'")

with open('/home/pc/Desktop/pfe/greedy_new_optimized_patched.py', 'w') as f:
    f.write(source)

print("Patched py!")
