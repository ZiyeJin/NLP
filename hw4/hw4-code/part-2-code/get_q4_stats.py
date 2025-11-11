import os
from transformers import T5TokenizerFast
from load_data import T5Dataset # Import our new dataset class

def get_q4_statistics():
    print("--- Starting Q4 Statistics ---")
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Load datasets
    train_dataset = T5Dataset(data_folder='data', split='train')
    dev_dataset = T5Dataset(data_folder='data', split='dev')
    
    datasets = {'Train': train_dataset, 'Dev': dev_dataset}
    stats = {}

    # --- Table 1: Before Pre-processing ---
    print("\n--- Q4: Table 1 (Data statistics BEFORE any pre-processing) ---")
    
    # Calculate stats for Table 1
    for split_name, dataset in datasets.items():
        stats[split_name] = {}
        stats[split_name]['Number of examples'] = len(dataset)
        
        # Tokenize raw NL
        nl_tokenized = tokenizer(dataset.raw_nl, padding=False, truncation=False)
        nl_lengths = [len(seq) for seq in nl_tokenized['input_ids']]
        stats[split_name]['Mean sentence length'] = sum(nl_lengths) / len(nl_lengths)
        
        # Tokenize raw SQL
        sql_tokenized = tokenizer(dataset.raw_sql, padding=False, truncation=False)
        sql_lengths = [len(seq) for seq in sql_tokenized['input_ids']]
        stats[split_name]['Mean SQL query length'] = sum(sql_lengths) / len(sql_lengths)
        
        # Vocab
        nl_vocab = set(token for seq in nl_tokenized['input_ids'] for token in seq)
        sql_vocab = set(token for seq in sql_tokenized['input_ids'] for token in seq)
        stats[split_name]['Vocabulary size (natural language)'] = len(nl_vocab)
        stats[split_name]['Vocabulary size (SQL)'] = len(sql_vocab)

    # Print Table 1
    print(f"{'Statistics Name':<35} | {'Train':<15} | {'Dev':<15}")
    print("-" * 67)
    print(f"{'Number of examples':<35} | {stats['Train']['Number of examples']:<15} | {stats['Dev']['Number of examples']:<15}")
    print(f"{'Mean sentence length':<35} | {stats['Train']['Mean sentence length']:<15.2f} | {stats['Dev']['Mean sentence length']:<15.2f}")
    print(f"{'Mean SQL query length':<35} | {stats['Train']['Mean SQL query length']:<15.2f} | {stats['Dev']['Mean SQL query length']:<15.2f}")
    print(f"{'Vocabulary size (natural language)':<35} | {stats['Train']['Vocabulary size (natural language)']:<15} | {stats['Dev']['Vocabulary size (natural language)']:<15}")
    print(f"{'Vocabulary size (SQL)':<35} | {stats['Train']['Vocabulary size (SQL)']:<15} | {stats['Dev']['Vocabulary size (SQL)']:<15}")
    print("-" * 67)

    # --- Table 2: After Pre-processing ---
    print("\n--- Q4: Table 2 (Data statistics AFTER pre-processing) ---")

    # Calculate stats for Table 2
    for split_name, dataset in datasets.items():
        # Number of examples is the same
        
        # Tokenize PROCESSED inputs (nl + schema)
        # We can just use the tokenized lengths from the dataset object
        nl_lengths = [len(seq) for seq in dataset.inputs['input_ids']]
        stats[split_name]['Mean sentence length'] = sum(nl_lengths) / len(nl_lengths)
        
        # SQL stats are the same as before
        
        # Vocab for processed inputs
        nl_vocab = set(token for seq in dataset.inputs['input_ids'] for token in seq)
        stats[split_name]['Vocabulary size (natural language)'] = len(nl_vocab)

    # Print Table 2
    print(f"{'Statistics Name':<35} | {'Train':<15} | {'Dev':<15}")
    print("-" * 67)
    print(f"{'Mean sentence length':<35} | {stats['Train']['Mean sentence length']:<15.2f} | {stats['Dev']['Mean sentence length']:<15.2f}")
    print(f"{'Mean SQL query length':<35} | {stats['Train']['Mean SQL query length']:<15.2f} | {stats['Dev']['Mean SQL query length']:<15.2f}")
    print(f"{'Vocabulary size (natural language)':<35} | {stats['Train']['Vocabulary size (natural language)']:<15} | {stats['Dev']['Vocabulary size (natural language)']:<15}")
    print(f"{'Vocabulary size (SQL)':<35} | {stats['Train']['Vocabulary size (SQL)']:<15} | {stats['Dev']['Vocabulary size (SQL)']:<15}")
    print("-" * 67)

if __name__ == "__main__":
    get_q4_statistics()