import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # This handles the download to the correct path
    nltk.download('punkt', quiet=True)
    
from transformers import T5TokenizerFast
import torch

# Import the schema helper from utils
from utils import get_schema_string

PAD_IDX = 0
# Using t5-small as per assignment (change to base only if needed and supported by compute)
T5_TOKENIZER = T5TokenizerFast.from_pretrained('google-t5/t5-small')

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Implementation of the T5 Dataset class.
        '''
        self.split = split
        self.tokenizer = T5_TOKENIZER
        
        # Load and flatten the schema
        # This assumes utils.py has the get_schema_string function
        self.schema_string = get_schema_string(os.path.join(data_folder, 'flight_database.schema'))
        
        # Paths to data files
        nl_path = os.path.join(data_folder, f'{split}.nl')
        sql_path = os.path.join(data_folder, f'{split}.sql')
        
        # Load raw text
        self.raw_nl = load_lines(nl_path)
        if os.path.exists(sql_path):
            self.raw_sql = load_lines(sql_path)
            assert len(self.raw_nl) == len(self.raw_sql)
        else:
            self.raw_sql = [""] * len(self.raw_nl) # For test set
            
        # Process and tokenize data
        self.inputs, self.outputs = self.process_data()

    def process_data(self):
        '''
        Tokenizes the data, adding the schema to the input.
        '''
        processed_inputs = []
        for nl in self.raw_nl:
            # CRITICAL FIX: Put Schema FIRST, Query LAST.
            # This ensures the query is never truncated if the schema is too long.
            input_text = f"translate English to SQL: schema: {self.schema_string} query: {nl}"
            processed_inputs.append(input_text)
            
        # Tokenize inputs
        # max_length=512 covers most schema+query combos
        tokenized_inputs = self.tokenizer(processed_inputs, padding=False, truncation=True, max_length=512)
        
        # Tokenize outputs (SQL)
        # max_length=256 gives plenty of room for complex queries
        tokenized_outputs = self.tokenizer(self.raw_sql, padding=False, truncation=True, max_length=256)
        
        return tokenized_inputs, tokenized_outputs

    def __len__(self):
        return len(self.raw_nl)

    def __getitem__(self, idx):
        input_ids = self.inputs['input_ids'][idx]
        attention_mask = self.inputs['attention_mask'][idx]
        
        output_ids = self.outputs['input_ids'][idx]
        
        # Create decoder_input_ids (starts with <pad>)
        decoder_input_ids = [PAD_IDX] + output_ids
        
        # Create decoder_target_ids (ends with <eos>)
        decoder_target_ids = output_ids + [self.tokenizer.eos_token_id]
        
        # For evaluation, we MUST return the raw SQL string
        raw_sql = self.raw_sql[idx]

        return {
            "encoder_input_ids": torch.tensor(input_ids, dtype=torch.long),
            "encoder_attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "decoder_target_ids": torch.tensor(decoder_target_ids, dtype=torch.long),
            "raw_sql": raw_sql
        }

def normal_collate_fn(batch):
    '''
    Collation function for train and dev.
    '''
    encoder_ids = [item['encoder_input_ids'] for item in batch]
    encoder_mask = [item['encoder_attention_mask'] for item in batch]
    decoder_inputs = [item['decoder_input_ids'] for item in batch]
    decoder_targets = [item['decoder_target_ids'] for item in batch]
    
    # We need this for the eval loop to compare against ground truth
    raw_sql_batch = [item['raw_sql'] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, raw_sql_batch

def test_collate_fn(batch):
    '''
    Collation function for the test set.
    '''
    encoder_ids = [item['encoder_input_ids'] for item in batch]
    encoder_mask = [item['encoder_attention_mask'] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=0)
    
    # Return empty lists for other items to match the train loop structure
    # The test loop expects 5 return values
    return encoder_ids, encoder_mask, [], [], []

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    
    # Use normal_collate_fn for train/dev, test_collate_fn for test
    collate_fn = test_collate_fn if split == "test" else normal_collate_fn 

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader

def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # This is for Part 3, leaving empty for now
    return [], [], [], [], []