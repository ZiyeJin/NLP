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
    nltk.download('punkt', quiet=True)
    
from transformers import T5TokenizerFast
import torch

from utils import get_schema_string

PAD_IDX = 0
T5_TOKENIZER = T5TokenizerFast.from_pretrained('google-t5/t5-small')

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Implementation of the T5 Dataset class.
        '''
        self.split = split
        self.tokenizer = T5_TOKENIZER
        
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
            
        # Process data
        self.inputs, self.outputs = self.process_data()

    def process_data(self):
        # Load schema string from utils.py
        schema_string = get_schema_string('data/flight_database.schema')
        
        # Create input strings (nl + schema)
        # This is our "pre-processing"
        processed_inputs = []
        for nl in self.raw_nl:
            input_text = f"translate English to SQL: query: {nl} schema: {schema_string}"
            processed_inputs.append(input_text)
            
        # Tokenize inputs and outputs
        # We handle padding in the collate_fn
        tokenized_inputs = self.tokenizer(processed_inputs, padding=False, truncation=True, max_length=512)
        tokenized_outputs = self.tokenizer(self.raw_sql, padding=False, truncation=True, max_length=128)
        
        return tokenized_inputs, tokenized_outputs

    def __len__(self):
        return len(self.raw_nl)

    def __getitem__(self, idx):
        # We need to create the decoder_target_ids which are the
        # decoder_input_ids shifted to the right.
        
        input_ids = self.inputs['input_ids'][idx]
        attention_mask = self.inputs['attention_mask'][idx]
        
        decoder_input_ids = [PAD_IDX] + self.outputs['input_ids'][idx]
        decoder_target_ids = self.outputs['input_ids'][idx] + [self.tokenizer.eos_token_id]
        
        # For evaluation, we also return the raw SQL string
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
    Collation function to perform dynamic padding.
    '''
    encoder_ids = [item['encoder_input_ids'] for item in batch]
    encoder_mask = [item['encoder_attention_mask'] for item in batch]
    decoder_inputs = [item['decoder_input_ids'] for item in batch]
    decoder_targets = [item['decoder_target_ids'] for item in batch]
    raw_sql_batch = [item['raw_sql'] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    
    # initial_decoder_inputs is just the <pad> token (batch_size, 1)
    # This is handled inside the T5 model, but we return it for consistency
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    
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
    
    # Not used in our main loop, but good to have
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)

    # We don't return raw_sql here since we don't have it for the test set
    return encoder_ids, encoder_mask, initial_decoder_inputs, [], []

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    # Use normal_collate_fn for all splits, as our new train_t5.py needs raw_sql for dev
    collate_fn = normal_collate_fn 

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

# load_prompting_data is not used by the T5 model, so we leave it
def load_prompting_data(data_folder):
    # TODO
    return [], [], [], [], []