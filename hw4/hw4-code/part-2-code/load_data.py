import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
# This line is commented out as it might cause issues in some environments
# nltk.download('punkt') 
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-base')
        # T5 uses pad_token_id (0) as the start token for decoding
        self.decoder_start_token_id = self.tokenizer.pad_token_id 
        
        self.data = self.process_data(data_folder, split)

    def process_data(self, data_folder, split):
        '''
        Loads and tokenizes the data.
        '''
        nl_path = os.path.join(data_folder, f"{split}.nl")
        sql_path = os.path.join(data_folder, f"{split}.sql")

        nl_lines = load_lines(nl_path)

        # For train and dev, we load both NL and SQL queries
        if split in ["train", "dev"]:
            sql_lines = load_lines(sql_path)
            assert len(nl_lines) == len(sql_lines), "Mismatch in number of examples"
            
            processed_data = []
            for nl, sql in tqdm(zip(nl_lines, sql_lines), desc=f"Tokenizing {split} set"):
                # We add a prefix for T5, which is a common practice for seq2seq tasks
                # This helps the model differentiate between tasks if it were multi-task trained
                input_text = f"translate Natural Language to SQL: {nl}"
                target_text = sql

                # Tokenize encoder input
                encoder_inputs = self.tokenizer(input_text, truncation=True, padding=False, return_tensors="pt")
                
                # Tokenize decoder input
                decoder_targets = self.tokenizer(target_text, truncation=True, padding=False, return_tensors="pt")

                processed_data.append({
                    "encoder_input_ids": encoder_inputs.input_ids.squeeze(0),
                    "encoder_attention_mask": encoder_inputs.attention_mask.squeeze(0),
                    "decoder_target_ids": decoder_targets.input_ids.squeeze(0),
                })
            return processed_data

        # For test set, we only have NL queries
        elif split == "test":
            processed_data = []
            for nl in tqdm(nl_lines, desc="Tokenizing test set"):
                input_text = f"translate Natural Language to SQL: {nl}"
                
                # Tokenize encoder input
                encoder_inputs = self.tokenizer(input_text, truncation=True, padding=False, return_tensors="pt")
                
                processed_data.append({
                    "encoder_input_ids": encoder_inputs.input_ids.squeeze(0),
                    "encoder_attention_mask": encoder_inputs.attention_mask.squeeze(0),
                })
            return processed_data
        
        else:
            raise ValueError(f"Unknown split: {split}")

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # For train/dev, return encoder and decoder inputs/targets
        if self.split in ["train", "dev"]:
            return (
                item["encoder_input_ids"],
                item["encoder_attention_mask"],
                item["decoder_target_ids"]
            )
        # For test, just return encoder inputs
        else:
            return (
                item["encoder_input_ids"],
                item["encoder_attention_mask"]
            )

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation.
    '''
    encoder_ids, encoder_masks, decoder_targets_list = zip(*batch)

    # Pad encoder inputs
    encoder_ids_padded = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask_padded = pad_sequence(encoder_masks, batch_first=True, padding_value=0) # Mask is 0 for padding

    # Pad decoder targets
    decoder_targets_padded = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)

    # Create decoder inputs by shifting targets to the right and adding start token
    # T5 uses the PAD token as the decoder start token.
    batch_size = encoder_ids_padded.shape[0]
    decoder_start_tokens = torch.full((batch_size, 1), PAD_IDX, dtype=torch.long)
    
    # Shift targets to the right: [START, t1, t2, ...]
    # We remove the last token from the padded targets to match lengths
    decoder_inputs_padded = torch.cat([decoder_start_tokens, decoder_targets_padded[:, :-1]], dim=-1)

    # The "initial_decoder_inputs" is just the start token for generation during eval
    # This matches the shape (batch_size, 1)
    initial_decoder_inputs = decoder_start_tokens

    return (
        encoder_ids_padded, 
        encoder_mask_padded, 
        decoder_inputs_padded, 
        decoder_targets_padded, 
        initial_decoder_inputs
    )


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.
    '''
    encoder_ids, encoder_masks = zip(*batch)

    # Pad encoder inputs
    encoder_ids_padded = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask_padded = pad_sequence(encoder_masks, batch_first=True, padding_value=0)

    # Create the initial decoder input (just the start token)
    batch_size = encoder_ids_padded.shape[0]
    initial_decoder_inputs = torch.full((batch_size, 1), PAD_IDX, dtype=torch.long)

    return (
        encoder_ids_padded, 
        encoder_mask_padded, 
        initial_decoder_inputs
    )


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_prompting_data(data_folder):
    # This is for Part 3, but we fill it in to make the file complete
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x