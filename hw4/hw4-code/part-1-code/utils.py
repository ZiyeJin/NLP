import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import AutoModelForSeq2SeqLM

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

print("Loading translation models...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Back-Translation will use device: {DEVICE}")

# 1. Load English -> German model
EN_TO_DE_TOKENIZER = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
EN_TO_DE_MODEL = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(DEVICE)

# 2. Load German -> English model
DE_TO_EN_TOKENIZER = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
DE_TO_EN_MODEL = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en").to(DEVICE)

print("Translation models loaded.")

def example_transform(example):
    example["text"] = example["text"].lower()
    return example

def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # raise NotImplementedError
    texts = example['text']
    
    try:
        # 1. Translate English to German (in a batch)
        batch = EN_TO_DE_TOKENIZER(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        generated_ids = EN_TO_DE_MODEL.generate(**batch)
        german_texts = EN_TO_DE_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)

        # 2. Translate German back to English (in a batch)
        batch = DE_TO_EN_TOKENIZER(german_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        generated_ids = DE_TO_EN_MODEL.generate(**batch)
        back_translated_texts = DE_TO_EN_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 3. Assign the new list of texts back
        example['text'] = back_translated_texts
            
    except Exception as e:
        print(f"Back-translation failed on a BATCH. Error: {e}")
        # If the batch fails, just return the original texts for this batch
        pass

    ##### YOUR CODE ENDS HERE ######

    return example
