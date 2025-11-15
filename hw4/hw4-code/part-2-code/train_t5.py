import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    # A learning rate of 1e-1 is very high for finetuning. 
    # Let's set a more reasonable default for finetuning, e.g., 1e-4 or 5e-5
    parser.add_argument('--learning_rate', type=float, default=1e-4) 
    parser.add_argument('--weight_decay', type=float, default=0.01) # Added a small weight decay

    parser.add_argument('--scheduler_type', type=str, default="linear", choices=["none", "cosine", "linear"], # Changed default
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1, # Changed default
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=20, # Changed default
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=3, # Changed default
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='t5_finetune', # Changed default
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    # Generation hyperparameters
    parser.add_argument('--max_gen_length', type=int, default=128,
                        help="Maximum length for generated SQL queries")

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    
    # Use experiment name from args
    experiment_name = args.experiment_name
    gt_sql_path = os.path.join(f'data/dev.sql')
    # This ground truth record path is from the PDF/README
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')

    # Get tokenizer from dataloader
    tokenizer = dev_loader.dataset.tokenizer

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader, tokenizer,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
                'dev/best_f1': best_f1,
                'epoch': epoch
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            save_model(checkpoint_dir, model, best=True) # Save best model
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False) # Save latest model

        if epochs_since_improvement >= args.patience_epochs:
            print(f"No improvement in {args.patience_epochs} epochs. Stopping early.")
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) # Ignore padding index

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # Forward pass
        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
            labels=decoder_targets # T5 can compute loss internally if labels are provided
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            # T5 loss is already averaged over non-pad tokens
            # To get total loss, we need to find non-pad tokens
            num_tokens = (decoder_targets != PAD_IDX).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return 0.0
    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, tokenizer, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluation loop.
    '''
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    all_generated_queries = []

    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_gen_length,
        decoder_start_token_id=model.config.decoder_start_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=4, # Use beam search for better results
    )

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_input in tqdm(dev_loader, desc="Evaluating"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # --- 1. Calculate Loss ---
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
                labels=decoder_targets
            )
            loss = outputs.loss
            
            num_tokens = (decoder_targets != PAD_IDX).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # --- 2. Generate SQL Queries ---
            # We use .generate() for inference
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                generation_config=gen_config
            )

            # Decode generated IDs to strings
            generated_queries = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            all_generated_queries.extend(generated_queries)

    # --- 3. Compute Metrics ---
    # Save the generated queries and their records
    save_queries_and_records(all_generated_queries, model_sql_path, model_record_path)

    # Compute metrics by comparing generated files to ground truth
    try:
        sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
            gt_sql_pth, model_sql_path, gt_record_path, model_record_path
        )
    except Exception as e:
        print(f"Error computing metrics: {e}. Returning 0s.")
        sql_em, record_em, record_f1, model_error_msgs = 0.0, 0.0, 0.0, ["Metric computation error"] * len(all_generated_queries)

    # Calculate error rate
    num_errors = sum(1 for msg in model_error_msgs if msg)
    error_rate = num_errors / len(all_generated_queries) if all_generated_queries else 0

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Perform inference on the test set and save results.
    '''
    model.eval()
    tokenizer = test_loader.dataset.tokenizer
    all_generated_queries = []

    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_gen_length,
        decoder_start_token_id=model.config.decoder_start_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=4, # Use beam search
    )

    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_input in tqdm(test_loader, desc="Test Inference"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            # --- Generate SQL Queries ---
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                generation_config=gen_config
            )

            # Decode generated IDs to strings
            generated_queries = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            all_generated_queries.extend(generated_queries)

    # --- Save results ---
    print(f"Saving test predictions to {model_sql_path} and {model_record_path}")
    save_queries_and_records(all_generated_queries, model_sql_path, model_record_path)
    print("Test inference complete.")


def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)

    # Resize model embeddings to match the new tokenizer (with SQL tokens)
    print("Resizing model token embeddings...")
    model.resize_token_embeddings(len(train_loader.dataset.tokenizer))
    
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    print("Loading best model for final evaluation...")
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    model_type = 'ft' if args.finetune else 'scr'
    experiment_name = args.experiment_name # Use the name from args
    tokenizer = dev_loader.dataset.tokenizer

    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    
    print("Running final evaluation on dev set...")
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader, tokenizer,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Final Dev set results: Loss: {dev_loss:.4f}, Record F1: {dev_record_f1:.4f}, Record EM: {dev_record_em:.4f}, SQL EM: {dev_sql_em:.4f}")
    print(f"Final Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    # The file names need to match the submission requirements
    # e.g., t5_ft_test.sql and t5_ft_test.pkl
    # Let's adjust the naming logic slightly to match the PDF for the main assignment
    # Q8 asks for: t5_ft_experiment_test.pkl and t5_ft_experiment.test.sql
    # This is slightly different from the README. Let's follow the PDF
    # The README says: {t5_ft, ft_scr, gemma}_test.sql
    # Let's use the README format as it's more standard.
    # We'll use t5_ft_test.sql for finetune and t5_scr_test.sql for scratch
    
    test_file_prefix = f"t5_{model_type}_test" # This will be t5_ft_test or t5_scr_test
    
    # --- Check for Extra Credit ---
    # The PDF (page 8) implies the EC file names are different.
    # Let's assume if we train from scratch, we use a different name
    if not args.finetune:
        # This is the extra credit "from scratch"
        test_file_prefix = "t5_ft_experiment_ec_test" # Matching PDF for EC
        print("Running for Extra Credit (from scratch)")
    else:
        # This is the main assignment
        test_file_prefix = "t5_ft_experiment_test" # Matching PDF Q8
        print("Running for Main Assignment (finetune)")

    model_sql_path = os.path.join(f'results/{test_file_prefix}.sql')
    model_record_path = os.path.join(f'records/{test_file_prefix}.pkl')
    
    print("Running inference on test set...")
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    
    print("All done!")
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()