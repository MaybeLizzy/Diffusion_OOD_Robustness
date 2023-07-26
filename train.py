import yaml
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, BertModel, BertConfig
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers.modeling_bert_ood import BertForMaskedLM
from transformers.modeling_roberta_diffusion import RobertaForMaskedLM
from torch.utils.data import DataLoader
from torch import optim
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import os
from tqdm import tqdm, trange
from data import load

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
task_list = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k']
task = 'trec'   # choose an in-domain dataset for training
max_len = 512
model_path = "roberta-large"  
data_path = './dataset'

def main():
    config = yaml.safe_load(open('./config.yml'))
    
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["lr"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    adam_epsilon = config["adam_epsilon"]
    max_grad_norm = config["max_grad_norm"]
    logging_steps = config["logging_steps"]
    save_steps = config["save_steps"]
    max_steps = config["max_steps"]
    warmup_steps = config["warmup_steps"]

    # Initializing a model (with random weights) from roberta-large style configuration
    bertconfig = AutoConfig.from_pretrained(model_path, batch_size=batch_size)
    model = RobertaForMaskedLM.from_pretrained(model_path, config=bertconfig)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    out_dir = os.path.join("./", task, task+"_lr"+str(lr)+"_grad"+str(gradient_accumulation_steps)+"_bat"+str(batch_size) +"_step"+str(max_steps))
  
    def collate_fn(batch):
        max_len = max([len(f["input_ids"]) for f in batch])
        input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
        input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]     
        labels = [f["input_ids"] + [-100] * (max_len - len(f["input_ids"])) for f in batch]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": labels,
        }
        return outputs
    train_dataset, dev_dataset, test_dataset = load(task, tokenizer, max_seq_length=None, is_id=True, state=None, ood=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)

    t_total = max_steps
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0    
    tr_loss = 0.0
    
    model = model.to(device)
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(epochs), desc="Epoch")
    for epoch in train_iterator:
        print('Epoch{}'.format(epoch))
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()                      
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if inputs.shape[1]>max_len:
                inputs=inputs[:,:max_len]
            if labels.shape[1]>max_len:
                labels=labels[:,:max_len]
            logits, loss= model(inputs, labels=labels, return_dict=True)  

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            
            if (step + 1) % gradient_accumulation_steps == 0:                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % logging_steps == 0:    
                    logit = logits.logits
                    #predicted_token_id = logit[0].argmax(axis=-1)              
                    # Log metrics                             
                    prefix = "lr"+str(lr)+"_grad"+str(gradient_accumulation_steps)+"_bat"+str(batch_size) +"_step"+str(global_step)     
                    result = evaluate(model, dev_loader, prefix)  
                    for key in result.keys():
                        print("{0}={1}".format(key, str(float(result[key]))))                  
                
                if global_step % save_steps == 0:
                    # Save model checkpoint 
                    checkpoint_prefix = "checkpoint"
                    output_dir = os.path.join(out_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    print("***** Saving model checkpoint to ***** ", output_dir)                                                          
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)                   
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    print("Saving optimizer and scheduler states to ", output_dir)
            
            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break
    
    print(" global_step = {}, average loss = {}".format(global_step, tr_loss / global_step))
    

def evaluate(model, dev_loader, prefix):
    print("***** Running evaluation {} *****".format(prefix))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for batch in tqdm(dev_loader, desc="Evaluating"):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if inputs.shape[1]>max_len:
            inputs=inputs[:,:max_len]
        with torch.no_grad():
            outputs, lm_loss = model(inputs, return_dict=True)
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity, "loss": eval_loss}

    return result


if __name__ == "__main__":
    main()