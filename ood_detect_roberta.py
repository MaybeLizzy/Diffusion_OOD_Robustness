import yaml
import torch
from transformers import AutoTokenizer
from transformers import set_seed
from transformers import (
    AutoTokenizer,
)
from tqdm import tqdm
import numpy as np
from data import load
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42
domain_list = ['rte', 'sst2', 'trec', 'imdb', 'mnli', '20ng', 'wmt16', 'multi30k']
score_list = ["diffusion", "diffusion+maha"]    
in_domain = 'sst2'   # choose in-domain dataset
score = "diffusion"   # choose a method
lamda = 0.99
data_path = './dataset'

def main():
    # Initializing a model (with random weights) from a previously-finetuned roberta model    
    from transformers.modeling_roberta_diffusion import RobertaForMaskedLM
    if in_domain == "trec":
        if score == "diffusion":
            dev_t = 600
            model_name_or_path = "./model/trec/loss"
        elif score == "diffusion+maha":
            dev_t = 550
            model_name_or_path = "./model/trec/maha"
        ood_domain = ['rte', 'multi30k', 'sst2', '20ng', 'mnli', 'wmt16', 'imdb', 'multi30k']     # out-of-domain dataset  
        max_len=None
    elif in_domain == "20ng":
        ood_domain = ['rte', 'trec', 'sst2', 'imdb', 'mnli', 'wmt16', 'multi30k']     # out-of-domain dataset
        model_name_or_path = "./model/20ng"
        max_len = None
        dev_t = 600
    elif in_domain=="sst2":
        ood_domain = ['trec', '20ng', 'rte', 'mnli', 'wmt16', 'multi30k']     # out-of-domain dataset
        model_name_or_path = "./model/sst2"
        max_len = None
        dev_t = 500            
    elif in_domain == "imdb":
        ood_domain = ['multi30k', 'trec', 'wmt16', 'rte', 'mnli', '20ng']     # out-of-domain dataset
        model_name_or_path = "./model/imdb"
        if score == 'diffusion':
            max_len = None
        else:
            max_len = 512
        dev_t = 700
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = RobertaForMaskedLM.from_pretrained(model_name_or_path)
     
    model.to(device)
    model.eval()
    set_seed()

    train_dataset, dev_dataset, test_dataset = load(in_domain, tokenizer, max_seq_length=max_len, is_id=True, state=None, ood=True)     
    
    if score != "diffusion":
        print("***** Calculate the loss and maha center of ID train split*****")
        train_loss = 0.0
        train_loss_list = []
        nb_eval_steps = 0
        bank_ = None
        label_bank_ = None
        for batch in tqdm(train_dataset, desc="Evaluating"):
            input_ids = torch.tensor(batch.input_ids).unsqueeze(dim=0).to(device)
            labels = torch.tensor([batch.labels]).to(device)
            if input_ids.shape[1]>512:
                input_ids=input_ids[:,:512]
            with torch.no_grad():
                nb_eval_steps += 1
                outputs, lm_loss = model(input_ids, return_dict=True, state="dev", dev_t=dev_t)
                train_loss += outputs.loss.item()                        
                train_loss_list.append(float(outputs.loss))
                # prepare ood
                pooled = outputs.hidden_states                   
                pooled=pooled[len(pooled)-2]  
                pooled = pooled.mean(1)                   
                if bank_ is None:
                    bank_ = pooled.clone().detach()
                    label_bank_ = labels.clone().detach()
                else:
                    bank = pooled.clone().detach()
                    label_bank = labels.clone().detach()
                    bank_ = torch.cat([bank, bank_], dim=0)
                    label_bank_ = torch.cat([label_bank, label_bank_], dim=0)            
            
        train_loss = train_loss / nb_eval_steps
        train_loss_list.sort(reverse = True)
        print("train largest, smallest:",train_loss_list[0], train_loss_list[len(train_loss_list)-1])      
        class_mean = bank_.mean(0)         
        centered_bank = (bank_ - class_mean).detach().cpu().numpy()         
        precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)          
        class_var = torch.from_numpy(precision).float().to(device)
        
    
    print("***** Calculate the loss of ID test split *****")
    print("  Num examples = {}".format(len(test_dataset)))
    eval_loss = 0.0
    nb_eval_steps = 0
    loss_list = []          
    maha_scores = []
    for batch in tqdm(test_dataset, desc="Evaluating"):
        input_ids = torch.tensor(batch.input_ids).unsqueeze(dim=0).to(device)
        labels = torch.tensor(batch.labels).to(device)
        if input_ids.shape[1]>512:
            input_ids=input_ids[:,:512]
        
        with torch.no_grad():            
            outputs, lm_loss = model(input_ids, return_dict=True, state="dev", dev_t=dev_t)
            loss = float(outputs.loss.item())
            eval_loss += loss            
            loss_list.append(loss)
            nb_eval_steps += 1
            if score != "diffusion":
                # maha score
                pooled = outputs.hidden_states
                pooled=pooled[len(pooled)-2]
                pooled = pooled.mean(1) 
                maha_score = []
                centered_pooled = pooled - class_mean.unsqueeze(0)
                ms = torch.diag(centered_pooled @ class_var @ centered_pooled.t())
                maha_score.append(ms)
                maha_score = torch.stack(maha_score, dim=-1)
                maha_score = maha_score.min(-1)[0]
                maha_scores.append(float(maha_score))          
            
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    result = {"perplexity": perplexity, "avg loss": eval_loss}
    print("result:",result)
    if len(maha_scores)!=0:
        print(sum(maha_scores)/len(maha_scores)) 

    

    for ood in ood_domain:
        print(ood)        
        if in_domain == "sst2" and ood == "multi30k":
            max_len = 128
        if in_domain == "imdb":
            if ood == "20ng" and score == "diffusion":
                max_len = None
            else:
                max_len = 512
        ood_train_dataset, ood_dev_dataset, ood_test_dataset = load(ood, tokenizer, max_seq_length=max_len, is_id=True, state=None,ood=False)     
                
        print("***** Calculate the loss of OOD test split *****")
        print("  Num examples = {}".format(len(ood_test_dataset)))
                    
        ood_eval_loss = 0.0
        ood_nb_eval_steps = 0
        ood_loss_list = []        
        true_ = []
        ood_maha_scores = []
        for batch in tqdm(ood_test_dataset, desc="Evaluating"):
            input_ids = torch.tensor(batch.input_ids).unsqueeze(dim=0).to(device)
            labels = torch.tensor(batch.labels).to(device)
            true_.append(int(labels))
            if input_ids.shape[1]>512:
                input_ids=input_ids[:,:512]
            with torch.no_grad():                
                outputs, lm_loss = model(input_ids, return_dict=True, state="dev", dev_t=dev_t)
                loss = float(outputs.loss.item())
                ood_eval_loss += loss           
                ood_loss_list.append(loss)
                ood_nb_eval_steps += 1
                if score != "diffusion":
                    # maha score
                    pooled = outputs.hidden_states
                    pooled=pooled[len(pooled)-2]
                    pooled = pooled.mean(1)
                    maha_score = []
                    centered_pooled = pooled - class_mean.unsqueeze(0)
                    ms = torch.diag(centered_pooled @ class_var @ centered_pooled.t())
                    maha_score.append(ms)
                    maha_score = torch.stack(maha_score, dim=-1)
                    maha_score = maha_score.min(-1)[0]
                    ood_maha_scores.append(float(maha_score))
                    
        ood_eval_loss = ood_eval_loss / ood_nb_eval_steps
        ood_perplexity = torch.exp(torch.tensor(ood_eval_loss))
        ood_result = {"ood perplexity": ood_perplexity, "avg ood loss": ood_eval_loss}
        print("ood_result:",ood_result)
        
        if len(ood_maha_scores)!=0:
            print(sum(ood_maha_scores)/len(ood_maha_scores)) 
        
        
        print("***** Report the AUROC and FAR95 accuracy *****")
        if score == "diffusion":
            pred_task = np.array(loss_list)
            pred_ood = np.array(ood_loss_list)
        elif score == "diffusion+maha":            
            pred_task = lamda * np.array(loss_list) + (1-lamda) * np.array(maha_scores)
            pred_ood = lamda * np.array(ood_loss_list) + (1-lamda) * np.array(ood_maha_scores)

        inl = np.zeros_like(pred_task).astype(np.int64)
        outl = np.ones_like(pred_ood).astype(np.int64)
        scores = np.concatenate([pred_task, pred_ood], axis=0)
        labels = np.concatenate([inl, outl], axis=0)
        auroc, fpr_95 = get_auroc(labels, scores), get_fpr_95(labels, scores)        
        
        print("method:{4}, t={5}, ID:{0}, OOD:{1}, auroc:{2}, fpr_95:{3} ".format(in_domain, ood, auroc, fpr_95, score, dev_t))
        

def set_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_auroc(key, prediction):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    return roc_auc_score(new_key, prediction)


def get_fpr_95(key, prediction):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    score = fpr_and_fdr_at_recall(new_key, prediction)
    return score        

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=1.):
    y_true = (y_true == pos_label)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))


if __name__ == "__main__":
    main()