import datasets
from datasets import load_dataset
import random
import torch

datasets.logging.set_verbosity(datasets.logging.ERROR)

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    '20ng': ("text", None),
    'trec': ("text", None),
    'imdb': ("text", None),
    'wmt16': ("en", None),
    'multi30k': ("text", None),
}


def load(task_name, tokenizer, max_seq_length=128, is_id=False, state=None,ood = None):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    print("Loading {}".format(task_name))
    if task_name in ('mnli', 'rte'):
        datasets = load_glue(task_name)
    elif task_name == 'sst2':
        datasets = load_sst2()
    elif task_name == '20ng':
        datasets = load_20ng()
    elif task_name == 'trec':
        datasets = load_trec(state=state)
    elif task_name == 'imdb':
        datasets = load_imdb()
    elif task_name == 'wmt16':
        datasets = load_wmt16()
    elif task_name == 'multi30k':
        datasets = load_multi30k()

    def preprocess_function(examples):
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key] + " " + examples[sentence2_key],)
        )
        if state == "no_max_length":
            result = tokenizer(inputs, return_tensors="pt")            
        else:
            result = tokenizer(*inputs, max_length=max_seq_length, padding="max_length",truncation=True)        
        result["labels"] = examples["label"] if 'label' in examples else 0        
        result["sentence"] = inputs
        result["tokenize"] = tokenizer.tokenize(inputs[0])
        if ood is False: # out domain
            result["ood_labels"] = 1
        else:    # in domain
            result["ood_labels"] = 0
        
        return result

    train_dataset = list(map(preprocess_function, datasets['train'])) if 'train' in datasets and is_id else None
    dev_dataset = list(map(preprocess_function, datasets['validation'])) if 'validation' in datasets and is_id else None
    test_dataset = list(map(preprocess_function, datasets['test'])) if 'test' in datasets else None
    return train_dataset, dev_dataset, test_dataset


def load_glue(task):
    datasets = load_dataset("glue", task)
    if task == 'mnli':
        test_dataset = [d for d in datasets['test_matched']] + [d for d in datasets['test_mismatched']]
        datasets['test'] = test_dataset
    return datasets


def load_20ng():
    all_subsets = ('18828_alt.atheism', '18828_comp.graphics', '18828_comp.os.ms-windows.misc', '18828_comp.sys.ibm.pc.hardware', '18828_comp.sys.mac.hardware', '18828_comp.windows.x', '18828_misc.forsale', '18828_rec.autos', '18828_rec.motorcycles', '18828_rec.sport.baseball', '18828_rec.sport.hockey', '18828_sci.crypt', '18828_sci.electronics', '18828_sci.med', '18828_sci.space', '18828_soc.religion.christian', '18828_talk.politics.guns', '18828_talk.politics.mideast', '18828_talk.politics.misc', '18828_talk.religion.misc')
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    for i, subset in enumerate(all_subsets):
        dataset = load_dataset('newsgroup', subset)['train']
        examples = [{'text': d['text'], 'label': i} for d in dataset]
        #random.shuffle(examples)
        num_train = int(0.8 * len(examples))
        num_dev = int(0.1 * len(examples))
        train_dataset += examples[:num_train]
        dev_dataset += examples[num_train: num_train + num_dev]
        test_dataset += examples[num_train + num_dev:]
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets


def load_trec(state=None):
    datasets = load_dataset('trec')
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    idxs = list(range(len(train_dataset)))
    #random.shuffle(idxs)
    tag = 1
    if state == "nonshift":
        num_reserve = int(len(train_dataset) * 0.1)
        dev_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label-coarse']} \
                                    for i in idxs[-num_reserve:] \
                                    if train_dataset[i]['label-coarse'] != tag]    
        train_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label-coarse']} \
                                    for i in idxs[:-num_reserve]\
                                    if train_dataset[i]['label-coarse'] != tag]
        test_dataset = [{'text': d['text'], 'label': d['label-coarse']} for d in test_dataset if d['label-coarse']!=tag]
        #print(dev_dataset)
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    elif state =="nonshift-ood":
        num_reserve = int(len(train_dataset) * 0.1)
        dev_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label-coarse']} \
                                    for i in idxs[-num_reserve:] \
                                    if train_dataset[i]['label-coarse'] == tag]    
        train_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label-coarse']} \
                                    for i in idxs[:-num_reserve]\
                                    if train_dataset[i]['label-coarse'] == tag]
        test_dataset = [{'text': d['text'], 'label': d['label-coarse']} for d in test_dataset if d['label-coarse']==tag]
        #print(dev_dataset)
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    elif state == None:    
        num_reserve = int(len(train_dataset) * 0.1)
        dev_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label-coarse']} for i in idxs[-num_reserve:]]
        train_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label-coarse']} for i in idxs[:-num_reserve]]
        test_dataset = [{'text': d['text'], 'label': d['label-coarse']} for d in test_dataset]
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets


def load_imdb():
    datasets = load_dataset('imdb')
    train_dataset = datasets['train']
    idxs = list(range(len(train_dataset)))
    #random.shuffle(idxs)
    num_reserve = int(len(train_dataset) * 0.1)
    dev_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label']} for i in idxs[-num_reserve:]]
    train_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label']} for i in idxs[:-num_reserve]]
    test_dataset = datasets['test']
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets


def load_wmt16():
    datasets = load_dataset('wmt16', 'de-en')
    #print(datasets)
    test_dataset = [d['translation'] for d in datasets['test']]
    datasets = {'test': test_dataset}
    return datasets


def load_multi30k():
    test_dataset = []
    for file_name in ('./dataset/multi30k/test_2016_flickr.en', './dataset/multi30k/test_2017_mscoco.en', './dataset/multi30k/test_2018_flickr.en'):
        with open(file_name, 'r') as fh:
            for line in fh:
                line = line.strip()
                if len(line) > 0:
                    example = {'text': line, 'label': 0}
                    test_dataset.append(example)
    datasets = {'test': test_dataset}
    return datasets


def load_sst2():
    def process(file_name):
        examples = []
        with open(file_name, 'r') as fh:
            for line in fh:
                splits = line.split()
                label = splits[0]
                text = " ".join(splits[1:]).rstrip(" ")
                examples.append(
                    {'sentence': text, 'label': int(label)}
                )
        return examples
    def split(example):   
        example["sentence"] = example["sentence"].rstrip(" ")          
        return example
    datasets = load_dataset('glue', 'sst2')
    datasets = datasets.map(split)
    train_dataset = datasets['train']
    dev_dataset = datasets['validation']
    test_dataset = process('./dataset/sst2/test.txt')
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets
