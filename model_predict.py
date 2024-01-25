import gzip
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time
import os
import transformers
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import pandas as pd
from transformers.modeling_utils import PreTrainedModel
from transformers import BertConfig


def load_imdb_data():
    df = pd.read_csv(os.environ["IMDB_PATH"])
    df = df.iloc[int(os.environ["N_TEST_REVIEWS"]):]
    texts = df['review'].tolist()
    labels = [1 if sentiment == "positive" else 0 for sentiment in df['sentiment'].tolist()]
    return texts, labels


def run_gzip_experiments(filenames, k, pbar_name):

    times = []
    preds = []
    for pos_file in tqdm(filenames, desc=pbar_name):
        with open(pos_file, "r") as file:
            request = file.readlines()
        pred, time = gzip_predict(request, k)
        
        preds.append(pred)
        times.append(time)

    return np.array(preds), np.array(times)


def get_gzip_experiments_results(args):
    pos_filenames = sorted(os.listdir(os.path.join(args.dir_name, "pos_reviews")))
    neg_filenames = sorted(os.listdir(os.path.join(args.dir_name, "neg_reviews")))

    pos_filenames = [os.path.join(args.dir_name, "pos_reviews", filename) for filename in pos_filenames]
    neg_filenames = [os.path.join(args.dir_name, "neg_reviews", filename) for filename in neg_filenames]

    pos_preds, pos_times = run_gzip_experiments(pos_filenames, args.k_neighbours, pbar_name="GZIP:pos")
    neg_preds, neg_times = run_gzip_experiments(neg_filenames, args.k_neighbours, pbar_name="GZIP:neg")

    return {"pred": pos_preds, "time": pos_times}, {"pred": neg_preds, "time": neg_times}


def make_gzip_report(args):
    pos_res, neg_res = get_gzip_experiments_results(args)

    report = {}
    time = np.concatenate((pos_res["time"], neg_res["time"]))
    report["time_ndarray"] = time
    report["avg_time"] = np.mean(time)

    labels = np.concatenate((np.ones(pos_res["pred"].shape[0]), np.zeros(neg_res["pred"].shape[0])))
    preds = np.concatenate((pos_res["pred"], neg_res["pred"]))

    f1 = f1_score(labels, preds)
    report["f1_score"] = f1

    return report


def gzip_predict(request, k):
    texts, labels = load_imdb_data()

    x1 = request[0]
    distances_from_x1 = []
    start = time.time()
    for x2 in texts:
        Cx1 = len(gzip.compress(x1.encode()))
        Cx2 = len(gzip.compress(x2.encode()))
        x1x2 = "".join([x1, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))

        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distances_from_x1.append(ncd)
    sorted_idx = np.argsort(np.array(distances_from_x1))
    top_k_class = np.array(labels)[sorted_idx[:k]].tolist()
    predict_class = max(set(top_k_class), key=top_k_class.count)
    end = time.time()

    return predict_class, end-start


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}


class BERTClassifier(PreTrainedModel):
    def __init__(self,
                 bert_model_name,
                 num_classes,
                 config):
        super(BERTClassifier, self).__init__(config=config)
        self.bert = BertModel.from_pretrained(bert_model_name,config=config)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.config=config

    def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            logits = self.fc(x)
            return logits
    

def train_step_bert(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()

    start = time.time()
    for batch in tqdm(data_loader, desc=f"BERT | {epoch} epoch"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
    end = time.time()
    return model, end-start


def evaluate_step_bert(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return f1_score(actual_labels, predictions)


def train_bert(bert_model_name='bert-base-uncased', num_epochs=1, num_classes=2, learning_rate=2e-5, batch_size=16):
    texts, labels = load_imdb_data()
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, 128)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, 128)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = BertConfig()
    model = BERTClassifier(bert_model_name, num_classes, config=conf).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    start = time.time()
    step_times = []
    f1_scores = []
    for epoch in range(num_epochs):
        model, step_time = train_step_bert(model, train_dataloader, optimizer, scheduler, device, epoch)
        f1 = evaluate_step_bert(model, val_dataloader, device)
        f1_scores.append(f1)
        step_times.append(step_time)
    end = time.time()

    return model, np.array(f1_scores), np.array(step_times), end-start


def make_bert_report():
    model, f1_scores, step_times, train_time = train_bert()

    report = {}
    report["step_execution_time"] = step_times
    report["avg_time"] = np.mean(step_times)
    report["f1_scores"] = [f"{f1:.5f}" for f1 in f1_scores]
    report["training_time"] = train_time


    return report, model


def bert_predict(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
    
    return "positive" if preds.item() == 1 else "negative"