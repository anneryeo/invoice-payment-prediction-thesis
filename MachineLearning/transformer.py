from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertConfig,
    BertForSequenceClassification
)
from torch.utils.data import Dataset
import torch
import pandas as pd
from .Utils.data_evaluation import data_evaluation


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if isinstance(self.texts, pd.DataFrame):
            text = self.texts.iloc[idx, 0]
        else:
            text = self.texts[idx]

        if text is None or (isinstance(text, float) and pd.isna(text)):
            text = ""
        else:
            text = str(text)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        if isinstance(self.labels, pd.Series):
            label = self.labels.iloc[idx]
        else:
            label = self.labels[idx]

        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


class TransformerPipeline:
    def __init__(self, X_train, X_test,
                 y_train, y_test,
                 args, parameters=None):
        self.args = args
        self.parameters = parameters or {}
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.results = None

        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

    def build_model(self):
        model_name = self.parameters.get("model_name", "distilbert-base-uncased")

        if "num_labels" in self.parameters:
            num_labels = self.parameters["num_classes"]
        else:
            unique_labels = set(self.y_train) | set(self.y_test)
            num_labels = len(unique_labels)
            print(f"[INFO] Auto-detected {num_labels} classes: {sorted(unique_labels)}")

        nn_transformer = self.parameters.get("nn_transformer", None)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if nn_transformer:
            cfg = nn_transformer[0]
            bert_config = BertConfig(
                vocab_size=self.tokenizer.vocab_size,
                num_hidden_layers=cfg["num_layers"],
                num_attention_heads=cfg["num_heads"],
                hidden_size=cfg["d_model"],
                num_labels=num_labels
            )
            self.model = BertForSequenceClassification(bert_config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )

        # Detect GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")
        self.model.to(device)

        train_dataset = TextDataset(self.X_train, self.y_train, self.tokenizer)
        test_dataset  = TextDataset(self.X_test, self.y_test, self.tokenizer)

        training_args = TrainingArguments(
            output_dir="./results",
            eval_steps=500,
            save_steps=500,
            learning_rate=self.parameters.get("learning_rate", 5e-5),
            per_device_train_batch_size=self.parameters.get("train_batch_size", 16),
            per_device_eval_batch_size=self.parameters.get("eval_batch_size", 16),
            num_train_epochs=self.parameters.get("epochs", 3),
            weight_decay=self.parameters.get("weight_decay", 0.01),
            logging_dir="./logs",
            logging_steps=10
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        return self

    def train(self):
        if self.trainer is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.trainer.train()
        return self

    def evaluation(self):
        eval_results = self.trainer.evaluate()
        self.results = data_evaluation(self.model, self.X_test, self.y_test)
        self.results.update(eval_results)
        return self

    def show_results(self):
        return self.results