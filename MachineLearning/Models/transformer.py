from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertConfig,
    BertForSequenceClassification,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shutil


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
        self.parameters = parameters
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.results = None

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

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
        test_dataset = TextDataset(self.X_test, self.y_test, self.tokenizer)

        training_args = TrainingArguments(
            output_dir=self.args.results_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.parameters.get("learning_rate", 5e-5),
            per_device_train_batch_size=self.parameters.get("train_batch_size", 16),
            per_device_eval_batch_size=self.parameters.get("eval_batch_size", 16),
            num_train_epochs=self.parameters.get("epochs", 3),
            weight_decay=self.parameters.get("weight_decay", 0.01),
            logging_dir="./logs",
            logging_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=16)]
        )
        self.test_dataset = test_dataset
        return self

    def train(self):
        if self.trainer is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.trainer.train()

        # Clean up checkpoints after best model is loaded
        shutil.rmtree("./results", ignore_errors=True)
        print("[INFO] Deleted checkpoint directory after loading best model.")

        return self

    def evaluation(self):
        # Hugging Face evaluation (loss, etc.)
        eval_results = self.trainer.evaluate()

        # Hugging Face predictions
        predictions = self.trainer.predict(self.test_dataset)
        y_pred = predictions.predictions.argmax(axis=-1)

        # Wrap into sklearn-style evaluation
        results = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision_macro": precision_score(self.y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(self.y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(self.y_test, y_pred, average="macro", zero_division=0),
        }

        # Merge Hugging Face eval metrics (like eval_loss) with sklearn metrics
        results.update(eval_results)
        self.results = results
        return self

    def show_results(self):
        return self.results