# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Named Entity Recognition (NER) is a fundamental task in Natural Language Processing (NLP) that involves identifying and classifying entities such as person names, organizations, locations, dates, and other predefined categories from unstructured text data. Traditional rule-based and statistical approaches often struggle with handling contextual dependencies and sequential information in sentences.

The objective of this project is to develop a Deep Learning-based Named Entity Recognition model using Long Short-Term Memory (LSTM) networks. The model should be capable of learning contextual relationships within sequences of words and accurately predicting entity labels for each token in a sentence.

The system will take annotated text data as input, preprocess it into suitable numerical representations (such as word embeddings), and train an LSTM model to perform sequence labeling. The performance of the model will be evaluated based on metrics such as accuracy, precision, recall, and F1-score.

The developed model aims to improve entity recognition performance by leveraging the sequential learning capability of LSTMs and handling long-range dependencies in text.

<img width="481" height="490" alt="image" src="https://github.com/user-attachments/assets/ce588892-74fb-4ab0-9eb9-a7944f527488" />

## DESIGN STEPS

STEP 1:
Load data, create word/tag mappings, and group sentences.

STEP 2:
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.

STEP 3:
Define dataset and DataLoader for batching.

STEP 4:
Build a bidirectional LSTM model for sequence tagging.

STEP 5:
Train the model over multiple epochs, tracking loss.




## PROGRAM

### Name: S V SHADHANASHREE

### Register Number: 212223230202

```python

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return self.fc(x)




model = BiLSTMTagger(len(word2idx) + 1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)

            loss = loss_fn(
                outputs.view(-1, len(tag2idx)),
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)

                loss = loss_fn(
                    outputs.view(-1, len(tag2idx)),
                    labels.view(-1)
                )

                val_loss += loss.item()

        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {total_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}"
        )

    return train_losses, val_losses


```

### OUTPUT

## Loss Vs Epoch Plot

<img width="893" height="635" alt="image" src="https://github.com/user-attachments/assets/a8aea5c6-c445-4fb5-8cdf-f6db195f6e3d" />


### Sample Text Prediction
<img width="407" height="512" alt="image" src="https://github.com/user-attachments/assets/898cc984-dd31-4575-acdb-b3878f0fefe0" />


## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
