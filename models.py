import torch
import pandas as pd
from tqdm import tqdm
import config,fair_utils, convert

# from transformers import BertTokenizer, BertModel

modelname = "google-bert/bert-large-uncased"
# tokenizer = BertTokenizer.from_pretrained(modelname)
# model = BertModel.from_pretrained(modelname)
# model.eval()  # Turn off dropout


from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForMaskedLM.from_pretrained(modelname)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def encode_passages(df, text_column='text', batch_size=16, max_length=512):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size)):
            batch_texts = df[text_column].iloc[i:i+batch_size].tolist()
            encoded = tokenizer(batch_texts, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding (first token)
            cls_embeddings = output.last_hidden_state[:, 0, :]  # shape: (batch, hidden_size)
            embeddings.append(cls_embeddings.cpu())

    # Concatenate all batches
    return torch.cat(embeddings, dim=0)


for modelname in config.models:
    res_file = f'{modelname}_msmarco-passage_dev_{retr_num}.res'
    doc_embeddings = encode_passages(df)
    print("Embedding shape:", doc_embeddings.shape)  # (num_docs, 768)
