from more_itertools import chunked
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pyterrier_dr import BiEncoder

class Contriever(BiEncoder):
    def __init__(self, model_name, batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size=batch_size, text_field=text_field, verbose=verbose)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'device===: {device}')
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.verbose = verbose
        self.batch_size = batch_size

    # Mean pooling
    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode_queries(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(list(chunk), return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                # res = self.model(**inps).last_hidden_state[:, 0] 
                res = self.model(**inps)
                res = self.mean_pooling(res[0], inps['attention_mask'])
                # res = res.last_hidden_state[:, 0] 
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(list(chunk), return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                # res = self.model(**inps).last_hidden_state[:, 0]
                res = self.model(**inps)
                res = self.mean_pooling(res[0], inps['attention_mask'])
                # res = res.last_hidden_state[:, 0]

                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    # @classmethod
    # def from_pretrained(cls, model_name, batch_size=32, text_field='text', verbose=False, device=None):
    #     model = AutoModel.from_pretrained(model_name)
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     config = AutoConfig.from_pretrained(model_name)
    #     res = cls(model, tokenizer, config, batch_size=batch_size, text_field=text_field, verbose=verbose, device=device)
    #     res.model_name = model_name
    #     return res
    #
    # def __repr__(self):
    #     if hasattr(self, 'model_name'):
    #         return f'HgfBiEncoder({repr(self.model_name)})'
    #     return 'HgfBiEncoder()'

