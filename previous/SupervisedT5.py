import itertools
import pyterrier as pt
import torch
from torch.nn import functional as F
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration

from pyterrier.transformer import TransformerBase


class SupervisedT5(TransformerBase):

    def __init__(self,
                 tok_model='t5-base',
                 model='/nfs/quality/t5train/t5-base-2-bak',
                 batch_size=100,
                 # text_field='text',
                 verbose=False):
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(tok_model, fast=True)
        self.model_name = model
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
        # self.text_field = text_field
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]
        self.model.lm_head.weight = torch.nn.Parameter(self.model.lm_head.weight[[self.REL, self.NREL]])

    def compute_probability(self, texts):
        #texts = sorted(texts, key=lambda x: len(x))
        scores = []
        it = range(0, len(texts), self.batch_size)
        dec_ids = torch.full(
            (self.batch_size, 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )
        if self.verbose:
            it = pt.tqdm(it, desc=self.model_name, unit='batches')
        for start_idx in it:
            rng = slice(start_idx, start_idx + self.batch_size)  # same as start_idx:start_idx+self.batch_size
            enc = self.tokenizer.batch_encode_plus([f'Document: {d} Relevant:' for d in texts[rng]], return_tensors='pt', padding='longest', max_length=512, truncation=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            enc['decoder_input_ids'] = dec_ids[:len(texts[rng])]
            if scores:
              scores[-1] = scores[-1].cpu().detach().tolist()
            #print(enc['input_ids'].shape)
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                result = self.model(**enc).logits
                result = result[:, 0]
                #scores += F.log_softmax(result, dim=1)[:, 0].cpu().detach().tolist()
                scores.append(F.log_softmax(result, dim=1)[:, 0])
            #scores.append(F.log_softmax(result, dim=1)[:, 0].cpu().detach().tolist())
        if scores:
          scores[-1] = scores[-1].cpu().detach().tolist()

        scores = list(itertools.chain.from_iterable(scores))
        #scores = list(itertools.chain.from_iterable(s.cpu().tolist() for s in scores))

        return scores

    def transform(self, df):
        input_text = df['text'].to_list()
        df["quality"] = self.compute_probability(input_text)
        return df

