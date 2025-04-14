import ir_datasets
import pyterrier as pt
if not pt.started():
    pt.init()
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW
import os
import torch
torch.manual_seed(0)
_logger = ir_datasets.log.easy()


root_dir = f'/root/retrievability-bias'
nfs_dir = f'/nfs/resources/cxj/retrievability-bias'
    

BATCH_SIZE = 16

OUTPUTS = ['true', 'false']
def iter_train_samples():
  dataset = ir_datasets.load('msmarco-passage/train')
  docs = dataset.docs_store()

  while True:
    for _, dida, didb in dataset.docpairs_iter():
      yield ' Document: ' + docs.get(dida).text + ' Relevant:', OUTPUTS[0]
      yield ' Document: ' + docs.get(didb).text + ' Relevant:', OUTPUTS[1]
        

train_iter = _logger.pbar(iter_train_samples(), desc='total train samples')

# t5_models = {'t5-base':'t5-base','google_t5_tiny': 'google/t5-efficient-tiny','t5-small':'t5-small'}
t5_models = {'t5-base':'t5-base'}

for model_name, model_path in t5_models.items():    
    model = T5ForConditionalGeneration.from_pretrained(model_path).cuda()
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    epoch = 1
    while epoch <= 5:
        with _logger.pbar_raw(desc=f'train {epoch}', total=16384 // BATCH_SIZE) as pbar:
            print(f'start training {model_name}-epoch-{epoch}')
            model.train()
            total_loss = 0
            count = 0
            for _ in range(16384 // BATCH_SIZE):
                inp, out = [], []
                for i in range(BATCH_SIZE):
                    i, o = next(train_iter)
                    inp.append(i)
                    out.append(o)
                inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.cuda()
                out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.cuda()
                loss = model(input_ids=inp_ids, labels=out_ids).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss = loss.item()
                count += 1
                pbar.update(1)
                pbar.set_postfix({'loss': total_loss/count})
        
        epoch += 1
        if epoch == 5: 
          root_model_save = f'{root_dir}/{model_name}-epoch-{epoch}'
          model.save_pretrained(f'{root_model_save}')
          tokenizer.save_pretrained(f'{root_model_save}')
            
          os.system(f'cp -r {root_model_save} {nfs_dir}/')
          print(f'saved {root_model_save} into {nfs_dir}')
          

    