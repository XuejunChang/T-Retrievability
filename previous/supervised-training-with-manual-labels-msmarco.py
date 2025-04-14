import ir_datasets
import pyterrier as pt
if not pt.started():
    # pt.init()
    pt.java.init()
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW
from Lz4PickleCache import *
import os
import sys
import time
import torch
from QT5_Model import *
torch.manual_seed(0)
_logger = ir_datasets.log.easy()

# work_name = sys.argv[1]
# root_dir = f'/root/{work_name}'
# nfs_dir = sys.argv[2]
# nfs_data_save = f'{nfs_dir}/data/{work_name}'
# new_model_name = sys.argv[3]
# init_model = sys.argv[4]

work_name="distill"
root_dir = f'/root/{work_name}'
nfs_data_save = f'/nfs/datasets/cxj/distill/trained_models/'
if not os.path.exists(nfs_data_save):
    os.makedirs(nfs_data_save)

new_model_name = sys.argv[1]
init_model = sys.argv[2]
total_epoch = int(sys.argv[3])
start_epoch = int(sys.argv[4])
print(new_model_name,init_model,total_epoch,start_epoch)

BATCH_SIZE = 16
samples_each_epoch = 16000
print('BATCH_SIZE===',BATCH_SIZE)
os.system(f'rm -rf {root_dir}/{new_model_name}-epoch-{start_epoch}/')

OUTPUTS = ['true', 'false']
def iter_train_samples():
  dataset = ir_datasets.load('msmarco-passage/train')
  docs = dataset.docs_store()

  while True:
    for _, dida, didb in dataset.docpairs_iter():
      yield ' Document: ' + docs.get(dida).text + ' Relevant:', OUTPUTS[0]
      yield ' Document: ' + docs.get(didb).text + ' Relevant:', OUTPUTS[1]
        

train_iter = _logger.pbar(iter_train_samples(), desc='total train samples')

model = T5ForConditionalGeneration.from_pretrained(init_model).cuda()
tokenizer = T5Tokenizer.from_pretrained("t5-base")
optimizer = AdamW(model.parameters(), lr=5e-5)

epoch = start_epoch
start = time.time()
while epoch <= total_epoch:
    new_model = f'{new_model_name}-epoch-{epoch}'
    print(f'start training {new_model}')
    with _logger.pbar_raw(desc=f'train {epoch}', total=samples_each_epoch // BATCH_SIZE) as pbar:
        print(f'start training {new_model}')
        model.train()
        total_loss = 0
        count = 0
        for _ in range(samples_each_epoch // BATCH_SIZE):
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
            print('total_loss:----->',total_loss)
            count += 1
            pbar.update(1)
            pbar.set_postfix({'avg-loss': total_loss/count})

    if epoch % 10 == 0 or epoch == total_epoch:
        root_dir_save = f'{root_dir}/{new_model}'
        print(f'saving to {root_dir_save}')
        model.save_pretrained(root_dir_save)
        tokenizer.save_pretrained(f'{root_dir_save}')
        print('saved to local disk.')
        os.system(f'cp -r {root_dir_save} {nfs_data_save}/')
        print(f'copied to {nfs_data_save}')

    epoch += 1

end = time.time()
print(f'training time: {end-start} seconds, {(end-start)/60/60} hours.')
