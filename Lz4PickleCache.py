import pickle
import lz4.frame
import struct
import pyterrier as pt

class Lz4PickleCache(pt.Indexer):
    def __init__(self, path):
        self.path = path

    def index(self, it):
        with lz4.frame.LZ4FrameFile(self.path, mode='x') as fout:
            for record in it:
                record_enc = pickle.dumps(record)
                fout.write(struct.pack('<Q', len(record_enc)))
                fout.write(record_enc)

    def get_corpus_iter(self):
        with lz4.frame.LZ4FrameFile(self.path, mode='r') as fin:
            while count := fin.read(struct.calcsize('<Q')):
                count, = struct.unpack_from('<Q', count)
                yield pickle.loads(fin.read(count))

    # def get_corpus_iter(self):
    #     with lz4.frame.LZ4FrameFile(self.path, mode='r') as fin:
    #         num = 0
    #         while count := fin.read(struct.calcsize('<Q')):
    #             count, = struct.unpack_from('<Q', count)
    #             yield pickle.loads(fin.read(count))
    #             num += 1
    #             if num % 10000 == 0:
    #                 print(f'num = {num}')


    # def get_corpus_iter(self):
    #     with lz4.frame.LZ4FrameFile(self.path, mode='r') as fin:
    #         end_position = fin.seek(0, 2)
    #         fin.seek(0, 0)
    #         with pt.tqdm(total=end_position, unit="B", unit_scale=True, desc="Decompressing") as pbar:
    #             while count1 := fin.read(struct.calcsize('<Q')):
    #                 count, = struct.unpack_from('<Q', count1)
    #                 yield pickle.loads(fin.read(count))
    #                 pbar.update(len(count1))
