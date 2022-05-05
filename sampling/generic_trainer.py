from tqdm import tqdm

class GenericTrainer:

    def __init__(self):
        self.batch_size = 128
        self.embedding_size = 100
        self.num_epochs = 100
        self.learning_rate = 0.001

    def precompute(self, batch, num_epochs, num_batches_per_epoch, update=None, single=False):
        batches = list()
        first = None

        for epoch in tqdm(range(num_epochs)):
            if single and epoch > 0:
                batches.extend(first)
                continue

            batch.reset()
            if update is not None:
                update(epoch)
            for i in range(num_batches_per_epoch):
                x_batch, y_batch = batch()
                # copy() is very necessary:
                # list will normally append updated references to the same (!)
                # tuple, so that each new addition also adapts all previous additions
                batches.append((x_batch.copy(), y_batch.copy()))
            if epoch == 0:
                first = batches[:]

        return batches
