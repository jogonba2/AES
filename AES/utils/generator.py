
class TrainGenerator:

    def __init__(self, dataset_file, batch_size):
        self.dataset_file = dataset_file
        self.batch_size = batch_size

    def generator(self):
        while True:
            fr = open(self.dataset_file, "r", encoding="utf8")
            fr.readline() # Skip header

            yield fr.readline()