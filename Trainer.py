from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

class Trainer:
    def __init__(model, dataset):
        self.model = model
        self.dataset = dataset
    
    def train(self, epochs = 100, batch_size = 32,  train_rate = 0.7, val_rate = 0.2, test_rate = 0.1):
        train_data, validation_data, test_data = self.__split_dataset(train_rate, val_rate, test_rate)
        
    
    
    def __split_dataset(self, train_rate = 0.7, val_rate = 0.2, test_rate = 0.1):
        dataset_len = len(self.dataset)
        data_train = data[:int(dataset_len*train_rate)]
        data_validation = data[int(dataset_len*train_rate):int(dataset_len*(train_rate+val_rate))]
        data_test = data[int(dataset_len*(train_rate+val_rate)):]
        return data_train, data_validation, data_test
    
