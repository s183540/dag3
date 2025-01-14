from torch.utils.data import Dataset
from elines_pakke.data import corrupt_mnist
import pdb
import torch



def test_my_dataset():
    # """Test the MyDataset class."""
    # dataset = MyDataset("data/raw")
    # assert isinstance(dataset, Dataset)
    
    train, test = corrupt_mnist()
    assert len(train) == 30000
    assert len(test) == 5000
    
    for dataset in [train, test]:
        #pdb.set_trace()
        for x, y in dataset:
            assert x.shape == (1, 28, 28) 
            assert y in range(10) # 10 classes
    train_targets = torch.unique(train.tensors[1]) # train.tensors[1] is the labels, train.tensors[0] is the images
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()


if __name__ =="__main__":
    test_my_dataset()

# def test_data():
#     dataset = MNIST(...)
#     assert len(dataset) == N_train for training and N_test for test
#     assert that each datapoint has shape [1,28,28] or [784] depending on how you choose to format
#     assert that all labels are represented