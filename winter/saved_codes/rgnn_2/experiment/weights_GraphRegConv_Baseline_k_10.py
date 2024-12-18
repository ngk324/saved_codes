import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from model.graphRegConvNet import GraphRegConv_GNN
from impl.graphRegressor_Graph_Conv import modelImplementation_GraphRegressor
from data_reader.cross_validation_reader import split_ids, get_graph_diameter
from utils.utils import printParOnFile
from random import seed
from data_processing.my_dataset import MyOwnDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

rnd_state = np.random.RandomState(seed(1))

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def getcross_validation_split(n_folds=2, batch_size=1):
    # local_dataset should be an instance of a Dataset or a processed list of data objects
    local_dataset = MyOwnDataset(pre_transform=get_graph_diameter)
    print("Full train dataset", local_dataset, len(local_dataset),local_dataset[len(local_dataset)-1])
    train_ids, test_ids, valid_ids = split_ids(
        rnd_state.permutation(len(local_dataset)), folds=n_folds
    )
    splits = []

    for fold_id in range(n_folds):
        loaders = []
        for split in [train_ids, test_ids, valid_ids]:
            # print(torch.from_numpy((train_ids if split.find('train') >= 0 else test_ids)[fold_id]))
            # print("---")
            gdata = local_dataset[torch.from_numpy(split[fold_id])]
            #print(gdata[0])
            loader = DataLoader(gdata,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)#,
                                #collate_fn=collate_batch)
            loaders.append(loader)
        splits.append(loaders)
        # print("---")

    return splits #0-train, 1-test, 2-valid


if __name__ == "__main__":    
    torch.cuda.empty_cache()
    n_epochs = 500
    out_dim = 16#86 # int(sys.argv[1]) #86
    n_units = 256
    lr = 0.0012
    drop_prob = 0.35 #0.35 #0.5
    weight_decay = 5e-4
    momentum = 0.9
    batch_size = 128
    n_folds = 10
    test_epoch = 1
    max_k = 10
    aggregator = "concat"
    test_name = "DeepGraphConv_linear_Baseline_Test"

    dataset_path = "~/AIRLab/DRDistributedDynamics/RecurrentDGNN/Dataset/weights"
    dataset_name = "weights"

    test_name = (
        test_name
        + "_data-"
        + dataset_name
        + "_aggregator-"
        + aggregator
        + "_nFold-"
        + str(n_folds)
        + "_lr-"
        + str(lr)
        + "_drop_prob-"
        + str(drop_prob)
        + "_weight-decay-"
        + str(weight_decay)
        + "_batchSize-"
        + str(batch_size)
        + "_nHidden-"
        + str(n_units)
        + "_maxK-"
        + str(max_k)
    )
    training_log_dir = os.path.join("./test_log/", test_name)
    if not os.path.exists(training_log_dir):
        os.makedirs(training_log_dir)

    printParOnFile(
        test_name=test_name,
        log_dir=training_log_dir,
        par_list={
            "dataset_name": dataset_name,
            "n_fold": n_folds,
            "learning_rate": lr,
            "drop_prob": drop_prob,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "test_epoch": test_epoch,
            "n_hidden": n_units,
            "max_k": max_k,
            "aggregator": aggregator,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss()

    dataset_cv_splits = getcross_validation_split(n_folds, batch_size)

    for split_id, split in enumerate(dataset_cv_splits):
        if split_id == 1:
            break

        loader_train = split[0]
        loader_test = split[1]
        loader_valid = split[2]
        
        model = GraphRegConv_GNN(
            loader_train.dataset.num_node_labels,
            n_units,
            out_dim,
            drop_prob=drop_prob,
            max_k=max_k,
        ).to(device)
        
        model_impl = modelImplementation_GraphRegressor(
            model, lr, criterion, device
        ).to(device)

        model_impl.set_optimizer(weight_decay=weight_decay)
        
        #local_dataset = MyOwnDataset(pre_transform=get_graph_diameter)
        #loader_new_test = local_dataset.get_data()
        #print(loader_train)
        #test_dataset = MyOwnDataset()
        #test_data1 = test_dataset.get_data()
        #print("Test dataset:",test_data1)
        #gdata = test_dataset#[torch.from_numpy()]
        #test_data = DataLoader(test_data1,batch_size=58)
        #print(test_data)

        model_impl.train_test_model(
            split_id,
            loader_train,
            loader_test,
            loader_valid,
            n_epochs,
            test_epoch,
            #test_data,
            aggregator,
            test_name,
            training_log_dir
        )