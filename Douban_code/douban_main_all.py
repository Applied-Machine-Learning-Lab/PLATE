import torch
import tqdm
import copy
import time
import numpy as np
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
#from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from dataset.douban import Douban, DoubanMusic, DoubanBook, DoubanMovie


from pdfm_user_autodis_v2 import PromptDeepFactorizationMachineModel_user_autodis
from pdfm_usermlp_v2 import PromptDeepFactorizationMachineModel_usermlp

def get_dataset(name, path):
    if name == 'douban':
        return Douban()
    elif name == 'douban_music':
        return DoubanMusic(path)
    elif name == 'douban_book':
        return DoubanBook(path)
    elif name == 'douban_movie':
        return DoubanMovie(path)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims

    if name == 'pdfm_usermlp':
        return PromptDeepFactorizationMachineModel_usermlp(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2, domain_id=0)
    elif name == 'pdfm_user_autodis':
        return PromptDeepFactorizationMachineModel_user_autodis(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2, domain_id=0, number=10, max_val=0.01, temperature=1e-5)
    else:
        raise ValueError('unknown model name: ' + name)
        
class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device).long(), target.to(device).long()
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device).long(), target.to(device).long()
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)

def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         job):
    device = torch.torch.device(device)
    dataset_name = ['douban_music','douban_book','douban_movie']
    ########## first split, then combine data from different domains #####################
    for i in range (len(dataset_name)):
        
        dataset_on = get_dataset(dataset_name[i], dataset_path+'/'+dataset_name[i]+'/ratings.dat')
        domain_id = np.full((dataset_on.items.shape[0],1),0)
        dataset_on.items = np.concatenate([domain_id,dataset_on.items],1) 
        
        train_length = int(len(dataset_on) * 0.8)
        valid_length = int(len(dataset_on) * 0.1)
        test_length = len(dataset_on) - train_length - valid_length
        #torch.manual_seed(0)
        train_dataset_on, valid_dataset_on, test_dataset_on = torch.utils.data.random_split(dataset_on, (train_length, valid_length, test_length),generator=torch.Generator().manual_seed(0))
        if i==0:
            dataset = copy.deepcopy(dataset_on) #must be deepcopy
            train_dataset = copy.deepcopy(dataset)
            valid_dataset = copy.deepcopy(dataset)
            test_dataset = copy.deepcopy(dataset)
            train_dataset.items = dataset_on.items[train_dataset_on.indices,:] #the indices contains the random index
            train_dataset.targets = dataset_on.targets[train_dataset_on.indices]
            valid_dataset.items = dataset_on.items[valid_dataset_on.indices,:]
            valid_dataset.targets = dataset_on.targets[valid_dataset_on.indices]
            test_dataset.items = dataset_on.items[test_dataset_on.indices,:]
            test_dataset.targets = dataset_on.targets[test_dataset_on.indices]
            
        else:
            #dataset = dataset + dataset_on
            train_dataset.items = np.concatenate((train_dataset.items, dataset_on.items[train_dataset_on.indices,:]), axis=0)
            train_dataset.targets = np.concatenate((train_dataset.targets, dataset_on.targets[train_dataset_on.indices]), axis=0)
            #print(dataset.items.shape)
            
            valid_dataset.items = np.concatenate((valid_dataset.items, dataset_on.items[valid_dataset_on.indices,:]), axis=0)
            valid_dataset.targets = np.concatenate((valid_dataset.targets, dataset_on.targets[valid_dataset_on.indices]), axis=0)
            
            test_dataset.items = np.concatenate((test_dataset.items, dataset_on.items[test_dataset_on.indices,:]), axis=0)
            test_dataset.targets = np.concatenate((test_dataset.targets, dataset_on.targets[test_dataset_on.indices]), axis=0)
            
            dataset.items = np.concatenate((dataset.items, dataset_on.items), axis=0)
            dataset.targets = np.concatenate((dataset.targets, dataset_on.targets), axis=0)
    
    dataset.field_dims[0] = 2718 #specific number for douban
    dataset.field_dims[1] = 5567 + 6777 + 9565
    
    train_dataset.field_dims[0] = 2718
    train_dataset.field_dims[1] = 5567 + 6777 + 9565
    valid_dataset.field_dims[0] = 2718
    valid_dataset.field_dims[1] = 5567 + 6777 + 9565
    test_dataset.field_dims[0] = 2718
    test_dataset.field_dims[1] = 5567 + 6777 + 9565
        
    #dataset = get_dataset(dataset_name, dataset_path+'/'+dataset_name[i]+'/ratings.dat') 
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8,shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset).to(device)
    
    if "pdfm" in model_name or "prompt" in model_name:
        model.Freeze1()
    
    #print(model.autodis_model.meta_embedding)
    
    #for name, param in model.named_parameters():
    #        print(name)
    
    param_count = 0
    for name, param in model.named_parameters():
        #print(param.shape)
        if not param.requires_grad:
            print(name)
            print(param.shape)
            param_count += param.view(-1).size()[0]
        #if name == 'embedding_prompt.embedding.weight':
        #    print(param)
    print(param_count)
    
    param_count = 0
    for name, param in model.named_parameters():
        param_count += param.view(-1).size()[0]
    print(param_count)
    
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    save_path=f'{save_dir}/{model_name}_v3_douban_train_{job}.pt'
    early_stopper = EarlyStopper(num_trials=5,save_path=save_path)
    
    start = time.time()
    
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        if not early_stopper.is_continuable(model, auc):
            
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    
    end = time.time()
    #checkpoint = torch.load(f'{save_dir}/{model_name}.pt.tar')
    #model.load_state_dict(checkpoint['state_dict']) #fix the bug
    
    model.load_state_dict(torch.load(save_path))
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')
    print('running time = ',end - start)
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default=['douban_music','douban_book','douban_movie'])
    parser.add_argument('--dataset_path', default='dataset/')
    parser.add_argument('--model_name', default='pdfm_user_autodis')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0',help='cpu, cuda:0')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--job', type=int, default=1)
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.job)
