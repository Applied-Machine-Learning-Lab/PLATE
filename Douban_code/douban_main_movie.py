import torch
import tqdm
import time
import os
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
         mode,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         tem,
         device,
         save_dir,
         freeze,
         job):
    device = torch.torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)   
    domain_id = np.full((dataset.items.shape[0],1),3)
    dataset.items = np.concatenate([domain_id,dataset.items],1) 
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length),generator=torch.Generator().manual_seed(0))
    '''
    print(dataset.targets[train_dataset.indices[100]])
    print(dataset.targets[valid_dataset.indices[100]])
    print(dataset.targets[test_dataset.indices[100]])
    '''
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset).to(device)
    
    #model = torch.load(f'{save_dir}/{model_name}.pt')
    if mode=='test':
        #model = torch.load(f'{save_dir}/{model_name}_v3_douban_train_{job}.pt')
        save_path=f'{save_dir}/{model_name}_v3_douban_train_{job}.pt'
        model.load_state_dict(torch.load(save_path))
    if "pdfm" in model_name or "prompt" in model_name:
        model.domain_id = 3
        if freeze==2:
            model.Freeze2()
        elif freeze==3:
            model.Freeze3()
        elif freeze==4:
            model.Freeze4()
        elif freeze==5:
            model.Freeze5()
        elif freeze==6:
            model.Freeze6()
        #model.Freeze3()
    if model_name=='pdfm_user_autodis':
        model.autodis_model.temperature = tem
    '''
    #model.phase = 2
    for name, param in model.named_parameters():
        #print(param.shape)
        print(name)
        print(param.shape)
    
    print('')
    '''        
    param_count = 0
    for name, param in model.named_parameters():
        #print(param.shape)
        if param.requires_grad:
            print(name)
            print(param.shape)
            param_count += param.view(-1).size()[0]
        if name == 'embedding_prompt.embedding.weight':
            a=param
    print(param_count)
    
    #auc = test(model, valid_data_loader, device)
    #print(f'val auc: {auc}')
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=5, save_path=f'{save_dir}/{model_name}_v3_{dataset_name}_{mode}_{job}_{freeze}.pt')
    
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
    save_path=f'{save_dir}/{model_name}_v3_{dataset_name}_{mode}_{job}_{freeze}.pt'
    model.load_state_dict(torch.load(save_path))
    
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')
    #print(model.embedding_prompt.embedding.weight)
    print('running time = ',end - start)
    #os.remove(f'{save_dir}/{model_name}_v3_{dataset_name}_{mode}_{job}_{freeze}.pt')   
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='douban_movie')
    parser.add_argument('--dataset_path', default='dataset/')
    parser.add_argument('--model_name', default='pdfm')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--tem', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0',help='cpu, cuda:0')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--freeze', type=int, default=5)
    parser.add_argument('--job', type=int, default=1)
    #args = parser.parse_args(args=[])
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.mode,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.tem,
         args.device,
         args.save_dir,
         args.freeze,
         args.job)
