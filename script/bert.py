import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer


# https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa
class BertClassifier(nn.Module):
    def __init__(self, freeze_bert = True, hidden_size=1024, num_layers=25, num_class=2):
        super(BertClassifier, self).__init__()
        #Instantiating BERT model obeject
        self.bert_layer = BertModel.from_pretrained(
            'bert-large-uncased', 
            output_hidden_states=True, 
            output_attentions=True,
            attention_probs_dropout_prob=0.6,
            hidden_dropout_prob=0.6
        )
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
                
        #Classification Layer
        self.weights = nn.Parameter(torch.rand(self.num_layers, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.fc = nn.Linear(self.hidden_size, num_class-1) 

    def forward(self, seq, attn_masks):
        '''
        Inputs:
          -seq: Tensor of shape [B,T] containing token ids of sequences
          -attn_masks: Tensor of shape [B, T] conatining attention masks to be used to avoid contribution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representation
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)[-2:]

        batch_size = seq.shape[0]
        ht_cls = torch.cat(cont_reps)[:, :1, :].view(self.num_layers, batch_size, 1, self.hidden_size)
        #Obtaining the representation of [CLS} head
        atten = torch.sum(ht_cls * self.weights.view(self.num_layers, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(self.num_layers, 1, 1, 1), dim=[0, 2])
        
        for i, dropout in enumerate(self.dropouts):
            if i==0:
                h = self.fc(dropout(feature))
            else:
                h += self.fc(dropout(feature))
                
            h = h / len(self.dropouts)
        return h
    
    def freeze_bert(self):
        for p in self.bert_layer.parameters():
            p.requires_grad = False
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args: 
          patience (int): How long to wait after last time validation loss improved.
                          Default: 7
          verbose (bool): If True, prints a message for each validation loss improvement.
                          Default: False
          delta (float): Minimun change in the monitored quantity to qualify as an improvement.
                          Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter; {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss descreased ({self.val_loss_min:.6f}) --> {val_loss:.6f}. Saving model ...')
            torch.save(model.state_dict(), 'checkpoint.pt')
            self.val_loss_min = val_loss

def train(net, criterion, opti, train_loader, val_loader, max_eps, patience, print_every):
    best_acc = 0
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for ep in range(1, max_eps+1):
        if ep > 1:
                net.freeze_bert()
        
        for it, (_, seq, attn_masks, labels) in enumerate(train_loader):
            #Clear gradients
            opti.zero_grad()
            #Converting these to cuda tensors
            seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
            
            # freeze bert layer from the 2 iteration
                
            #Obtaining the logits from the model
            logits = net(seq, attn_masks)

            #Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            #Back propagating the gradients
            loss.backward()

            #Optimization step
            opti.step()

            if (it+1) % print_every == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epochs {} complete. Loss : {} Accuracy : {}".format(it+1, ep, loss.item(), acc))
        
        # print score
        train_f1, train_acc, train_loss = evaluate(net, criterion, train_loader)
        print("Epoch {} complete! Train F1 : {}, Train Accuracy : {}, Train Loss : {}".format(ep, train_f1, train_acc, train_loss))
        val_f1, val_acc, val_loss = evaluate(net, criterion, val_loader)
        print("Epoch {} complete! Validation F1 : {}, Validation Accuracy : {}, Validation Loss : {}".format(ep, val_f1, val_acc, val_loss))

        early_stopping(val_loss, net)

        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}".format(best_acc, val_acc))
            best_acc = val_acc
            
        if early_stopping.early_stop:
            print("Early stopping")
            break

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def get_class_from_logits(logits):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return (probs > 0.5).long().squeeze().tolist()

def evaluate(net, criterion, dataloader):
    net.eval()
    mean_acc, mean_loss = 0, 0
    count = 0
    
    y, y_pred = [], []

    with torch.no_grad():
        for _, seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
            logits = net(seq, attn_masks)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1
            
            y_pred += get_class_from_logits(logits)
            y += labels.tolist()
    
    f1 = f1_score(y, y_pred, average=None)[0]
    return f1, mean_acc / count, mean_loss / count

class TweetDataset(Dataset):
  
    def __init__(self, df, maxlen):

        #Store the contents of the file in a pandas dataframe
        self.df = df

        #Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

        self.maxlen= maxlen
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the dataframe
        Id = self.df.loc[index, 'id']
        sentence = self.df.loc[index, 'text']
        label = self.df.loc[index, 'target']

        #Preprocessing the text to be suitable for BERT 
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]']

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return Id, tokens_ids_tensor, attn_mask, label

# http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
def main(train_loader, valid_loader, freeze_bert=False, lr=1e-5, print_every=2000, max_eps=5, patience=1):
    net = BertClassifier(freeze_bert = freeze_bert)

    criterion = nn.BCEWithLogitsLoss()

    # AdamW
    param_optimizer = list(net.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opti = optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-6)

    net.cuda()

    train(net, criterion, opti, train_loader, valid_loader, max_eps, patience, print_every)

    return net

def predict(net, test_loader):
    y_pred = {}
    
    for ids, seq, attn_masks, labels in test_loader:
            #Converting these to cuda tensors
            ids, seq, attn_masks = ids.cuda(), seq.cuda(), attn_masks.cuda()

            #Obtaining the logits from the model
            logits = net(seq, attn_masks)
            
            for Id, label in zip(ids, get_class_from_logits(logits)):
                y_pred[Id.long().item()] = label
    
    return pd.DataFrame([[index, label] for (index, label) in sorted(y_pred.items(), key=lambda x: x[0])], columns=('id', 'target'))

if __name__ == "__main__":
    df, df_test = pd.read_csv('./data/train.csv'), pd.read_csv('./data/test_leak.csv')
    df_train, df_valid = train_test_split(df, test_size=0.15, random_state=42, stratify=df.target.values.tolist())
    df_train, df_valid = df_train.reset_index(), df_valid.reset_index()

    maxlen, batch_size = 40, 8

    #Creating instances of training and validation set
    train_set = TweetDataset(df = df, maxlen = maxlen)
    test_set = TweetDataset(df = df_test, maxlen = maxlen)

    #Creating intsances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size = batch_size, num_workers = 5)
    test_loader = DataLoader(test_set, batch_size = batch_size, num_workers = 5)

    net = main(
        train_loader, 
        train_loader, 
        freeze_bert=False, 
        lr=1e-5, 
        print_every=2000, 
        max_eps=5, 
        patience=1
    )

    submit = predict(net, test_loader)
    submit.to_csv('./output/submit.csv', index=None)
