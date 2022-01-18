

import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from matplotlib import font_manager

fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
#fontP.set_size(14)


def get_embd(text, emb_type='cls'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True)
    model = model.to(device)
    model.eval()

    text_ids = tokenizer.encode(text)
    text_ids = torch.LongTensor(text_ids)
    
    text_ids = text_ids.to(device)
    text_ids = text_ids.unsqueeze(0)

    with torch.no_grad():
        out = model(input_ids=text_ids)

    if emb_type == 'cls':
        return out[1]
    elif emb_type == 'last_4':
        hidden_states = out[2]
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
        cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1)
        return cat_sentence_embedding
    elif emb_type == 'last':
        hidden_states = out[2]      
        sentence_embedding = torch.mean(hidden_states[-1], dim=1)
        return sentence_embedding

    else:
        print('Invalid embedding type.')
        exit()

def get_all_labels(label_path):
    with open(label_path, 'r') as f:
        data = f.readlines()
    
    data = [x.rstrip() for x in data]
    label2name = {}
    name2label = {}
    for line in data:
        sample = line.split(' ')
        label2name[int(sample[0])] = sample[2]
        name2label[sample[2]]      = int(sample[0])
    
    return label2name, name2label

def save_embd(label_path, embd_type='last'):
    label2name, name2label = get_all_labels(label_path)

    
    all_food = list(name2label.keys())
    all_embd = {}
    for food in all_food:
        embd = get_embd(food, emb_type=embd_type) 

        embd = embd.cpu().numpy()
        all_embd[food] = embd
        

    pickle.dump(all_embd, open('embd_last.pkl', 'wb'))
    print('embedding saved to embd_{}.pkl'.format(embd_type))

def draw_pca(words, embd):
    n_components = 4
    twodim = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True, random_state=0).fit_transform(embd)[:,:2]
    

    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)
    plt.savefig('pca.png')
    


if __name__ == '__main__':

    label_path = '../food_data/label2name.txt'
    #save_embd(label_path, embd_type='last')

    embd_dict = pickle.load(open('embd_last.pkl', 'rb'))
    words = list(embd_dict.keys())
    embd  = np.array(list(embd_dict.values())).reshape(1000, 768)
    draw_pca(words, embd)