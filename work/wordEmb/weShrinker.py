import pickle
import numpy as np
import pandas as pd

with open("allWords",'rb') as f:
    megaset = pickle.load(f)
    
'''    
filename = "../../dataset/wordembeddings/embeddings_snap_s512_e15.txt"

we = {row[0] : np.asarray(row[1:]) for _, row in pd.read_csv(filename, skiprows=1, delim_whitespace=True, encoding='latin').iterrows()}

with open(f'{filename}_m','w') as f:
    for el in megaset:
        try:
            emb = we[el]
            f.write(f'{el}')
            for v in emb:
                f.write(f' {v}')
            f.write('\n')
        except:
            pass
            
            
filename = "../../dataset/wordembeddings/embeddings_snap_s512_e30.txt"

we = {row[0] : np.asarray(row[1:]) for _, row in pd.read_csv(filename, skiprows=1, delim_whitespace=True, encoding='latin').iterrows()}

with open(f'{filename}_m','w') as f:
    for el in megaset:
        try:
            emb = we[el]
            f.write(f'{el}')
            for v in emb:
                f.write(f' {v}')
            f.write('\n')
        except:
            pass
            
'''            
filename = "../../dataset/wordembeddings/embeddings_snap_s512_e50.txt"

we = {row[0] : np.asarray(row[1:]) for _, row in pd.read_csv(filename, skiprows=1, delim_whitespace=True, encoding='latin').iterrows()}

with open(f'{filename}_m','w') as f:
    for el in megaset:
        try:
            emb = we[el]
            f.write(f'{el}')
            for v in emb:
                f.write(f' {v}')
            f.write('\n')
        except:
            pass