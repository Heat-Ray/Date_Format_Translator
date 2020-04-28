import numpy as np
from nmt_utils import *
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *

def pad_encode(data, sz, vocab_dict):
    ret_list = []
    if len(data) > sz:
        data = data[:sz]
    for i in data:
        ret_list.append(vocab_dict[i])

    if len(ret_list) < sz:
        while len(ret_list) != sz:
            ret_list.append(vocab_dict['<pad>'])

    return ret_list

def machine_to_human(mach, machine_inv_vocab):
    human_readble = []
    for i in mach:
        human_readble.append(machine_inv_vocab[i])
        
    return human_readble

data, hv , mv, iv = load_dataset(10000)

json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("model.h5")
print("Loaded model from disk")

mv['<pad>'] = 11
iv[11] = '<pad>'

model.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(0.005),metrics=['accuracy'])


i = 0
while i != 2:
    i = int(input('Enter 1 to check date\nEnter 2 to exit'))
    if i not in [1,2]:
        print('Invalid input')
        continue
    if i is 1:
        dt = input('Enter the human readable date : ')
        p = np.array(pad_encode(dt, 25, hv))
        p = p.reshape((-1, 25, 1))
        preds = model.predict(p)

        lis = []
        for i in preds[0]:
            i = list(i)
            lis.append(i.index(max(i)))

        print(dt,'predicted to be',('').join((machine_to_human(lis, iv)))[:10])
    
