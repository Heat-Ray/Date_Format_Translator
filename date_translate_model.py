import numpy as np
from nmt_utils import *
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from sklearn.model_selection import train_test_split

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

def pre_proc(data, hv, mv, szi, szo):
  inp, out = zip(*data)
  inp = np.array([pad_encode(i, szi, hv) for i in inp])
  out = np.array([pad_encode(t, szo, mv) for t in out])

  return inp, out


data, hv , mv, iv = load_dataset(10000)
print('Data generated\n',data[:10])

mv['<pad>'] = 11
iv[11] = '<pad>'

data, data_test = train_test_split(data, test_size = 0.1, random_state = 25)

train_inp, train_out = pre_proc(data, hv, mv, 25, 25)
test_inp, test_out = pre_proc(data_test, hv, mv, 25, 25)

train_inp = train_inp.reshape(*train_inp.shape, 1)
train_out = train_out.reshape(*train_out.shape, 1)
test_inp = test_inp.reshape(*test_inp.shape, 1)
test_out = test_out.reshape(*test_out.shape, 1)

learning_rate = 0.005
    
model = Sequential()
model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=train_inp.shape[1:]))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))

model.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(learning_rate),metrics=['accuracy'])
model.summary()

model.fit(train_inp, train_out,epochs=11, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_inp, test_out)
print('Test accurracy',test_acc)

p = np.array(pad_encode('21 april 2001', 25, hv))
p = p.reshape((-1, 25, 1))
preds = model.predict(p)

lis = []
for i in preds[0]:
  i = list(i)
  lis.append(i.index(max(i)))

print('21 april 2001 predicted to be',('').join((machine_to_human(lis, iv)))[:10])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")
