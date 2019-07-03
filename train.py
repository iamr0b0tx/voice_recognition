#basic imports
import glob, os
import IPython
from random import randint
from itertools import combinations

#data processing
import librosa
import numpy as np

from progressbar import ProgressBar

#modelling
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from matplotlib import pyplot as plt

from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, Adam

CWD = '.'

def audio2vector(file_path, max_pad_len=400):
    
    #read the audio file
    audio, sr = librosa.load(file_path, mono=True)
    #reduce the shape
    audio = audio[::3]
    
    #extract the audio embeddings using MFCC
    mfcc = librosa.feature.mfcc(audio, sr=sr) 
    
    #as the audio embeddings length varies for different audio, we keep the maximum length as 400
    #pad them with zeros
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc



def get_training_data():
    
    pairs = []
    labels = []
    
    dirs = sorted(os.listdir('{}/data'.format(CWD)))
    
    for di, d in enumerate(dirs):
        if di > 0:
            break
        
        print(d)
        
        #audio files
        afs = glob.glob('{}/data/{}/*.wav'.format(CWD, d))
        
        n = len(afs)
        combs = combinations(range(0, n), 2)


        for i, j in combs:
            ri = randint(0, len(dirs)-1)
            while ri == di:
                ri = randint(0, len(dirs)-1)
                               
            #other audio files
            oafs = glob.glob('{}/data/{}/*.wav'.format(CWD, dirs[ri]))
            
            k = randint(0, len(oafs)-1)
            x, y, z = audio2vector(afs[i]), audio2vector(afs[j]), audio2vector(oafs[k])
            
            #print(i, j, k)
            
            #genuine pair
            pairs.append([x, y])
            labels.append(1)

            #imposite pair
            pairs.append([y, z])
            labels.append(0)
            
            
    return np.array(pairs), np.array(labels)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def build_base_network(input_shape):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.1))
    
    model.add(Dense(256))
    model.add(Dropout(0.1))
    
    model.add(Dense(128))
    model.add(Dropout(0.1))
    
    return model


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


def predict(afs, y, threshold):
    print(afs.shape)
    acc = 0
    preds = model.predict([afs[:, 0], afs[:, 1]])
    for i in range(len(preds)):
        p = preds[i][0]
        z = int(p < threshold)
        if z == y[i]:
            acc += 1
        print(z, y[i])
    print('acc = {}%'.format(acc*100/len(preds)))

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        
        # identity name
        self.name = name
        
        # image file name
        self.file = file

    def __repr__(self):
        return self.path()

    def path(self):
        return os.path.join(self.base, self.name, self.file) 
    
def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.wav':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

# initialize network architecture
input_dim = (20, 400, 1)

audio_a = Input(shape=input_dim)
audio_b = Input(shape=input_dim)

base_network = build_base_network(input_dim)

feat_vecs_a = base_network(audio_a)
feat_vecs_b = base_network(audio_b)

difference = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

# initialize training params
epochs = 32
batch_size = 16
optimizer = Adam() #RMSprop()

# initialize the network
model = Model(input=[audio_a, audio_b], output=difference)
model.compile(loss=contrastive_loss, optimizer=optimizer)

def main():
    XX, Y = get_training_data()
    X = XX.reshape(tuple(list(XX.shape) + [1]))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # call datasets
    audio_1 = X_train[:, 0]
    audio_2 = X_train[:, 1]

    # train model
    model.fit([audio_1, audio_2], y_train, validation_split=.25, batch_size=batch_size, verbose=2, epochs=epochs)

    # save weights
    model.layers[2].save_weights('weights/weights.h5')

    # load datasets
    metadata = load_metadata('{}/data'.format(CWD))

    # load emebeddings
    num_images = metadata.shape[0]
    progress = ProgressBar(num_images)
    progress.start()

    embedded = np.zeros((num_images, 128))

    for i, af in enumerate(metadata):
        #print(m.image_path())    
        av = audio2vector(af.path())
        
        # obtain embedding vector for image
        embedded[i] = model.layers[2].predict(np.expand_dims(av.reshape(20, 400, 1), axis=0))[0]
        progress.update(i+1)

    distances = [] # squared L2 distance between pairs
    identical = [] # 1 if same identity, 0 otherwise

    num = len(metadata)

    for i in range(num - 1):
        for j in range(1, num):
            distances.append(distance(embedded[i], embedded[j]))
            identical.append(1 if metadata[i].name == metadata[j].name else 0)
            
    distances = np.array(distances)
    identical = np.array(identical)

    start = distances[distances.argmin()]
    end = distances[distances.argmax()]
    dx = np.diff(distances)
    step = dx[dx > 0].mean()/2

    print(start, end, step)

    thresholds = np.arange(start, end, step)

    f1_scores = [f1_score(identical, distances < t) for t in thresholds]
    acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

    opt_idx = np.argmax(f1_scores)

    # Threshold at maximal F1 score
    opt_tau = thresholds[opt_idx]

    # Accuracy at maximal F1 score
    opt_acc = accuracy_score(identical, distances < opt_tau)

    # predict
    predict(X_test, y_test, opt_tau)

    with open('weights/threshold.txt', 'w') as f:
    	f.write(str(opt_tau/2))

if __name__ == '__main__':
    main()
