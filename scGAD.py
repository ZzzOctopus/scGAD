from __future__ import division

import pickle as pkl
import numpy as np
import pandas as pd
import sys
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.cluster import KMeans, SpectralClustering
import scipy.sparse as sp
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2, l1
from keras import backend as K
import keras

import argparse

from GAT import GraphAttention
from utils import load_data, performance

parser = argparse.ArgumentParser()

parser.add_argument('dataset_str', type=str)
parser.add_argument('n_clusters', type=int)
parser.add_argument('--subtype_path', default=None, type=str)
parser.add_argument('--k', default=4, type=int)
parser.add_argument('--method', default='pearson', type=str)
parser.add_argument('--PCA_dim', default=256, type=int)
parser.add_argument('--F1', default=64, type=int)
parser.add_argument('--F2', default=16, type=int)
parser.add_argument('--n_attn_heads', default=4, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_reg', default=0, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--c1', default=0.5, type=float)
parser.add_argument('--c2', default=0.5, type=float)

args = parser.parse_args()

if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists('result/'):
    os.makedirs('result/')

dataset_str = args.dataset_str
n_clusters = args.n_clusters
if args.k == 1:
    dropout_rate = 0.  # To avoid absurd results
else:
    dropout_rate = args.dropout_rate


# Paths
data_path = 'dataset/' + dataset_str + '.csv'
GAT_autoencoder_path = 'logs/GATae_' + dataset_str + '.h5'
pred_path = 'result/pred_' + dataset_str + '.txt'


# Read data
start_time = time.time()
A, X = load_data(data_path, dataset_str, args.PCA_dim, args.n_clusters, args.method ,args.k)

N = X.shape[0]  # Number of nodes in the graph
F = X.shape[1]  # Original feature dimension

# Loss functions
def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def maie_class_loss(y_true, y_pred):
    loss_E = mae(y_true, y_pred)
    return loss_E

def inner_product(y_true, y_pred):
    return K.mean(K.exp(-1 * A * K.sigmoid(K.dot(y_pred, K.transpose(y_pred)))))

# Model definition
X_in = Input(shape=(F,))
A_in = Input(shape=(N,))

dropout1 = Dropout(dropout_rate)(X_in)
graph_attention_1 = GraphAttention(args.F1,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout1, A_in])

dropout2 = Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(args.F2,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout2, A_in])

dropout3 = Dropout(dropout_rate)(graph_attention_2)
graph_attention_3 = GraphAttention(args.F1,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout3, A_in])

dropout4 = Dropout(dropout_rate)(graph_attention_3)
graph_attention_4 = GraphAttention(F,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout4, A_in])

# Build GAT autoencoder model
GAT_autoencoder = Model(inputs=[X_in, A_in],
                        outputs=[graph_attention_2,
                                 graph_attention_4])
optimizer = Adam(lr=args.lr)
GAT_autoencoder.compile(optimizer=optimizer,
                        loss=[inner_product,maie_class_loss],
                        loss_weights=[args.c1, args.c2])

# GAT_autoencoder.summary()

# Callbacks
es_callback = EarlyStopping(monitor='loss', min_delta=0.1, patience=50)
tb_callback = TensorBoard(batch_size=N)
mc_callback = ModelCheckpoint(GAT_autoencoder_path,
                              monitor='loss',
                              save_best_only=True,
                              save_weights_only=True)

# Train GAT_autoencoder model
start_time = time.time()
print(X.shape,A.shape)
GAT_autoencoder.fit([X, A],[A, X], epochs=args.epochs, batch_size=N, verbose=0, shuffle=False)

end_time = time.time()
run_time = (end_time - start_time) / 60
print('Pre-train: run time is %.2f '%run_time, 'minutes')

# Construct a model for hidden layer
hidden_model = Model(inputs=GAT_autoencoder.input, outputs=graph_attention_2)
hidden = hidden_model.predict([X, A], batch_size=N)
hidden = hidden.astype(float)

true_path = args.subtype_path
y = pd.read_csv(true_path, header=None, low_memory=False)
y = np.array(y)
y = y.ravel()
print(y.shape)

'''
mid_str = dataset_str
hidden = pd.DataFrame(hidden)
hidden.to_csv('result/hidden_'+mid_str+'.csv', sep='\t')
'''
# pd.DataFrame(hidden).to_csv('result/hidden_'+dataset_str+'.csv', index=False, header=None)

u, s, v = sp.linalg.svds(hidden, k=n_clusters, which='LM')
kmeans = KMeans(n_clusters=n_clusters).fit(u)
predict_labels = kmeans.predict(u)
pd.DataFrame(predict_labels).to_csv('result/pre_labels.csv', index=False, header=None)
performance(y, predict_labels)