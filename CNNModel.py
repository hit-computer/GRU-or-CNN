#coding:utf-8
from utils import NormalInit, OrthogonalInit, add_to_params
import theano
import numpy as np
import theano.tensor as T

class SentenceEncoder_CNN(): #用CNN学习句子向量表示
    def init_params(self, word_embedding_param):
        # Initialzie W_emb to given word embeddings
        assert(word_embedding_param != None)
        self.W_emb = word_embedding_param

        """ sent weights """
        self.Filter1 = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_encoder), name='Filter1'+self.name))
        self.Filter2 = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, 2*self.rankdim, self.qdim_encoder), name='Filter2'+self.name))
        self.Filter3 = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, 3*self.rankdim, self.qdim_encoder), name='Filter3'+self.name))
        
        self.b_1 = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'), name='cnn_b1'+self.name))
        self.b_2 = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'), name='cnn_b2'+self.name))
        self.b_3 = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'), name='cnn_b3'+self.name))

    # This function takes as input word indices and extracts their corresponding word embeddings
    def approx_embedder(self, x):
        return self.W_emb[x]
    
    def ConvLayer1(self, q1):
        output = T.dot(q1, self.Filter1) + self.b_1
        return output
    
    def ConvLayer2(self, q1, q2):
        output = T.dot(T.concatenate([q1, q2], axis=1), self.Filter2) + self.b_2
        return output
    
    def ConvLayer3(self, q1, q2, q3):
        output = T.dot(T.concatenate([q1, q2, q3], axis=1), self.Filter3) + self.b_3
        return output
    
    def Convolution(self, x, mask):
        xe = self.approx_embedder(x)
        _mask = self.tmp[mask]
        
        _res1, _ = theano.scan(self.ConvLayer1, sequences=[xe])
        _res2, _ = theano.scan(self.ConvLayer2, sequences=[xe[:-1], xe[1:]])
        _res3, _ = theano.scan(self.ConvLayer3, sequences=[xe[:-2],xe[1:-1],xe[2:]])
        
        hidden1 = T.tanh(T.max(_res1*_mask, axis=0)).dimshuffle('x',0,1)
        hidden2 = T.tanh(T.max(_res2*_mask[:-1], axis=0)).dimshuffle('x',0,1)
        hidden3 = T.tanh(T.max(_res3*_mask[:-2], axis=0)).dimshuffle('x',0,1)
        
        return T.mean(T.concatenate([hidden1, hidden2, hidden3], axis=0), axis=0)
        #return hidden3
        #return (hidden1 + hidden2 + hidden3)/3.0
        #return x[:5]
        #return (hidden1 + hidden2)/2.0
    
    def build_encoder(self, x, mask): #x是一个matrix
        res = self.Convolution(x, mask)
        
        return res
        
    def __init__(self, word_embedding_param, name, config):
        self.name = name
        self.rankdim = config.w_dim
        self.qdim_encoder = config.h_dim
        self.params = []
        self.rng = np.random.RandomState(23333)
        self.init_params(word_embedding_param)
        a = np.zeros((2, self.qdim_encoder))
        a[1] = 1
        self.tmp = theano.shared(value=a)