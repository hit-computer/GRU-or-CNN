#coding:utf-8
from utils import NormalInit, OrthogonalInit, add_to_params
import theano
import numpy as np
import theano.tensor as T

class SentenceEncoder():
    def init_params(self, word_embedding_param):
        # Initialzie W_emb to given word embeddings
        assert(word_embedding_param != None)
        self.W_emb = word_embedding_param

        """ sent weights """
        self.W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_encoder), name='W_in'+self.name))
        self.W_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim_encoder, self.qdim_encoder), name='W_hh'+self.name))
        self.b_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'), name='b_hh'+self.name))
        
        self.W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_encoder), name='W_in_r'+self.name))
        self.W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_encoder), name='W_in_z'+self.name))
        self.W_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim_encoder, self.qdim_encoder), name='W_hh_r'+self.name))
        self.W_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim_encoder, self.qdim_encoder), name='W_hh_z'+self.name))
        self.b_z = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'), name='b_z'+self.name))
        self.b_r = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'), name='b_r'+self.name))

    # This function takes as input word indices and extracts their corresponding word embeddings
    
    def approx_embedder(self, x):
        return self.W_emb[x]

    def GRU_sent_step(self, x_t, m_t, ph_t):
        hr_tm1 = ph_t

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_r) + T.dot(hr_tm1, self.W_hh_r) + self.b_r)
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_z) + T.dot(hr_tm1, self.W_hh_z) + self.b_z)
        h_tilde = T.tanh(T.dot(x_t, self.W_in) + T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
        h_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
        
        m_t = m_t.dimshuffle(0, 'x') #make a column out of a 1d vector (N to Nx1)
        h_t = (m_t) * h_t + (1 - m_t) * ph_t
        
        # return both reset state and non-reset state
        return h_t, r_t, z_t, h_tilde

    def build_encoder(self, x, mask, prev_state): #x是一个matrix
        xe = self.approx_embedder(x)
        
        hs_0 = prev_state
        _res, _ = theano.scan(self.GRU_sent_step,
                          sequences=[xe, mask],\
                          outputs_info=[hs_0, None, None, None])#每次循环输入GRU_sent_step是一个矩阵，shape为N*w_dim(N为x的列维度)

        # Get the hidden state sequence
        h = _res[0] #返回f_enc函数每次调用的第一个输出值，在RGU中h[i]会作为f_enc第i+1次迭代的输入，得到h[i+1]
        return h, mask

    def __init__(self, word_embedding_param, name, config):
        self.name = name
        self.rankdim = config.w_dim
        self.qdim_encoder = config.h_dim
        self.params = []
        self.rng = np.random.RandomState(23333)
        self.init_params(word_embedding_param)
