#coding:utf-8
import theano, argparse, random
import numpy as np
import cPickle,time
import theano.tensor as T
from collections import OrderedDict, Counter
import logging
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Coherence')

margin = 0.6 #正负例得分间隔
iter = 6 #迭代次数
learning_rate = 0.0003
test_freq = 1 #每迭代多少次进行一次测试
h_dim = 300 #句子向量维度
vocab_size = 60000
w_dim = 100 #词向量维度
neg_sample = 10
up_dim = 500 #句子联合表示向量维度

def NormalInit(rng, sizeX, sizeY, scale=0.01, sparsity=-1):
    """ 
    Normal Initialization
    """
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    
    if sparsity < 0:
        sparsity = sizeY
     
    sparsity = np.minimum(sizeY, sparsity)
    values = np.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals
        
    return values.astype(theano.config.floatX)
    
def OrthogonalInit(rng, sizeX, sizeY, sparsity=-1, scale=1):
    """ 
    Orthogonal Initialization
    """

    sizeX = int(sizeX)
    sizeY = int(sizeY)

    assert sizeX == sizeY, 'for orthogonal init, sizeX == sizeY'

    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = np.minimum(sizeY, sparsity)

    values = np.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals

    # Use SciPy:
    if sizeX*sizeY > 20000000:
        import scipy
        u,s,v = scipy.linalg.svd(values)
    else:
        u,s,v = np.linalg.svd(values)
    values = u * scale
    return values.astype(theano.config.floatX)

def add_to_params(params, new_param):
    params.append(new_param)
    return new_param

def ReadDate(file1, file2): #选90W作为训练数据，10W作为测试数据
    Que = []
    Ans = []
    allword = []
    with open(file1,'r') as fq, open(file2,'r') as fa:
        for line in fq:
            tmp = line.split()
            Que.append(tmp)
            allword += tmp
        for line in fa:
            tmp = line.split()
            Ans.append(tmp)
            allword += tmp
    
    assert(len(Que)==len(Ans))
    traindata = []
    testdata = []
    c = Counter(allword)
    vocab = [i[0] for i in c.most_common(vocab_size-1)]
    for q,a in zip(Que[:900000],Ans[:900000]):
        traindata.append((q,a))
    for q,a in zip(Que[900000:950000],Ans[900000:950000]):
        testdata.append((q,a,1))
    for q in Que[950000:]:
        a = Ans[random.randint(0,200000)]
        testdata.append((q,a,0))
    
    return traindata, testdata, vocab

def SoftMax(x):
    x = T.exp(x - T.max(x, axis=x.ndim-1, keepdims=True))
    return x / T.sum(x, axis=x.ndim-1, keepdims=True)
    
print 'Loading the data...'
traindata, testdata, vocab = ReadDate('100.q', '100.a')#'100.q'全是question，'100.a'是对应的answers，请替换成自己的文件。
print len(traindata), len(testdata)
print ' Done'
str_to_id = dict([(j,i) for i,j in enumerate(vocab)]+[('OOV',vocab_size-1)])
assert(len(str_to_id)==vocab_size)

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
        
        m_t = m_t.dimshuffle(0, 'x')
        h_t = (m_t) * h_t + (1 - m_t) * ph_t
        
        # return both reset state and non-reset state
        return h_t, r_t, z_t, h_tilde

    def build_encoder(self, x, mask, prev_state): #x是一个matrix
        xe = self.approx_embedder(x)
        
        hs_0 = prev_state
        _res, _ = theano.scan(self.GRU_sent_step,
                          sequences=[xe, mask],\
                          outputs_info=[hs_0, None, None, None])

        # Get the hidden state sequence
        h = _res[0] #返回f_enc函数每次调用的第一个输出值，在RGU中h[i]会作为f_enc第i+1次迭代的输入，得到h[i+1]
        return h, mask

    def __init__(self, word_embedding_param, name):
        self.name = name
        self.rankdim = w_dim
        self.qdim_encoder = h_dim
        self.params = []
        self.rng = np.random.RandomState(23333)
        self.init_params(word_embedding_param)


print 'Build model...'
rng = np.random.RandomState(23455)
params = []
W_emb = add_to_params(params, theano.shared(value=NormalInit(rng, vocab_size, w_dim), name='W_emb'))

T_que = T.imatrix('question')
T_ans = T.imatrix('answer')
T_neg = T.imatrix('neg_sample')
M_que = T.imatrix('question')
M_ans = T.imatrix('answer')
M_neg = T.imatrix('neg_sample')

Question_Encoder = SentenceEncoder(W_emb, 'Question')
Answer_Encoder = SentenceEncoder(W_emb, 'Answer')

que_ph = theano.shared(value=np.zeros((1, h_dim), dtype='float32'), name='que_ph')
ans_ph = theano.shared(value=np.zeros((1, h_dim), dtype='float32'), name='ans_ph')
neg_ph = theano.shared(value=np.zeros((neg_sample, h_dim), dtype='float32'), name='neg_ph')

que_h, _ = Question_Encoder.build_encoder(T_que, T.eq(M_que,1), que_ph)
ans_h, _ = Answer_Encoder.build_encoder(T_ans, T.eq(M_ans,1), ans_ph)
neg_h, _test_mask = Answer_Encoder.build_encoder(T_neg, T.eq(M_neg,1), neg_ph)

que_emb = que_h[-1]
ans_emb = ans_h[-1]
neg_emb = neg_h[-1]

W_up = add_to_params(params, theano.shared(value=NormalInit(rng, 2*h_dim, up_dim), name='W_up'))
W_up_b = add_to_params(params, theano.shared(value=np.zeros((up_dim,), dtype='float32'), name='W_up_b'))
Sen_U = add_to_params(params, theano.shared(value=NormalInit(rng, up_dim, 1), name='Sen_U'))
Sen_b = add_to_params(params, theano.shared(value=np.zeros((1,), dtype='float32'), name='Sen_b'))

join_emb = T.concatenate([que_emb, ans_emb], axis=1)
join_hidden = T.tanh(T.dot(T.concatenate([que_emb, ans_emb], axis=1), W_up)+W_up_b)
#join_hidden = T.tanh(T.dot(W_up, join_emb.T)+W_up_b)
f_x = T.nnet.sigmoid(T.dot(join_hidden, Sen_U)+Sen_b)

neg_join_hidden = T.tanh(T.dot(T.concatenate([T.repeat(que_emb,neg_sample,axis=0), neg_emb], axis=1), W_up)+W_up_b)
f_neg = T.nnet.sigmoid(T.dot(neg_join_hidden, Sen_U)+Sen_b)

cost = T.maximum(0, margin - f_x.sum() + f_neg)
training_cost = cost.sum()

def sharedX(value, name=None, borrow=False, dtype=None):
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)

def Adam(grads, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    i = sharedX(0.)
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in grads.items():
        m = sharedX(p.get_value() * 0.)
        v = sharedX(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates
    
def compute_updates(training_cost, params):
    updates = []
     
    grads = T.grad(training_cost, params)
    grads = OrderedDict(zip(params, grads))

    # Clip stuff
    c = np.float32(1.)
    clip_grads = []
    
    norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
    normalization = T.switch(T.ge(norm_gs, c), c / norm_gs, np.float32(1.))
    notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
     
    for p, g in grads.items():
        clip_grads.append((p, T.switch(notfinite, np.float32(.1) * p, g * normalization)))
    
    grads = OrderedDict(clip_grads)

    updates = Adam(grads, learning_rate)

    return updates

updates = compute_updates(training_cost, params+Question_Encoder.params+Answer_Encoder.params)

train_model = theano.function([T_que, T_ans, T_neg, M_que, M_ans, M_neg],[training_cost],updates=updates, on_unused_input='ignore', name="train_fn")
#train_model = theano.function([T_que, T_ans, T_neg, M_que, M_ans, M_neg],[f_x, f_neg, cost], on_unused_input='ignore', name="train_fn")
test_model = theano.function([T_que, T_ans, M_que, M_ans], [f_x], on_unused_input='ignore', name="train_fn")
print 'function build finish!'


print 'Training...'
for step in range(iter):
    print 'iter: ',step
    cost = 0
    length = 0
    stime = time.time()
    for idx in range(len(traindata)):
        if idx % 200000 == 0:
            print 'training on ', idx 
        data = traindata[idx]
        que = data[0]
        ans = data[1]
        #print ' '.join(que)
        #print ' '.join(ans)
        if not que or not ans:
            continue
        #_range = range(len(traindata))
        #_range.pop(idx)
        #nsample = random.sample(_range, neg_sample)
        nsample = []
        n_traindata = len(traindata)
        neg_matrix = []
        max_lenght = 0
        while len(nsample) < neg_sample:
            _rand = random.randint(10, n_traindata-10)
            if _rand != idx and _rand not in nsample:
                tmp = []
                if not traindata[_rand][1]:
                    continue
                for wd in traindata[_rand][1]:
                    if wd in str_to_id:
                        tmp.append(str_to_id[wd])
                    else:
                        tmp.append(str_to_id['OOV'])
                neg_matrix.append(tmp)
                max_lenght = max(max_lenght, len(tmp))
                nsample.append(_rand)
        #print len(nsample)
        
        for i_ in nsample:
            tmp = []
            for wd in traindata[i_][1]:
                if wd in str_to_id:
                    tmp.append(str_to_id[wd])
                else:
                    tmp.append(str_to_id['OOV'])
            neg_matrix.append(tmp)
            max_lenght = max(max_lenght, len(tmp))
            
        neg_mask = []
        new_neg_matrix = []
        for i in range(max_lenght):
            tmp = []
            tmp_mask = []
            for j in range(neg_sample):
                if i < len(neg_matrix[j]):
                    tmp.append(neg_matrix[j][i])
                    tmp_mask.append(1)
                else:
                    tmp.append(0)
                    tmp_mask.append(0)
            new_neg_matrix.append(tmp)
            neg_mask.append(tmp_mask)
        
        #print neg_matrix
        #print new_neg_matrix
        #print neg_mask
        
        que_array = []
        que_mask = []
        for wd in que:
            if wd in str_to_id:
                que_array.append([str_to_id[wd]])
            else:
                que_array.append([str_to_id['OOV']])
            que_mask.append([1])
        ans_array = []
        ans_mask = []
        for wd in ans:
            if wd in str_to_id:
                ans_array.append([str_to_id[wd]])
            else:
                ans_array.append([str_to_id['OOV']])
            ans_mask.append([1])
        
        que_matrix = np.array(que_array, dtype=np.int32)
        ans_matrix = np.array(ans_array, dtype=np.int32)
        neg_matrix = np.array(new_neg_matrix, dtype=np.int32)
        
        que_mask = np.array(que_mask, dtype=np.int32)
        ans_mask = np.array(ans_mask, dtype=np.int32)
        neg_mask = np.array(neg_mask, dtype=np.int32)
        
        c = train_model(que_matrix, ans_matrix, neg_matrix, que_mask, ans_mask, neg_mask)[0]
        #print c
        
        if np.isinf(c) or np.isnan(c):
            continue
        cost += c
        length += 1
        #break
    #f = foo()
    etime = time.time()    
    print 'Cost: ', cost/length
    print 'cost time: ', etime-stime,'s'
    
    if step%test_freq == 0: # and step:
        print 'Test...'
        fw_valid = open('1_VALID_%d.txt'%step, 'w')
        test_length = 0
        test_right = 0
        for data in testdata:
            que = data[0]
            ans = data[1]
            label = data[2]
            
            if not que or not ans:
                continue
            
            test_length += 1
            que_array = []
            que_mask = []
            for wd in que:
                if wd in str_to_id:
                    que_array.append([str_to_id[wd]])
                else:
                    que_array.append([str_to_id['OOV']])
                que_mask.append([1])
            ans_array = []
            ans_mask = []
            for wd in ans:
                if wd in str_to_id:
                    ans_array.append([str_to_id[wd]])
                else:
                    ans_array.append([str_to_id['OOV']])
                ans_mask.append([1])
            
            que_matrix = np.array(que_array, dtype=np.int32)
            ans_matrix = np.array(ans_array, dtype=np.int32)
            
            que_mask = np.array(que_mask, dtype=np.int32)
            ans_mask = np.array(ans_mask, dtype=np.int32)
            
            prob = test_model(que_matrix, ans_matrix, que_mask, ans_mask)[0]
            prob = prob[0][0]
            
            if label == 1 and prob > 0.5:
                test_right += 1
            if label == 0 and prob < 0.5:
                test_right += 1
            fw_valid.write('Prob: ' + str(prob) + ' ' + str(label) + '\r\n')
        accuracy = 1.0 * test_right / test_length
        fw_valid.write('\r\n'+'Accuracy: ' + str(accuracy))
        fw_valid.close()
        print 'Accuracy: ', accuracy
        #vals = dict([(x.name, x.get_value()) for x in [Wd_out, bd_out]])
        #np.savez('models/model_%d.npz'%step, **vals)
        print 'Test Done' 
        