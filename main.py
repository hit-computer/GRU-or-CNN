#coding:utf-8
import theano, random
import numpy as np
import cPickle,time
import theano.tensor as T
from collections import OrderedDict, Counter
import logging
from utils import compute_updates, NormalInit, add_to_params
from GRUModel import SentenceEncoder
from CNNModel import SentenceEncoder_CNN

logging.basicConfig(level=logging.DEBUG)

class Configuration(object):
    margin = 0.6 #正负例得分间隔
    iter = 6 #迭代次数
    learning_rate = 0.0003
    test_freq = 1 #每迭代多少次进行一次测试
    h_dim = 300 #句子向量维度
    vocab_size = 60000
    w_dim = 100 #词向量维度
    neg_sample = 10
    up_dim = 500 #句子联合表示向量维度
    CNN_Flag = True #是否使用CNN，为False时使用GRU
    save_file = 'test_res' #测试结果保存文件名

config = Configuration()

def ReadDate(file1, file2): #选90W作为训练数据，10W作为测试数据
    Que = []
    Ans = []
    allword = []
    with open(file1,'r') as fq, open(file2,'r') as fa:
        for line in fq:
            tmp = line.split()
            allword += tmp
            if config.CNN_Flag:
                while len(tmp) < 3 and len(tmp) > 0: #当使用CNN模型时，需要做padding
                    tmp.append('OOV')
                Que.append(tmp)
            else:
                Que.append(tmp)
        for line in fa:
            tmp = line.split()
            allword += tmp
            if config.CNN_Flag:
                while len(tmp) < 3 and len(tmp) > 0:
                    tmp.append('OOV')
                Ans.append(tmp)
            else:
                Ans.append(tmp)
    
    assert(len(Que)==len(Ans))
    traindata = []
    testdata = []
    c = Counter(allword)
    vocab = [i[0] for i in c.most_common(config.vocab_size-1)]
    for q,a in zip(Que[:900000],Ans[:900000]):
        traindata.append((q,a))
    for q,a in zip(Que[900000:950000],Ans[900000:950000]):
        testdata.append((q,a,1))
    for q in Que[950000:]:
        a = Ans[random.randint(0,200000)]
        testdata.append((q,a,0))
    
    return traindata, testdata, vocab

    
print 'Loading the data...'
traindata, testdata, vocab = ReadDate('100w.q', '100w.a')#'100w.q'全是question，'100w.a'是对应的answers，请替换成自己的文件。
print len(traindata), len(testdata)
print ' Done'
str_to_id = dict([(j,i) for i,j in enumerate(vocab)]+[('OOV',config.vocab_size-1)])
assert(len(str_to_id)==config.vocab_size)


print 'Build model...'
rng = np.random.RandomState(23455)
params = []
W_emb = add_to_params(params, theano.shared(value=NormalInit(rng, config.vocab_size, config.w_dim), name='W_emb'))

T_que = T.imatrix('question')
T_ans = T.imatrix('answer')
T_neg = T.imatrix('neg_sample')
M_que = T.imatrix('question')
M_ans = T.imatrix('answer')
M_neg = T.imatrix('neg_sample')

if config.CNN_Flag == False:
    print 'use GRU model...'
    Question_Encoder = SentenceEncoder(W_emb, 'Question', config)
    Answer_Encoder = SentenceEncoder(W_emb, 'Answer', config)

    que_ph = theano.shared(value=np.zeros((1, config.h_dim), dtype='float32'), name='que_ph')
    ans_ph = theano.shared(value=np.zeros((1, config.h_dim), dtype='float32'), name='ans_ph')
    neg_ph = theano.shared(value=np.zeros((config.neg_sample, config.h_dim), dtype='float32'), name='neg_ph')

    que_h, _ = Question_Encoder.build_encoder(T_que, T.eq(M_que,1), que_ph)
    ans_h, _ = Answer_Encoder.build_encoder(T_ans, T.eq(M_ans,1), ans_ph)
    neg_h, _test_mask = Answer_Encoder.build_encoder(T_neg, T.eq(M_neg,1), neg_ph)

    que_emb = que_h[-1]
    ans_emb = ans_h[-1]
    neg_emb = neg_h[-1]
    
else:
    print 'use CNN model...'
    Question_Encoder = SentenceEncoder_CNN(W_emb, 'Question', config)
    Answer_Encoder = SentenceEncoder_CNN(W_emb, 'Answer', config)
    
    que_emb = Question_Encoder.build_encoder(T_que, T.eq(M_que,1))
    ans_emb = Answer_Encoder.build_encoder(T_ans, T.eq(M_ans,1))
    neg_emb = Answer_Encoder.build_encoder(T_neg, T.eq(M_neg,1))

W_up = add_to_params(params, theano.shared(value=NormalInit(rng, 2*config.h_dim, config.up_dim), name='W_up'))
W_up_b = add_to_params(params, theano.shared(value=np.zeros((config.up_dim,), dtype='float32'), name='W_up_b'))
Sen_U = add_to_params(params, theano.shared(value=NormalInit(rng, config.up_dim, 1), name='Sen_U'))
Sen_b = add_to_params(params, theano.shared(value=np.zeros((1,), dtype='float32'), name='Sen_b'))

join_emb = T.concatenate([que_emb, ans_emb], axis=1)
join_hidden = T.tanh(T.dot(T.concatenate([que_emb, ans_emb], axis=1), W_up)+W_up_b)
#join_hidden = T.tanh(T.dot(W_up, join_emb.T)+W_up_b)
f_x = T.nnet.sigmoid(T.dot(join_hidden, Sen_U)+Sen_b)

neg_join_hidden = T.tanh(T.dot(T.concatenate([T.repeat(que_emb, config.neg_sample, axis=0), neg_emb], axis=1), W_up)+W_up_b)
f_neg = T.nnet.sigmoid(T.dot(neg_join_hidden, Sen_U)+Sen_b)

cost = T.maximum(0, config.margin - f_x.sum() + f_neg)
training_cost = cost.sum()

updates = compute_updates(training_cost, params+Question_Encoder.params+Answer_Encoder.params, config)

train_model = theano.function([T_que, T_ans, T_neg, M_que, M_ans, M_neg],[training_cost],updates=updates, on_unused_input='ignore', name="train_fn")
#train_model = theano.function([T_que, T_ans, T_neg, M_que, M_ans, M_neg],[que_emb, ans_emb, neg_emb], on_unused_input='ignore', name="train_fn")
test_model = theano.function([T_que, T_ans, M_que, M_ans], [f_x], on_unused_input='ignore', name="train_fn")
print 'function build finish!'


print 'Training...'
for step in range(1, config.iter+1):
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
        while len(nsample) < config.neg_sample:
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
            for j in range(config.neg_sample):
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
        #print que_matrix.shape, ans_matrix.shape, neg_matrix.shape
        #a, b, c = train_model(que_matrix, ans_matrix, neg_matrix, que_mask, ans_mask, neg_mask)
        #print a.shape, b.shape, c.shape
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
    
    if step%config.test_freq == 0:
        print 'Test...'
        fw_valid = open(config.save_file+'_%d.txt'%step, 'w')
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
            #break
        accuracy = 1.0 * test_right / test_length
        fw_valid.write('\r\n'+'Accuracy: ' + str(accuracy))
        fw_valid.close()
        print 'Accuracy: ', accuracy
        #vals = dict([(x.name, x.get_value()) for x in [Wd_out, bd_out]])
        #np.savez('models/model_%d.npz'%step, **vals)
        print 'Test Done' 
        