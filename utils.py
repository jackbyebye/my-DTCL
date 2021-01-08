
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from scipy.special import comb
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans


def normalize(seq):
    return 2 * (seq - np.min(seq)) / (np.max(seq) - np.min(seq)) - 1

# def read_dataset(opts, dataset_type, label_dict=None, if_n=False):
#     '''
#     normal_cluster: 代表正常的标签，在所有数据集中，将数据占比多的一方视为正常数据
#     split: 分割数据的段数
#     '''
#     if dataset_type == 'train':
#         data = np.loadtxt(opts['train_file'])
#     elif dataset_type == 'test':
#         data = np.loadtxt(opts['test_file'])
#     elif dataset_type == 'v':
#         data = np.loadtxt(opts['test_file'])
    
            
#     label = data[:,0]
#     label = label + 10
#     label = -1 * label
    
#     if label_dict is None:
#         label_dict = {}
#         label_list = np.unique(label)
#         for idx in range(len(label_list)):
#             label_dict[str(label_list[idx])] = idx#key：-1*原始label，value：新label

#     o_label = list(label_dict.keys())
#     for l in o_label:
#         label[label == float(l)] = label_dict[l]
        
#     label = label.astype(int)
#     data = data[:,1:]

#     #----------------------------------------
    
#     if dataset_type == 'test' and 'MIT' in opts['test_file']:
#         tmp_data = []
#         tmp_label = []
#         for item in np.unique(label):
#             tmp_data.append(data[label == item][0:50])
#             tmp_label.append(label[label == item][0:50])
#         data = np.concatenate(tmp_data, axis=0)
#         label = np.concatenate(tmp_label, axis=0)
        
#     #----------------------------------------
#     if if_n == True:
#         for i in range(data.shape[0]):
#             data[i] = normalize(data[i])

    
#     #数据集中的类别数量
#     print(dataset_type)       
#     print('Number of class: ', len(np.unique(label)))
#     print('Number of sample:', data.shape[0])
#     print('Time Series Length: ', data.shape[1])
#     return data, label, label_dict

def shuffle_timeseries(data, rate=0.2):
    # 打乱一定比率的数据
    ordered_index = np.arange(len(data))
    ordered_index.astype(int)
    # 选定要打乱的index
    shuffled_index = np.random.choice(ordered_index, size=int(np.floor(rate*len(data))), replace=False)
    ordered_index[shuffled_index] = -1
    # 打乱
    shuffled_index = np.random.permutation(shuffled_index)
    ordered_index[ordered_index == -1] = shuffled_index
    data = data[ordered_index]
    
    return data

def data_aug(x_train,y_train,bs = 8):
    bs = max(1,int(bs/2))
    nb_datas = x_train.shape[0]
    idx = np.random.choice(range(nb_datas),bs,replace=False)
    ##random cut
    # rnd_idx = np.random.choice(range(x_train.shape[1]//5),1)[0]
    # x = x_train[idx,rnd_idx:]
    #not random cut
    x = x_train[idx]
    # mus = x.sum(axis = 1,keepdims=True)
    type = np.random.choice(range(3),1)
    if type==0:
        ##norm noise 
        print(type)
        sigma = x.std(axis = 1,keepdims=True)
        noise = np.random.randn(x.shape[0],x.shape[1],x.shape[2])
        x1 = noise * 0.1*sigma 
        noise = np.random.randn(x.shape[0],x.shape[1],x.shape[2])
        x2 = noise * 0.1*sigma 
        x12 = np.concatenate((x+x1,x+x2),axis = 0)
    elif type==1:
        ##random shuffle
        first = 0
        print(type)
        for x_i in x:
            if first==0:
                rnd_idx = np.random.choice(range(x_i.shape[0]),1)[0]
                tmp = np.concatenate((x_i[rnd_idx:],x_i[0:rnd_idx]),axis = 0)
                x_aug = np.expand_dims(tmp,0)
                first = 1
            else:
                rnd_idx = np.random.choice(range(x_i.shape[0]),1)[0]
                tmp = np.concatenate((x_i[rnd_idx:],x_i[0:rnd_idx]),axis = 0)
                tmp = np.expand_dims(tmp,0)
                x_aug = np.concatenate((x_aug,tmp),axis = 0)
        for x_i in x:
            rnd_idx = np.random.choice(range(x_i.shape[0]),1)[0]
            tmp = np.concatenate((x_i[rnd_idx:],x_i[0:rnd_idx]),axis = 0)
            tmp = np.expand_dims(tmp,0)
            x_aug = np.concatenate((x_aug,tmp),axis = 0)        
        x12 = x_aug
    elif type==2:
        ##random cut
        pass
        rnd_idx = np.random.choice(range(x_train.shape[1]//5),1)[0]
        x1 = x
        x2 = x
        x2[:,:rnd_idx]=0
        x12 = np.concatenate((x1,x2),axis = 0)

    assert x12.shape[0] == 2*bs
    return x12,np.zeros([x12.shape[0],1])
def readucr(filename, delimiter='\t'):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

def read_dataset(dataset_name='Beef',if_n=False):

    # datasets_dict = {}
    # dataset_name = 'Beef'
    # import pdb;pdb.set_trace()
    root_dir='../../UCRArchive_2018/'
    file_name = root_dir  + '/' + dataset_name + '/' + dataset_name
    x_train, y_train = readucr(file_name + '_TRAIN.tsv')
    x_test, y_test = readucr(file_name + '_TEST.tsv')
    if if_n == True:
        for i in range(x_train.shape[0]):
            x_train[i] = normalize(x_train[i]) 
    if if_n == True:
        for i in range(x_test.shape[0]):
            x_test[i] = normalize(x_test[i]) 
    # nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    ##infinity loop

    return x_train,y_train,x_test,y_test
    # while 1:
    #     input_batch, target_batch = data_aug(x_train,y_train,bs = bs)
    #     yield input_batch, target_batch


# def construct_classification_dataset(dataset):
#     real_dataset = copy.deepcopy(dataset)
#     fake_dataset = []
#     for seq in real_dataset:
#         fake_dataset.append(shuffle_timeseries(seq))
#     fake_dataset = np.array(fake_dataset)
    
#     label = np.array([1]*fake_dataset.shape[0] + [0]*real_dataset.shape[0])
#     dataset = np.concatenate([fake_dataset, real_dataset], axis=0)
    
#     label = np.random.permutation(label)
#     dataset = np.random.permutation(dataset)
    
#     print('dataset shape: ', dataset.shape)
#     print('label shape:', label.shape)
    
#     return dataset, label

def truncatedSVD(matrix, K):
    svd = TruncatedSVD(n_components=K)
    truncated_matrix = svd.fit_transform(matrix)
    return truncated_matrix
    
def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)
def ri_score(y_true, y_pred):
    tp_plus_fp = comb(np.bincount(y_true), 2).sum()
    tp_plus_fn = comb(np.bincount(y_pred), 2).sum()
    A = np.c_[(y_true, y_pred)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(y_true))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def nmi_score(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')

def cluster_using_kmeans(embeddings, K):
    return KMeans(n_clusters=K).fit_predict(embeddings)

def show_train_test_curve(opts, train, test, index=''):        
    file_name = '{}_{}_en_{}_lambda_{}_train_curve.png'.format(index, opts['indicator'], opts['encoder_hidden_units'], opts['lambda'])

    x = np.arange(len(train))
    x *= opts['test_every_epoch']
    
    plt.plot(x, train, label='train')
    plt.plot(x, test, label='test')
    plt.title('{} curve'.format(opts['indicator']))
    plt.xlabel('epoch')
    plt.ylabel(opts['indicator'])
    plt.ylim((0,1))
    plt.legend()
    plt.savefig(opts['img_path']+'/'+file_name)
    plt.close()
    