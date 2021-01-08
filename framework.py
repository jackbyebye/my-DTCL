
import tensorflow as tf
import numpy as np
from rnns import dilated_encoder, single_layer_decoder
from classification import classifier
from kmeans import kmeans
from utils import truncatedSVD, ri_score, cluster_using_kmeans, nmi_score,data_aug,rand_index_score
from scipy.special import comb
from sklearn.cluster import KMeans

class DTCR():
    def __init__(self, opts):
        self.opts = opts
        
        tf.reset_default_graph()
#         import pdb;pdb.set_trace()
        self.creat_network()
        self.init_optimizers()
        
    
    def creat_network(self):
        opts = self.opts
        self.encoder_input = tf.placeholder(dtype=tf.float32, shape=(None, opts['input_length'], opts['input_dims']), name='encoder_input')
        self.decoder_input = tf.placeholder(dtype=tf.float32, shape=(None, opts['input_length'], opts['input_dims']), name='decoder_input')
        # self.encoder_input = tf.placeholder(dtype=tf.float32, shape=(None, None, opts['input_dims']), name='encoder_input')
        # self.decoder_input = tf.placeholder(dtype=tf.float32, shape=(None, None, opts['input_dims']), name='decoder_input')

        # self.classification_labels = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='classification_labels')
        
        
        # seq2seq
        with tf.variable_scope('seq2seq'):
            self.D_ENCODER = dilated_encoder(opts)
            self.h = self.D_ENCODER.encoder(self.encoder_input)
            
            self.S_DECOER = single_layer_decoder(opts)
            recons_input = self.S_DECOER.decoder(self.h, self.decoder_input)
            self.recons_input  = recons_input
            # self.h_fake, self.h_real = tf.split(self.h, num_or_size_splits=2, axis=0)
            
        # classifier
        # with tf.variable_scope('classifier'):
        #     self.CLS = classifier(opts)
        #     output_without_softmax = self.CLS.cls_net(self.h)
        
        # K-means
        # with tf.variable_scope('kmeans'):
        #     self.KMEANS = kmeans(opts)
        #     # update F
        #     kmeans_obj = self.KMEANS.kmeans_optimalize(self.h_real)


        # add_contrastive_loss
        hidden = self.h
        hidden_norm=True
        temperature=1.0
        weights=1.0
        LARGE_NUM = 1e9
        # Get (normalized) hidden1 and hidden2.
        if hidden_norm:
            hidden = tf.keras.backend.l2_normalize(hidden, -1)
        hidden1, hidden2 = tf.split(hidden, 2, 0)
        batch_size = tf.shape(hidden1)[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
        logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

        loss_a = tf.losses.softmax_cross_entropy(
            labels, tf.concat([logits_ab, logits_aa], 1), weights=weights)
        loss_b = tf.losses.softmax_cross_entropy(
            labels, tf.concat([logits_ba, logits_bb], 1), weights=weights)
        self.contrastive_loss = loss_a + loss_b


        
        
        # L-reconstruction
        self.loss_reconstruction = tf.losses.mean_squared_error(self.encoder_input, recons_input)
        # L-classification
        # self.loss_classification = tf.losses.softmax_cross_entropy(self.classification_labels, output_without_softmax)
        # L-kmeans
        # self.loss_kmeans = kmeans_obj
        
        
    def init_optimizers(self):
        lambda_1 = self.opts['lambda']
        
        # vars
        seq2seq_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='seq2seq')
        # cls_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
        # end2end_vars = seq2seq_vars + cls_vars
        end2end_vars = seq2seq_vars
        
        # kmeans_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='kmeans')
        # import pdb;pdb.set_trace()
        # loss
        # self.loss_dtcr = self.loss_reconstruction + self.loss_classification + lambda_1 * self.loss_kmeans
        if self.opts['reconstruction_loss'] and self.opts['sim_loss']:
            self.loss_dtcr = self.loss_reconstruction + lambda_1 * self.contrastive_loss
        elif self.opts['reconstruction_loss'] and (not self.opts['sim_loss']):
            self.loss_dtcr = self.loss_reconstruction
        elif (not self.opts['reconstruction_loss']) and self.opts['sim_loss']:
            self.loss_dtcr = self.contrastive_loss

    
        
        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=5e-3)
        

        # update vars
        self.train_op = optimizer.minimize(self.loss_dtcr, var_list=end2end_vars)
    
    def update_kmeans_f(self, train_h):    
        new_f = truncatedSVD(train_h, self.opts['cluster_num'])
        self.KMEANS.update_f(new_f)        
        
    def train(self, x_train,y_train,x_test,y_test,bs = 4):
        '''
        x_train: shape: (2*batchsize, timestep, dim), 前半部分是fake data
        cls_label: shape: (2*batchsize)
        
        train_data/shape: (batchsize, timestep, dim)
        train_label/test_label: (batchsize)
        
        '''
        opts =self.opts
        
        # processing data and label
        # cls_data = np.expand_dims(cls_data, axis=2)        
        # cls_label_ = np.zeros(shape=(cls_label.shape[0], len(np.unique(cls_label))))
        # cls_label_[np.arange(cls_label_.shape[0]), cls_label] = 1

        
        # feed dict
        # feed_d = {self.encoder_input: cls_data,
        #           self.decoder_input: np.zeros_like(cls_data),
        #           self.classification_labels: cls_label_}

        # session
        config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver(max_to_keep=200)
        print('vars_num: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
#         import pdb;pdb.set_trace()
        # init:
        # train_h = sess.run(self.h_real, feed_dict=feed_d)
        # self.update_kmeans_f(train_h)
        # print('init k-means vars.')
        
        # train
        train_list = []
        test_list = []
        
        best_indicator = 0
        best_epoch = -1
        for step in range(opts['max_iter']):
            print("step:",step)
            if opts['use_kmeans'] and step >= 50:
                if step % opts['kmeans_step'] ==0:
                    train_embedding = self.test(sess, x_train)
                    kmeans = KMeans(n_clusters=len(np.unique(y_train)), random_state=0).fit(train_embedding)
                idx = []
                for i in range(opts['cluster_num']):
                    idx.append(np.random.choice(np.where(kmeans.labels_==i)[0],1)[0])
                ##to do data aug
                # import pdb;pdb.set_trace()
                feed_d = {self.encoder_input: np.concatenate((x_train[idx],x_train[idx]),axis = 0),
                self.decoder_input: np.zeros_like(np.concatenate((x_train[idx],x_train[idx]),axis = 0))}
            else:
                # import pdb;pdb.set_trace()
                if self.opts['data_aug']:
                    if step==0:
                        print("use data aug")
                    x_train_aug,y_train_aug=data_aug(x_train,y_train,bs = bs)
                    feed_d = {self.encoder_input: x_train_aug,
                    self.decoder_input: np.zeros_like(x_train_aug)}
                else:
                    if step==0:
                        print("not use data aug")
                    feed_d = {self.encoder_input: np.concatenate((x_train,x_train),axis = 0),
                    self.decoder_input: np.zeros_like(np.concatenate((x_train,x_train),axis = 0))}                

            _, loss, l_recons, l_sim = sess.run([self.train_op, self.loss_dtcr, self.loss_reconstruction,self.contrastive_loss], feed_dict=feed_d)
            print('loss: {}, l_recons: {}, l_sim: {}, step: {}'.format(loss, l_recons, l_sim, step))
            
            # if epoch % opts['alter_iter'] == 0:
            #     train_h = sess.run(self.h_real, feed_dict=feed_d)
            #     self.update_kmeans_f(train_h)
            #     print('update F matrix in k-means loss, epoch: {}.'.format(epoch))
                
            if step % opts['test_every_epoch'] == 0:
                # import pdb;pdb.set_trace()
                train_embedding = self.test(sess, x_train)
                test_embedding = self.test(sess, x_test)
                
                # kmeans
                # pred_train = cluster_using_kmeans(train_embedding, opts['cluster_num'])
                # pred_test = cluster_using_kmeans(test_embedding, opts['cluster_num'])
                
                # performance
                # if opts['indicator'] == 'RI':
                #     score_train = ri_score(train_label, pred_train)
                #     score_test = ri_score(test_label, pred_test)
                # elif opts['indicator'] == 'NMI':
                #     score_train = nmi_score(train_label, pred_train)
                #     score_test = nmi_score(test_label, pred_test)     
                # res = classifier.predict(x_test,model_path="./model/"+ dataset_name + "_last_model.hdf5")
                
                kmeans = KMeans(n_clusters=len(np.unique(y_train)), random_state=0).fit(train_embedding)
                ri = rand_index_score(kmeans.labels_, [int(i) for i in y_train])   
                kmeans_test = KMeans(n_clusters=len(np.unique(y_train)), random_state=0).fit(test_embedding)
                ri_test = rand_index_score(kmeans_test.labels_, [int(i) for i in y_test])             
                print('{}: train: {}\ttest:{}'.format(opts['indicator'], ri, ri_test))
                # performance list
                train_list.append(ri)
                test_list.append(ri_test)
                
                if ri_test > best_indicator:
                    best_indicator = ri_test
                    best_step = step
                    saver.save(sess,'model/'+opts['model_name']+'.ckpt',global_step=0)  
                    print("save model!" + str(step) + ' model_name: ' +opts['model_name'])

                if opts['save_for_vs']:
                    saver.save(sess,'model_for_vs/'+opts['model_name']+ str(ri_test) + '.ckpt',global_step=0)
                    print("save model!" + str(step) + ' model_name: ' +opts['model_name'] + "ri=" + str(round(ri_test,4)))  
        if opts['save_model']:
            saver=tf.train.Saver()
            saver.save(sess,'model/'+opts['model_name']+'.ckpt',global_step=step)                
        sess.close()
               
        return best_indicator, best_step, train_list, test_list
                    
    def test(self, sess, test_data):
        
        # test_data = np.expand_dims(test_data, axis=2)
        
        feed_d = {self.encoder_input: test_data}
        
        h = sess.run(self.h, feed_dict=feed_d)
        return h
    # def ensemble(self,sess,)
    def recons(self,sess,test_data):
        feed_d = {self.encoder_input: test_data,self.decoder_input: np.zeros_like(test_data)}
        recons = sess.run(self.recons_input,feed_dict=feed_d)
        return recons