'''
Created on 2020年7月14日

@author: Shaoyu Dou

update by cy 20210108
'''
config_dtcr = {}

# seq2seq
config_dtcr['encoder_hidden_units'] = [50,30,30] # or [50,30,30], [100,50,50]
config_dtcr['dilations'] = [1,4,16]

# classification
config_dtcr['classifier_hidden_units'] = [128,2]

# dataset setting]
config_dtcr['train_file'] = 'UCRArchive_2018/{0}/{0}.TRAIN.tsv'.format('TwoLeadECG')
config_dtcr['test_file'] = 'UCRArchive_2018/{0}/{0}.TEST.tsv'.format('TwoLeadECG')

config_dtcr['training_samples_num'] = 23 # config in main file
config_dtcr['cluster_num'] = 2 # config in main file
config_dtcr['input_length'] = 82 # config in main file
config_dtcr['data_aug'] = True
config_dtcr['use_kmeans'] = False
config_dtcr['kmeans_step'] = 1
config_dtcr['model_name'] = 'dtcr'
config_dtcr['save_model'] = True
config_dtcr['save_for_vs'] = False

# loss
config_dtcr['lambda'] = 1 # or 1, 1e-1, 1e-2, 1e-3
config_dtcr['reconstruction_loss'] = True
config_dtcr['sim_loss'] = True

# train setting
config_dtcr['max_iter'] = 2001 # config
config_dtcr['alter_iter'] = 10
config_dtcr['test_every_epoch'] = 5
config_dtcr['img_path'] = 'train_img' # config
# config_dtcr['save_path'] = 'train_img'

# performance setting
config_dtcr['indicator'] = 'RI' # 'RI' or 'NMI'

