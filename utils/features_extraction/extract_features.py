import os
import sys
import argparse
import numpy as np
import pickle

import yaml
from features_util import extract_features
from collections import Counter
import pandas as pd
from database import SER_DATABASES
import random


def main(config):

    params={'window': config['window'],
        'win_length': config['win_length'],
        'hop_length': config['hop_length'],
        'ndft': config['ndft'],
        'nfreq':config['nfreq'],
        'segment_size':config['segment_size'],
        'mixnoise':config['mixnoise'],
        }
    
    dataset  = config['dataset']
    features = config['features']
    dataset_dir = config['dataset_dir']
    mixnoise = config['mixnoise']

    if config['save_dir'] is not None:
        out_filename = config['save_dir'] + '\\' + dataset + '_' + config['save_label'] +'200'+'.pkl'
    else:
        out_filename = 'None'

    print('\n')
    print('*'*50)
    print('\nFEATURES EXTRACTION')
    print(f'\t{"Dataset":>20}: {dataset}')
    print(f'\t{"Features":>20}: {features}')
    print(f'\t{"Dataset dir.":>20}: {dataset_dir}')
    print(f'\t{"Features file":>20}: {out_filename}')
    print(f'\t{"Add noise version":>20}: {mixnoise}')
    print(f"\nPARAMETERS:")
    for key in params:
        print(f'\t{key:>20}: {params[key]}')
    print('\n')

    # Random seed
    seed_everything(111)

    if dataset == 'IEMOCAP':
        # This is the 4-class, improvised data set
        #emot_map = {'ang':0,'sad':1,'hap':2,'neu':3}
        #include_scripted = False
        
        # Some publication works combined 'happy' and 'excited' into one 'happy' class,
        #   enable below for the 5531 dataset
        emot_map = {'ang':0,'sad':1,'hap':2, 'exc':2, 'neu':3}
        include_scripted = True 
        
        #Initialize database
        database = SER_DATABASES[dataset](dataset_dir, emot_map=emot_map, 
                                        include_scripted = include_scripted)

    #Get file paths and label in database
    speaker_files = database.get_files()

    #Extract features
    features_data = extract_features(speaker_files, features, params)
    print(type(features_data["3M"]))
    
    #Save features
    if config['save_dir'] is not None:
        
        with open(out_filename, "wb") as fout:
                pickle.dump(features_data, fout)

    #Print classes statistic
        
    print(f'\nSEGMENT CLASS DISTRIBUTION PER SPEAKER:\n')
    classes = database.get_classes()
    n_speaker=len(features_data)
    n_class=len(classes)
    class_dist= np.zeros((n_speaker,n_class),dtype=np.int64)
    speakers=[]
    data_shape=[]
    for i,speaker in enumerate(features_data.keys()):
        #print(f'\tSpeaker {speaker:>2}: {sorted(Counter(features_data[speaker][2]).items())}')
        cnt = sorted(Counter(features_data[speaker]["seg_label"]).items())
        
        for item in cnt:
            #print(item)
            class_dist[i][item[0]]=item[1]
        #print(class_dist)
        speakers.append(speaker)
        if mixnoise == True:
            data_shape.append(str(features_data[speaker]["seg_spec"][0].shape))
        else:
            data_shape.append(str(features_data[speaker]["seg_spec"].shape))
    class_dist = np.vstack(class_dist)
    #print(class_dist)
    df = {"speakerID": speakers,
          "shape (N,C,F,T)": data_shape}
    
    for c in range(class_dist.shape[1]):
        df[classes[c]] = class_dist[:,c]
    
    class_dist_f = pd.DataFrame(df)
    class_dist_f = class_dist_f.to_string(index=False) 
    print(class_dist_f)
     
    print('\n')
    print('*'*50)
    print('\n')



# seeding function for reproducibility
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, default="configs\\feature_config.yaml", 
                        help="Path to feature config file")
    
    args = parser.parse_args()

    config = None

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"配置文件未找到到：{args.config}")
    
    main(config)