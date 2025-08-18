import os
from collections import defaultdict, OrderedDict
from pathlib import Path
from pkl_viewer import data

IEMOCAP_EMO_CODES = {'neu': ['neu', 'neutral'],
                     'hap': ['hap', 'happy', 'happiness'],
                     'sad': ['sad', 'sadness'],
                     'ang': ['ang', 'angry', 'anger'],
                     'sur': ['sur', 'surprise', 'surprised'],
                     'fea': ['fea', 'fear'],
                     'dis': ['dis', 'disgust', 'disgusted'],
                     'fru': ['fru', 'frustrated', 'frustration'],
                     'exc': ['exc', 'excited', 'excitement'],
                     'oth': ['oth', 'other', 'others']}

class IEMOCAP_database():
    """
    IEMOCAP database contains data from 10 actors, 5 male and 5 female,
    during their affective dyadic interaction. The database consists of
    5 sessions, containing both improvised and scripted sessions. Each session
    consists of 2 unique speakers: 1 male and 1 female.

    For each session, the utterances are organized into conversation folders
        eg. Ses01F_impro01/                     -> improvised conversation 01 of Session 01
                |-- Ses01F_impro01_F000.wav     -> speaker F, utterance 000
                |-- Ses01F_impro01_M000.wav     -> speaker M, utterance 000
                |-- ...

    This function extract utterance filenames and labels for improvised sessions,
    organized into dictionary of {'speakerID':[(conversation_wavs,lab),(wavs,lab),...,(wavs,lab)]}

        > speakerID eg. 1M: Session 1, Male speaker
    
    Database Reference:
        (2008). IEMOCAP: Interactive emotional dyadic motion capture database. 
        Language Resources and Evaluation.
    
    Authors:
        Busso, Carlos
        Bulut, Murtaza
        Lee, Chi-Chun
        Kazemzadeh, Abe
        Mower, Emily
        Kim, Samuel
        Chang, Jeannette
        Lee, Sungbok
        Narayanan, Shrikanth
    
    Download request link:
        https://sail.usc.edu/iemocap/iemocap_release.htm
    """
    def __init__(self, database_dir, emot_map = {'ang': 0, 'sad': 1, 'hap': 2, 'neu': 3}, include_scripted=False):
        # Path
        self.database_dir = Path(database_dir)
        
        # Emotion to label mapping for features
        self.emot_map = emot_map

        # IEMOCAP Session name
        self.sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

        # IEMOCAP available emotion classes
        self.all_emo_classes = IEMOCAP_EMO_CODES.keys()

        # to include scripted session
        self.include_scripted = include_scripted
    
    def get_speaker_id(self, session, gender):
        return session[-1]+gender
    
    def get_classes(self):

        classes = {}
        for key, value in self.emot_map.items():
            if value in classes.keys():
                classes[value] += '+'+key
            else:
                classes[value] = key
        return classes
    
    def get_files(self):
        """
        Get all the required .wav file paths for each speaker and organized into
        dictionary:
            keys   -> speaker ID
            values -> list of (.wav filepath, label) tuples for corresponding speaker
        """
        emotions = self.emot_map.keys()
        dataset_dir = self.database_dir
        all_speaker_files = defaultdict()
        total_num_files = 0
        for session_name in dataset_dir.iterdir():
            if session_name not in self.sessions:
                continue
            
            wav_dir = Path(dataset_dir) / session_name / "sentences/wav"
            label_dir = Path(dataset_dir) / session_name / "dialog/EmoEvaluation"
            


    
