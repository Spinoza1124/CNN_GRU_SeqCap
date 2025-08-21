from pathlib import Path
from collections import defaultdict, OrderedDict

IEMOCAP_EMO_CODES = {
    'neu':['neu', 'neutral'],
    'hap':['hap', 'happy', 'happiness'],
    'sad':['sad', 'sadness'],
    'ang':['ang', 'angry', 'anger'],
    'sur':['sur', 'surprise', 'surprised'],
    'fea':['fea', 'fear'],
    'dis':['dis', 'disgust', 'disgusted'],
    'fru':['fru', 'frustrated', 'frustration'],
    'exc':['exc', 'excited', 'excitement'],
    'oth':['oth', 'other', 'others']
}

class IEMOCAP_database():
    def __init__(self, database_dir, emo_map={'ang':0, 'sad':1, 'hap':2, 'neu':3}, include_scripted=False):
        self.database_dir = database_dir
        self.emo_map = emo_map
        self.include_scripted = include_scripted

        self.sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        self.all_emo_classes = IEMOCAP_EMO_CODES.keys()
    
    def get_speaker_id(self, session, gender):
        return session[-1]+gender
    
    def get_classes(self):
        classes = {}
        for key, value in self.emo_map.items():
            if value in classes.keys():
                classes[value] += '+'+key
            else:
                classes[value] = key

    def get_files(self):

        emotions = self.emo_map.keys()

        dataset_dir = Path(self.database_dir)

        all_speaker_files = defaultdict()

        total_num_files = 0

        for session_name in dataset_dir.iterdir():
            if session_name not in self.sessions:
                continue
            
            wav_dir = dataset_dir / session_name / 'sentences/wav'
            label_dir = dataset_dir / session_name / 'dialog/EmoEvaluation'

            M_wav, F_wav = list(), list()