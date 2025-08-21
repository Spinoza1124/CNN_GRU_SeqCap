import pickle
import numpy as np
import os

def create_test_iemocap_data():
    """
    创建模拟的IEMOCAP数据用于测试
    """
    # 创建data目录
    os.makedirs('data', exist_ok=True)
    
    # 模拟IEMOCAP数据结构
    # 每个Session包含多个样本，每个样本有特征和标签
    data = {}
    
    # 创建5个Session的数据
    for session_id in range(1, 6):
        session_key = f'Session{session_id}'
        data[session_key] = {}
        
        # 每个Session创建两个说话人的数据
        for speaker_id in ['M', 'F']:  # Male, Female
            speaker_key = f'{session_key}_{speaker_id}'
            
            # 每个说话人创建20个样本
            num_samples = 20
            features = []
            labels = []
            
            for i in range(num_samples):
                # 创建模拟特征 (假设是39维的MFCC特征，100帧)
                feature = np.random.randn(100, 39).astype(np.float32)
                features.append(feature)
                
                # 创建模拟标签 (4个情感类别: 0-angry, 1-happy, 2-sad, 3-neutral)
                label = np.random.randint(0, 4)
                labels.append(label)
            
            data[session_key][speaker_key] = {
                'features': features,
                'labels': labels
            }
    
    # 保存数据
    with open('data/IEMOCAP_features.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print("✅ 测试数据创建完成!")
    print(f"数据结构:")
    for session in data.keys():
        print(f"  {session}: {list(data[session].keys())}")
        for speaker in data[session].keys():
            num_samples = len(data[session][speaker]['features'])
            print(f"    {speaker}: {num_samples} 个样本")

if __name__ == '__main__':
    create_test_iemocap_data()