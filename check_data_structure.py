import pickle
import numpy as np

def check_data_structure(data_path):
    """检查数据文件的结构"""
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"数据类型: {type(data)}")
        print(f"顶层键数量: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        if isinstance(data, dict):
            print("\n顶层键:")
            for i, key in enumerate(sorted(data.keys())):
                print(f"  {i+1}. {key} (类型: {type(data[key])})")
                if i >= 10:  # 只显示前10个键
                    print(f"  ... 还有 {len(data) - 10} 个键")
                    break
            
            # 检查第一个键的结构
            first_key = list(data.keys())[0]
            first_value = data[first_key]
            print(f"\n第一个键 '{first_key}' 的值:")
            print(f"  类型: {type(first_value)}")
            
            if isinstance(first_value, dict):
                print(f"  子键数量: {len(first_value)}")
                print("  子键:")
                for subkey in list(first_value.keys())[:5]:  # 只显示前5个子键
                    subvalue = first_value[subkey]
                    print(f"    {subkey}: {type(subvalue)}")
                    
                    if isinstance(subvalue, np.ndarray):
                        print(f"      形状: {subvalue.shape}, 数据类型: {subvalue.dtype}")
                    elif isinstance(subvalue, list):
                        print(f"      长度: {len(subvalue)}")
                        if len(subvalue) > 0:
                            print(f"      第一个元素类型: {type(subvalue[0])}")
                            if isinstance(subvalue[0], np.ndarray):
                                print(f"      第一个元素形状: {subvalue[0].shape}")
            elif isinstance(first_value, np.ndarray):
                print(f"  形状: {first_value.shape}, 数据类型: {first_value.dtype}")
            elif isinstance(first_value, list):
                print(f"  长度: {len(first_value)}")
                if len(first_value) > 0:
                    print(f"  第一个元素类型: {type(first_value[0])}")
        
        elif isinstance(data, list):
            print(f"列表长度: {len(data)}")
            if len(data) > 0:
                print(f"第一个元素类型: {type(data[0])}")
        
    except Exception as e:
        print(f"读取文件时出错: {e}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = 'data/IEMOCAP_multi.pkl'
    
    print(f"检查数据文件: {data_path}")
    check_data_structure(data_path)