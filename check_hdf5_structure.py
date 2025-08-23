#!/usr/bin/env python3
"""Check HDF5 file structure"""

import h5py

def check_hdf5_structure(file_path):
    """Check the structure of an HDF5 file"""
    print(f"📁 检查HDF5文件结构: {file_path}")
    print("=" * 60)
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"✅ 文件打开成功")
            print(f"📊 文件大小: {f.id.get_filesize() / (1024*1024):.2f} MB")
            print(f"🔑 顶级键: {list(f.keys())}")
            print()
            
            print("📂 详细结构:")
            for key in f.keys():
                item = f[key]
                print(f"  {key}: {type(item)}")
                
                if hasattr(item, 'shape'):
                    print(f"    Shape: {item.shape}")
                elif hasattr(item, 'keys'):
                    print(f"    Subkeys: {list(item.keys())}")
                    
                    # Check subkeys recursively
                    for subkey in item.keys():
                        subitem = item[subkey]
                        if hasattr(subitem, 'shape'):
                            print(f"      {subkey}: {subitem.shape}")
                        else:
                            print(f"      {subkey}: {type(subitem)}")
                
                print()
                
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    check_hdf5_structure("data/sionna/sionna_5g_simulation.h5")
