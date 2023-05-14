#### 这个文件是用来自动对ply逐一训练并生成skeleton的，需要把它放置在Point2Skeleton的同级路径内，另外，同级还应有vertex_count.pkl文件
##### 记得data/pointclouds/里面要有pulmonaryArterys_ply这个文件夹！

import pickle
import subprocess
import time

#事前准备：建立存放结果的文件夹
result = subprocess.run(f"mkdir ./all_results",shell=True,stdout=subprocess.PIPE)
print(result.stdout.decode('utf-8'))

# 获取字典
with open("vertex_count.pkl","rb") as f:
    ply_verts_dictXXXXX = pickle.load(f)
print(f"dict getted!\n len(dict):{len(ply_verts_dictXXXXX)}")

#截短dict，用作测试
def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

ply_verts_dict = dict_slice(ply_verts_dictXXXXX,0,1)

# ply_verts_dict = ply_verts_dictXXXXX        #正式使用的时候就要带上这一句了。

#清空results，weights和training-weights文件夹
result = subprocess.run("rm -rf ./Point2Skeleton/results/* ./Point2Skeleton/weights/* ./Point2Skeleton/training-weights/*",shell=True,stdout=subprocess.PIPE)
print(result.stdout.decode('utf-8'))

print("results/,weights/ and training-weights/ : CLEARED.\n")



#循环开始：
for k,v in ply_verts_dict.items():

    print(f"[{time.asctime()}]  now training {k} .........")
    #更改"Point2Skeleton/data/data-split/all-train.txt"
    with open("./Point2Skeleton/data/data-split/all-train.txt","w") as f:
        f.write(f"pulmonaryArterys_ply/{k[:-4]}\n"*32)

    #执行train.py
    result = subprocess.run(f"cd ./Point2Skeleton/src/;python train.py --pc_list_file ../data/data-split/all-train.txt --data_root ../data/pointclouds/ --point_num {v} --skelpoint_num 100 --gpu 0",shell=True,stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

    #移动training-weights文件夹里的文件到weights文件夹
    result = subprocess.run("mv ./Point2Skeleton/training-weights/* ./Point2Skeleton/weights/",shell=True,stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

    #更改"Point2Skeleton/data/data-split/all-test.txt"
    with open("./Point2Skeleton/data/data-split/all-test.txt","w") as f:
        f.write(f"pulmonaryArterys_ply/{k[:-4]}")

    #执行test.py
    result = subprocess.run(f"cd ./Point2Skeleton/src/;python test.py --pc_list_file ../data/data-split/all-test.txt --data_root ../data/pointclouds/ --point_num {v} --skelpoint_num 100 --gpu 0 --load_skelnet_path ../weights/weights-skelpoint.pth --load_gae_path ../weights/weights-gae.pth --save_result_path ../results/",shell=True,stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

    #从Point2Skeleton/data/pointclouds/pulmonaryArterys_ply里找到原文件，复制其并放入result文件夹。
    result = subprocess.run(f"cp ./Point2Skeleton/data/pointclouds/pulmonaryArterys_ply/{k} ./Point2Skeleton/results/",shell=True,stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
    
    #把weights文件夹里的内容移入result文件夹
    result = subprocess.run(f"mv ./Point2Skeleton/weights/* ./Point2Skeleton/results/",shell=True,stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
    #再将result文件夹的内容全部移动到一个叫all_results/s***_pulmonaryArterys的新文件夹里面
    result = subprocess.run(f"mkdir ./all_results/{k[:-4]}/;mv ./Point2Skeleton/results/* ./all_results/{k[:-4]}/",shell=True,stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))


print(f"[{time.asctime()}]    CONGRATULATIONS!!!\nALL PLY FILES WORKS ARE FINISHED.")