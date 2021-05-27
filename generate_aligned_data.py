# -*- coding:utf-8 -*-
import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import argparse

data_path = '/lustre/home/acct-hpc/hpcwlz/PAPI/PAPI_multiplex_test/Benchmark/hpcbenchmark-collection/Rodinia'
dic_event = {0:'brins:all',1:'brins:cond',2:'brmis:all',3:'brmis:cond',4:'dtlm:m',5:'dtlm:s',6:'l1lh',7:'l1lm',8:'l2lh',9:'l2lm',10:'ldram',11:'ich',12:'icm',13:'uistall',14:'urstall',15:'inst'}
dic_workload = {0:'heartwall',1:'leukocyte',2:'streamcluster',3:'kmeans',4:'lud',5:'bfs',6:'knn',7:'b+tree',8:'srad',9:'backprop',10:'hotspot',11:'lavaMD',12:'nw',13:'pathfinder',14:'hotspot3D'}


parser = argparse.ArgumentParser()
parser.add_argument('--workload',type=int)
parser.add_argument('--event_id',type=int)
#parser.add_argument('--event_len',type=int,default=200)
args = parser.parse_args()

workloads = [i for i in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,i))]
print(workloads)
#del workloads[6]

#choose WORKLOAD here
i = workloads[args.workload]
#print(i)

raw_path = os.path.join(data_path,i,'raws4')
# raw_path = os.path.join(raw_path,'raws')
mpx_paths = [os.path.join(raw_path,i) for i in os.listdir(raw_path) if 'mpx' in i]
ocoe_paths = [os.path.join(raw_path,i) for i in os.listdir(raw_path) if 'ocoe' in i]
mpx_file = [os.path.join(mpx_path,'mpx.bin') for mpx_path in mpx_paths]
ocoe_file1 = [os.path.join(ocoe_path,'ocoe-1.bin') for ocoe_path in ocoe_paths]
ocoe_file2 = [os.path.join(ocoe_path,'ocoe-2.bin') for ocoe_path in ocoe_paths]
ocoe_file3 = [os.path.join(ocoe_path,'ocoe-3.bin') for ocoe_path in ocoe_paths]


def unpack_bin_file(filename,num_events):
    f = open(filename, "rb")
    size = os.path.getsize(filename)
    nsamples = int(size / ( 8 * num_events * 3))
    
    te = np.zeros((nsamples, num_events))
    tr = np.zeros((nsamples, num_events))
    count = np.zeros((nsamples, num_events))
    read_buf = []
    for i in range(int(size/8)):
        data = f.read(8)
        datall = struct.unpack("q", data)[0]
        read_buf.append(datall)

    for t in range(nsamples):
        for ne in range(num_events):
            count[t][ne] = read_buf[t*num_events*3+ne*3+0]
            te[t][ne] = read_buf[t*num_events*3+ne*3+1]
            tr[t][ne] = read_buf[t*num_events*3+ne*3+2]
            
    return count, te, tr


def concat_ocoe(filename1,filename2,filename3):
    count1,te1,tr1=unpack_bin_file(filename1,4)
    count2,te2,tr2=unpack_bin_file(filename2,4)
    count3,te3,tr3=unpack_bin_file(filename3,5)
    Min = min(count1.shape[0],count2.shape[0],count3.shape[0])
    count1 = count1[0:Min,]
    count2 = count2[0:Min,]
    count3 = count3[0:Min,]
    te1=te1[0:Min,]
    te2=te2[0:Min,]
    te3=te3[0:Min,]
    tr1,tr2=tr1[0:Min,],tr2[0:Min,]
    #tr2=tr2[0:Min,]
    tr3=tr3[0:Min,]
    count = np.concatenate((count1,count2,count3),axis=1)
    tr = np.concatenate((tr1,tr2,tr3),axis=1)
    te = np.concatenate((te1,te2,te3),axis=1)
    return count,te,tr

'''
count_mpx,te_mpx,tr_mpx = unpack_bin_file(mpx_file[0],12)
count_mpx_new = np.diff(count_mpx,axis=0)
te_mpx_new = np.diff(te_mpx,axis=0)
tr_mpx_new = np.diff(tr_mpx,axis=0)
print(count_mpx_new.T[args.event_id])
print(te_mpx_new[args.event_id])
print(tr_mpx_new[args.event_id])
'''


#select EVENT & SAMPLE here
sample_len=2
mpx=np.zeros((sample_len,800))
ocoe=np.zeros((sample_len,800))
for i in range(0,2):    
    sum_mpx=0
    count_mpx,te_mpx,tr_mpx=unpack_bin_file(mpx_file[i],13)
    count_ocoe,te_ocoe,tr_ocoe=concat_ocoe(ocoe_file1[i],ocoe_file2[i],ocoe_file3[i])
    Min = min(count_mpx.shape[0],count_ocoe.shape[0]) #将行数对齐
    #Min = count_mpx.shape[0]
    count_ocoe = count_ocoe[0:Min]
    count_mpx = count_mpx[0:Min]
    tr_mpx = tr_mpx[0:Min,]
    te_mpx=te_mpx[0:Min,]
    #接下来做一阶差分
    count_mpx_new = np.diff(count_mpx,axis=0)
    count_ocoe_new = np.diff(count_ocoe,axis=0)
    te_mpx_new = np.diff(te_mpx,axis=0)
    tr_mpx_new = np.diff(tr_mpx,axis=0)+1
    t = te_mpx_new/tr_mpx_new    
    print(t.shape,count_mpx_new.shape)

    data_mpx=(count_mpx_new*te_mpx_new/tr_mpx_new).T[args.event_id]
    #data_mpx = (count_mpx_new*te_mpx_new/tr_mpx_new).T
    data_ocoe=count_ocoe_new.T[args.event_id]
    #data_ocoe = count_ocoe_new[0:Min-3].T
    #print(data_mpx.shape)

    
    for j in range(0,Min-3):
        #sum_mpx+=data_mpx[j-1]
        mpx[i][j] = data_mpx[j]
        ocoe[i][j] = data_ocoe[j]
    #ocoe[i][0] = count_ocoe[:,args.event_id][Min-1]
    #mpx[i][0] = sum_mpx
    #data_mpx = mpx[0:Min-2].T
    #print(tr_mpx_new/te_mpx_new)
    

'''
mpx = np.zeros((800,13))
print(count_mpx_new.shape,t.shape)
for i in range(0,13):
    for j in range(0,Min-3):
        mpx[j][i] = 0.5*t[j][i]*(count_mpx_new[j][i]+count_mpx_new[j+1][i])

data_mpx = mpx[0:Min-3].T
'''

#print(tr_mpx_new[0,])
#print(te_mpx_new[0,])
print(te_mpx_new[:,0].shape)

all_ocoe,all_mpx=np.log10(data_ocoe+1),np.log10(data_mpx[1:]+1)
print(data_mpx.shape,data_ocoe.shape)
#all_ocoe,all_mpx=np.array(all_ocoe[:,0:Min]),np.array(all_mpx[:,0:Min])
np.save('./mpx_data.npy',mpx[:,0:Min-3]+1)
np.save('./ocoe_data.npy',ocoe[:,0:Min-3]+1)
#print('Processing ',dic_event[args.event_id])

