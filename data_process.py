import numpy as np
import pandas as pd

X = pd.read_csv('projectiles.csv',header=None)
data=X.values

time=data[:,0]
print time
data=data[:,1:]
rows=0
process_data=np.zeros([1425,6])

for i in range(0,len(data)-2):
    if time[i+2]==0 or time[i+1]==0:continue
    for j in range(0,3):
        process_data[rows][j*2]=data[i+j][0]
        process_data[rows][j*2+1]=data[i+j][1]
    rows=rows+1

process_data=process_data[0:rows,:]

process_data_csv=pd.DataFrame(process_data,columns=['x1','y1','x2','y2','rx','ry'])
process_data_csv.to_csv('data.csv',index=False)
