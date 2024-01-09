import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

df=pd.read_csv('E:/ML PROGRAMS/18AI72/3) LOCAL WEIGHT REGRESSION/tips.csv')
features=np.array(df.total_bill)
label=np.array(df.tip)

def kernel(data,point,xmat,k):
    m,n=np.shape(xmat)
    diff=np.mat(np.eye((m)))
    for i in range(m):
        d=point-data[i]
        diff[i,i]=np.exp(d*d.T/(-2.0*k**2))
    return diff

def we(data,point,xmat,ymat,k):
    wei=kernel(data,point,xmat,k)
    return (data.T*(wei*data)).I*(data.T*(wei*ymat.T))


def lwr(xmat,ymat,k):
    m,n=np.shape(xmat)
    ypred=np.zeros(m)
    for i in range(m):
        ypred[i]=xmat[i]*we(xmat,xmat[i],xmat,ymat,k)
    return ypred

m=features.shape[0]
mtip=np.mat(label)
data=np.hstack((np.ones((m,1)),np.mat(features).T))

ypred=lwr(data,mtip,0.5)
indices=data[:,1].argsort(0)
xsort=data[indices][:,0]

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(features,label,color='blue')
ax.plot(xsort[:,1],ypred[indices],color='red',linewidth=3)
plt.xlabel('w')
plt.ylabel('x')
plt.show()
