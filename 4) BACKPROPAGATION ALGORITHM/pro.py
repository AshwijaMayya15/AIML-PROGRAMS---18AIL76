import numpy as np 

X=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
X=X/np.amax(X)
y=y/100

epoch=5
lr=0.1
inputneuron=2
hiddenneuron=3
outputneuron=1

wh=np.random.uniform(size=(inputneuron,hiddenneuron))
bh=np.random.uniform(size=(1,hiddenneuron))
wout=np.random.uniform(size=(hiddenneuron,outputneuron))
bout=np.random.uniform(size=(1,outputneuron))

def sigmoid(x):
    return 1/1+np.exp(-x)

def deri(x):
    return x*(1-x)

for i in range(epoch):

    hinp1=np.dot(X,wh)
    hinp=hinp1+bh
    hlayer=sigmoid(hinp)
    outinp1=np.dot(hlayer,wout)
    outinp=outinp1+bout
    output=sigmoid(outinp)

    EO=y-output
    outgrad=deri(output)
    d_out=EO*outgrad
    EH=d_out.dot(wout.T)

    hiddengrad=deri(hlayer)
    h_out=EH*hiddengrad

    wout+=hlayer.T.dot(d_out)*lr
    wh+=X.T.dot(h_out)*lr
    print("EPOCH",i+1,"STARTS")
    print("INPUT:\n", str(X))
    print("ACTUAL OUTPUT:\n",str(y))
    print("PREDICTED OUTPUT:\n",output)

print("INPUT:\n", str(X))
print("ACTUAL OUTPUT:\n",str(y))
print("PREDICTED OUTPUT:\n",output)




