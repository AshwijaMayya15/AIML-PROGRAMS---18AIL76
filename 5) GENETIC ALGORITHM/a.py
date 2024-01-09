import numpy as np 
import matplotlib.pyplot as plt 

def gen_pop(size,items):
    return np.random.randint(2,size=(size,items))

def fitness(pop,w,v,c):
    penalty=np.maximum(0,np.dot(pop,v)-c)
    return np.dot(pop,w)-penalty.max()

def ts(pop,fitness,s=5):
    return pop[np.argmax(np.random.choice(fitness,size=s))]

def crossover(p1,p2):
    point=np.random.randint(len(p1))
    return np.concatenate([p1[:point],p2[point:]])

def mutate(child,mr):
    return np.logical_xor(child,np.random.rand(len(child))<mr)

def ga(w,v,c,size=50,generation=100,mr=0.1):
    pop,hist=gen_pop(size,len(w)),[]
    for _ in range(generation):
        fit=fitness(pop,w,v,c)
        best=pop[np.argmax(fit)]
        hist.append(np.max(fit))
        pop=np.array([crossover(ts(pop,fit),ts(pop,fit)) for _ in range(size//2)])

        if size%2 !=0:
            pop=np.vstack([pop,np.zeros_like(pop[0])])
        if len(pop)<size:
            ap=gen_pop(1,len(w))
            pop=np.vstack([pop,ap])

        pop[np.argmin(fit)]=best
        pop=np.array([mutate(ind,mr) for ind in pop])

    plt.plot(hist)
    plt.title('s')
    plt.xlabel('w')
    plt.ylabel('w')
    plt.show()

    return best,np.max(hist)

w=[2,3,4,5,6,7]
v=[11,12,13,14,15,16]
c=20
bs,op=ga(w,v,c)
print(f"Best solution: {bs}")
print(f"Optimal solution: {op}")

