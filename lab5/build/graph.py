import matplotlib.pyplot as plt
import numpy as np

def read(filename):
    err=[]
    n=[]
    with open(f"../TextFiles/{filename}.txt", 'r') as file:
        for line in file:
            new_line = line.split(" ")
            err.append(int(new_line[0]))
            n.append(float(new_line[1]))
    return err,n

def read2(filename):
    err=[]
    n=[]
    with open(f"../TextFiles/{filename}.txt", 'r') as file:
        for line in file:
            new_line = line.split(" ")
            err.append(float(new_line[0]))
            n.append(float(new_line[1]))
    return err,n

err,n=read("buffon1")



plt.loglog(err,n)

plt.show()

plt.figure()
frac,err=read2("buffon2")
plt.loglog(frac,err)
plt.show()