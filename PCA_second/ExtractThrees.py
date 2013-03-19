import numpy as np

prefile=open("./optdigits.tra","r")
alllines=prefile.readlines()
prefile.close()

cnt = 0
postfile=open("./threes.txt","a")
for line in alllines:
    if int(line[-2])==3:
        postfile.write(line)
        cnt += 1
postfile.close()
print"the number of 3:"
print cnt




