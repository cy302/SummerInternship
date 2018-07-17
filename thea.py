import csv
import os
import numpy as np
import pickle

os.chdir("/Users/cy302/Downloads")

results = []
with open("snapshot.txt") as inputfile:
    for row in csv.reader(inputfile):
        results.append(row)

results = results[2:-1]

for i in range(len(results)):
    temp = results[i][0].split()
    if len(temp) == 5:
        temp2 = temp[1]
        a = temp2[:-5]
        b = temp2[-5:]
        temp.pop(1)
        temp.insert(1, a)
        temp.insert(2, b)
    temp[-4:] = [float(j) for j in temp[-4:]]
    temp[0] = int(temp[0][:-4])
    results[i] = temp

results = np.array(results)
atom_list_unique = list(range(1, 201))
atom_list = [int(i[0]) for i in results]
atom_list = np.array(atom_list)
center = np.zeros([200, 3])
coord = results[:, [3, 4, 5]]
coord = np.array([[float(i) for i in j] for j in coord])
for i in range(len(atom_list_unique)):
    ind = np.where(atom_list == atom_list_unique[i])
    center[i, :] = np.mean(coord[ind], axis=0)

center = np.ndarray.tolist(center)

count = 1
for i in center:
    i.insert(0, str(count)+"ITIC")
    count += 1

center = np.array(center)
np.savetxt("center.txt", center)

with open("center.txt", "wb") as filehandle:
    for c in center:
        filehandle.write("%s\n" % c)


