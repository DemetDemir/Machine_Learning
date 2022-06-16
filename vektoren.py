import numpy as np
import math

#Addition
vector1 = [4,2,3]
vector2 = [3,5,3]
result = []

def addition(vector1, vector2):
  for i in range(len(vector1)):
    result.append(vector1[i] + vector2[i])
  #print(result)

addition(vector1, vector2)

"""
A = np.array([[4,3,1], [2,-1,3],[3,8,1]])
#print(A)
A[0][0]



result = [] # final result
for i in range(len(A)):

    row = [] # the new row in new matrix
    for j in range(len(vector1)):
        
        product = 0 # the new element in the new row
        for v in range(len(A[i])):
            product += A[i][v] * vector1[v][j]
        row.append(product) # append sum of product into the new row
        
    result.append(row) # append the new row into the final result

"""
#print(result)


vecotr1 = np.array(vector1)
vector2 = np.array(vector2)

result = vector1+vector2
print(result)