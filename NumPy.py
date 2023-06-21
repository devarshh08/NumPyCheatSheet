"""
Refer to Numpy docs for more info:
https://numpy.org/doc/stable/reference/arrays.html
"""

import numpy as np
import random

sep = "________________________________________________________________"

#making an array
a = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
print(a)
print(sep)

#shape of array
print(a.shape)
print(sep)

# Get Dimension
print(a.ndim)
print(sep)

# Get Type
print(a.dtype)
print(sep)

# Get Size
print(a.itemsize)
print(sep)

# Get total size
print(a.nbytes)
print(sep)

# Get number of elements
print(a.size)
print(sep)

#get a specific element[r, c]
print(a[1, 5])
print(sep)

#get a specific row
print(a[0, :])
#this will get row of index 0 i.e first row
print(sep)

#get a specific column
#lets say we want third column
print(a[:, 2])
print(sep)

#getting a little more fancy [startindex:endindex:stepsize]
print(a[0, 1:6:2])
print(a[0, 1:-1:2])
#code to print 2, 4, 6
print(sep)

#changing an element
a[0,0] = 0
print(a)
print(sep)

#replacing an entire column
a[:, 0] = [5]
print(a)
a[:, 0] = [1, 8]
print(a)
print(sep)

#replacing an entire row
a[0, :] = [69]
print(a)
a[0, :] = [1, 2, 3, 4, 5, 6, 7]
print(a)
print(sep)

#3-d example
b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(b)
print(sep)

#printing second element of second row of first array
print(b[0, 1, 1])
print(sep)

#you can play around with dimensions

#this will give second row of all arrays
print(b[:, 1, :])
print(sep)

#this will give first row of first array
print(b[0, 0, :])
print(sep)

#this will give first column of first array
print(b[0, :, 0])
print(sep)

#replacing
b[:, 1, :] = [[9, 9], [8, 8]]
print(b)
print(sep)

#initializing different types of arrays

#all 0s matrix
print(np.zeros(5))
print(sep)

#making 4d array
print(np.zeros((2, 3, 3, 2)))
print(sep)

#all 1s matrix
print(np.ones(4))
print(sep)

#you can even specify data types
print(np.ones((1, 2, 3, 4), dtype = "int32"))
print(sep)

#any other number => numpy.full(shape, values)
print(np.full((3, 3), 69, dtype = 'float32'))
print(sep)

#any other number => full_like method
#(allows us to reuse shape of another array used before) numpy.full_like(array like we want it to be shaped, values)
print(np.full_like(a, 5))
print(sep)

#initialize matrix of random numbers
print(np.random.rand(5, 5, 5))
print(sep)
#this will give array of random nos b/w 0 and 1

#random integer values matrix
print(np.random.randint(1,100 , size = (5, 5)))
print(sep)

#identity matrix
print(np.identity(5))
print(sep)

#repeating an array three times
#it will repeat inner part on axis
r1 = np.repeat(a, 2, axis = 0)
print(r1)
print(sep)

#Question : initializing an array
output = np.ones((5, 5))
print("We started with:")
print(output)

z = np.zeros((3, 3))
print("We have new 3x3 matrix z")
print(z)

z[1, 1] = 9
print("We changed middle value of z from 0 to 9")
print(z)

output[1:4, 1:4] = z
print("Now we change middle row and column of output to z")
print("We use code: output[1:4, 1:4] = z")
print(output)
print(sep)

#be careful when copying arrays
a = np.array([1, 2, 3])
#we want to make b a copy of a
b = a
print(b)

#we change element of b
b[0] = 100
print(b)

print(a)
print(sep)
#but even a's element has changed, it has 100 instead of 1
#this is because when we did b = a, we set variable name b to a, we are pointing at the same thing a is pointing
#We didnt tell numpy to make a copy of the contents of a.

#we can use copy function for copying
a = np.array([1, 2, 3, 4])
b = a.copy()
print(b)

#we change element of b
b[0] = 100
print(b)

print(a)
print(sep)

#math capabilities of numpy

#elementwise addition and subtraction:
print(a+2)
print(sep)

print(a-2)
print(sep)

print(a*2)
print(sep)

print(a/2)
print(sep)

b = np.array([1, 0, 1, 0])
print(a + b)
print(sep)

print(a**2)
print(sep)

#taking the sin
print(np.sin(a))
print(sep)

#Linear Algebra
a = np.ones((2,3))
b = np.full((3, 2), 2)

print(np.matmul(a, b))
print(sep)

#find the determinant
c = np.identity(3) #det of identity matrix is 1
print(np.linalg.det(c))
print(sep)

"""
Reference docs: 
https://docs.scipy.org/doc/numpy/reference/routines.linalg.html

Determinant
Trace
Singular Vector Decomposition
Eigenvalues
Matrix Norm
Inverse
Etc... """

#statistics with numpy

stats = 