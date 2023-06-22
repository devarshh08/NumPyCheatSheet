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

stats = np.array(([1, 2, 3], [4, 5, 6]))

print(np.min(stats))
print(sep)

#similar like with min, all applies for max
print(np.max(stats))
print(sep)

#this axis = 0 will print full row
print(np.max(stats, axis = 0))
print(sep)

#axis = 1 will print full column
print(np.max(stats, axis = 1))
print(sep)

#sum
#this will add while matrix
print(np.sum(stats))
print(sep)

#this will add columns
print(np.sum(stats, axis = 0))
print(sep)

#this will add rows
print(np.sum(stats, axis = 1))
print(sep)

"""
1-D array has 1 axis that is axis 0, 2d array has 2 axes which are axis 0 and axis 1, 3d array has 3 axes, which are axis 0, axis 1 and axis 2 and so on...
more about axis
https://www.geeksforgeeks.org/how-to-set-axis-for-rows-and-columns-in-numpy/
"""

#Reorganizing arrays
before = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(before)
print(sep)

after = before.reshape((2, 2, 2))
print(after)
print(sep)

#vertically stacking vectors
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])

print(np.vstack([v1, v2, v1, v2]))
print(sep)

#horizontally stacking vectors
h1 = np.ones((2, 4))
h2 = np.zeros((2, 2))

print(np.hstack((h1, h2)))
print(sep)

#load data from file
filedata = np.genfromtxt("E:\\CODING\\GitHub Repos\\NumPyCheatSheet\\data.txt", delimiter = ",")
print(filedata)
print(sep)

#to convert type
print(filedata.astype('int32',))
print(sep)

#Boolean masking and Advanced indexing
print(filedata > 50)
#we can use different types of combinations
print(sep)

#we can get values as well
print(filedata[filedata > 50])
print(sep)

#You can index with a list in numpy
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a[[1, 2, 8]])
print(sep)

print(np.any(filedata > 50, axis = 0))
#this checks column wise if any value is greater than 50, it tells true even if single value is greater than 50
print(sep)

print(np.all(filedata > 50, axis = 0))
#this also checks columnwise, but only returns true if all values of column are greater than 50
print(sep)

#all values >50 but <100
print((filedata > 50) & (filedata < 100))
print(sep)

#broadcasting in NumPy arrays

# var1 = np.array([1, 2, 3])
# var2 = np.array([1, 2, 3, 4])
# print(var1+var2)
#since size of both arrays is different, so upon adding we get broadcasting error

#for broadcasting, the dimension should be equal

var1 = np.array([1, 2, 3])
var2 = np.array([[1], [2], [3]])
print(var1+var2)
print(sep)

#iterating arrays

a1 = np.array([1, 2, 3, 4, 5])
for i in a1:
    print(i)

print(sep)

#iterating 2 dimensional array
a2 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
for i in a2:
    for j in i:
        print(j)

print(sep)

#for 3d we use 3 nested loops and so on...
#we can also use function nditer

for i in np.nditer(a2):
    print(i)

print(sep)

#to get index valus as well as data
for i, d in np.ndenumerate(a2):
    print(i, d)

print(sep)

#joining arrays
v11 = np.array([[1, 2], [3, 4]])
v22 = np.array([[5, 6], [7, 8]])
arr_new = np.concatenate((v11, v22), axis = 0)
print("Axis = 0")
print(arr_new)
print("VS")
print("Axis = 1")
arr_new = np.concatenate((v11, v22), axis = 1)
print(arr_new)
print(sep)

#splitting arrays
arr_split = np.array_split(a1, 3)
print(arr_split)
print(type(arr_split))
print(sep)

#search array
arr = np.array([1, 2, 3, 4, 4, 3, 2, 4, 4, 2, 6])
x = np.where(arr == 4)
print(x) #this prints index nos of where 4 is in the array
print(sep)

#search sorted array
#this performs a binary search in array and returns index where specified value would be inserted to maintain the search order

arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x1 = np.searchsorted(arr1, 5)
print(x1) #this will tell position where we can place the entered number(i.e 5)
print(sep)

arr1 = np.array([1, 2, 3, 4, 9, 10])
x1 = np.searchsorted(arr1, [5, 6, 7], side = "right") #side = right means it starts to search from right
print(x1) #this will tell where we can place the whole list of numbers\
print(sep)

#sort an array
arr1 = np.array([1, 5, 45, 69, 78, 34, 69, 2, 9 ])
print(np.sort(arr1))
print(sep)

#sorting 2d array
arr2 = np.array([[1, 2], [3, 4]])
print(np.sort(arr2))
print(sep)

#sorting string
str1 = np.array(['a', 'b', 'g', 'h', 'z', 'b', 'B', 'A', 'M', 'Z'])
print(np.sort(str1))
print(sep)

#filter array : getting some elements out of an existing array and creating a new array out of them
str2 = np.array(['a', 'b', 'g', 'h', 'z'])
f = [True, False, False, True, True]
new_st = str2[f] #we only get the true data
print(new_st)
print(sep)

#NumPy array functions
#shuffle
np.random.shuffle(arr1)
print(arr1)
print(sep)

#unique
var1 = np.array([1, 2, 3, 4, 2, 5, 2, 6, 2, 7])
x = np.unique(var1)
print(x)
print(sep)

y = np.unique(var1, return_index = True, return_counts = True) #we also get index no and counting
print(y)
print(sep)

#resize
#np.resize(array, (row, column))
y = np.resize(var1, (2,3))
print(y)
print(sep)

#flatten
#this can help convert 2d array to 1d array
#Order of conversion : {C (Flatten in row major : DEFAULT), F (flatten in column major - Fortran Style), A (flatten in column major order if 'a' is fortran *contiguous* in memory, row-major otherwise), K(flatten in order the elements occur in memory)}
print(y.flatten())
print(sep)

print(y.flatten(order = "F"))
print(sep)

#ravel
print(np.ravel(y))
print(sep)

print(np.ravel(y, order = "F"))
print(sep)

#Insert and Delete
v = np.insert(var1, (2, 3), 30)
print(v)
print(sep)

#this does not insert float value, it converts to integer and inserts
v = np.insert(var1, (1), 1.55)
print(v)
print(sep)

#for 2d array
z = np.array([[1, 2, 3], [4, 5, 6]])

z = np.insert(y, 2 , 6, axis = 0)
print(z)
print(sep)

z = np.insert(y, 2 , 6, axis = 1)
print(z)
print(sep)

#insert multiple data
z = np.array([[1, 2, 3], [4, 5, 6]])

z = np.insert(y, 2 , [6, 7, 8], axis = 0)
print(z)
print(sep)

z = np.insert(y, 2 , [6, 7], axis = 1)
print(z)
print(sep)

#similarly like insert we can use delete

#matrix in np
var = np.matrix([[1, 2, 3], [4, 5, 6]])
                