#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1
import numpy as np


# In[4]:


#2
null_vector = np.zeros(10)
print(null_vector)


# In[8]:


#3
import numpy as np

null_vector = np.ones(5)
print(null_vector)


# In[11]:


#4
import numpy as np

vector = np.arange(10, 50)
print(vector)


# In[12]:


#5
import numpy as np

vector = np.arange(10, 50)
print(vector[::-1])


# In[14]:


#6
import numpy as np

matrix = np.arange(9).reshape(3,3)
print(matrix)


# In[15]:


#7
import numpy as np

arr = np.array([1,2,0,0,4,0])
indices = np.nonzero(arr)
print(indices)


# In[16]:


#8
import numpy as np

identity_matrix = np.identity(3)
print(identity_matrix)



# In[17]:


#9
import numpy as np

random_array = np.random.rand(3, 3, 3)
print(random_array)


# In[19]:


#10
import numpy as np

random_array = np.random.rand(10,10)
print(np.min(random_array ))
print(np.max(random_array ))


# In[20]:


#11
import numpy as np

random_vector = np.random.rand(30)
mean_value = np.mean(random_vector)
print(mean_value)


# In[21]:


#12
import numpy as np

# create a 5x5 array with all zeros
arr = np.zeros((5,5))

# fill the border with ones
arr[0,:] = 1
arr[-1,:] = 1
arr[:,0] = 1
arr[:,-1] = 1

print(arr)


# In[22]:


#13
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1


# In[23]:


#14
import numpy as np

# create a 5x3 matrix with random values
A = np.random.rand(5,3)

# create a 3x2 matrix with random values
B = np.random.rand(3,2)

# multiply the matrices
C = np.dot(A,B)

print(C)


# In[24]:


#15
import numpy as np

# Create a 1D array
arr = np.array([1, 4, 6, 8, 3, 5, 7, 9, 2])

# Create a boolean mask for elements between 3 and 8
mask = (arr > 3) & (arr < 8)

# Negate the elements between 3 and 8 in place
arr[mask] *= -1

print(arr)


# In[26]:


#16
import numpy as np
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)


# In[27]:


#17
import numpy as np

# Create two 1D arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([3, 4, 5, 6, 7])

# Find common values between the two arrays
common_values = np.intersect1d(arr1, arr2)

print(common_values)


# In[28]:


#18
from datetime import datetime, timedelta

# Get today's date
today = datetime.now().date()

# Get yesterday's date
yesterday = today - timedelta(days=1)

# Get tomorrow's date
tomorrow = today + timedelta(days=1)

print("Yesterday:", yesterday)
print("Today:", today)
print("Tomorrow:", tomorrow)


# In[29]:


#19
from datetime import datetime, timedelta

# Set the start date to July 1, 2016
start_date = datetime(2016, 7, 1)

# Set the end date to July 31, 2016
end_date = datetime(2016, 7, 31)

# Iterate through the dates in July 2016
current_date = start_date
while current_date <= end_date:
    print(current_date.date())
    current_date += timedelta(days=1)


# In[33]:


#20
import random

# Create a random vector of size 10
random_vector = [random.randint(1, 100) for _ in range(10)]

# Sort the random vector
random_vector.sort()

print(random_vector)


# In[34]:


#21
# Find the maximum value in the random vector
max_value = max(random_vector)

# Find the index of the maximum value
max_index = random_vector.index(max_value)

# Replace the maximum value with 0
random_vector[max_index] = 0

print(random_vector)


# In[35]:


#22
import numpy as np

# Assuming float_array is your float (32 bits) array
float_array = np.array([1.5, 2.7, 3.8], dtype=np.float32)

# Convert the float array to an integer array in place
float_array.view(dtype=np.int32)[:] = float_array.view(dtype=np.int32)

print(float_array)


# In[47]:


#23


# In[48]:


#24
import numpy as np

# Create a sample matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Calculate the mean of each row
row_means = np.mean(matrix, axis=1, keepdims=True)

# Subtract the mean of each row from the corresponding row
matrix_centered = matrix - row_means

print("Original matrix:")
print(matrix)
print("\nMatrix with row means subtracted:")
print(matrix_centered)


# In[50]:


#25
import numpy as np

# Create a sample array
array = np.array([[1, 4, 3],
                  [5, 2, 6],
                  [7, 8, 9]])

n = 1

sorted_indices = np.argsort(array[:, n])

sorted_array = array[sorted_indices]

print("Original array:")
print(array)

print("\nArray sorted by the nth column:")
print(sorted_array)


# In[49]:


#26
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 0, 6],
                [7, 8, 9],
                [0, 0, 0]])

null_columns = np.all(arr == 0, axis=1)
null_column_indices = np.where(null_columns)[0]

print("Null columns indices:", null_column_indices)


# In[ ]:




