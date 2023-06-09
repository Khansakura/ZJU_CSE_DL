#!/usr/bin/env python
# coding: utf-8

# # 作业第一周  Python、Numpy练习
1.编写程序打印1-100中所有能被3整除的奇数
# In[53]:


import numpy as np
numlist = np.arange(1,101,1)
num=[]
for x in numlist:
    if x % 3 == 0  and x % 2 != 0:
        num.append(x)
print(num)


# In[ ]:




2.读取文本文件"data/story.txt"，寻找出文件第一行与第二行相同的所有单词
提示：可以使用字符串的split()方法分割单词，使用字符串的replace()方法替换删除不需要的标点符号。
# In[54]:


filename = 'data/story.txt'
f = open(filename,'r')
lines = f.readlines()
line1 = lines[0].rstrip().lower()
line2 = lines[1].rstrip().lower()
for char in '-.,\n':
    line1 = line1.replace(char, ' ')
    line2 = line2.replace(char, ' ')
words1 = set(line1.split())
words2 = set(line2.split())
common_words = words1 & words2
print(common_words)
f.close()


# In[ ]:




3. Create a 10x10 array with random values and display the minimum and the maximum value of the array.
# In[55]:


import numpy as np
array = np.random.rand(10, 10)
print(array)
print("the minimum value:", array.min())
print("the maximum value:", array.max())


# In[ ]:




4. Create a vector of [1,2,… 15] and a vector of [7,8…12], then reshape them to a 5x3 matrix and a 3x2 matrix respectively, finally multiply  the 5x3 matrix by  the 3x2 matrix (real matrix product，should be 5x2), and print the result.
# In[56]:


import numpy as np
vector1 = np.arange(1,16)
vector2 = np.arange(7,13)
matrix1 = vector1.reshape(5,3)
matrix2 = vector2.reshape(3,2)
matrix = np.dot(matrix1, matrix2)
print(matrix)



# In[ ]:




