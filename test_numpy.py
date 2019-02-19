# encoding=utf-8
import numpy as np
# test 1
a = np.array([1, 2, 3])
print a
# test 2
b = np.array([[1, 2], [3, 4]])
print b
# test 3
student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
print student
# test 4
a = np.array([[1, 2, 3], [4, 5, 6]])
b = a.reshape(3, 2)
print b
# test 5
a = np.arange(24)
b = a.reshape(2, 4, 3)
print b
# test 6
x = np.array([1, 2, 3, 4, 5])
print x.flags
# test 7
y = np.empty([3, 2], dtype = int)
print y
# test 8
x = np.zeros(5)
print x
# test 9
x = np.zeros((5,), dtype = np.int)
print x
# test 10
s =  'Hello World'
a = np.frombuffer(s, dtype =  'S1')
print a
# test 11
list = range(5)
it = iter(list)
x = np.fromiter(it, dtype = float)
print x
# test 12
x = np.arange(10, 20, 2, dtype = float)
print x
# test 13
stupid = np.linspace(10, 20, 5)
soft = np.linspace(10, 20, 5, endpoint = False)
ware = np.linspace(10, 20, 5, retstep = True)
print stupid
print soft
print ware
# test 14
a = np.logspace(1, 10, num = 10, base = 2, dtype = int)
print a
# test 15
a = np.arange(10)
s = slice(2, 7, 2)
print a[s]
# test 16
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print a
print a[1:]
# test 17
a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
print '数组是:'
print a
print '\n'
print '第二列:'
print a[...,1]
print '\n'
print '第二行:'
print a[1,...]
print '\n'
print '第二列及其之后:'
print a[...,1:]
# test 18
x = np.array([[1, 2], [3, 4], [5, 6]])
y = x[[0, 1, 2], [0, 1, 0]]
print y
# test 19
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print a
rows = np.array([[0, 0], [3, 3]])
cols = np.array([[0, 2], [0, 2]])
y = x[rows, cols]
print y
# test 20
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print x
y = x[1:4, 1:3]
print y
z = x[1:4, [1,2]]
print z
# test 21
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print x
print x[x > 5]
# test 22
a = np.array([1, 2+6j, 5, 3.5+5j])
print a[np.iscomplex(a)]
# test 23
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
c = a * b
print c
# test 24
a = np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20], [30, 30, 30]])
b = np.array([1, 2, 3])
print a + b
# test 25
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print a
for x in np.nditer(a):
    print x
# test 26
a = np.arange(0 ,60, 5)
a = a.reshape(3, 4)
print a
b = a.T
print b
for x in np.nditer(b):
    print x
# test 27
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print  '原始数组是：'
print a
print  '\n'
print  '原始数组的转置是：'
b = a.T
print b
print  '\n'
print  '以 C 风格顺序排序：'
c = b.copy(order='C')
print c
for x in np.nditer(c):
    print x
print  '\n'
print  '以 F 风格顺序排序：'
c = b.copy(order='F')
print c
for x in np.nditer(c):
    print x
# test 28
a = np.arange(0, 60, 5)
a = a.reshape(3,4)
print  '原始数组是：'
print a
print  '\n'
print  '以 C 风格顺序排序：'
for x in np.nditer(a, order =  'C'):
    print x
print  '\n'
print  '以 F 风格顺序排序：'
for x in np.nditer(a, order =  'F'):
    print x
# test 29
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print a
for x in np.nditer(a, op_flags = ['readwrite']):
    x[...] = 2 * x
print a
# test 30
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print a
for x in np.nditer(a, flags = ['external_loop'], order = 'F'):
    print x
# test 31
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print a
b = np.array([1, 2, 3, 4], dtype = int)
print b
for x, y in np.nditer([a, b]):
    print "%d:%d" % (x, y)
# test 32
a = np.arange(8).reshape(2, 4)
print a
print a.flat[5]
# test 33
a = np.arange(8).reshape(2, 4)
print a
print a.flatten()
print a.flatten(order = 'F')
print a
# test 34
a = np.arange(8).reshape(2, 4)
print a
print a.ravel()
print a.ravel(order = 'F')
print a
# test 35
a = np.arange(12).reshape(3, 4)
print a
print np.transpose(a)
# test 36
a = np.arange(8).reshape(2, 2, 2)
print a
print np.rollaxis(a, 2)
print np.rollaxis(a, 2, 1)
# test 37
a = np.arange(8).reshape(2, 2, 2)
print a
print np.swapaxes(a, 2, 0)
# test 38
x = np.array([[1], [2], [3]])
y = np.array([4, 5, 6])

# 对 y 广播 x
b = np.broadcast(x,y)
# 它拥有 iterator 属性，基于自身组件的迭代器元组

print '对 y 广播 x：'
r,c = b.iters
print r.next(), c.next()
print r.next(), c.next()
print '\n'
# shape 属性返回广播对象的形状

print '广播对象的形状：'
print b.shape
print '\n'
# 手动使用 broadcast 将 x 与 y 相加
b = np.broadcast(x,y)
c = np.empty(b.shape)

print '手动使用 broadcast 将 x 与 y 相加：'
print c.shape
print '\n'
c.flat = [u + v for (u,v) in b]

print '调用 flat 函数：'
print c
print '\n'
# 获得了和 NumPy 内建的广播支持相同的结果

print 'x 与 y 的和：'
print x + y
# test 39
a = np.arange(4).reshape(1, 4)
print a
print np.broadcast_to(a, (4, 4))
# test 40
x = np.array([[1, 2], [3, 4]])
print x
y = np.expand_dims(x, axis = 0)
print y
print x.shape, y.shape
y = np.expand_dims(x, axis = 1)
print y
print x.ndim, y.ndim
print x.shape, y.shape
# test 41
x = np.arange(9).reshape(1, 3, 3)
print x
y = np.squeeze(x)
print y
print x.shape, y.shape
# test 42
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print a, b
print np.concatenate((a, b))
print np.concatenate((a, b), axis = 1)
# test 43
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print a, b
print np.stack((a, b), 0)
print np.stack((a, b), 1)
# test 44
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print a, b
print np.hstack((a, b))
# test 45
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print a, b
print np.vstack((a, b))
# test 46
a = np.arange(9)
b = np.split(a, 3)
print b
b = np.split(a, [4, 7])
print b
# test 47
a = np.arange(16).reshape(4, 4)
b = np.hsplit(a, 2)
print b
# test 48
a = np.arange(16).reshape(4, 4)
b = np.vsplit(a, 2)
print b
# test 49
a = np.array([[1, 2, 3], [4, 5, 6]])
print a
print a.shape
b = np.resize(a, (3,2))
print b
print b.shape
b = np.resize(a, (3, 3))
# test 50
a = np.array([[1, 2, 3], [4, 5, 6]])
print np.append(a, [7, 8, 9])
print np.append(a, [[7, 8, 9]], axis = 0)
print np.append(a, [[5, 5, 5], [7, 8, 9]], axis = 1)
# test 51
a = np.array([[1, 2], [3, 4], [5, 6]])
print a
print np.insert(a, 3, [11, 12]) #未传递 Axis 参数， 在插入之前输入数组会被展开。
print np.insert(a, 1, [11], axis = 0) #传递了 Axis 参数， 会广播值数组来配输入数组。
print np.insert(a, 1, 11, axis = 1)
# test 52
a = np.arange(12).reshape(3, 4)
print a
print np.delete(a, 5)
print np.delete(a, 1, axis = 1)
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print np.delete(a, np.s_[::2])
# test 53
a = np.array([5,2,6,2,7,5,6,8,2,9])
print a
u = np.unique(a)
print u
u,indices = np.unique(a, return_index = True)
print indices
u,indices = np.unique(a, return_inverse = True)
print u
print indices
print u[indices]
u,indices = np.unique(a, return_counts = True)
print u
print indices
# test 54
print np.char.add(['hello'],[' xyz'])
print np.char.add(['hello', 'hi'],[' abc', ' xyz'])
print np.char.multiply('Hello ',3)
print np.char.center('hello', 20,fillchar = '*')
print np.char.capitalize('hello world')
print np.char.title('hello how are you?')
print np.char.lower(['HELLO','WORLD'])
print np.char.lower('HELLO')
print np.char.upper('hello')
print np.char.upper(['hello','world'])
print np.char.split ('hello how are you?')
print np.char.split ('YiibaiPoint,Hyderabad,Telangana', sep = ',')
print np.char.splitlines('hello\nhow are you?')
print np.char.splitlines('hello\rhow are you?')
print np.char.strip('ashok arora','a')
print np.char.strip(['arora','admin','java'],'a')
print np.char.join(':','dmy')
print np.char.join([':','-'],['dmy','ymd'])
print np.char.replace ('He is a good boy', 'is', 'was')
# test 55
a = np.char.encode('hello', 'cp500')
print a
print np.char.decode(a,'cp500')
# test 56
a = np.array([0,30,45,60,90])
print np.sin(a*np.pi/180)
print np.cos(a*np.pi/180)
print np.tan(a*np.pi/180)
# test 57
a = np.array([1.0,5.55,123,0.567,25.532])
print np.around(a)
print np.around(a, decimals =  1)
print np.around(a, decimals =  -1)
# test 58
a = np.array([-1.7,  1.5,  -0.2,  0.6,  10])
print a
print np.floor(a)
print np.ceil(a)
# test 59
a = np.arange(9, dtype = np.float_).reshape(3,3)
print a
b = np.array([10,10,10])
print b
print np.add(a,b)
print np.subtract(a,b)
print np.multiply(a,b)
print np.divide(a,b)
# test 60
a = np.array([0.25,  1.33,  1,  0,  100])
print a
print np.reciprocal(a)
b = np.array([100], dtype =  int)
print b
print np.reciprocal(b)
# test 61
a = np.array([10,100,1000])
print a
print np.power(a,2)
b = np.array([1,2,3])
print b
print np.power(a,b)
# test 62
a = np.array([10,20,30])
b = np.array([3,5,7])
print a
print b
print np.mod(a,b)
print np.remainder(a,b)
# test 63
a = np.array([-5.6j,  0.2j,  11.  ,  1+1j])
print a
print np.real(a)
print np.imag(a)
print np.conj(a)
print np.angle(a)
print np.angle(a, deg =  True)
# test 64
a = np.array([[3,7,5],[8,4,3],[2,4,9]])
print a
print np.amin(a,1)
print np.amin(a,0)
print np.amax(a)
print np.amax(a, axis =  0)
# test 65
a = np.array([[3,7,5],[8,4,3],[2,4,9]])
print a
print np.ptp(a)
print np.ptp(a, axis =  1)
print np.ptp(a, axis =  0)
# test 66
a = np.array([[30,40,70],[80,20,10],[50,90,60]])
print a
print np.percentile(a,50)
print np.percentile(a,50, axis =  1)
print np.percentile(a,50, axis =  0)
# test 67
a = np.array([[30,65,70],[80,95,10],[50,90,60]])
print a
print np.median(a)
print np.median(a, axis =  0)
print np.median(a, axis =  1)
# test 68
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print a
print np.mean(a)
print np.mean(a, axis =  0)
print np.mean(a, axis =  1)
# test 69
a = np.array([1,2,3,4])
print a
print np.average(a)
wts = np.array([4,3,2,1])
print np.average(a,weights = wts)
print np.average([1,2,3,  4],weights =  [4,3,2,1], returned =  True)
# test 70
print np.std([1,2,3,4])
print np.var([1,2,3,4])
# test 71
a = np.array([[3,7],[9,1]])
print a
print np.sort(a)
print np.sort(a, axis =  0)
dt = np.dtype([('name',  'S10'),('age',  int)])
a = np.array([("raju",21),("anil",25),("ravi",  17),  ("amar",27)], dtype = dt)
print a
print np.sort(a, order =  'name')
# test 72
x = np.array([3,  1,  2])
print x
y = np.argsort(x)
print y
print x[y]
for i in y:
    print x[i]
# test 73
nm =  ('raju','anil','ravi','amar')
dv =  ('f.y.',  's.y.',  's.y.',  'f.y.')
ind = np.lexsort((dv,nm))
print ind
print  [nm[i]  +  ", "  + dv[i]  for i in ind]
# test 74
a = np.array([[30,40,70],[80,20,10],[50,90,60]])
print  '我们的数组是：'
print a
print  '\n'
print  '调用 argmax() 函数：'
print np.argmax(a)
print  '\n'
print  '展开数组：'
print a.flatten()
print  '\n'
print  '沿轴 0 的最大值索引：'
maxindex = np.argmax(a, axis =  0)
print maxindex
print  '\n'
print  '沿轴 1 的最大值索引：'
maxindex = np.argmax(a, axis =  1)
print maxindex
print  '\n'
print  '调用 argmin() 函数：'
minindex = np.argmin(a)
print minindex
print  '\n'
print  '展开数组中的最小值：'
print a.flatten()[minindex]
print  '\n'
print  '沿轴 0 的最小值索引：'
minindex = np.argmin(a, axis =  0)
print minindex
print  '\n'
print  '沿轴 1 的最小值索引：'
minindex = np.argmin(a, axis =  1)
print minindex
# test 75
a = np.array([[30,40,0],[0,20,10],[50,0,60]])
print a
print np.nonzero (a)
# test 76
x = np.arange(9.).reshape(3,  3)
print x
y = np.where(x >  3)
print y
print x[y]
# test 77
x = np.arange(9.).reshape(3,  3)
print x
condition = np.mod(x,2)  ==  0
print condition
print np.extract(condition, x)
# test 78
# 存储在计算机内存中的数据取决于 CPU 使用的架构。 它可以是小端(最小有效位存储在最小地址中)或大端(最小有效字节存储在最大地址中)。
a = np.array([1,  256,  8755], dtype = np.int16)
print  '我们的数组是：'
print a
print  '以十六进制表示内存中的数据：'
print map(hex,a)
# byteswap() 函数通过传入 true 来原地交换
print  '调用 byteswap() 函数：'
print a.byteswap(True)
print  '十六进制形式：'
print map(hex,a)
# test 79
a = np.arange(6)
print a
print id(a)
b = a
print b
print id(b)
b.shape =  3,2
print b
print a
# test 80
a = np.arange(6).reshape(3,2)
print a
b = a.view()
print b
print id(a)
print id(b)
b.shape =  2,3
print b
print a
# test 81
a = np.array([[10,10],  [2,3],  [4,5]])
print a
b = a.copy()
print b
print  '我们能够写入 b 来写入 a 吗？'
print b is a
b[0,0]  =  100
print b
print a
# test 82
import numpy.matlib
print np.matlib.empty((2,2))
print np.matlib.zeros((2,2))
print np.matlib.ones((2,2))
print np.matlib.eye(n =  3, M =  4, k =  0, dtype =  float)
print np.matlib.identity(5, dtype =  float)
print np.matlib.rand(3,3)
i = np.matrix('1,2;3,4')
print i
j = np.asarray(i)
print j
k = np.asmatrix (j)
print k
# test 83
a = np.array([1,2,3,4,5])
np.save('outfile',a)
b = np.load('outfile.npy')
print b
# test 84
a = np.array([1,2,3,4,5])
np.savetxt('out.txt',a)
b = np.loadtxt('out.txt')
print b