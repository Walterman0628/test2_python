#encoding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# test 1
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print s
# test 2
dates = pd.date_range('20170101', periods=7)
print dates
print '--' * 16
df = pd.DataFrame(np.random.rand(7, 4), index=dates, columns=list('ABCD'))
print df
print(df.head())
print("--------------" * 10)
print(df.tail(3))
print("index is :" )
print(df.index)
print("columns is :" )
print(df.columns)
print("values is :" )
print(df.values)
print(df.describe())
# test 3
df2 = pd.DataFrame({
    'A' : 1.,
    'B' : pd.Timestamp('20170102'),
    'C' : pd.Series(1, index=list(range(4)), dtype='float32'),
    'D' : np.array([3] * 4, dtype='int32'),
    'E' : pd.Categorical(['test', 'train', 'test', 'train']),
    'F' : 'foo'
})
print df2
# test 4
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df.T)
print(df.sort_index(axis=1, ascending=False))
print(df.sort_values(by='B'))
print(df['A'])
print(df[0:3])
print("========= 指定选择日期 ========")
print(df['20170102':'20170103'])
print(df.loc[dates[0]])
print(df.loc[:,['A','B']])
print(df.loc['20170102':'20170104',['A','B']])
print(df.loc[dates[0],'A'])
print(df.at[dates[0],'A'])
print(df.iloc[3])
print(df.iloc[3:5,0:2])
print(df.iloc[[1,2,4],[0,2]])
print(df.iloc[1:3,:])
print(df.iat[1,1])
print(df[df.A > 0])
# test 5
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
print df2
print df2[df2['E'].isin(['two', 'four'])]
# test 6
data = np.array(['a','b','c','d'])
s = pd.Series(data)
print s
s = pd.Series(data,index=[100,101,102,103])
print s
# test 7
data = {'a' : 0., 'b' : 1., 'c' : 2.}
s = pd.Series(data)
print s
# test 8
s = pd.Series(5, index=[0, 1, 2, 3])
print s
# test 9
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'])
print df
# test 10
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data)
print df
# test 11
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print df
# test 12
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
      'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
print df
# test 13
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
      'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
df['three']=pd.Series([10,20,30],index=['a','b','c'])
print df
df['four']=df['one']+df['three']
print df
del df['one']
print df
df.pop('two')
print df
# test 14
df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])
df = df.append(df2)
print df
df = df.drop(0)
print df
# test 15
data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)),
        'Item2' : pd.DataFrame(np.random.randn(4, 2))}
p = pd.Panel(data)
print p['Item1']
print p.major_xs(1)
print p.minor_xs(1)
#  test 16
s = pd.Series(np.random.randn(4))
print s.axes
print s.empty
print s.ndim
print s.size
print s.values
print s.head(2)
print s.tail(2)
# test 17
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Minsu','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}
df = pd.DataFrame(d)
print df
print df.T
print df.axes
print df.dtypes
print df.empty
print df.ndim
print df.shape
print df.size
print df.values
print df.head(2)
print df.tail(2)
# test 18
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Minsu','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])}
df = pd.DataFrame(d)
print df
print df.sum()
print df.sum(1)
print df.mean()
print df.std()
print df.describe()
print df.describe(include=['object'])
print df.describe(include=['number'])
print df.describe(include='all')
# test 19
def adder(ele1,ele2):
   return ele1+ele2
df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.pipe(adder,2)
print df
df.apply(np.mean)
print df
df.apply(lambda x: x.max() - x.min())
print df
df.applymap(lambda x:x*100)
print df
# test 20
N=20
df = pd.DataFrame({
   'A': pd.date_range(start='2016-01-01',periods=N,freq='D'),
   'x': np.linspace(0,stop=N-1,num=N),
   'y': np.random.rand(N),
   'C': np.random.choice(['Low','Medium','High'],N).tolist(),
   'D': np.random.normal(100, 10, size=(N)).tolist()
})
df_reindexed = df.reindex(index=[0,2,5], columns=['A', 'C', 'B'])
print df_reindexed
# test 21
df1 = pd.DataFrame(np.random.randn(10,3),columns=['col1','col2','col3'])
df2 = pd.DataFrame(np.random.randn(7,3),columns=['col1','col2','col3'])
df1 = df1.reindex_like(df2)
print df1
# test 22
df1 = pd.DataFrame(np.random.randn(6,3),columns=['col1','col2','col3'])
df2 = pd.DataFrame(np.random.randn(2,3),columns=['col1','col2','col3'])
print df2.reindex_like(df1)
print ("Data Frame with Forward Fill limiting to 1:")
print df2.reindex_like(df1,method='ffill',limit=1)
# test 23
df1 = pd.DataFrame(np.random.randn(6,3),columns=['col1','col2','col3'])
print df1
print ("After renaming the rows and columns:")
print df1.rename(columns={'col1' : 'c1', 'col2' : 'c2'},
index = {0 : 'apple', 1 : 'banana', 2 : 'durian'})
# test 24
df = pd.DataFrame(np.random.randn(4,3),columns=['col1','col2','col3'])
for key,value in df.iteritems():
   print key,value
for row_index,row in df.iterrows():
   print row_index,row
for row in df.itertuples():
    print row
# test 25
unsorted_df = pd.DataFrame(np.random.randn(10,2),index=[1,4,6,2,3,5,9,8,0,7],columns = ['col2','col1'])
sorted_df=unsorted_df.sort_index()
print sorted_df
sorted_df = unsorted_df.sort_index(ascending=False)
print sorted_df
sorted_df=unsorted_df.sort_index(axis=1)
print sorted_df
# test 26
unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})
sorted_df = unsorted_df.sort_values(by='col1')
print sorted_df
sorted_df = unsorted_df.sort_values(by=['col1','col2'])
print sorted_df
# test 27
s = pd.Series(['Tom', 'William Rick', 'John', 'Alber@t', np.nan, '1234','SteveMinsu'])
print s.str.lower()
print s.str.upper()
print s.str.len()
print s.str.strip()
print s.str.split(' ')
print s.str.cat(sep=' <=> ')
print s.str.get_dummies()
print s.str.contains(' ')
print s.str.replace('@','$')
print s.str.repeat(2)
print s.str.count('m')
print s.str.startswith ('T')
print s.str.endswith('t')
print s.str.find('e')
print s.str.swapcase()
print s.str.islower()
print s.str.isupper()
# test 28
print "display.max_columns = ", pd.get_option("display.max_columns")
print "before set display.max_rows = ", pd.get_option("display.max_rows")
pd.set_option("display.max_rows",80)
print "after set display.max_rows = ", pd.get_option("display.max_rows")
pd.reset_option("display.max_rows")
print "reset display.max_rows = ", pd.get_option("display.max_rows")
pd.describe_option("display.max_rows")
# test 29
s = pd.Series([1,2,3,4,5,4])
print s.pct_change()
df = pd.DataFrame(np.random.randn(5, 2))
print df.pct_change()
# test 30
s1 = pd.Series(np.random.randn(10))
s2 = pd.Series(np.random.randn(10))
print s1.cov(s2) # 协方差
# test 31
frame = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
print frame['a'].cov(frame['b'])
print frame.cov() # 协方差
# test 32
frame = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
print frame['a'].corr(frame['b'])
print frame.corr() # 相关
# test 33
s = pd.Series(np.random.np.random.randn(5), index=list('abcde'))
print s.rank()
# test 34
df = pd.DataFrame(np.random.randn(10, 4),index = pd.date_range('1/1/2020', periods=10),columns = ['A', 'B', 'C', 'D'])
print df.rolling(window=3).mean()
print df.expanding(min_periods=3).mean()
print df.ewm(com=0.5).mean()
# test 35
df = pd.DataFrame(np.random.randn(10, 4),
      index = pd.date_range('1/1/2000', periods=10),
      columns = ['A', 'B', 'C', 'D'])
print df
r = df.rolling(window=3,min_periods=1)
print r.aggregate(np.sum)
print r['A'].aggregate(np.sum)
print r[['A','B']].aggregate(np.sum)
print r['A'].aggregate([np.sum,np.mean])
print r[['A','B']].aggregate([np.sum,np.mean])
print r.aggregate({'A' : np.sum,'B' : np.mean})
# test 36
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])
df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
print df
print df['one'].isnull()
print df['one'].notnull()
print df['one'].sum()
print df.fillna(0)
print df.fillna(method='pad') # 缺失值向前填充
print df.fillna(method='backfill') # 缺失值向后填充
print df.dropna()
print df.dropna(axis=1)
# test 37
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
print df
print df.groupby('Team')
print df.groupby('Team').groups
print df.groupby(['Team','Year']).groups
grouped = df.groupby('Year')
for name,group in grouped:
    print name
    print group
print grouped.get_group(2014)
print grouped['Points'].agg(np.mean)
grouped = df.groupby('Team')
print grouped.agg(np.size)
print grouped['Points'].agg([np.sum, np.mean, np.std])
score = lambda x: (x - x.mean()) / x.std()*10
print grouped.transform(score)
print df.groupby('Team').filter(lambda x: len(x) >= 3)
# test 38
left = pd.DataFrame({
         'id':[1,2,3,4,5],
         'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
         'subject_id':['sub1','sub2','sub4','sub6','sub5']})
right = pd.DataFrame(
         {'id':[1,2,3,4,5],
         'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
         'subject_id':['sub2','sub4','sub3','sub6','sub5']})
print left
print "========================================"
print right
rs = pd.merge(left,right,on='id')
print rs
rs = pd.merge(left,right,on=['id','subject_id'])
print rs
rs = pd.merge(left, right, on='subject_id', how='left')
print rs
rs = pd.merge(left, right, on='subject_id', how='right')
print rs
rs = pd.merge(left, right, on='subject_id', how='outer')
print rs
rs = pd.merge(left, right, on='subject_id', how='inner')
print rs
# test 39
one = pd.DataFrame({
         'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
         'subject_id':['sub1','sub2','sub4','sub6','sub5'],
         'Marks_scored':[98,90,87,69,78]},
         index=[1,2,3,4,5])
two = pd.DataFrame({
         'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
         'subject_id':['sub2','sub4','sub3','sub6','sub5'],
         'Marks_scored':[89,80,79,97,88]},
         index=[1,2,3,4,5])
rs = pd.concat([one,two])
print rs
rs = pd.concat([one,two],keys=['x','y'])
print rs
rs = pd.concat([one,two],keys=['x','y'],ignore_index=True)
print rs
rs = pd.concat([one,two],axis=1)
print rs
rs = one.append(two)
print rs
rs = one.append([two,one,two])
print rs
# test 40
print pd.datetime.now()
time = pd.Timestamp('2018-11-01')
print time
time = pd.date_range("12:00", "23:59", freq="30min").time
print time
time = pd.date_range("12:00", "23:59", freq="H").time
print time
time = pd.to_datetime(pd.Series(['Jul 31, 2009','2019-10-10', None]))
print time
time = pd.to_datetime(['2009/11/23', '2019.12.31', None])
print time
# test 41
datelist = pd.date_range('2020/11/21', periods=5)
print datelist
datelist = pd.date_range('2020/11/21', periods=5, freq='M')
print datelist
datelist = pd.bdate_range('2011/11/03', periods=5)
print datelist
# test 42
timediff = pd.Timedelta(days=2)
print timediff
s = pd.Series(pd.date_range('2012-1-1', periods=3, freq='D'))
td = pd.Series([ pd.Timedelta(days=i) for i in range(3) ])
df = pd.DataFrame(dict(A = s, B = td))
print df
df['C']=df['A']+df['B']
df['D']=df['C']-df['B']
print df
# test 43
s = pd.Series(["a","b","c","a"], dtype="category")
print s
cat = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
print cat
cat = cat=pd.Categorical(['a','b','c','a','b','c','d'], ['c', 'b', 'a'])
print cat
cat = cat=pd.Categorical(['a','b','c','a','b','c','d'], ['c', 'b', 'a'],ordered=True)
print cat
# test 44
cat = pd.Categorical(["a", "c", "c", np.nan], categories=["b", "a", "c"])
df = pd.DataFrame({"cat":cat, "s":["a", "c", "c", np.nan]})
print df.describe()
print "============================="
print df["cat"].describe()
# test 45
s = pd.Categorical(["a", "c", "c", np.nan], categories=["b", "a", "c"])
print s.categories
print s.ordered
# test 46
s = pd.Series(["a","b","c","a"], dtype="category")
s.cat.categories = ["Group %s" % g for g in s.cat.categories]
print s.cat.categories
s = pd.Series(["a","b","c","a"], dtype="category")
s = s.cat.add_categories([4])
print s.cat.categories
s = pd.Series(["a","b","c","a"], dtype="category")
print s
print s.cat.remove_categories("a")
# test 47
cat = pd.Series([1,2,3]).astype("category", categories=[1,2,3], ordered=True)
cat1 = pd.Series([2,2,2]).astype("category", categories=[1,2,3], ordered=True)
print cat>cat1
# test 48
df = pd.DataFrame(np.random.randn(10,4),index=pd.date_range('2018/12/18',
   periods=10), columns=list('ABCD'))
df.plot()
df = pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d'])
df.plot.bar()
df.plot.bar(stacked=True)
df.plot.barh(stacked=True)
# plt.show()
# test 49
df = pd.DataFrame({'a':np.random.randn(1000)+1,'b':np.random.randn(1000),'c':
np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])
df.plot.hist(bins=20)
df.hist(bins=20)
# plt.show()
# test 50
df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df.plot.box()
# plt.show()
# test 51
df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
df.plot.area()
# plt.show()
# test 52
df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
df.plot.scatter(x='a', y='b')
# plt.show()
# test 53
df = pd.DataFrame(3 * np.random.rand(4), index=['a', 'b', 'c', 'd'], columns=['x'])
df.plot.pie(subplots=True)
# plt.show()
# test 54
df=pd.read_csv("temp.csv")
print df
df=pd.read_csv("temp.csv",index_col=['S.No'])
print df
df = pd.read_csv("temp.csv", dtype={'Salary': np.float64})
print df.dtypes
df=pd.read_csv("temp.csv", names=['a', 'b', 'c','d','e'])
print df
df=pd.read_csv("temp.csv",names=['a','b','c','d','e'],header=0)
print df
df=pd.read_csv("temp.csv", skiprows=2)
print df
# test 55
ts = pd.Series(np.random.randn(10))
ts[2:-2] = np.nan
sts = ts.to_sparse()
print sts
print sts.to_dense()
# test 56
print pd.Series([True]).bool()
# test 57
s = pd.Series(range(5))
print s==4
# test 58
s = pd.Series(list('abc'))
s = s.isin(['a', 'c', 'e'])
print s
