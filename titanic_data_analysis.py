"""
    时间：2020.08.06
    项目：泰坦尼克号幸存者分析
    语言：python
    作者：aiitx luke
"""

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing

data_train = pd.read_csv("D:/pycharmpro_20192/python_project/titanic_20200806/titanic/train.csv")


fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
plt.title(u"Rescued situation")  # 标题
plt.ylabel(u"Num of people")

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.ylabel(u"Num of people")
plt.title(u"class distribution")

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"age")  # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title(u"Distribution of rescued by age")

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"age")  # plots an axis lable
plt.ylabel(u"density")
plt.title(u"Age distribution by class")
plt.legend((u'1 class', u'2 class', u'3 class'), loc='best')  # sets our legend for our graph.

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"Number of people on board at port")
plt.ylabel(u"num of people")
plt.show()

# 查看各乘客等级获救情况
fig = plt.figure()
fig.set(alpha=0.2)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'rescued': Survived_1, u'not rescued': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"Rescued status by passenger class")
plt.xlabel(u"Passenger class")
plt.ylabel(u"Number of people")
plt.show()

# 查看不同性别获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({u'male': Survived_m, u'female': Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"Rescued by gender")
plt.xlabel(u"gender")
plt.ylabel(u"Number of people")
plt.show()

# 然后我们再来看看各种舱级别情况下各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.65)  # 设置图像透明度，无所谓
plt.title(u"Rescued according to cabin class and gender")

ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels([u"rescued", u"not rescued"], rotation=0)
ax1.legend([u"female/highClass"], loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"rescued", u"not rescued"], rotation=0)
plt.legend([u"female/lowClass"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"rescued", u"not rescued"], rotation=0)
plt.legend([u"male/highClass"], loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"rescued", u"not rescued"], rotation=0)
plt.legend([u"male/lowClass"], loc='best')

plt.show()


# 查看各登船港口的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'rescued': Survived_1, u'not rescued': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"According to the port of embarkation")
plt.xlabel(u"port")
plt.ylabel(u"num of people")

plt.show()

# 堂兄弟/妹，孩子/父母有几人，对是否获救的影响
g = data_train.groupby(['SibSp', 'Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)
# 无特别的规律

# ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴把
# cabin只有204个乘客有值，我们先看看它的一个分布
df = data_train.Cabin.value_counts()
print(df)

# 有无cabin的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({u'yes': Survived_cabin, u'no': Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"According to yes/no a cabin")
plt.xlabel(u"Cabin")
plt.ylabel(u"num of people")
plt.show()

# 简单数据预处理
# 使用 RandomForestClassifier 填补缺失的年龄属性


def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].iloc[:,:].values
    unknown_age = age_df[age_df.Age.isnull()].iloc[:,:].values

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedages = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedages

    return df, rfr


def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_cabin_type(data_train)
print(data_train)

# 对类目型的特征因子化
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df

# 收敛变化幅度大的数据
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df[['Age']])
df['Age_scaled'] = scaler.fit_transform(df[['Age']], age_scale_param)
fare_scale_param = scaler.fit(df[['Fare']])
df['Fare_scaled'] = scaler.fit_transform(df[['Fare']], fare_scale_param)
df


# 逻辑回归建模
from sklearn import linear_model

# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.iloc[:,:].values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', solver='liblinear', tol=1e-6)
clf.fit(X, y)

clf


# test.csv也和train.csv同样的预处理
data_test = pd.read_csv("D:/pycharmpro_20192/python_project/titanic_20200806/titanic/test.csv")
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].iloc[:,:].values
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test = set_cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age']], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Fare']], fare_scale_param)
df_test

# 做预测取结果
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': predictions.astype(np.int32)})
result.to_csv("D:/pycharmpro_20192/python_project/titanic_20200806/titanic/logistic_regression_predictions.csv", index=False)
result
