import pandas as pd
import os

HOUSING_PATH = os.path.join("datasets", "housing")
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_data()

import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

import numpy as np

np.random.seed(42)
def split_train_test(data, test_ratio):
    # 先生成随机数
    shuffled_indices = np.random.permutation(len(data))
    # 取前面的百分比的随机数作为测试集
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
# 随机生成数据集
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

from zlib import crc32
def test_set_check1(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

import hashlib
def test_set_check2(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
def test_set_check3(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio
# 利用test_set_check来进行测试集的选择。确保稳定的随机性。
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:test_set_check3(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# 进行分层抽样
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 图像可视化
housing = strat_train_set.copy()
# housing.plot(kind="scatter", x="longitude", y="latitude")
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#     s=housing["population"]/100, label="population", figsize=(10,7),
#     c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#     sharex=False)
# plt.legend()
#
# 属性间的相关关系探讨
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])

housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
# 用中位数填充缺省值
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
# 用中位数填充
imputer = SimpleImputer(strategy="median")
# 丢掉文本属性，再计算中位数
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
# x是Numpy格式，转为Pandas DataFrame格式
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
# 处理文本属性
try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    from future_encoders import OrdinalEncoder # Scikit-Learn < 0.20
# 取出文本属性部分: housing_cat是Series属性，要转为array属性
housing_cat = np.array(housing["ocean_proximity"])
housing_cat = housing_cat.reshape(-1, 1)
ordinal_encoder = OrdinalEncoder()
# 由于只有一个特征，要使用array.reshape(-1,1)
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# 采用二进制编码 Onehot
try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# 自定义转换器
from sklearn.base import BaseEstimator, TransformerMixin
# 比直接的3.4.5.6要更安全
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")
]
# 特征组合成新的特征
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):   # 没有*args和**kwargs的构造函数
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self     # 什么都不做

    def transform(self, X, y=None):
        # 每个房子的房间数量；每个房子的人口；每个房子的卧室数量(optional)
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# 做一个pipeline实现数值属性的所有预处理
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 中位数填充、组合属性、标准化数据
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder(add_bedrooms_per_room=False)),
    ('std_scaler', StandardScaler()),
])
# 预处理流水线：先中位数填充缺省值；再增加额外的组合属性，然后进行标准化缩放
housing_num_tr = num_pipeline.fit_transform(housing_num)

# 处理数值和分类属性数据的流水线
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)

# 5. 训练一个线性回归模型
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# 看一些训练数据的预测情况，同时预处理要是一样的
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

# 测量训练集上的RMSE均方误差
housing_predictions = lin_reg.predict(housing_prepared)
# MSE
from sklearn.metrics import mean_squared_error
# housing_labels还是series类型
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# MAE
from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(housing_labels, housing_predictions)

# 画出所有的数据的预测与标签的值的对比
housing_labels = np.array(housing_labels)
housing_predictions_id = sorted(range(len(housing_labels)), key=lambda k:housing_labels[k], reverse=False)
housing_predictions_show = housing_predictions[housing_predictions_id]
housing_labels = np.array(housing_labels)
housing_labels_show = housing_labels[housing_predictions_id]
x_axis_data = [i for i in range(len(housing_predictions))]
plt.plot(x_axis_data, housing_labels_show)
plt.plot(x_axis_data, housing_predictions_show)
plt.show()

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
# 画出拟合图像
# housing_predictions_id = sorted(range(len(housing_labels)), key=lambda k:housing_labels[k], reverse=False)
# housing_predictions_show = housing_predictions[housing_predictions_id]
# housing_labels = np.array(housing_labels)
# housing_labels_show = housing_labels[housing_predictions_id]
# x_axis_data = [i for i in range(len(housing_predictions))]
# plt.plot(x_axis_data, housing_labels_show)
# plt.show()
# plt.plot(x_axis_data, housing_predictions_show)
# plt.show()

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
# 线性模型 10 fold
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)



