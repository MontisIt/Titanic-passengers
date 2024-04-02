
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# ����� �������
# train_data.head()

# ������ ������� ����������
# print(train_data.shape)

# ���������� ������ ����� � �������� ��������� �������
# print(train_data.isnull().sum())

# ������ ������������
# print(sns.countplot(x='Survived', data=train_data))#Axes(0.125,0.11;0.775x0.77)

# ��� � ����� �����,���������� ���������\����� � ���������� �������\������ �� �����
features = ["Sex", "Pclass", "SibSp", "Parch"]

y = train_data[ "Survived" ]  # ���������� � �������� � �������� ���������� �������� � ���������� y

#��������� ������� 'Sex' � ���� ��������� ����������.
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

#� ������� �������� �������� �������� ������ ���� ������, 
#������� ������������ ��� �������� ������� � �������� ���������.
model=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)

model.fit(X,y)# ������� ������

prediction = model.predict(X_test)# ������ ������������

output=pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived':prediction})
output.to_csv('my_submission.csv', index=False)  # ��������� �������� ��������� � ��������� ��� � csv ����