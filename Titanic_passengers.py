
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Вывод таблицы
# train_data.head()

# оценим размеры датафрейма
# print(train_data.shape)

# количество пустых ячеек в столбцах обучающей выборки
# print(train_data.isnull().sum())

# оценим выживаемость
# print(sns.countplot(x='Survived', data=train_data))#Axes(0.125,0.11;0.775x0.77)

# пол и класс каюты,количество родителей\детей и количество братьев\сестер на борту
features = ["Sex", "Pclass", "SibSp", "Parch"]

y = train_data[ "Survived" ]  # Информацию о выживших и погибших пассажирах поместим в переменную y

#превратим столбец 'Sex' в пару фиктивных переменных.
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

#С помощью ансамбля решающих деревьев обучим нашу модель, 
#сделаем предсказание для тестовой выборки и сохраним результат.
model=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)

model.fit(X,y)# обучаем модель

prediction = model.predict(X_test)# делаем предсказание

output=pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived':prediction})
output.to_csv('my_submission.csv', index=False)  # формируем итоговый датафрейм и сохраняем его в csv файл