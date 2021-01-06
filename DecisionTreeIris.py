from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target


X_train, X_test, Y_train, Y_test = train_test_split(df[iris.feature_names],df['target'], random_state=0)

dt = tree.DecisionTreeClassifier(max_depth=2)

dt.fit(X_train, Y_train)
compare_test = np.array([Y_test==dt.predict(X_test)])
accuracy = np.sum(compare_test)/compare_test.shape[1]

print('Accuracy of model using test set:', accuracy)

tree.plot_tree(dt)

plt.show()


