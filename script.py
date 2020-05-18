import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')

# perform exploratory analysis here:
ax1 = plt.subplot()
ax1.scatter(df['DoubleFaults'], df['Losses'], alpha=0.3)
ax1.set_xlabel('Double Faults')
ax1.set_ylabel('Losses')
plt.show()
plt.clf()

## perform single feature linear regressions here:
X = df[['DoubleFaults']]
y = df[['Losses']]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
regr = LinearRegression()
regr.fit(x_train, y_train)
y_predict = regr.predict(x_test)

print(regr.score(x_test, y_test))

ax2 = plt.subplot()
ax2.scatter(y_predict, y_test, alpha=0.3)
ax2.set_ylabel('Real values')
ax2.set_xlabel('Predicted values')
plt.show()
plt.clf()

ax3 = plt.subplot()
ax3.scatter(X, y, alpha=0.3)
ax3.plot(x_test, y_predict)
plt.show()
plt.clf








## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:
