import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#loading data:
df = pd.read_csv('tennis_stats.csv')

#investigating potential relationship between 'Aces' and 'Winnings':
ax1 = plt.subplot()
ax1.scatter(df['Aces'], df['Winnings'], alpha=0.3)
ax1.set_xlabel('Aces')
ax1.set_ylabel('Winnings')
plt.show()
plt.clf()

##linear regression model which predicts losses by double faults:
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
ax3.set_ylabel('Losses')
ax3.set_xlabel('Double Faults')
plt.show()
plt.clf

## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:
