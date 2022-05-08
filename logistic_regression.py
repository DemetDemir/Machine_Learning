#import matplotlib.pyplot as plt
#import numpy as np
from sklearn.linear_model import LogisticRegression
from Perzeptron_scikit import*
from plott_decision import*

#Illustration how logistic Regression uses Sigmoid Function to determine the probability 
#if an object is class 1 or 0 
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()


lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)


plot_decision_region(X_combined_std, y_combined, classifier=lr)
plt.xlabel('Länge des Blüttenblattes [standardisiert]')
plt.ylabel('Breite des Blütenblattes [standardisiert]')
plt.legen(loc='upper left')
plt.show()


print(lr.predict_proba(X_test_std[0, :].reshape(1,-1)))