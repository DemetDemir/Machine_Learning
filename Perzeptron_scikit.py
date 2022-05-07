from sklearn import datasets, metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Standardisierung der Merkmale (Standardnormalverteilung)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Trainieren des Models auf den Trainingsdaten
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

#Vorhersage basierend auf den Test-Daten
y_pred = ppn.predict(X_test_std)

#Ausgabe Fehlklassifizierungen und Korrektklassifizierungen
print('Fehlklassifizieurte Exemplare: %d' % (y_test != y_pred).sum())
print('Korrektklassifizierungsrate: %.2f' % accuracy_score(y_test, y_pred))

#Plotten der Entscheidungsgrenze
def plot_decision_region(X, y, classifier, resolution=0.02):

    #Markierungen und Farben einstellen
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #Plotten der Entscheidungsgrenze 
    
    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() +1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() +1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution ) , np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_region(X=X_combined_std, y=y_combined, classifier=ppn)
plt.xlabel=('Länge des Blütenblattes')
plt.ylabel=('Breite des Blütenblattes')
plt.legend(loc='upper left')
plt.show()



