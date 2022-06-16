import numpy as np

#create a numpy array with 3 rows and 5 columns
a = np.arange(15).reshape(3,5)
a

#dimension of array (n,m) n rows, m columns
a.shape



a.ndim

#Initialiaize Array
a = np.array([1,2,3])
print(a)

b = np.array([[5.0, 4.2, 6.5 ], [4.3, 2.6, 3.7]])
print(b)

print(b.ndim)
print(b.shape)
print(len(b))
#type
print(b.dtype)
#size
b.itemsize
print(b.size)

a = np.array([[1,2,3,4,5,6,7], [8,9,10,11,12,13,14]])
a.shape
a[1, 5]
a[1, -2]

#[startindex:endindex,stepsize]
#[:, :] erster Wert ist die Zeile, zweiter Wert ist die Spalte
#Alle Elemente des Arrays
print(a[:, :])
print(" ")
#Alle Elemente der ersten Zeile
print(a[0, :])
print(" ")
#Alle Elemente der zweiten Zeile
print(a[1, :])
print(" ")
#Alle Elemente der ersten Reihe
print(a[:, 0])
print(" ")
#Alle Elemente der zweiten Reihe
print(a[:, 1])#
print(" ")
#Erste und zweite Reihe
print(a[0:2, 0:2])
print(" ")
#Spezifische Reihe [4,11]
print(a[:, 3])
print(" ")
#
print(a[0, 1:6:2])


#Spezielles Element ändern
a[1, 5] = 20
print(a[1,5])
print(" ")
#Komplette Reihe auf den Wert 5 setzen
a[:, 2] = 5
print(a)
print(" ")
#Reihe auf die Werte 1,2 ändern
a[:, 2] = [1,2]
print(a)
print(" ")

#3 Dimesionales Array mit random Werten erstellen
a=np.random.rand(3,3)

print(a)
print("")
print(a.shape)
print("")

#Return evenly spaced values within a given interval.
#numpy.arange([start, ]stop, [step, ]dtype=None, *, like=None)
#array von -3 bis 3
a = np.arange(-3, 4, 1)
print(a)
print("")

#numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
#Return evenly spaced numbers over a specified interval.
#Returns num evenly spaced samples, calculated over the interval [start, stop].
#Erstelle 10 Werte zwischen -3 und +3 
a = np.linspace(-3,3,5, axis=0)
print(a)
print("")


b = np.random.rand(3,3)
print(b)
print("")

#3D Array 
a = np.array( [ [ [1,2], [3,4] ], [ [5,6], [7,8] ] ] )
print(a)
print(a.ndim)
print("")

#Access 8
print(a[1, 1, 1])
#Access 3
print(a[0, 1, 0])
#Access 5
print(a[1,0,0])
#Acces 1,2
print(a[0, 0, :])
#Acces 1,3
print(a[0, :, 0])
#Access 5,6,7,8
print(a[1, :, :])

#Assign subset to new variable
print(a[1, 0, :])
print(a)
print("")

#Change the value
c = a[1, 0, :]
c = [11,22]
print(c)
print("")

a[1, 1, : ] = [11,12]
print(a)
print("")

a[1, : ,0] = [30,40]
print(a)
print("")

#Initialize all zeros Vector
z = np.zeros(5)
print(z)
print("")

#Initalize 2x3
z1 = np.zeros((2,3))
print(z1)
print("")

#Initialize 5x5
z2 = np.zeros((5,5))
print(z2)
print("")


#Initalize 3D
z3 = np.zeros((3,2,3))
#print(z3)
print("")

#4D
z4 = np.zeros((3,2,3,2))
print(z4)

#All ones matrix
ones = np.ones((2,2,2))
print(ones)

#All one value matrix. First parameter shape, second value
g = np.full((2,2), 99)
print(g)

#Reuse size of g as parameter
f = np.full_like(g, 4)
print(f)

#Pass in shape as Argument for random
j = np.random.random_sample(f.shape)
print(j)

#Random integer values
k = np.random.randint(7, size=(3,3))
print(k)
print("")

l = np.random.randint(-3,4, size=(3,3))
print(l)

#Identity Matrix only nxn
i = np.identity(6)
print(i)

#Repeat array
arr = np.array([[1,2,3]])
print(arr)
print("")

k = np.repeat(arr, 3, axis=0)
print(k)
print("")

#Erstelle Array mit 1 Rand 1er, 2er Rand nullen, mitte 9
#1. Initialisiere 5x5 Matrix mit 1en
A = np.ones((5,5))
print(A)
print("")

#2. Wähle inneren Kreis aus und setze ihn auf 0
#print( A[1:4, 1:4 ] )
tmp = A[1:4, 1:4] 
tmp = np.zeros(tmp.shape)
A[1:4, 1:4]  = tmp
print(A)
print("")

#Wähle mittleren Wert aus und setze ihn auf 9
A[2, 2] = 9
print(A)

#Be carefull when copying array
#If you change the copy, it changes the original array
a = ([1,2,3])
print(a)
print("")

b = a
print(b)
print("")
b[0] = 100
print(b)
print(a)
#now a has 100 as the first element to
#to avoid this, use copy function

k = np.array([2,3,4])
j = np.copy(k)
#oder j = k.copy
j[0] = 100
print(k)
print(j)


#Elementwise calculation
a = np.array([1,2,3,4])
#Add 2 to every value
a = a + 2
print(a)
print("")

#Subtract 5 from every value
a = a -5
print(a)
print("")

#Multiply by a scalar
a = a * -1
print(a)
print("")

#Divide by a scalar
a = a / 2
print(a)
print("")

#Vector by Vector arithmetic
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])

#Add to vectors
c = a+b
print(c)
print("")

#Subtract two vectors
c = a -b
print(c)
print("")

#Multiply two vectors
c = a * b
print(c)
print("")

#Divide two vectors
c = a / b
print(c)
print("")

#Potenzieren
a = a ** 2
print(a)
print("")

#Berechne Sinus
a = np.sin(a)
print(a)
print("")

#Berechne Cosinus
a = np.cos(a)
print(a)
print("")

#Dot Product
a = np.array( [[1],[2],[3]] )
print(a)
print("")

b = np.array([[1,2,3]] )
print(b)
print("")

c = np.dot(a,b)
print(c)

#Matrix multiplication

a = np.ones((2,3))
b = np.full((3,2), 2)
print(a)
print("")
print(b)
print("")

c = np.matmul(a,b)
print(c)
print("")


#Anderes Beispiel
a = np.array( [[1,2,3], [4,5,6]] )
print(a)
print(a.shape)
print("")

b = np.array( [[6,7],[8,9],[10,11]]  )
print(b)
print("")

c = np.matmul(a,b)
print(c)
print("")

d = np.dot(a,b)
print(d)

#Berechne Determinante
c = np.identity(3)
determinante = np.linalg.det(c)
print(c)
print("")
print(determinante)
#Eigenvalues: np.linalg.eig()
#Inverse of Matrix: np.linalg.inv


#Transponieren 
c = np.random.randint(7, size=(3,3))
print(c)
print("")


d = np.transpose(np.copy(c))
print(d)

#Statistics
stats = np.array( [[1,2,3], [4,5,6]] )
print(stats)
print("")

minimum = np.min(stats)
print(minimum)
print("")

maximum = np.max(stats)
print(maximum)
print("")

#Row based
min = np.min(stats, axis=0)
print(min)
print("")

max = np.max(stats, axis=1)
print(max)

#Summiere alle Elemente einer Matrix
f = np.sum(stats)
print(f)
print("")

g = np.sum(stats, axis=0)
print(g)
print("")

print(stats)
z = np.sum(stats, axis=1)
print(z)

#Reorganize
before = np.random.randint(7, size=(2,4))
print(before)
print("")

after = before.reshape((8,1))
print(after)
print("")

after = before.reshape((4,2))
print(after)
print("")

after = before.reshape((2,2,2))
print(after)


#Vertically stacking
a = np.array( [1,2,3,4 ] )
b = np.array( [5,6,7,8 ] )

vstacked = np.vstack([a,b])
print(vstacked)
print("")
#mehrere aufeinander

v = np.vstack([a,b,a,b,b])
print(v)

#Horizontal stack
a = np.array( [1,2,3,4 ] )
b = np.array( [5,6,7,8 ] )

h1 = np.hstack((a,b))
print(h1)

#Boolean Masking <,>, >=, <=, !=, ==
z = np.array( [[1,2,3], [4,5,10], [7,8,9]] )
print(z)
print("")

k = z > 5
print(k)
print("")

#Gib alle Werte zurück die größer als 5 sind
n = z[z>5]
print(n)

#Gib True zurück wenn in einer Spalte der Wert größer gleich 0.5 ist
f = np.random.rand(3,6)
print(f)
print("")

b = np.any(f >= 0.5, axis=0)
print(b)
print("")

#Gib alle Werte zurück die größer gleich 0.5 sind und kleiner gleich 0.9
n = f[(f>=0.5) & (f<=0.9)]
print(n)
print("")

#Gegenteil vom obigen als nicht größer als 0.5 & kleiner als 0.9
p = f[~(f>=0.5) & (f<=0.9)]
print(p)
print("")


t = np.arange(1,31,1)
print(t)
print("")

t = t.reshape(6,5)
print(t)
print("")

n = t[2:4, 0:2]
print(n)
print("")

v = t[[0,1,2,3], [1,2,3,4]]
print(v)