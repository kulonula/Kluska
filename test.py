import pickle as pkl
from predict import classification

x_data,y_data = pkl.load(open('ula.pkl',mode = 'rb'))
x = x_data[:500]
y = y_data[:500]

cl = classification(x,x,y,5)
