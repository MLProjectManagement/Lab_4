
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
