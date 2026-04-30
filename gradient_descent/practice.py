import pandas as pd
import numpy as np

df = pd.read_csv('test_scores.csv')
x = np.array(df['math'])
y = np.array(df['cs'])

def gradient_descent(x,y):
    m_curr = 0
    b_curr = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002

    for i in range(iterations):
        y_pred = m_curr * x + b_curr
        cost = (1/n)*sum((y-y_pred)**2)
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum((y-y_pred))
        m_curr = m_curr - learning_rate*md
        b_curr = b_curr - learning_rate*bd

        print('m {}, b {}, cost {}, iteration {}'.format(m_curr,b_curr,cost,i))


gradient_descent(x,y)


