
from sklearn import metrics
#target values
y=[0,0,0,0]
#prediction probabilities
yh=[[1, 0],[1, 0],[1, 0],[1, 0]] #in this case all perfect
#yh=[[0.4, 0.6],[0.5, 0.5],[0.5, 0.5],[0.6, 0.4]]
#yh=[[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5]] # all half
metrics.log_loss(y, yh, labels=[0,1]) #define the labels