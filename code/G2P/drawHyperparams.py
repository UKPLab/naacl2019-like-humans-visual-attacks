import random,numpy as np

opt = random.choice(['Adam',"RMSprop","Adagrad","Adadelta","Adamax","Nadam","sgd"])
layers = random.choice([1, 2, 3, 4])
dropout_values = np.random.uniform(0.1,0.75) #random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.75])
#units = random.choice([50, 100, 200, 300, 400, 500])
units = int(np.random.uniform(30,500))
#learning_rate = random.choice([0.002,0.001,0.005,0.01,0.1,0.5,1.0])
#learning_rate = np.random.uniform(0.002,1.0)

if opt in ["Adam","RMSprop"]:
  mean=0.001
elif opt in ["Adagrad","sgd"]:
  mean=0.01
elif opt in ["Adadelta"]:
  mean=1.0
elif opt in ["Adamax","Nadam"]:
  mean=0.002
else:
  mean=None

learning_rate=np.random.normal(mean,mean/5)
if learning_rate<=0: learning_rate = random.choice([mean,mean,10*mean,5*mean,0.1*mean])

init_name = random.choice(["rnormal","runiform","varscaling","orth","lecun_uniform","glorot_normal","glorot_uniform","he_normal","he_uniform","id"])

# 100 adam 2 0.5 0.001 orthogonal 0.1
print(units,opt,layers,dropout_values,learning_rate,init_name)
