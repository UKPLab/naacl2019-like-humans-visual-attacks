from keras import optimizers,initializers


def getInitializer(init_name,learning_rate,opt,functions):

 if init_name=="rnormal":
  init = initializers.RandomNormal()
 elif init_name=="runiform":
  init = initializers.RandomUniform()
 elif init_name=="varscaling":
  init = initializers.VarianceScaling()
 elif init_name=="orth":
  init = initializers.Orthogonal()
 elif init_name=="id":
  init = initializers.Identity()
 elif init_name=="lecun_uniform":
  init = initializers.lecun_uniform()
 elif init_name=="glorot_normal":
  init = initializers.glorot_normal()
 elif init_name=="glorot_uniform":
  init = initializers.glorot_uniform()
 elif init_name=="he_normal":
  init = initializers.he_normal()
 elif init_name=="he_uniform":
  init = initializers.he_uniform()

 if opt=="Adam":
  optimizer = optimizers.Adam(lr=learning_rate)
 elif opt=="Adagrad":
  optimizer = optimizers.Adagrad(lr=learning_rate)
 elif opt=="Adadelta":
  optimizer = optimizers.Adadelta(lr=learning_rate)
 elif opt=="Adamax":
  optimizer = optimizers.Adamax(lr=learning_rate)
 elif opt=="Nadam":
  optimizer = optimizers.Nadam(lr=learning_rate)
 elif opt=="sgd":
  optimizer = optimizers.SGD(lr=learning_rate)
 elif opt=="RMSprop":
  optimizer = optimizers.RMSprop(lr=learning_rate)

 if functions.startswith("maxout"):
  functions,maxout_k = functions.split("-")
  maxout_k = int(maxout_k)
 else:
  maxout_k = 3
 if functions.startswith("leakyrelu"):
  if "-" in functions:
    functions,maxout_k = functions.split("-")
    maxout_k = float(maxout_k)
  else: maxout_k = 0.01

 return init,optimizer,functions,maxout_k
