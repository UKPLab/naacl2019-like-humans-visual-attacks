def readDoc(fn):
  docs=[]
  a,b,c=[],[],[]
  for line in open(fn):
    line = line.strip()
    if line=="":
      if a!=[]:
        docs.append((a,b,c))
      a,b,c=[],[],[]
    else:
      x = line.split("\t")
      a.append(x[0])
      b.append(x[1])
      c.append(x[2])
  if a!=[]:
    docs.append((a,b,c))
  return docs
