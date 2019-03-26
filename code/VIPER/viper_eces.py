import sys,random

def readD(fn):
  h = {}
  for line in open(fn):
    line = line.strip()
    x = line.split()
    a,b = x[0].strip(),x[1].strip()
    h[a] = b
  return h

words = []
prob=float(sys.argv[1])
h = readD(sys.argv[2])



for line in sys.stdin:
  line = line.strip()
  if True:
    word = line
    ww = []
    for w in word:
      p = random.random()
      if p<prob:
        d = h.get(w,w) 
      else: d=w
      ww.append((d,w))
    words.append(( "".join([c[0] for c in ww]), "".join([c[-1] for c in ww]) ))
    truth = " ".join([w[0] for w in words])
    disturbed = " ".join([w[1] for w in words])
    print("\t".join([truth,disturbed]))
    words = []
