import sys

EMPTY="___"

for line in sys.stdin:
  line = line.strip()
  a,b = line.split("\t")
  a = a.split()
  b = b.split()
  if len(a)>len(b):
    b = b+[EMPTY]*(len(a)-len(b))
  elif len(a)<len(b):
    m = len(b)-len(a)
    c = "".join(b[-m-1:])
    b = b[:-m-1]+[c]
    
  for i in range(len(a)):
    print("{}\t{}\t{}".format(i+1,a[i],b[i]))
  print()
