import sys,numpy as np
from readDocs import readDoc
import editdistance as edd

def mod(x,rep):
  if x==rep: return x
  return x.replace(rep,"")


def eval(fn,prob=0,verbose=False,sep=":"):
 docs = readDoc(fn)
 #print(docs[0])
 ignore=["_","___","EMPTY"]

 start=['n', 'G', 't', '5', '@', 's']
 start=docs[0][1]
 start=[mod(x,"_MYJOIN_") for x in start if x not in ignore]
 start="".join(start)
 printString=["k","r","A","s","b","F","z"]
 printString=["r","E","<","@","d","5","i"]

 avg_ed=[]

 if verbose:
   print(len(docs))
 index=0
 outlist=[]

 for doc in docs:
  graph,truth,pred = doc
  truth = "".join([mod(x,sep) for x in truth if x not in ignore])
  pred = "".join([mod(x,sep) for x in pred if x not in ignore])
  #print(graph,truth,pred)
  if truth==start:
    index+=1
    if verbose:
      print(" ".join(graph),"\t"," ".join(truth),"\t"," ".join(pred))
    if avg_ed!=[]:
      outlist.append(np.mean(avg_ed))
    #print("New")
    avg_ed = []
  ed = edd.eval(truth,pred)
  if np.random.random()<prob or truth=="".join(printString):
    if verbose:
      print("{}\t{}\t{}\t{}\t{}".format(index," ".join(graph)," ".join(truth)," ".join(pred),ed))
  avg_ed.append(ed) 

 if avg_ed!=[]:
  outlist.append(np.mean(avg_ed))
 return outlist


#### main
if __name__ == "__main__":

 try:
  sep=sys.argv[2]
 except IndexError:
  sep=":"

 print(eval(sys.argv[1],prob=1.0,verbose=True,sep=sep))
