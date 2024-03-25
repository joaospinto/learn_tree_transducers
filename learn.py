import sys
from copy import deepcopy

def separateOuterCommas(term):
    l=[]
    bf=""
    c=0
    for x in term:
        if x=="(":
            c+=1
            bf+=x
        elif x==")":
            c-=1
            bf+=x
        elif x=="," and c==0:
            l.append(bf)
            bf=""
        else:
            bf+=x
    l.append(bf)
    return l

class RankedTree:
    def __init__(self,term):
        if not isinstance(term,str):
            self.root=term
            self.children=[]
        elif term.find("(")==-1:
            self.root=term
            self.children=[]
        else:
            self.root=term[:term.find("(")]
            self.children=[RankedTree(child) for child in separateOuterCommas(term[term.find("(")+1:term.rfind(")")])]

    def __str__(self):
        out=str(self.root)
        if len(self.children)!=0:
            out+="("
            i=0
            while i<len(self.children):
                out+=str(self.children[i])
                if i<len(self.children)-1:
                    out+=","
                i+=1
            out+=")"
        return out

    def __eq__(self,obj):
        return str(self)==str(obj)

    def __neq__(self,obj):
        return not self==obj

    def __hash__(self):
        return hash(str(self))

    def isLeaf(self):
        return len(self.children)==0

    def gcp(self,obj):
        if self.root==obj.root:
            out=RankedTree(self.root)
            i=0
            while i<len(self.children):
                out.children.append(self.children[i].gcp(obj.children[i]))
                i+=1
            return out
        else:
            return RankedTree("#")

    def getSymbolsToRanks(self):
        out={self.root:len(self.children)}
        if len(self.children)==0:
            return out
        else:
            for child in self.children:
                d=child.getSymbolsToRanks()
                for c in d.keys():
                    out[c]=d[c]
            return out

    def getStatePaths(self,state):
        if self.isLeaf():
            if isinstance(self.root,PairStateNumber) and self.root.getState()==state:
                return [Path([])]
            else:
                return []
        else:
            l=[]
            i=0
            while i<len(self.children):
                for x in self.children[i].getStatePaths(state):
                    l.append(Path([(self.root,i+1)])+x)
                i+=1
            return l

    def getBottomPaths(self):
        if self.isLeaf():
            if self.root=="#":
                return [Path([])]
            else:
                return []
        else:
            l=[]
            i=0
            while i<len(self.children):
                for x in self.children[i].getBottomPaths():
                    l.append(Path([(self.root,i+1)])+x)
                i+=1
            return l

    def getPairStateNumberPaths(self):
        if self.isLeaf():
            if isinstance(self.root,PairStateNumber):
                return [Path([])]
            else:
                return []
        else:
            l=[]
            i=0
            while i<len(self.children):
                for x in self.children[i].getPairStateNumberPaths():
                    l.append(Path([(self.root,i+1)])+x)
                i+=1
            return l

    def hasPath(self,u):
        if u.isEmpty():
            return True
        elif self.isLeaf():
            return False
        else:
            return u.getTopLetter()==self.root and self.children[u.getTopIndex()-1].hasPath(u.getSuccessor())

    def hasNPath(self,pair): # we denote npaths as a pair (path,symbol)
        u=pair[0]
        s=pair[1]
        if u.isEmpty():
            return s==self.root
        elif self.isLeaf():
            return False
        else:
            return u.getTopLetter()==self.root and self.children[u.getTopIndex()-1].hasNPath((u.getSuccessor(),s))

    def getRoot(self):
        return self.root

    def getSubtree(self,path):
        if path.isEmpty():
            return self
        else:
            if self.isLeaf():
                raise RuntimeError("Cannot get the subtree at a path which is not in this tree.")
            elif not path.getTopLetter()==self.root:
                raise RuntimeError("Cannot get the subtree at a path which is not in this tree.")
            else:
                return self.children[path.getTopIndex()-1].getSubtree(path.getSuccessor())

    def changeAtPath(self,path,new):
        if not self.hasPath(path):
            raise RuntimeError("Cannot change an unexisting path of a tree.")
        else:
            if path.isEmpty():
                return new
            else:
                out=RankedTree(self.root)
                out.children=self.children
                out.children[path.getTopIndex()-1]=self.children[path.getTopIndex()-1].changeAtPath(path.getSuccessor(),new)
                return out

def gcp(treeList):
    if len(treeList)==0:
        raise RuntimeError("Cannot compute the greatest common prefix of an empty set of trees.")
    elif len(treeList)==1:
        return treeList[0]
    else:
        out=treeList[0].gcp(treeList[1])
        for x in treeList[2:]:
            out=out.gcp(x)
        return out

class Path:
    def __init__(self,path):
        self.path=path
    def __eq__(self,obj):
        return str(self)==str(obj)
    def __neq__(self,obj):
        return not self==obj
    def __add__(self,obj): # path concatenation
        return Path(self.path+obj.path)
    def __str__(self):
        return str(self.path)
    def __lt__(self,obj):
        if len(self.path)!=len(obj.path):
            return len(self.path)<len(obj.path)
        else:
            return str(self.path)<str(obj.path)
    def __hash__(self):
        return hash(str(self.path))
    def getLength(self):
        return len(self.path)
    def isEmpty(self):
        return len(self.path)==0
    def getTopLetter(self):
        return self.path[0][0]
    def getTopIndex(self):
        return self.path[0][1]
    def getLastLetter(self):
        return self.path[-1][0]
    def getLastIndex(self):
        return self.path[-1][1]
    def getSuccessor(self,n=None):
        if n==None:
            n=1
        return Path(self.path[n:])
    def isStrictExtension(self,obj):
        if self.isEmpty()==True:
            return False
        elif obj.isEmpty()==True:
            return True
        else:
            return self.getTopLetter()==obj.getTopLetter() and self.getTopIndex()==obj.getTopIndex() and self.getSuccessor().isStrictExtension(obj.getSuccessor())
    
class Pair:
    def __init__(self,u,v):
        self.u=u
        self.v=v
    def __add__(self,obj):
        return Pair(self.u+obj.u,self.v+obj.v)
    def __eq__(self,obj):
        if obj==None:
            return False
        return self.u==obj.u and self.v==obj.v
    def __neq__(self,obj):
        return not self==obj
    def __lt__(self,obj):
        if self.u==obj.u:
            return self.v<obj.v
        else:
            return self.u<obj.u
    def __str__(self):
        return "("+str(self.u)+","+str(self.v)+")"
    def __hash__(self):
        return hash(str(self))

    def isIOPath(self,sample):
        maxPathOut=sample.maxPathOut(self.u)
        if maxPathOut==None:
            return False
        elif not maxPathOut.hasPath(self.v):
            return False
        else:
            return sample.getResidual(self)!=None and maxPathOut.getSubtree(self.v).getRoot()=="#"
    def getU(self):
        return self.u
    def getV(self):
        return self.v

class Alphabet:
    def __init__(self):
        self.letters=set()
        self.ranks={}
    def addSymbol(self,symbol,rank):
        if not symbol in self.letters:
            self.letters.add(symbol)
            self.ranks[symbol]=rank
        elif rank!=self.ranks[symbol]:
            raise RuntimeError("Every symbol must have a fixed number of descendents in every tree.")
        else:
            pass
    def getLetters(self):
        return self.letters
    def getRank(self,letter):
        return self.ranks.get(letter)

class Sample:
    def __init__(self):
        self.map={}
        self.in_ab=Alphabet()
        self.out_ab=Alphabet()

    def getInputFromConsole(self):
        done=False
        while done!=True:
            ex_input=input("Insert the next input:")
            ex_output=input("Insert the corresponding output:")
            if not (isGround(ex_input) and isGround(ex_output) and len(ex_input)>0 and len(ex_output)>0):
                raise RuntimeError("Make sure you insert ground nonempty terms.")
            self.addExample(ex_input,ex_output)
            areYouDone=input("Done?[y/n]")
            if areYouDone=="y":
                done=True
            elif areYouDone=="n":
                done=False
            else:
                raise RuntimeError("Invalid answer.")

    def addExample(self,ex_input,ex_output):
        inTree=RankedTree(ex_input)
        outTree=RankedTree(ex_output)
        inSymbolsToRanks=inTree.getSymbolsToRanks()
        for c in inSymbolsToRanks.keys():
            self.in_ab.addSymbol(c,inSymbolsToRanks.get(c))
        outSymbolsToRanks=outTree.getSymbolsToRanks()
        for c in outSymbolsToRanks.keys():
            self.out_ab.addSymbol(c,outSymbolsToRanks.get(c))
        if inTree in self.map.keys():
            raise RuntimeError("You have already introduced that input.")
        self.map[inTree]=outTree

    def getInputFromFile(self,fileName):
        with open(fileName,'r') as f:
            for line in f:
                if line[0]=="#":
                    continue
                if not "|" in line:
                    raise RuntimeError("Invalid line, could not find the separator |.")
                ex_input=line[:line.find("|")]
                ex_output=line[line.find("|")+1:-1]
                if not (isGround(ex_input) and isGround(ex_output) and len(ex_input)>0 and len(ex_output)>0):
                    raise RuntimeError("Make sure you insert ground nonempty terms.")
                self.addExample(ex_input,ex_output)

    def getInput(self):
        readSource=input("Should the input be read from the console or from a file? [i/f] ")
        if readSource=="f":
            fileName=input("Insert the file name: ")
            self.getInputFromFile(fileName)
        elif readSource=="c":
            self.getInputFromConsole()

    def getInputAlphabet(self):
        return self.in_ab

    def maxPathOut(self,u):
        l=[]
        for inp in self.map.keys():
            out=self.map.get(inp)
            if inp.hasPath(u):
                l.append(out)
        if len(l)==0:
            return None
        else:
            return gcp(l)

    def maxNPathOut(self,pair):
        l=[]
        for inp in self.map.keys():
            out=self.map.get(inp)
            if inp.hasNPath(pair):
                l.append(out)
        if len(l)==0:
            return None
        else:
            return gcp(l)

    def getResidual(self,pair): # returns None if the residual is non-functional
        residual={}
        u=pair.getU()
        v=pair.getV()
        for inp in self.map.keys():
            out=self.map.get(inp)
            if inp.hasPath(u):
                if out.hasPath(v):
                    subtreeInp=inp.getSubtree(u)
                    if subtreeInp in residual.keys():
                        if not residual[subtreeInp]==out.getSubtree(v):
                            return None
                    else:
                        residual[inp.getSubtree(u)]=out.getSubtree(v)
                else:
                    raise RuntimeError("It does not make sense to compute the residual of (u,v) if v is not in max-out(u).")
        if len(residual.keys())==0:
            return None
        return residual

    def getIOPaths(self):
        l1=[]
        l2=[Pair(Path([]),v) for v in self.maxPathOut(Path([])).getBottomPaths()]
        warning=False
        while len(l2)>0:
            pair=l2[0]
            for letter in self.in_ab.getLetters():
                if self.in_ab.getRank(letter)==0:
                    continue
                maxout=self.maxNPathOut((pair.getU(),letter))
                if maxout==None:
                    continue
                bottomPathList=maxout.getBottomPaths()
                for v in bottomPathList:
                    candidates=[]
                    i=1
                    while i<=self.in_ab.getRank(letter):
                        u=pair.getU()+Path([(letter,i)])
                        if Pair(u,v).isIOPath(self):
                            candidates.append(i)
                        i+=1
                    if len(candidates)>1: # we use an heuristic to select a candidate
                        if warning==False:
                            print("Warning: condition (O) fails to hold, therefore some IO-paths were ignored, but we could still proceed.")
                            warning=True
                        if not v.isEmpty() and v.getLastIndex() in candidates:
                            u=pair.getU()+Path([(letter,v.getLastIndex())])
                        else:
                            u=pair.getU()+Path([(letter,min(candidates))])
                        l2.append(Pair(u,v))
                    elif len(candidates)==1:
                        u=pair.getU()+Path([(letter,candidates[0])])
                        l2.append(Pair(u,v))
                    #elif not v in self.maxPathOut(pair.getU()).getBottomPaths():
                    else:
                        raise RuntimeError("Either this function is not top-down or some IO-path of the function is not an IO-path of the sample.")
            l2.remove(pair)
            l1.append(pair)
        return l1

    def partition(self,ioPathList):
        states=[]
        warning=False
        while len(ioPathList)>0:
            p=min(ioPathList)
            q=State([p],self.getResidual(p))
            for state in states:
                candidates=[]
                if q.isMergeable(state):
                    candidates.append(state)
                if len(candidates)>0:
                    index=states.index(min(candidates))
                    states[index]+=q
                    if len(candidates)>1:
                        warning=True
                    break
            else:
                states.append(q)
            ioPathList.remove(p)
        if warning:
            print("Warning: condition (N) fails to hold.")
        return set(states)

    def computeRHS(self,states):
        rhs={}
        for q in states:
            rhs[q]={}
            for letter in self.in_ab.getLetters():
                rank=self.in_ab.getRank(letter)
                gcpArg=[self.maxNPathOut((p.getU(),letter)).getSubtree(p.getV()) for p in q.getIOPathList() if self.maxNPathOut((p.getU(),letter))!=None]
                if len(gcpArg)==0:
                    raise RuntimeError("Could not compute the transition function for the symbol {0} at the state {1}, therefore condition .".format(letter,str(q)))
                rhs[q][letter]=gcp(gcpArg)
                l=rhs.get(q).get(letter).getBottomPaths()
                for bottomPath in l:
                    for state in states:
                        for x in state.getIOPathList():
                            for p in q.getIOPathList():
                                if x.getV()==p.getV()+bottomPath and x.getU().isStrictExtension(p.getU()):
                                    index=state.getIOPathList().index(x)
                                    i=state.getIOPathList()[index].getU().getLastIndex()
                                    rhs[q][letter]=rhs.get(q).get(letter).changeAtPath(bottomPath,RankedTree(PairStateNumber(state,i)))
        return rhs

    def computeAxiom(self,states):
        axiom=self.maxPathOut(Path([]))
        l=axiom.getBottomPaths()
        for bottomPath in l:
            for state in states:
                if Pair(Path([]),bottomPath) in state.getIOPathList():
                    axiom=axiom.changeAtPath(bottomPath,RankedTree(PairStateNumber(state,0)))
        return axiom

    def hasFunctionalResidual(self,pair):
        maxPathOut=self.maxPathOut(pair.u)
        if maxPathOut==None:
            return False
        elif not maxPathOut.hasPath(pair.v):
            return False
        else:
            return self.getResidual(pair)!=None

def isGround(term):
    c=0
    for x in term:
        if x=="(":
            c+=1
        elif x==")":
            c-=1
        if c<0:
            return False
    return c==0

class State:
    def __init__(self,ioPathList,residual):
        self.ioPathList=ioPathList
        self.residual=residual
    def __str__(self):
        return str(min(self.ioPathList))
    def __eq__(self,obj):
        return self.ioPathList[0]==obj.ioPathList[0]
    def __neq__(self,obj):
        return not self==obj
    def __hash__(self):
        return hash(self.ioPathList[0])
    def __lt__(self,obj):
        return min(self.ioPathList)<min(obj.ioPathList)
    def __add__(self,obj):
        newRes={}
        for x in self.residual:
            newRes[x]=self.residual.get(x)
        for x in obj.residual:
            newRes[x]=obj.residual.get(x)
        return State(self.ioPathList+obj.ioPathList,newRes)
    def isMergeable(self,obj):
        for inp in self.residual.keys():
            if inp in obj.residual.keys() and not obj.residual[inp]==self.residual[inp]:
                return False
        return True
    def addPath(self,ioPath,residual):
        self.ioPathList.add(ioPath)
        for inp in residual.keys():
            if inp in self.residual.keys():
                if not residual.get(inp)==self.residual.get(inp):
                    raise RuntimeError("Cannot merge in the same state two io-paths whose combined residual is not functional; this set cannot be characteristic.")
            else:
                self.residual[inp]=residual.get(inp)
    def getIOPathList(self):
        return self.ioPathList
    def getResidual(self):
        return self.residual

class PairStateNumber:
    def __init__(self,state,number):
        self.state=state
        self.number=number
    def __str__(self):
        return "<"+str(self.state)+","+str(self.number)+">"
    def getState(self):
        return self.state
    def getNumber(self):
        return self.number

class Transducer:
    def __init__(self,states,rhs,axiom,input_ab):
        self.states=states
        self.rhs=rhs
        self.axiom=axiom
        self.input_ab=input_ab
    def __str__(self):
        out="States:\n"
        if len(self.states)>0:
            for state in self.states:
                out+=str(state)+","
            out=out[:-1]
        out+="\n"
        out+="Axiom:\n"+str(self.axiom)+"\n"
        out+="Transition function:"
        for state in self.states:
            for letter in self.input_ab.getLetters():
                out+="\n("+str(state)+","+letter+") ---> "+str(self.rhs.get(state).get(letter))
        return out
    def run(self,inputTree):
        out=deepcopy(self.axiom)
        statePaths=out.getPairStateNumberPaths()
        for v in statePaths:
            state=out.getSubtree(v).getRoot().getState()
            subOut=self.runFromState(inputTree,state)
            if subOut==None:
                return None
            out=out.changeAtPath(v,subOut)
        return out
    def runFromState(self,inputTree,state):
        if self.rhs.get(state)==None:
            return None
        if self.rhs.get(state).get(inputTree.getRoot())==None:
            return None
        out=deepcopy(self.rhs.get(state).get(inputTree.getRoot()))
        statePaths=out.getPairStateNumberPaths()
        for v in statePaths:
            newState=out.getSubtree(v).getRoot().getState()
            successorNumber=out.getSubtree(v).getRoot().getNumber()
            successorInput=inputTree.getSubtree(Path([(inputTree.getRoot(),successorNumber)]))
            subOut=self.runFromState(successorInput,newState)
            if subOut==None:
                return None
            out=out.changeAtPath(v,subOut)
        return out
    def getAxiom(self):
        return self.axiom
    def getStates(self):
        return self.states
    def getRHS(self):
        return self.rhs
    def getState(self,p):
        for state in self.states:
            if p in state.ioPathList:
                return state
        return None
    def mergeStates(self,q1,q2):
        trans=deepcopy(self)
        trans.states.remove(q1)
        trans.states.remove(q2)
        del trans.rhs[q1]
        del trans.rhs[q2]
        newRes={}
        for x in q1.residual.keys():
            newRes[x]=q1.residual.get(x)
        for x in q2.residual.keys():
            if x in newRes.keys():
                newRes[x]=q2.residual.get(x).gcp(q1.residual.get(x))
                for v in newRes.get(x).getBottomPaths():
                    if isinstance(q1.residual.get(x).getSubtree(v).getRoot(),PairStateNumber):
                        newRes[x]=newRes.get(x).changeAtPath(v,q1.residual.get(x).getSubtree(v))
                    else:
                        newRes[x]=newRes.get(x).changeAtPath(v,q2.residual.get(x).getSubtree(v))
            else:
                newRes[x]=q2.residual.get(x)
        newState=State(q1.ioPathList+q2.ioPathList,newRes)
        trans.states.add(newState)
        trans.rhs[newState]={}
        for x in self.rhs.get(q1).keys():
            trans.rhs[newState][x]=deepcopy(self.rhs.get(q1).get(x))
        for x in self.rhs.get(q2).keys():
            if x in self.rhs.get(q1).keys():
                trans.rhs[newState][x]=deepcopy(self.rhs.get(q1).get(x).gcp(self.rhs.get(q2).get(x)))
                for v in trans.rhs.get(newState).get(x).getBottomPaths():
                    if isinstance(self.rhs.get(q1).get(x).getSubtree(v).getRoot(),PairStateNumber):
                        trans.rhs[newState][x]=trans.rhs.get(newState).get(x).changeAtPath(v,deepcopy(self.rhs.get(q1).get(x).getSubtree(v)))
                    else:
                        trans.rhs[newState][x]=trans.rhs.get(newState).get(x).changeAtPath(v,deepcopy(self.rhs.get(q2).get(x).getSubtree(v)))
            else:
                trans.rhs[newState][x]=deepcopy(self.rhs.get(q2).get(x))
        for q in trans.rhs.keys():
            for letter in trans.input_ab.getLetters():
                if trans.rhs.get(q).get(letter)==None:
                    continue
                for v in trans.rhs.get(q).get(letter).getStatePaths(q1):
                    trans.rhs[q][letter]=trans.rhs.get(q).get(letter).changeAtPath(v,RankedTree(PairStateNumber(newState,trans.rhs.get(q).get(letter).getSubtree(v).getRoot().getNumber())))
                for v in trans.rhs.get(q).get(letter).getStatePaths(q2):
                    trans.rhs[q][letter]=trans.rhs.get(q).get(letter).changeAtPath(v,RankedTree(PairStateNumber(newState,trans.rhs.get(q).get(letter).getSubtree(v).getRoot().getNumber())))
        for v in trans.axiom.getStatePaths(q1):
            trans.axiom=trans.axiom.changeAtPath(v,RankedTree(PairStateNumber(newState,0)))
        for v in trans.axiom.getStatePaths(q2):
            trans.axiom=trans.axiom.changeAtPath(v,RankedTree(PairStateNumber(newState,0)))
        return trans

def learnTD(sample):
    states=sample.partition(sample.getIOPaths())
    rhs=sample.computeRHS(states)
    axiom=sample.computeAxiom(states)
    input_ab=sample.getInputAlphabet().getLetters()
    return Transducer(states,rhs,axiom,input_ab)

def learnTD2(sample,currentIOPath=None):
    if currentIOPath==None:
        axiom=sample.maxPathOut(Path([]))
        l=axiom.getBottomPaths()
        states=set()
        rhs={}
        for p in l:
            t=learnTD2(sample,Pair(Path([]),p))
            if t==None:
                raise RuntimeError("This function is not top-down.")
            states=states.union(t.getStates())
            axiom=axiom.changeAtPath(p,t.getAxiom())
            rhs=dict(list(rhs.items())+list(t.getRHS().items()))
        return minimize(Transducer(states,rhs,axiom,sample.getInputAlphabet()),sample)
    else:
        residual=sample.getResidual(currentIOPath)
        newState=State([currentIOPath],residual)
        domain=sample.getInputAlphabet().getLetters()
        rhs={}
        rhs[newState]={}
        states=set([newState])
        axiom=RankedTree(PairStateNumber(newState,0))
        for letter in domain:
            t=sample.maxNPathOut((currentIOPath.getU(),letter))
            if t==None:
                continue
            rhs[newState][letter]=t.getSubtree(currentIOPath.getV())
            for p in rhs.get(newState).get(letter).getBottomPaths():
                i=1
                candidates=[]
                while i<=sample.getInputAlphabet().getRank(letter):
                    suffix=Pair(Path([(letter,i)]),p)
                    pair=currentIOPath+suffix
                    if sample.hasFunctionalResidual(pair):
                        candidates.append(i)
                    i+=1
                for c in candidates:
                    suffix=Pair(Path([(letter,c)]),p)
                    trans=learnTD2(sample,currentIOPath+suffix)
                    if trans!=None:
                        rhs=dict(list(rhs.items())+list(trans.getRHS().items()))
                        states=states.union(trans.getStates())
                        rhs[newState][letter]=rhs.get(newState).get(letter).changeAtPath(p,RankedTree(PairStateNumber(trans.getAxiom().getRoot().getState(),c)))
                        break
                else:
                    return None #not a topdown set
        return minimize(Transducer(states,rhs,axiom,sample.getInputAlphabet()),sample)

def minimize(trans,sample):
    i=0
    states=list(trans.getStates())
    while i<len(states):
        j=i+1
        while j<len(states):
            newTrans=trans.mergeStates(states[i],states[j])
            for inp in sample.map.keys():
                if not sample.map.get(inp)==newTrans.run(inp):
                    j+=1
                    break
            else:
                trans=newTrans
                del states[j]
                states[i]=trans.getState(states[i].ioPathList[0])
        i+=1
    return trans
    

def main():
    print("This program learns total deterministic top-down tree transducers from examples.\nPlease note that # is reserved as an internal symbol, so it should not be included in any input.\nDon't use any separators that are not requested, eg. spaces. Don't use brackets after rank-zero symbols.\nShould you wish to read the sample from a file, please save it in a file called sample in the same directory as this program; each line should have a input/output pair, separated by the symbol |. Insert all trees in an analogous form: a(b,c(d,e)).")
    try:
        sample=Sample()
        sample.getInput()
        transducer=learnTD2(sample)
        for inp in sample.map.keys():
            if not sample.map.get(inp)==transducer.run(inp):
                raise RuntimeError("The learnt transducer does not match the given output on example {0}. If no warning message was produced, this means that condition (N) fails to hold.".format(str(inp)))
        print(transducer)
        return 0
    except RuntimeError as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
        return 1

if __name__ == "__main__":
    sys.exit(main())
