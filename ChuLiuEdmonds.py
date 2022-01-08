import numpy as np
from itertools import product, combinations

# ======================================================================
def possible_paths(start, edges):
  step_performed = False
  paths_arr = []
  for i,e in enumerate(edges):
    if e[0] == start:
      paths = possible_paths(e[1], [x for j,x in enumerate(edges) if j != i])
      if len(paths) == 0:
        paths_arr.append( [e] )
      for p in paths:
        paths_arr.append( [e]+p )

  return paths_arr
# ======================================================================


# ======================================================================
def find_cycle(v_arr, e_arr):
  step_from = [-1] * len(e_arr)
  step_to = [-1] * len(e_arr)
  new_e_arr = []
  for i,e in enumerate(e_arr):
    for j,v in enumerate(v_arr):
      if e[0] in v:
        step_from[i] = j
      if e[1] in v:
        step_to[i] = j
      if step_from[i] != -1 and step_to[i] != -1:
        new_e_arr.append( (step_from[i], step_to[i], i) )
        break

  paths_arr = []
  for i,first_step in enumerate(new_e_arr):
    edges = [e for j,e in enumerate(new_e_arr) if j != i]
    paths = possible_paths(first_step[1], edges)
    if len(paths) == 0:
      paths_arr.append( [first_step] )
    for p in paths:
      paths_arr.append( [first_step] + p )

  cycle = []
  for path in paths_arr:
    visited = [path[0][0]]
    for i, step in enumerate(path):
      if step[1] in visited:
        if i+1 > len(cycle):
          cycle = path[:i+1]
        break
      visited.append( step[1] )
  
  return [e_arr[x[2]] for x in cycle]
# ======================================================================


# ======================================================================
def is_arborescence(v_arr, e_arr):
  times_visited = np.zeros( len(v_arr) )

  for e in e_arr:
    for i,v in enumerate(v_arr):
      if e[1] in v:
        times_visited[i] += 1
        if times_visited[i] > 1:
          return False

  if sum(times_visited) != len(v_arr)-1:
    return False

  # find root
  for i,x in enumerate(times_visited):
    if x == 0:
      root = i
      break

  # build arborescence
  from_v = [root]
  to_v = [i for i in range(len(v_arr)) if i != root]

  step_from = [-1] * len(e_arr)
  step_to = [-1] * len(e_arr)
  for i,e in enumerate(e_arr):
    for j,v in enumerate(v_arr):
      if e[0] in v:
        step_from[i] = j
      if e[1] in v:
        step_to[i] = j
      if step_from[i] != -1 and step_to[i] != -1:
        break

  used = [False]*len(e_arr)
  while True:
    step_performed = False
    for i,e in enumerate(e_arr):
      if used[i]:
        continue

      if (step_from[i] in from_v) and (step_to[i] in to_v):
        step_performed = True
        used[i] = True
        from_v.append( step_to[i] )
        to_v.remove( step_to[i] )

    if all(used):
      return True
    
    if not step_performed:
      return False
# ======================================================================


# ======================================================================
# This is about fully connected graph
class graph_cls():
  def __init__(self):
    self.v = []
    self.e = []


  def add_vertices(self, v_arr):
    for v in v_arr:
      self.v.append( [v] )


  def add_edges(self, e_arr ):
    for e in e_arr:
      self.e.append( e )


  def indx_from_to(self, edge):
    indx_from = -1
    indx_to = -1
    for i,v in enumerate(self.v):
      if edge[0] in v:
        indx_from = i
      if edge[1] in v:
        indx_to = i
      if indx_from != -1 and indx_to != -1:
        return indx_from, indx_to


  def choose_root(self):
    smm = [0]*len(self.v)
    for e in self.e:
      indx_from, indx_to = self.indx_from_to(e)
      smm[indx_to] += e[2]
      
    root = max(range(len(smm)), key=smm.__getitem__)
    return root


  def best_edges(self):
    # lowest scoring incoming edge
    bst = [-1]*len(self.v)
    for i,e in enumerate(self.e):
      for indx_v,v in enumerate(self.v):
        if e[1] in v:
          break
          
      if bst[indx_v] != -1:
        if e[2] < self.e[bst[indx_v]][2]:
          bst[indx_v] = i
      else:
        bst[indx_v] = i

    return bst
    

  def MWSArborescence(self, root=None):
    if root == None:
      root = self.choose_root()
    #print("\nroot = ", root)

    # find best edges
    bst = self.best_edges()
    #print("bst:", bst)
    e_bst = [self.e[j] for i,j in enumerate(bst) if not root in self.v[i]]
    #print("e:",e_bst )
    w_bst = [self.e[i][2] for i in bst]

    # check if these edges form the arborescence
    if is_arborescence(self.v, e_bst):
      #print("Arborescence!!!")
      return e_bst
    #else: 
      #print("arborescence: False")
    #print("arborescence:", is_arborescence(self.v, e_bst) )

    # find any cycle
    e_cycle = find_cycle( self.v, e_bst )
    #print("cycle:", e_cycle)
    v_cycle = []
    indx_arr = []
    for e in e_cycle:
      indx_from, indx_to = self.indx_from_to(e)
      if not indx_from in indx_arr:
        indx_arr.append( indx_from )
        for v in self.v[indx_from]:
          v_cycle.append( v )
      if not indx_to in indx_arr:
        indx_arr.append( indx_to )
        for v in self.v[indx_to]:
          v_cycle.append( v )
    #print("cycle verteces:", v_cycle)

    # create new graph
    graph = graph_cls()
    graph.v.append( v_cycle )
    for v in self.v:
      if len(set(v) & set(v_cycle)) != 0:
        continue
      graph.add_vertices( v )
    #for v in graph.v:
      #print(v)
    for e in self.e:
      if e[0] in v_cycle and e[1] in v_cycle:
        continue
      if (not e[0] in v_cycle) and (e[1] in v_cycle):
        indx_from, indx_to = self.indx_from_to(e)
        graph.e.append( (e[0], e[1], e[2]-w_bst[indx_to]) )
        continue
      if (e[0] in v_cycle) and (not e[1] in v_cycle):
        graph.e.append( e )
        continue
      if (not e[0] in v_cycle) and (not e[1] in v_cycle):
        graph.e.append( e )
        continue
    
    #print("new graph:")
    #graph.prnt()
    sub_graph_e_bst = graph.MWSArborescence(root=root)
    #print("sub graph:", sub_graph_e_bst)
    # restore weights
    for i,sub_e in enumerate(sub_graph_e_bst):
      for e in self.e:
        if e[0] == sub_e[0] and e[1] == sub_e[1]:
          sub_graph_e_bst[i] = e
          break
    #print("restored: ", sub_graph_e_bst)
    
    sub_graph_e_bst_to = []
    for e in sub_graph_e_bst:
      indx_from, indx_to = self.indx_from_to(e)
      sub_graph_e_bst_to.append( indx_to )

    # remove from e_bst
    remove = []
    for i,e in enumerate(e_bst):
      indx_from, indx_to = self.indx_from_to(e)
      if indx_to in sub_graph_e_bst_to:
        remove.append( i )
    
    for i in sorted(remove,reverse=True):
      e_bst.pop(i)
    return e_bst + sub_graph_e_bst
    

  def prnt(self):
    for v in self.v:
      print(v)
      for i,e in enumerate(self.e):
        if e[1] in v:
          #print("G.add_edges( [", e,"] )")
          print(e)
# ======================================================================


# ======================================================================
def find_MWSArborescence(weights_mtrx):
  G = graph_cls()

  nv = weights_mtrx.shape[0]
  G.add_vertices( [i for i in range(nv)] )

  for i in range(nv-1):
    for j in range(i+1,nv):
      G.add_edges( [(i,j,weights_mtrx[i,j]), (j,i,weights_mtrx[j,i])] )

  bst_score = np.amax(weights_mtrx)*(nv-1)+1
  for root in range(nv):
    tst = G.MWSArborescence(root=root)
    score = sum([x[2] for x in tst])
    if score < bst_score:
      bst_score = score
      arborescence = tst
      bst_root = root

  # sort arborescence
  arborescence_sorted = []

  v_visited = [bst_root]
  used = [False]*(nv-1)
  while not all(used):
    add = []
    for i,e in enumerate(arborescence):
      if used[i]:
        continue
      for v in v_visited:
        if e[0] == v:
          add.append( e[1] )
          arborescence_sorted.append( e )
          used[i] = True
    v_visited += add

  return arborescence_sorted
# ======================================================================


# Straightforward search of the MWSArborescence ========================
def straightforward_test(weights_mtrx, arborescence):
  G = graph_cls()

  nv = weights_mtrx.shape[0]
  G.add_vertices( [i for i in range(nv)] )

  for i in range(nv-1):
    for j in range(i+1,nv):
      G.add_edges( [(i,j,weights_mtrx[i,j]), (j,i,weights_mtrx[j,i])] )

  #print("\nstraightforward_test")
  def variants(npaths, v_from, v_to):
    #print("n = ", npaths, v_from, v_to)
    arr = []
    if npaths == len(v_to):
      v_in = list( combinations(v_to,npaths))
      #print("0! v_in = ", v_in[0])
      for v_out in list( product(v_from,repeat=npaths) ):
        arr.append( [(list(v_out)[i],v_in[0][i]) for i in range(npaths)] )
      #print("0! arr = ", arr)
      return arr
    else:
      v_in = list( combinations(v_to,npaths))
      for v_out in list( product(v_from,repeat=npaths) ):
        for v in v_in:
          add = [(v_out[i],v[i]) for i in range(npaths)]

          v_from_new = list(v)
          v_to_new = list(set(v_to).difference(set(v)))
          #print("add = ", add, "new",v_from_new,v_to_new)

          for n in range(1,len(v_to_new)+1):
            for tmp in variants(n, v_from_new, v_to_new ):
              #print(add + list(tmp))
              arr.append( add + tmp )
      return arr

  # create all possible arborescences with weights and store
  # the best
  score = np.amax(weights_mtrx)*(nv-1)+1
  bst_edges = []
  for r in range(nv):
  #for r in range(1,2):
    #print("r = ", r)
    v_arr = [i for i in range(nv) if i != r]
    for npaths in range(1,len(v_arr)+1):
    #for npaths in range(2,3):
      for var in variants(npaths, [r], v_arr):
        weight = sum([weights_mtrx[e[0],e[1]] for e in var])
        #print(var, weight)
        if weight <= score:
          score = weight
          bst_edges.append( (var,weight) )

  # sort arborescences and check whether the input one
  # is in
  bst_sorted = sorted( bst_edges, key=lambda x: x[1] )
  for x in bst_sorted:
    #print(x)
    if x[1] != bst_sorted[0][1]:
      return False
    match = True
    for a in x[0]:
      Found = False
      for e in arborescence:
        if e[0] == a[0] and e[1] == a[1]:
          Found = True
          break
      if not Found:
        #print("Not Found = ", a)
        match = False
        break
    if match:
      return True
# ======================================================================
