"""
Check if a graph is convex. If not, print values of alpha, c, r, for which the graph is
(alpha, c, r) nearly-convex. Refer to our paper for definitions of convex and nearly-convex
graphs. Note that "convex" means "concave" in this case as we are maximizing!

Please quote our paper in your work: T Nguyen, Y Abbasi-Yadkori, B Kveton, A Shameli, A Rao.
"Sample Efficient Graph-Based Optimization with Noisy Observations."
Artificial Intelligence and Statistics. 2019.

"""
import pickle
import numpy as np

# load data
filename = 'graph_abtest.pkl'
with open('data/' + filename, "rb") as f_in:
    f, A = pickle.load(f_in)


# ===== General info: size, function values, degrees ================================ #
f = np.array(f)
n = f.shape[0]  # number of nodes
max_node = np.argmax(f)  # node with f_max
max_f = f[max_node]

print('Graph name:  {}'.format(filename))
print('Graph size:  {} nodes'.format(n))
print('Function values:  mean={:.3f}  std={:.3f}  min={:.3f}  max={:.3f} @ node {}'.format(
    f.mean(), f.std(), f.min(), f.max(), np.argmax(f)
))

degrees = np.array([len(A[i]) for i in range(n)])  # Graph degree
print('Degree of nodes:  mean={:.1f}   std={:.1f}   min={}   max={}'.format(
    degrees.mean(), degrees.std(), degrees.min(), degrees.max()
))

for x in range(n):
    for y in A[x]:
        if x not in A[y]:
            print('Graph is directed, which is unexpected! E.g. edge ({},{}) is unidirectional!'.format(x, y))
            quit()

print('Graph is undirected (as supposed)!')


# ===== Convex check: Is the graph convex? ========================================== #

# return the largest constant m among all convex path from start_node to end_node
def find_most_convex_path(start_node, end_node):
    global f, A
    stack, path, m_path, max_path = [[(start_node, float('inf'))]], [], [], {'m': -1.0, 'path': None}
    while stack:
        if not stack[-1]:  # stack head is empty, pop and reduce path
            stack.pop()
            if path:
                path.pop()
                m_path.pop()
        else:
            node, m = stack[-1].pop()
            path.append(node)
            m_path.append(m)
            if node == end_node:  # found a path to end_node
                if m > max_path['m']:
                    max_path = {'m': m, 'path': path.copy()}
            nnodes = []
            for nn in A[node]:
                d2 = f[nn] - f[node]
                d1 = float('inf')
                if len(path) > 1:
                    d1 = f[node] - f[path[-2]]
                if 0 < d2 < d1:
                    m_nn = min(m, (d1-d2)/d2)
                    if m_nn > max_path['m']:
                        nnodes.append((nn, m_nn))
            stack.append(nnodes)
    return max_path['m'], max_path['path']


# convex constant m of the graph
m_convex = float('inf')
for x in range(n):
    m, _ = find_most_convex_path(x, max_node)
    if m < m_convex:
        m_convex = m
    if m_convex <= 0:
        break

if m_convex > 0:
    print('The graph is convex! Convexity constant m={:.5f}'.format(m_convex))
    quit()


# ===== Non-convex info: No of maxima, cluster sizes ================================ #
print('The graph is not convex!')


# from start_node find the hill climbing path which achieve highest function value at the end
def hill_climb(start_node):

    global f, A, max_node
    stack, path, visited_nodes, max_path = [[start_node]], [], [], [start_node]
    while stack:
        if not stack[-1]:  # stack head is empty, pop and reduce path
            stack.pop()
            if path:
                path.pop()
        else:
            node = stack[-1].pop()
            if node not in visited_nodes:
                visited_nodes.append(node)
                path.append(node)
                if node == max_node:  # found a path to global maximum -> stop
                    return path
                nnodes = [nn for nn in A[node] if f[nn] > f[node] and nn not in visited_nodes]
                stack.append(nnodes)
                if not nnodes and f[max_path[-1]] < f[path[-1]]:  # path ended, if better -> store to max_path
                        max_path = list(path)

    return max_path


# build max hill climbing path for all nodes
H = []  # H[i] = [h, path] max achievable value (and the path) by hill climbing from node i
for i in range(n):
    path = hill_climb(i)
    H.append([f[path[-1]].item(), path])

# check nodes that failed convexity conditions
maxima = {max_node: {'f': max_f, 'counter': 1}}
for i in range(n):
    nmax = H[i][1][-1]  # maximum node achievable from node i
    if nmax not in maxima:
        maxima[nmax] = {'f': f[nmax], 'counter': 0}
    maxima[nmax]['counter'] += 1

# maxima
mxs = [i for i in maxima]
print('There are {} maxima, ranked as follows:'.format(len(mxs)))
print('(Cluster is the set of all nodes that could reach the given maximum via some ascending path)')
mxs.sort(key=lambda i: f[i], reverse=True)
for i in range(len(mxs)):
    print('    #{}\t\tnode={}\tf={:.4f}\tcluster_size={}'.format(i+1, mxs[i], f[mxs[i]], maxima[mxs[i]]['counter']))


# ===== Nearly-convex parameters ==================================================== #

def max_drop(x):
    global f, A
    md = max([f[z] - f[x] for z in A[x]])
    if md < 0:
        md = 0
    return md


delta = [max_drop(x) for x in range(n)]


def alpha_set(alpha):
    global f, max_f, delta
    aset = []
    for x in range(n):
        if x == max_node or delta[x] >= alpha * (max_f - f[x]):
            aset.append(x)
    return aset


def min_max_c_path_to_concave_set(start_node, max_len, aset):
    stack, path, c_path, min_path = [[(start_node, 0.)]], [], [], {'c': float('inf'), 'path': None}
    while stack:
        if not stack[-1]:  # stack head is empty, pop and reduce path
            stack.pop()
            if path:
                path.pop()
                c_path.pop()
        else:
            node, c = stack[-1].pop()
            path.append(node)
            c_path.append(c)
            if node in aset:  # found a path to cset
                if c < min_path['c']:
                    min_path = {'c': c, 'path': path.copy()}
            nnodes = []
            for nn in A[node]:
                if len(path) < max_len and nn not in path:
                    c_nn = max(f[nn] - f[start_node], c)
                    if c_nn < min_path['c']:
                        nnodes.append((nn, c_nn))
            stack.append(nnodes)
            if len(path) > 15:
                print(c, path)
    return min_path['c'], min_path['path']


# build minimax path P[c, p] is the minimax value of c and the path. P[i] = [-1, None] if path to cset
def find_nearly_convex_set(max_r, aset):
    # print('\nFinding near concave set with max_path_len =', max_r)
    r = c = -1
    P = [[-1, None] for _ in range(n)]  #P[i][0] is c of node i, P[i][1] is the path
    nearly_csize = 0
    for i in range(n):
        if i not in aset:  # node is is not in concave set containing max_f
            c_i, path_i = min_max_c_path_to_concave_set(i, max_r, aset)
            if path_i is not None:
                # print('node={} c={:.5f} p={}'.format(i, c_i, path_i))
                P[i] = [c_i, path_i]
                nearly_csize += 1
                if c_i > c:
                    c = c_i
                if len(path_i) > r:
                    r = len(path_i)

    # print('Nearly concave set size: {} ({:.1f}%),  r={},  c={:.5f}'.format(
    #    len(aset) + nearly_csize, 100*(len(aset) + nearly_csize)/n, r, c))
    return r, c, len(aset) + nearly_csize


ncsets = []
for a in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for max_r in [2,3,4]:
        r, c, ncset_count = find_nearly_convex_set(max_r, alpha_set(a))
        ncsets.append([a, r, c, ncset_count])
        print('alpha={} r={} c={:.4f} size={} ({:.1f}%)'.format(a, r, c, ncset_count, 100.*ncset_count/n))
# pickle.dump(ncsets, open('data/ncsets.pkl', 'wb'))
