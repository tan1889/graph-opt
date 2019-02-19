"""Generate synthetic concave graph: Nodes of the graph is the grid [x, y] where x,y in {-R...-1,0,1... R}.
Each node is connected to all of its neighbors including diagonal.
f.append(0.5 - (x**2 + y**2) / (4.*R**2)) this is a parabola f range is [0, 0.5]. It is easy to see that this graph
is m-strongly concave with m = 2 / (2R-3) """
import pickle
import random


def neighbors(i):
    x0, y0 = V[i]
    nbs = []
    for a in [-1, 0, 1]:
        for b in [-1, 0, 1]:
            if (a != 0 or b != 0) and -R <= x0 + a <= R and -R <= y0 + b <= R:
                nbs.append([x0+a, y0+b])
    return nbs


def get_node_id(x, y):
    def bin_search(i, j):
        if i + 1 >= j:
            if V[i][0] == x and V[i][1] == y:
                return i
            elif V[j][0] == x and V[j][1] == y:
                return j
            raise Exception('something is wrong, could not find the index')
        k = (i+j)//2
        if x < V[k][0] or (x == V[k][0] and y <= V[k][1]):
            return bin_search(i, k)
        else:
            return bin_search(k, j)
    return bin_search(0, N-1)


F_MAX = 0.8

# for N = 1e3, 1e4, 1e5, 1e6 set R = 16, 50, 158, 500
for R in [10, 100]:  # radius, x, y in {-R, ...,-1, 0, 1, ... R}
    filename = 'data/synthetic_graph_r{}.pkl'.format(R)

    N = (2*R + 1)**2    # number of nodes

    V = []
    for x in range(-R, R+1):
        for y in range(-R, R+1):
            V.append([x, y])

    assert len(V) == N, 'something is wrong, graph size does not match!'

    A = [[] for _ in range(N)]
    for i in range(N):
        # add the 8 neighboring nodes
        for x, y in neighbors(i):
            j = get_node_id(x, y)
            assert j is not None and V[j][0] == x and V[j][1] == y, 'get_node_id returned incorrect node'
            A[i].append(j)
        # add random edges so that the degree is 15
        while 15 - len(A[i]) > 0:
            j = i
            while j == i or j in A[i]:
                j = random.randint(0, N-1)
            A[i].append(j)
            A[j].append(i)
        # assert len(A[i]) == 15, 'something is wrong, degree should be 15'

    f = []

    for i in range(N):
        x, y = V[i]
        f.append(F_MAX - F_MAX * (x**2 + y**2) / (2.*R**2))
        if f[-1] < 0:
            print('negative f, something is not right')

    with open(filename, "wb") as f_out:
        pickle.dump((f, A), f_out)
    print('The graph is written to', filename)
