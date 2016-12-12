__author__ = 'daksh'

from collections import deque
from deepLearningLibrary.connections import *

GRAY, BLACK = 0, 1

def constructGraph(layers):
    g = {}
    for i in layers:
        g[i] = [connection.toLayer for connection in i.outConnections
                if not isinstance(connection,RecurrentConnection)]
    return g


def topological(graph):
    order, enter, state = deque(), set(graph), {}

    def dfs(node):
        state[node] = GRAY
        for k in graph.get(node, ()):
            sk = state.get(k, None)
            if sk == GRAY:
                raise(NetworkNotDAG())
            if sk == BLACK: continue
            enter.discard(k)
            dfs(k)
        order.appendleft(node)
        state[node] = BLACK

    while enter: dfs(enter.pop())
    return list(order)