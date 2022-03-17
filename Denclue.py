import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import collections as c
import networkx as nx
with open('/Users/lvjiazhi/PycharmProjects/pythonProject4/data/7.in') as f:
    lines = f.readlines()
    x, y = [], []
    for line in lines:
        a, b = list(map(float, line.strip().split()))
        x.append(a)
        y.append(b)
    x = np.array(x)
    y = np.array(y)
plt.scatter(x,y)
plt.show();
H = 0.5  # Smoothing parameter
K = 0.5  # Linkage distance
DELTA = 0.2  # Speed of convergence
XI = 0.01  # Denoising parameter
MAX_TIMES = 50
def sqrs(x):
    return (x ** 2).sum()


def slen(x):
    return sqrs(x) ** 0.5


def dist(x, y):
    return sqrs(x - y) ** 0.5


def k_gauss(x):
    return math.exp(-0.5 * sqrs(x)) / math.sqrt((2 * math.pi))


def get_z(x, y, f):
    m, n = x.shape
    z = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            z[i, j] = f(x[i, j], y[i, j])
    return z
class Denclue2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = len(x)
        assert(self.n == len(y))
        self.ps = [np.array([self.x[i], self.y[i]]) for i in range(self.n)]
        self.attrs = []
        self.bel = []
        self.is_out = []
        self.cluster_id = []

    def render_dens_fig(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        EPN = 50
        X = np.linspace(0, 10, EPN)
        Y = np.linspace(0, 10, EPN)
        X, Y = np.meshgrid(X, Y)
        Z = get_z(X, Y, lambda x, y: self.f_gauss(np.array([x, y])))
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        #plt.savefig(path)
        plt.show()

    def f_gauss(self, x): #密度值
        s = 0
        for p in self.ps:
            s += k_gauss((x - p) / H)
        return s / (self.n * (H ** 2))
    def delta_f(self, x):
        s = np.array([0., 0.])
        for p in self.ps:
            s += k_gauss((x - p) / H) * (p - x)
        return s / ((H ** 4) * self.n)
    def next_pos(self, x):
        d = self.delta_f(x)
        return x + d * DELTA / slen(d)
    def get_max(self, start):
        x = start
        raddi=0
        for i in range(MAX_TIMES):
            y = self.next_pos(x)
            if self.f_gauss(y) < self.f_gauss(x):
                break
            radius_new = np.linalg.norm(y - x)
            raddi+=radius_new
            x = y
        return x,raddi
    def climbs(self):
        for i in range(self.n):
            print("clms", i, self.ps[i])
            mx,radius = self.get_max(self.ps[i])
            self.attrs.append(mx)
            self.is_out.append(radius)
dc = Denclue2D(x, y)
dc.climbs()
g_clusters = nx.Graph()
for i in dc.ps:
    dc.bel.append(dc.f_gauss(i))
for j1 in range(dc.n):
    g_clusters.add_node(j1, attr_dict={'attractor': dc.attrs[j1],'density':dc.bel[j1],'radius':dc.is_out[j1]})
for j1 in range(dc.n):
    for j2 in (x for x in range(dc.n) if x != j1):
        if g_clusters.has_edge(j1, j2):
            continue
        diff = np.linalg.norm(g_clusters.nodes[j1]['attr_dict']['attractor'] - g_clusters.nodes[j2]['attr_dict']['attractor'])
        if diff <= (2*DELTA):
            g_clusters.add_edge(j1, j2)
clusters = list(nx.connected_components(g_clusters))
den=[]
count=0
for j in range(len(clusters)):
    num=0
    max=0
    for i in clusters[j]:
        if(g_clusters.nodes[i]['attr_dict']['density']>max):
            num=i
            max=g_clusters.nodes[i]['attr_dict']['density']
    den.append(num)
for i in range(len(den)):
    if g_clusters.nodes[den[i]]['attr_dict']['density']>=XI:
        count+=1
alter={}
for i in range(dc.n):
    alter[i]=False
label={}
for i in range(dc.n):
    for j in range(len(clusters)):
        for k in clusters[j]:
            label[k]=j
merge_cluster={}
for i in range(len(clusters)):
    merge_cluster[i]=[]
merge_cluster2={}
for i in range(len(clusters)):
    merge_cluster2[i]=[]
for i in range(dc.n):
    for j in range(i+1,dc.n):
        if(np.linalg.norm(dc.ps[i]-dc.ps[j])<=0.5 \
        and label[i] !=label[j] \
        and label[j] not in merge_cluster2[label[i]]):
            merge_cluster2[label[i]].append(label[j])
            if(g_clusters.nodes[i]['attr_dict']['density']<XI \
            and g_clusters.nodes[j]['attr_dict']['density']<XI \
            and label[j] not in merge_cluster[label[i]]):
                merge_cluster[label[i]].append(label[j])
for i in range(len(merge_cluster)):
    for j in merge_cluster[i]:
        for n in merge_cluster2[i]:
            if(j==n):
                merge_cluster2[i].remove[j]
for i in range(len(merge_cluster2)):
    for j in merge_cluster2[i]:
        for k in merge_cluster2[j]:
            if(k not in merge_cluster2[i]):
                merge_cluster2[i].append(k)
for i in range(len(merge_cluster2)):
    for j in merge_cluster2[i]:
        for k in label:
            if(label[k]==j and alter[k]==False):
                label[k]=i
count=0
label_color=[]
for i in range(len(label)):
    if((label[i]) not in label_color):
        label_color.append(label[i])
        count+=1
cata=-1
for i in range(len(label_color)):
    cata+=1
    for j in range(len(label)):
        if(label[j]==label_color[i]):
            label[j]=cata
colors = ['r', 'g', 'b','orange','pink','purple','yellow','green','gray','gold','hotpink','navy','brown']
for i in range(len(label)):
    if(g_clusters.nodes[i]['attr_dict']['density'])>=XI:
         plt.scatter(dc.ps[i][0],dc.ps[i][1],c=colors[label[i]])
    else:
         plt.scatter(dc.ps[i][0],dc.ps[i][1],c='black')
plt.show();