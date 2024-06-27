
import time
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from main import loadtxt, acc_features

def hyperbolicity_sample(G, num_samples=50000):
    curr_time = time.time()
    hyps = []
    for i in tqdm(range(num_samples)):
        curr_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        try:
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    print('Time for hyp: ', time.time() - curr_time)
    return max(hyps)
def acc_features(drug_mic_update, drug_sim_ts, micro_sim_ts, x, y):
    # (mi的相似度拼接mi与疾病的联系) 堆叠 (疾病与mi的联系拼接疾病的相似度_
    # drug_sim_ts, (1373, 1373); drug_mic_update, (1373, 173); micro_sim_ts, (173, 173);
    a = torch.cat((drug_sim_ts, drug_mic_update), dim=1)
    b = torch.cat((drug_mic_update.T, micro_sim_ts), dim=1)
    mat = torch.cat((a,b), dim=0)
    return mat

if __name__ == '__main__':
    data, mic_sim, drug_sim = loadtxt()
    # data = torch.tensor(drug_mic)
    r, c = data.shape
    M = np.zeros((r, r))
    M[:r, :c] = data
    M[:c, :r] += data.T
    graph = nx.from_numpy_matrix(M)
    nx.draw(graph)
    # graph = nx.from_scipy_sparse_matrix(data)
    print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
    hyp = hyperbolicity_sample(graph)
    print('Hyp: ', hyp)

