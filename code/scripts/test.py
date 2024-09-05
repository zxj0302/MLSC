import torch
import sys
import random
import re

class Sampler:
    def __init__(self, cons):
        self.bounds = {'min': 0, 'max': torch.iinfo(torch.int64).max, 'set': None}
        self.attributes = {a: self.bounds[a[-3:]] for a in [f'{pre}_{suf}' for pre in (['num_nodes', 'num_edges', 'degree_avg', 'degree_max', 'diameter', 'clustering'] + [f'{c}{p}' for c in 'in' for p in range(30)]) for suf in ['min', 'max']] + ['domain_set']}
        self.constrains = {key: cons[key] for key in cons if key in self.attributes}

        lambda_generator = lambda attr: (
                'lambda g: ' + (f'torch.max(g.gt_{"non" if attr[0] == "n" else ""}induced_le5[:, {attr[:-4][1:]}]).item()' if bool(re.match(r"^[in](|1|2)[0-9]_(min|max)$", attr)) else f'g.{attr[:-4]}' if attr in self.attributes else (lambda: (_ for _ in ()).throw(ValueError(f"Attribute {attr} not valid")))()) + {"min": ">=", "max": "<=", "set": "in"}[attr[-3:]] + str(self.constrains[attr])
        )
        # print([lambda_generator(c) for c in self.constrains])
        self.conditions = [eval(lambda_generator(c)) for c in self.constrains]
        print(f'Constrains: {self.constrains}')

    def sample(self, data, k=sys.maxsize, s=None):
        random.seed(s)
        filtered = [g for g in data if all(c(g) for c in self.conditions)]
        print(f'Number of graphs after filtering: {len(filtered)}, number of graphs to sample: {min(k, len(filtered))}')
        return random.sample(filtered, min(k, len(filtered)))


if __name__ == '__main__':
    data = torch.load('RWDataset/many_small/graph_featured.pt')
    s = Sampler({'num_nodes_min': 100, 'num_nodes_max': 500, 'degree_avg_max': 300, 'degree_max_min': 280})
    # the second parameter is the number of graphs to sample, the third parameter is the seed for random sampling
    res = s.sample(data, 200, 0)
    for r in res:
        print(r)