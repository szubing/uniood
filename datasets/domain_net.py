import os

from datasets.benchmark import MultiDomainsBenchmark, read_split_from_txt


class DomainNet(MultiDomainsBenchmark):
    dataset_name = 'domain_net'
    domains = {'painting': 'painting_train.txt', 
               'real': 'real_train.txt', 
               'sketch': 'sketch_train.txt'}

    def __init__(self, data_dir, source_domain, target_domain, n_share=None, n_source_private=None):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.source_path = os.path.join(self.dataset_dir, self.domains[source_domain])
        self.target_path = os.path.join(self.dataset_dir, self.domains[target_domain])
        source_total = read_split_from_txt(self.source_path, self.dataset_dir)
        target_total = read_split_from_txt(self.target_path, self.dataset_dir)

        super().__init__(source_total=source_total, target_total=target_total, n_share=n_share, n_source_private=n_source_private)
