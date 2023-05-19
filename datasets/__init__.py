from datasets.office31 import Office31
from datasets.officehome import OfficeHome
from datasets.domain_net import DomainNet
from datasets.visda import VisDa2017

dataset_classes = {
    "office31": Office31,
    "officehome": OfficeHome,
    "domainnet": DomainNet,
    'visda': VisDa2017,
}