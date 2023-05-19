from methods.source_only import SourceOnly
from methods.clip_distillation import ClipDistill
from methods.dance import Dance
from methods.ovanet import OVANet
from methods.uniot import UniOT
from methods.clip_zero_shot import ClipZeroShot
from methods.clip_cross_model import ClipCrossModel
from methods.wise_ft import WiSE_FT


"""
Note that each method class should be a subclass of SourceOnly.
"""
method_classes = {'SO': SourceOnly,
                  'DANCE': Dance,
                  'OVANet': OVANet,
                  'UniOT': UniOT,
                  'ClipZeroShot': ClipZeroShot,
                  'ClipCrossModel': ClipCrossModel,
                  'WiSE-FT': WiSE_FT,
                  'ClipDistill': ClipDistill,
                  }

