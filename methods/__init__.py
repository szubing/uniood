from methods.source_only import SourceOnly
from methods.clip_distillation import ClipDistill
from methods.dance import Dance
from methods.ovanet import OVANet
from methods.uniot import UniOT
from methods.clip_zero_shot import ClipZeroShot
from methods.clip_cross_model import ClipCrossModel
from methods.wise_ft import WiSE_FT
from methods.distill import Distill
from methods.oracle import Oracle
from methods.auto_distillation import AutoDistill
from methods.auto_distillation_only_calibration import AutoDistill_only_Cal

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
                  'Distill': Distill,
                  'AutoDistill': AutoDistill,
                  'Auto_only_cal':AutoDistill_only_Cal,
                  'Oracle': Oracle
                  }

