"""Integer datapath for EvT on ZCU102 -- see `../hw_flow.md`.

`fixed_point`  Qm.n primitives (pack/unpack, format choice)
`hw_quant`     the datapath itself: pack a fp32 model into integer constants,
               and run it (INT8 GEMMs, INT32 accumulators, Qm.n non-linear I/O,
               bf16 LayerNorm)
`calibration`  activation-scale calibration on the fp32 model
`data_utils`   checkpoint / dataloader / accuracy helpers
"""

from . import fixed_point
from . import hw_quant
from . import calibration
from . import data_utils

__all__ = ["fixed_point", "hw_quant", "calibration", "data_utils"]
