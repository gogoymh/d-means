"""d-means: k-means steered by any typed decision model: Jev, Clef, a chat LLM, or your own function.

Decision models live in `dmeans.deciders`; nothing here is tied to one of them.
"""

from .deciders import CachedDecider, Decider, FunctionDecider, as_decider
from .dmeans import DMeans
from .emeans import EMeans
from .lloyd import lloyd
from .omeans import OMeans

__all__ = ["DMeans", "EMeans", "OMeans", "CachedDecider", "Decider", "FunctionDecider", "as_decider", "lloyd"]
__version__ = "0.1.0"
