from utils.lookahead import *
from utils.lamb import Lamb
from utils.lamb_v3 import Lamb as Lamb_v3

def La_Lamb(params, alpha=0.5, k=6, *args, **kwargs):
     lamb = Lamb(params, *args, **kwargs)
     return Lookahead(lamb, alpha, k)

def La_Lamb_v3(params, alpha=0.5, k=6, *args, **kwargs):
     lamb = Lamb_v3(params, *args, **kwargs)
     return Lookahead(lamb, alpha, k)