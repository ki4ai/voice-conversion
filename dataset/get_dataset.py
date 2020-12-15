from .KETTS_male import KETTS_30m
from .KETTS_female import KETTS_30f
from .KETTS76 import KETTS76
from .KNTTS import KNTTS
from .KNTTS_two import KNTTS_two
from .BC2013 import BC2013
from .LJSpeech11 import LJSpeech11
from .ETRI import ETRI

def get_dataset(dbname, dbroot):
    if dbname == 'KETTS76':
        return KETTS76(dbroot)
    if dbname == 'KNTTS':
        return KNTTS(dbroot)
    if dbname == 'KNTTS_two':
        return KNTTS_two(dbroot)
    elif dbname == 'KETTS_male':
        return KETTS_30m(dbroot)
    elif dbname == 'KETTS_female':
        return KETTS_30f(dbroot)
    elif dbname == 'BC2013':
        return BC2013(dbroot)
    elif dbname == 'LJSpeech11':
        return LJSpeech11(dbroot)
    elif dbname == 'ETRI':
        return ETRI(dbroot)
    else:
        raise ValueError(f'{dbroot}')
