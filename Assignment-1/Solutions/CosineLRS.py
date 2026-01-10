import math

def lr_cos_sch(t:int, a_max:float, a_min:float, tw:int, tc:int):
    alpha=float
    if t<tw:
        alpha=(t/tw)*a_max
    elif t<=tc and t>=tw:
        b1=(t-tw)/(tc-tw)
        b1cos=1+math.cos(b1*math.pi)
        alpha = a_min+0.5*b1cos*(a_max-a_min)
    else:
        alpha=a_min
    return alpha