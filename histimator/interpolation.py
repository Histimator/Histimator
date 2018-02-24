
def PolyInterpValue(alpha, I0, Iup, Idown, alpha0 = 0):
    polycoeff = np.zeros(6)
    pow_up       =  pow(Iup/I0, alpha0)
    pow_down     =  pow(Idown/I0,  alpha0)
    logHi        =  math.log(Iup)  
    logLo        =  math.log(Idown )
    
    pow_up_log   = 0.0 if Iup <= 0.0 else pow_up * logHi
    pow_down_log = 0.0 if Idown <= 0.0 else -pow_down * logLo
    pow_up_log2  = 0.0 if Iup <= 0.0 else pow_up_log * logHi
    pow_down_log2= 0.0 if Idown <= 0.0 else -pow_down_log * logLo

    S0 = (pow_up+pow_down)/2
    A0 = (pow_up-pow_down)/2
    S1 = (pow_up_log+pow_down_log)/2
    A1 = (pow_up_log-pow_down_log)/2
    S2 = (pow_up_log2+pow_down_log2)/2
    A2 = (pow_up_log2-pow_down_log2)/2
    a = 1./(8*alpha0) * (15*A0 - 7*alpha0*S1 + alpha0*alpha0*A2)
    b = 1./(8*alpha0*alpha0) * (-24 + 24*S0 - 9*alpha0*A1 + alpha0*alpha0*S2)
    c = 1./(4*pow(alpha0, 3))*( -5*A0 + 5*alpha0*S1 - alpha0*alpha0*A2)
    d = 1./(4*pow(alpha0, 4))*( 12 - 12*S0 +  7*alpha0*A1 - alpha0*alpha0*S2)
    e = 1./(8*pow(alpha0, 5))*(    +  3*A0 -  3*alpha0*S1 + alpha0*alpha0*A2)
    f = 1./(8*pow(alpha0, 6))*( -8 +  8*S0 -  5*alpha0*A1 + alpha0*alpha0*S2)
   
   # evaluate the 6-th degree polynomial using Horner's method
    return 1. + x * (a + x * ( b + x * ( c + x * ( d + x * ( e + x * f ) ) ) ) )

class Interpolate(object):
    def __init__(self, scheme):
        self.scheme = scheme
    def __call__(self, alpha, I0, Iup, Idown):
        if self.scheme == 0:
            return self.PiecewiseLinear(alpha, I0, Iup, Idown)
        elif self.scheme == 1:
            return self.PiecewiseExponential(alpha, I0, Iup, Idown)
        elif self.scheme == 2:
            return self.QuadInterLinExtra(alpha, I0, Iup, Idown)
        elif self.scheme == 2:
            return self.PolyInterExpExtra(alpha, I0, Iup, Idown)

                
    def PiecewiseLinear(self, alpha, I0, Iup, Idown):
        if alpha < 0:
            return (1 + alpha*(I0-Idown))
        else:
            return (1 + alpha*(Iup - I0))
    def PiecewiseExponential(self, alpha, I0, Iup, Idown):
        if alpha < 0:
            return pow((Idown / I0),-alpha)
        else:
            return pow((Iup / I0),alpha)
    def QuadInterLinExtra(alpha, I0, Iup, Idown):
        a = (Iup + Idown)/2. - I0
        b = (Iup - Idown)/2.
        if alpha > 1.:
            return (1. + (b + 2.*a)*(alpha - 1.) + Iup - I0)
        elif alpha < -1.:
            return (1. + (b - 2.*a)*(alpha + 1.) + Idown - I0)
        else:
            return (1. + (a*alpha*alpha + b*alpha))

    def PolyInterExpExtra(alpha, I0, Iup, Idown, alpha0 = 0):
        if alpha >= alpha0 :
            return pow(Iup/I0, alpha)
        elif alpha <= alpha0:
            return pow(Idown/I0, -alpha)
        elif x != 0:
            return PolyInterpValue(alpha, I0, Iup, Idown, alpha0 = 0)

