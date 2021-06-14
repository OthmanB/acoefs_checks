import numpy as np
import matplotlib.pyplot as plt
from scipy import special, integrate
import itertools

def gauss_filter(theta, theta0, delta):
	F=np.exp(-(theta - theta0)**2/(2*delta**2)) + np.exp(-(theta - np.pi + theta0)**2/(2*delta**2)) 
	return F

def gate_filter(theta, theta0, delta):
	try:
		l=len(theta)
		F=np.zeros(l, dtype=float)
		g1=np.where(np.bitwise_and(theta >= (theta0 - delta/2), theta <= (theta0 + delta/2)))
		g2=np.where(np.bitwise_and(theta >= (np.pi - theta0 - delta/2), theta <= (np.pi - theta0 + delta/2)))
		for g in g1:
			F[g]=1
		for g in g2:
			F[g]=1
	except:
		if (theta >= (theta0 - delta/2)) and (theta <= (theta0 + delta/2)) or (theta >= (np.pi - theta0 - delta/2) and  theta <= (np.pi - theta0 + delta/2)):
			F=1
		else:
			F=0
	return F

def gauss_filter_cte(theta0, delta):
	'''
		A function that calculate what is the max value of a double-gaussian filter
		that is symetrical towards pi/2 and bounded between [0, pi] and of width delta
                     -                      -
		          -    -                  -   -
		         -       -              -      -
		        -         -            -        -
		       -            -         -          - 
		      -              -       -            -
		    -                  -   -                -
		  -                      -                    -
		--+----------+-----------+---------------------+----> theta
		  0         theta0      pi/2    pi-theta0      pi
	'''
	theta= np.linspace(0, np.pi, 100)
	F=gauss_filter(theta, theta0, delta)
	#F=F/max(F)
	return max(F)

def show_filters(theta0=np.pi/6, delta=np.pi/10):
	theta=np.linspace(0, np.pi, 1000)
	F1=gate_filter(theta, theta0, delta)
	F2=gauss_filter(theta, theta0, delta)
	F2=F2/gauss_filter_cte(theta0, delta)
	plt.plot(theta, F1)
	plt.plot(theta, F2)
	plt.show()


def Alm_gate(_theta, _phi, _l, _m, _theta0, _delta):
	Y=Ylm2(_theta, _phi, _l, _m)
	F=gate_filter(_theta, _theta0, _delta)
	return Y*F

def Alm_gauss(_theta, _phi, _l, _m, _theta0, _delta):
	Y=Ylm2(_theta, _phi, _l, _m)
	F=gauss_filter(_theta, _theta0, _delta)
	Fmax=gauss_filter_cte(_theta0, _delta)
	return Y*F/Fmax

def Ylm2(_theta, _phi, _l, _m):
    _s = special.sph_harm(_m, _l, _phi, _theta)
    return (_s.real**2 + _s.imag**2) * np.sin(_theta)

def integrate_ylm2(l, m, phi_range, theta_range):
    result = integrate.dblquad(Ylm2,
         phi_range[0], phi_range[1], theta_range[0], theta_range[1], args=(l, m,))
    return result

def integrate_Alm(l, m, phi_range, theta_range, theta0, delta, ftype='gate'):
    if ftype == 'gate':
    	result = integrate.dblquad(Alm_gate,
         phi_range[0], phi_range[1], theta_range[0], theta_range[1], args=(l, m, theta0, delta,))
    if ftype == 'gauss':
    	result = integrate.dblquad(Alm_gauss,
         phi_range[0], phi_range[1], theta_range[0], theta_range[1], args=(l, m, theta0, delta,))
    if ftype != 'gate' and ftype != 'gauss':
    	print("Wrong filter type argument: ")
    	print("Use only:")
    	print("    ftype='gate' or  ftype='gauss'")
    	print("The program will exit now ")
    	exit()
    return result


def Alm(l,m, theta0=np.pi/2, delta=2*8.4*np.pi/180, ftype='gate'):
    phi_range = [0, 2.*np.pi]
    theta_range = [0, np.pi]
    integral=integrate_Alm(l, m, phi_range, theta_range, theta0, delta, ftype=ftype)
    return integral[0]

def test_gate_filter():
	delta = np.pi/6;
	theta0= np.pi/2;
	for i in range(21):
		theta=i*np.pi/20;
		F=gate_filter(theta, theta0, delta);
		print("theta : ", theta,"     F=",F)

def test_gauss_filter():
	delta = np.pi/6;
	theta0= np.pi/2;
	Fmax=gauss_filter_cte(theta0, delta)
	for i in range(21):
		theta=i*np.pi/20;
		F=gauss_filter(theta, theta0, delta);
		print("theta : ", theta,"     F=",F/Fmax)
# -----

def test_integrate_ylm2(l):
    phi_range = [0, 2.*np.pi]
    theta_range = [0, np.pi/4.]
    print("Ylm2(", l, "):")
    for m in range(-l, l+1):
        integral=integrate_ylm2(l, m, phi_range, theta_range)
        print("(l=" , l, ", m=", m, ") =" , "   ", integral[0])

def test_integrate_Alm(l, theta0=np.pi/2, delta=np.pi/6):
    phi_range = [0, 2.*np.pi]
    theta_range=[0, np.pi]
    for m in range(-l, l+1):
        integral=integrate_Alm(l, m, phi_range, theta_range, theta0, delta, ftype='gate')
        print("Alm_gate(l=" , l, ", m=", m, ") =" , "   ", integral[0])
        integral=integrate_Alm(l, m, phi_range, theta_range, theta0, delta, ftype='gauss')
        print("Alm_gauss(l=" , l, ", m=", m, ") =" , "   ", integral[0])

