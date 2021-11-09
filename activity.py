import numpy as np
import matplotlib.pyplot as plt
from scipy import special, integrate
import itertools
from subprocess import Popen, PIPE

def gauss_filter(theta, theta0, delta):
	F=np.exp(-(theta - theta0)**2/(2*delta**2)) + np.exp(-(theta - np.pi + theta0)**2/(2*delta**2)) 
	return F

def gauss_filter_2pi(theta, theta0, delta):
	# This function includes 3 Gaussian in order to cover all possible possibilities
	# When covering the range [0, 2pi] for theta0
	F=np.exp(-(theta - theta0)**2/(2*delta**2)) + np.exp(-(theta - np.pi + theta0)**2/(2*delta**2))+np.exp(-(theta - 2*np.pi + theta0)**2/(2*delta**2))
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

def gate_filter_2pi(theta, theta0, delta):
	# This function includes 3 Gate in order to cover all possible possibilities
	# When covering the range [0, 2pi] for theta0
	try:
		l=len(theta)
		F=np.zeros(l, dtype=float)
		g1=np.where(np.bitwise_and(theta >= (theta0 - delta/2), theta <= (theta0 + delta/2)))
		g2=np.where(np.bitwise_and(theta >= (np.pi - theta0 - delta/2), theta <= (np.pi - theta0 + delta/2)))
		g3=np.where(np.bitwise_and(theta >= (2*np.pi - theta0 - delta/2), theta <= (2*np.pi - theta0 + delta/2)))
		for g in g1:
			F[g]=1
		for g in g2:
			F[g]=1
		for g in g3:
			F[g]=1
	except:
		if (theta >= (theta0 - delta/2)) and (theta <= (theta0 + delta/2)) or (theta >= (np.pi - theta0 - delta/2) and  theta <= (np.pi - theta0 + delta/2)) or (theta >= (2*np.pi - theta0 - delta/2) and  theta <= (2*np.pi - theta0 + delta/2)):
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

def gauss_filter_cte_2pi(theta0, delta):
	'''
		A function that calculate what is the max value of a triple-gaussian filter
		that is symetrical towards pi/2 and 3pi/2 and bounded between [0, 2pi] and of width delta
          |    -                                             -    
		  |  -    -                                        -   -                                     |
		  | -       -                                     -      -                                   |
		  |-         -                                   -        -                                  |
		  -            -                                -          -                                -|
		  |             -                              -            -                              - |
		  |              -                           -                -                           -  |
		  |                -                       -                    -                        -   |
		--+-----+----------------+----------------------+------+-------------------------------------+------> theta
		  0   theta0           pi/2                    pi    pi-theta0                              2pi
	'''
	theta= np.linspace(0, 2*np.pi, 100)
	F=gauss_filter_2pi(theta, theta0, delta)
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

# -----

def test_integrate_ylm2(l):
    phi_range = [0, 2.*np.pi]
    theta_range = [0, np.pi/4.]
    print("Ylm2(", l, "):")
    for m in range(-l, l+1):
        integral=integrate_ylm2(l, m, phi_range, theta_range)
        print("(l=" , l, ", m=", m, ") =" , "   ", integral[0])


def Alm_gate(_theta, _phi, _l, _m, _theta0, _delta):
	Y=Ylm2(_theta, _phi, _l, _m)
	F=gate_filter(_theta, _theta0, _delta)
	return Y*F

def Alm_gate_2pi(_theta, _phi, _l, _m, _theta0, _delta):
	Y=Ylm2(_theta, _phi, _l, _m)
	F=gate_filter_2pi(_theta, _theta0, _delta)
	return Y*F

def Alm_gauss(_theta, _phi, _l, _m, _theta0, _delta):
	Y=Ylm2(_theta, _phi, _l, _m)
	F=gauss_filter(_theta, _theta0, _delta)
	Fmax=gauss_filter_cte(_theta0, _delta)
	return Y*F/Fmax

def Alm_gauss_2pi(_theta, _phi, _l, _m, _theta0, _delta):
	Y=Ylm2(_theta, _phi, _l, _m)
	F=gauss_filter_2pi(_theta, _theta0, _delta)
	Fmax=gauss_filter_cte_2pi(_theta0, _delta)
	return Y*F/Fmax

def Ylm2(_theta, _phi, _l, _m):
    _s = special.sph_harm(_m, _l, _phi, _theta)
    return (_s.real**2 + _s.imag**2) * np.sin(_theta)

def integrate_ylm2(l, m, phi_range, theta_range):
    result = integrate.dblquad(Ylm2,
         phi_range[0], phi_range[1], theta_range[0], theta_range[1], args=(l, m,))
    return result

def integrate_Alm(l, m, phi_range, theta_range, theta0, delta, ftype='gate'):
	if delta != 0:
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
	else:
		result=[0,0] # When delta is 0, obviously the result is 0
	return result

def integrate_Alm_2pi(l, m, phi_range, theta_range, theta0, delta, ftype='gate'):
	if delta != 0:
		if ftype == 'gate':
			result = integrate.dblquad(Alm_gate_2pi,
		     phi_range[0], phi_range[1], theta_range[0], theta_range[1], args=(l, m, theta0, delta,))
		if ftype == 'gauss':
			result = integrate.dblquad(Alm_gauss_2pi,
		     phi_range[0], phi_range[1], theta_range[0], theta_range[1], args=(l, m, theta0, delta,))
		if ftype != 'gate' and ftype != 'gauss':
			print("Wrong filter type argument: ")
			print("Use only:")
			print("    ftype='gate' or  ftype='gauss'")
			print("The program will exit now ")
			exit()
	else:
		result=[0,0] # When delta is 0, obviously the result is 0
	return result

def Alm_cpp(l, theta0, delta, ftype, raw=False, use2pi=False):
	if use2pi == False:
		use2pi=0
	else:
		use2pi=1
		print('NOT SUPPORTED: IF YOU WANT TO TRY THIS YOU NEED TO : (1) Recompile cpp with activation of use2pi in main.cpp and (2) uncomment the Popen call that use use2pi')
		exit()
	try:
		#process = Popen(["./Alm", str(l), str(theta0), str(delta), ftype, use2pi], stdout=PIPE, stderr=PIPE)
		process = Popen(["./Alm", str(l), str(theta0), str(delta), ftype], stdout=PIPE, stderr=PIPE)
		(output, err) = process.communicate()
		exit_code = process.wait()
		#print(output)
		output=output.decode("utf-8") 
		if raw == False:
			r=output.split('\n')
			#config=[]
			l=[]
			m=[]
			Alm=[]
			for line in r:
				line=line.strip()
				#print("line =", line)
				try:
					if line[0] != "#":
						s=line.split()
						l.append(int(s[0]))
						m.append(float(s[1]))
						Alm.append(float(s[2]))
				except:
					err=True
			return np.asarray(l),np.asarray(m),np.asarray(Alm)
		else:
			return output, err
	except: # Handling processes that does not exist, aka, when  the Alm file is not available
		error=True
		print("Error: Could not execute the Alm C++ program. The most likely explanation is that it is not in the current directory")
		return -1, error

def Alm(l,m, theta0=np.pi/2, delta=2*8.4*np.pi/180, ftype='gate', cpp=True, use2pi=False):
    phi_range = [0, 2.*np.pi]
    theta_range = [0, np.pi]
    if use2pi==False:
    	integral=integrate_Alm(l, m, phi_range, theta_range, theta0, delta, ftype=ftype)
    else:
    	integral=integrate_Alm_2pi(l, m, phi_range, theta_range, theta0, delta, ftype=ftype)
    return integral[0]
