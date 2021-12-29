# Ensemble of functions to perform a fit of a2 terms with emcee of the gaussianised errors 
from activity import Alm
from activity import Alm_cpp
from acoefs import eval_acoefs
import numpy as np
import emcee
from multiprocessing import Pool
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import corner
import make_grid as mkgrid
from scipy import interpolate

## Definition of the Gaussian likelihood
def likelihood_gauss(xm, xobs, sigobs):
	return np.sum(-(np.array(xm)-np.array(xobs))**2/np.array(sigobs)**2)

# Definition of the Gaussian likelihood
# WARNING IF THIS IS NOT COMMENTED AND YOU DON'T KNOW WHY, PLEASE COMMENT AND USE THE FUNCTION ABOVE
# THIS FUNCTION IGNORES THE xobs AND THEREFORE TEST A NO-MEASUREMENT CASE (PRIOR ONLY)
#def likelihood_gauss(xm, xobs, sigobs):
#	return np.sum(-(np.array(xm))**2)


# Definition of a uniform prior in log space
def prior_uniform(x, xmin, xmax):
	if x >= xmin and x <= xmax:
		p=-np.log(np.abs(xmax-xmin))
	else:
		p=-np.inf
	return p

def prior_jeffreys(x, xmin, xmax):
	if (x <= xmax and x>=0):
		prior=1./(x+xmin)
		norm=np.log((xmax+xmin)/xmin)
		p=np.log(prior/norm)
	else:
		p=-np.inf
	return p

def read_mcmcobs(file):
	return read_obsfiles(file, real_data=True)

# Read files that contain observed a2 coefficients
def read_obsfiles(file, read_a4=False, read_a6=False, real_data=False):
	pos_a2_med=4
	pos_a4_med=10
	pos_a6_med=16
	f=open(file, 'r')
	txt=f.read()
	f.close()
	txt=txt.split('\n')
	en=[]
	el=[]
	nu_nl=[]
	a2=[]
	sig_a2=[]
	a4=[]
	sig_a4=[]
	a6=[]
	sig_a6=[]
	skip=0
	#print(txt)
	#print('----')
	if real_data == True:
		for t in txt:
			s=t.split()
			if s != '' and s !=[]:
				#print(s[0])
				if s[0] == '#' or s[0] == [] or s[0] == '':
					skip=skip+1
				else:
					#print(len(en))
					if len(en) != 0:
						en.append(en[-1]+1)
					else:
						en.append(1)
					el.append(int(float(s[0])))
					nu_nl.append(float(s[1]))
					a2.append(float(s[pos_a2_med]))
					em=float(s[pos_a2_med]) - float(s[pos_a2_med-1])
					ep=float(s[pos_a2_med+1]) - float(s[pos_a2_med])
					sig_a2.append(np.sqrt(em**2 + ep**2)/2)
					if read_a4 == True:
						a4.append(float(s[pos_a4_med]))
						em=float(s[pos_a4_med]) - float(s[pos_a4_med-1])
						ep=float(s[pos_a4_med+1]) - float(s[pos_a4_med])
						sig_a4.append(np.sqrt(em**2 + ep**2)/2)
					if read_a6 == True:
						a6.append(float(s[pos_a6_med]))
						em=float(s[pos_a6_med]) - float(s[pos_a6_med-1])
						ep=float(s[pos_a6_med+1]) - float(s[pos_a6_med])
						sig_a6.append(np.sqrt(em**2 + ep**2)/2)
		return en, el, nu_nl, a2, sig_a2, a4, sig_a4, a6, sig_a6
	else:
		for t in txt:
			done=False
			s=t.split()
			if s != '' and s !=[]:
				#print(s[0])
				if s[0] == '#' or s[0] == [] or s[0] == '':
					skip=skip+1
				else:
					if len(en) != 0:
						en.append(en[-1]+1)
					else:
						en.append(1)
					if s[0] == '!' and s[1] == 'l':
						el=s[2:]
						done=True
					if s[0] == '!' and s[1] == 'nu_nl_obs':
						nu_nl=s[2:]
						done=True
					if done !=True:
						a1.append(float(s[0]))
						a2.append(float(s[1]))
						a3.append(float(s[2]))
						a4.append(float(s[3]))
						a5.append(float(s[4]))
						a6.append(float(s[5]))
						sig_a1.append(float(s[6]))
						sig_a2.append(float(s[7]))
						sig_a3.append(float(s[8]))
						sig_a4.append(float(s[9]))
						sig_a5.append(float(s[10]))
						sig_a6.append(float(s[11]))
		return en, el, nu_nl, a2, sig_a2, a3, sig_a3, a4, sig_a4, a5, sig_a5, a6, sig_a6

def Qlm(l,m):
	Dnl=2./3
	Qlm=(l*(l+1) - 3*m**2)/((2*l - 1)*(2*l + 3))
	return Qlm*Dnl

def nu_CF(nu_nl, Dnu, a1, l, m, a1_unit='nHz'):
   Dnu_sun=135.1
   numax_sun=3150.
   R_sun=6.96342e5 #in km
   M_sun=1.98855e30 #in kg
   G=6.667e-8
   rho_sun=M_sun*1e3/(4*np.pi*(R_sun*1e5)**3/3) #in g.cm-3
   rho=(Dnu/Dnu_sun)**2 * rho_sun
   eta0=3./(4.*np.pi*rho*G); 
   if a1_unit == 'nHz':
   	return eta0*nu_nl * (a1*1e-9)**2 * Qlm(l,m)
   if a1_unit == 'microHz':
   	return eta0*nu_nl * (a1*1e-6)**2 * Qlm(l,m)
   if a1_unit != 'nHz' and a1_unit != 'microHz':
   	print('a1 must be provided either in nHz or in microHz')
   	print('use the a1_unit argument of the nu_CF() function to set it properly')
   	exit()

def nu_AR(nu_nl, epsilon_nl, theta0, delta, ftype, l, m):
	return nu_nl*epsilon_nl*Alm(l,m, theta0=theta0, delta=delta, ftype=ftype)

def a2_CF(nu_nl, Dnu, a1, l):
	nu_nlm=[]
	for m in range(-l, l+1):
		perturb=nu_CF(nu_nl, Dnu, a1, l, m)
		nu_nlm.append(nu_nl + perturb)
	acoefs=eval_acoefs(l, nu_nlm)
	#print(nu_nlm)
	return acoefs[1] # returns only a2

def a2_AR(nu_nl, epsilon_nl, theta0, delta, ftype):
	nu_nlm=[]
	for m in range(-l, l+1):
		perturb=nu_AR(nu_nl, epsilon_nl, theta0, delta, ftype, l, m)
		nu_nlm.append(nu_nl + perturb)
	acoefs=eval_acoefs(l, nu_nlm)
	return acoefs[1] # returns only a2

# Compute the a2 coefficient for the theoretical model and provided key parameters of that model
def a2_model(nu_nl, Dnu, a1, epsilon_nl, theta0, delta, ftype, l):
	nu_nlm=[]
	for m in range(-l, l+1):	
		perturb_CF=nu_CF(nu_nl, Dnu, a1, l, m)
		perturb_AR=nu_AR(nu_nl, epsilon_nl, theta0, delta, ftype, l, m)
		nu_nlm.append(nu_nl + perturb_CF + perturb_AR)
	#print(nu_nlm)
	acoefs=eval_acoefs(l, nu_nlm)
	return acoefs[1] # returns only a2 

# Compute the a-coefficients for the theoretical model and provided key parameters of that model
# Use Alm_cpp instead of Alm in python... much faster. Refer to test_convergence.py to see the accuracy
def a_model_cpp(nu_nl, Dnu, a1, epsilon_nl, theta0, delta, ftype, l):
	nu_nlm=[]
	el, em, Alm=Alm_cpp(l, theta0=theta0, delta=delta, ftype=ftype) # Array of m E [-l,l]
	for m in range(-l, l+1):	
		perturb_CF=nu_CF(nu_nl, Dnu, a1, l, m)
		perturb_AR=nu_nl*epsilon_nl*Alm[m+l]
		nu_nlm.append(nu_nl + perturb_CF + perturb_AR)
	#print(nu_nlm)
	acoefs=eval_acoefs(l, nu_nlm)
	return acoefs # returns all a-coeficients 

def a2_model_cpp(nu_nl, Dnu, a1, epsilon_nl, theta0, delta, ftype, l):
	acoefs=a_model_cpp(nu_nl, Dnu, a1, epsilon_nl, theta0, delta, ftype, l)
	return acoefs[1] # returns only the a2 coefficient

def a_model_interpol(nu_nl, Dnu, a1, epsilon_nl, theta0, delta0, l, interpolator_l1, interpolator_l2, interpolator_l3):
	nu_nlm=[]
	for m in range(-l, l+1):	
		if l==1:
			Alm=interpolator_l1[l+m](theta0, delta0)
		if l==2:
			Alm=interpolator_l2[l+m](theta0, delta0)
		if l==3:
			Alm=interpolator_l3[l+m](theta0, delta0)
		if l>=4:
			print("l>=4 is not implemented yet in a2_model_interpol")
			exit()
		if l<1:
			print("l<1 is not authorized in a2_model_interpol")
			exit()
		perturb_CF=nu_CF(nu_nl, Dnu, a1, l, m)
		perturb_AR=nu_nl*epsilon_nl*Alm
		nu_nlm.append(nu_nl + perturb_CF + perturb_AR)
	#print(nu_nlm)
	acoefs=eval_acoefs(l, nu_nlm)
	return acoefs

def a2_model_interpol(nu_nl, Dnu, a1, epsilon_nl, theta0, delta0, l, interpolator_l1, interpolator_l2):
	coefs=a_model_interpol(nu_nl, Dnu, a1, epsilon_nl, theta0, delta0, l, interpolator_l1, interpolator_l2)
	return acoefs[1]

def priors_model(nu_nl_obs, epsilon_nl0, epsilon_nl1, theta0, delta):
	# Reject out or range solutions for theta0
	#pena=prior_uniform(theta0, 0, np.pi/2)
	pena=prior_uniform(theta0, 0, 2*np.pi/3)
	# Reject absurd negative solutions and large 'spots' that exceed a pi/4 stellar coverage
	pena=pena+prior_uniform(delta, 0, np.pi/4)

	#if theta0 >= (np.pi/2 - delta/2):
	#	dim=1
	#	pena=pena-1.
	#else:
	#	dim=2

	#pena=pena+prior_jeffreys(delta, 0.02, np.pi/4)
	# impose the negativity of the epsilon coefficient, as it is for the Sun
	for i in range(len(nu_nl_obs)):
		epsilon_nl=epsilon_nl0 + epsilon_nl1*nu_nl_obs[i]*1e-3 # linear term for epsilon_nl
		pena=pena+prior_uniform(epsilon_nl, -0.1, 0.001)
	return pena

def do_simfile(file, Dnu, epsi0, N0, Nmax, a1, epsilon_nl,  theta0, delta, ftype, do_a4=False, do_a6=False,
		relerr_a2=None, relerr_a4=None, relerr_a6=None, relerr_epsilon_nl=None, lmax=2):
	#		
	l_list=[]
	nu_nl_list=[]
	a2_list=[]
	a2_true_list=[]
	a2_err_list=[]
	a4_list=[]
	a4_true_list=[]
	a4_err_list=[]
	a6_list=[]
	a6_true_list=[]
	a6_err_list=[]
	epsilon_nl_list=[]
	epsilon_nl_true_list=[]
	for l in range(1, lmax+1):
		for n in range(N0, Nmax+1):
			nu_nl=(n + l/2 + epsi0)*Dnu
			#
			e_nl_true=epsilon_nl[0] + epsilon_nl[1]*nu_nl
			if relerr_epsilon_nl != None:
				e_nl=np.random.normal(e_nl_true, (relerr_epsilon_nl*np.abs(e_nl_true))/np.sqrt(Nmax-N0))
			else:
				e_nl=e_nl_true
			#
			acoefs=a_model_cpp(nu_nl, Dnu, a1, e_nl, theta0, delta, ftype, l)
			a2_true=acoefs[1]
			if relerr_a2 != None:
				a2=np.random.normal(a2_true, (relerr_a2[0] + relerr_a2[1]*np.abs(a2_true))/np.sqrt(Nmax-N0))
			else:
				a2=a2_true
				relerr_a2=[0,0]
			if do_a4 == True:
				a4_true=acoefs[3]
				if relerr_a4 != None and l != 1: # l=1 has a4=0 by definition
					a4=np.random.normal(a4_true, (relerr_a4[0] + relerr_a4[1]*np.abs(a4_true))/np.sqrt(Nmax-N0))
				else:
					a4=a4_true
					relerr_a4=[0,0]
			if do_a6 == True:
				a6_true=acoefs[5]	
				if relerr_a6 != None and l != 1 and l !=2: # l=1 have a4=0, a6=0 and l=2 have a6=0 by definition
					a6=np.random.normal(a6_true, (relerr_a6[0] + relerr_a6[1]*np.abs(a6_true))/np.sqrt(Nmax-N0))
				else:
					a6=a6_true
					relerr_a6=[0,0]
			l_list.append(l)
			nu_nl_list.append(nu_nl)
			epsilon_nl_list.append(e_nl)
			epsilon_nl_true_list.append(e_nl_true)
			a2_err_list.append((relerr_a2[0] + relerr_a2[1]*a2_true)*1e3)
			a2_list.append(a2*1e3)
			a2_true_list.append(a2_true*1e3)
			if do_a4 == False and do_a6 == False:
				print(l, nu_nl, a2, a2_true, e_nl, e_nl_true)
			if do_a4 == True and do_a6 == False:
				if l>=2:
					a4_err_list.append((relerr_a4[0] + relerr_a4[1]*a4_true)*1e3)
					a4_list.append(a4*1e3)
					a4_true_list.append(a4_true*1e3)
				else:
					a4_err_list.append(1) # Error must be set to something non-zero with the current do_stats_ongrid routine. 
					a4_list.append(0)
					a4_true_list.append(0)					
				print(l, nu_nl, a2, a2_true, a4, a4_true, e_nl, e_nl_true)
			if do_a4 == False and do_a6 == True:
				if l>=3:
					a6_err_list.append((relerr_a6[0] + relerr_a6[1]*a6_true)*1e3)
					a6_list.append(a6*1e3)
					a6_true_list.append(a6_true*1e3)
				else:
					a6_err_list.append(1) # Error must be set to something non-zero with the current do_stats_ongrid routine.
					a6_list.append(0)
					a6_true_list.append(0)					
				print(l, nu_nl, a2, a2_true, a6, a6_true, e_nl, e_nl_true)
			if do_a4 == True and do_a6 == True:
				if l>=2:
					a4_err_list.append((relerr_a4[0] + relerr_a4[1]*a4_true)*1e3)
					a4_list.append(a4*1e3)
					a4_true_list.append(a4_true*1e3)
				else:
					a4_err_list.append(1) # Error must be set to something non-zero with the current do_stats_ongrid routine.
					a4_list.append(0)
					a4_true_list.append(0)					
				if l>=3:
					a6_err_list.append((relerr_a6[0] + relerr_a6[1]*a6_true)*1e3)
					a6_list.append(a6*1e3)
					a6_true_list.append(a6_true*1e3)
				else:
					a6_err_list.append(1) # Error must be set to something non-zero with the current do_stats_ongrid routine.
					a6_list.append(0)
					a6_true_list.append(0)					
				print(l, nu_nl, a2, a2_true, a4, a4_true, a6, a6_true, e_nl, e_nl_true)
	#
	f=open(file, 'w')
	f.write("# Table of SIMULATED aj coefficient in function of nu(n,l)"+"\n")
	f.write("# Created using fit_a2sig.py :: do_simfile()"+" Version 9 Nov 2021\n")
	f.write("# Dnu ="  + str(Dnu) +"\n")
	f.write("# epsilon =" +str(epsi0)+"\n")
	f.write("# N0 =" +str(N0)+"\n")
	f.write("# Nmax =" + str(Nmax)+"\n")
	f.write("# a1 =" + str(a1)+"\n")
	f.write("# epsilon_nl =" + str(epsilon_nl)+"\n")
	f.write("# theta0 =" + str(theta0)+"\n")
	f.write("# delta =" + str(delta)+"\n")
	f.write("# ftype =" + str(ftype)+"\n")
	f.write("# relerr_a2 =" + str(relerr_a2)+"\n")
	f.write("# do_a4 =" + str(do_a4)+"\n")
	f.write("# relerr_a4 =" + str(relerr_a4)+"\n")
	f.write("# do_a6 =" + str(do_a6)+"\n")
	f.write("# relerr_a6 =" + str(relerr_a6)+"\n")
	f.write("# relerr_epsilon_nl =" + str(relerr_epsilon_nl) +"\n")
	if do_a4 == False and do_a6 == False:
		f.write("# Col(0):l, Col(1):nu, Col(2)-Col(6):a2 (for P(a2)=[2.25,16,50,84,97.75]), Col(7): a2_true, Col(8): epsilon_nl Col(9): epsilon_nl_true\n")
		for i in range(len(nu_nl_list)):
			f.write("{0:1d}  	 {1:0.6f}   {2:0.6f}   {3:0.6f}   {4:0.6f}   {5:0.6f}   {6:0.6f}   {7:0.6f}   {8:0.8f}   {9:0.8f}".format(l_list[i], nu_nl_list[i], 
				a2_list[i]-2*a2_err_list[i], a2_list[i]-a2_err_list[i], a2_list[i], a2_list[i]+a2_err_list[i], a2_list[i]+2*a2_err_list[i], 
				a2_true_list[i], epsilon_nl_list[i], epsilon_nl_true_list[i])+"\n")
	if do_a4 == True and do_a6 == False:
		f.write("# Col(0):l, Col(1):nu, Col(2)-Col(6):a2 (for P(a2)=[2.25,16,50,84,97.75]), Col(7): a2_true,  Col(8-12): a4, Col(13): a4_true, Col(14): epsilon_nl Col(16): epsilon_nl_true\n")		
		for i in range(len(nu_nl_list)):
			f.write("{0:1d}  	 {1:0.6f}   {2:0.6f}   {3:0.6f}   {4:0.6f}   {5:0.6f}   {6:0.6f}   {7:0.6f}   {8:0.6f}    {9:0.6f}    {10:0.6f}    {11:0.6f}    {12:0.6f}    {13:0.6f}    {14:0.8f}   {15:0.8f}".format(l_list[i], nu_nl_list[i], 
				a2_list[i]-2*a2_err_list[i], a2_list[i]-a2_err_list[i], a2_list[i], a2_list[i]+a2_err_list[i], a2_list[i]+2*a2_err_list[i], 
				a2_true_list[i], 
				a4_list[i]-2*a4_err_list[i], a4_list[i]-a4_err_list[i], a4_list[i], a4_list[i]+a4_err_list[i], a4_list[i]+2*a4_err_list[i],
				a4_true_list[i], 
				epsilon_nl_list[i], epsilon_nl_true_list[i])+"\n")
	if do_a4 == False and do_a6 == True:
		f.write("# Col(0):l, Col(1):nu, Col(2)-Col(6):a2 (for P(a2)=[2.25,16,50,84,97.75]), Col(7): a2_true,  Col(8-12): a6, Col(13): a6_true, Col(14): epsilon_nl Col(16): epsilon_nl_true\n")		
		for i in range(len(nu_nl_list)):
			f.write("{0:1d}  	 {1:0.6f}   {2:0.6f}   {3:0.6f}   {4:0.6f}   {5:0.6f}   {6:0.6f}   {7:0.6f}   {8:0.6f}    {9:0.6f}    {10:0.6f}    {11:0.6f}    {12:0.6f}    {13:0.6f}    {14:0.8f}   {15:0.8f}".format(l_list[i], nu_nl_list[i], 
				a2_list[i]-2*a2_err_list[i], a2_list[i]-a2_err_list[i], a2_list[i], a2_list[i]+a2_err_list[i], a2_list[i]+2*a2_err_list[i], 
				a2_true_list[i], 
				a6_list[i]-2*a6_err_list[i], a6_list[i]-a6_err_list[i], a6_list[i], a6_list[i]+a6_err_list[i], a6_list[i]+2*a6_err_list[i],
				a6_true_list[i], 
				epsilon_nl_list[i], epsilon_nl_true_list[i])+"\n")
	if do_a4 == True and do_a6 == True:
		f.write("# Col(0):l, Col(1):nu, Col(2)-Col(6):a2 (for P(a2)=[2.25,16,50,84,97.75]), Col(7): a2_true,  Col(8-12): a4, Col(13): a4_true, Col(13-17): a6, Col(18): a6_true, Col(19): epsilon_nl Col(20): epsilon_nl_true\n")		
		for i in range(len(nu_nl_list)):
			f.write("{0:1d}  	 {1:0.6f}   {2:0.6f}   {3:0.6f}   {4:0.6f}   {5:0.6f}   {6:0.6f}   {7:0.6f}   {8:0.6f}    {9:0.6f}    {10:0.6f}    {11:0.6f}    {12:0.6f}    {13:0.6f}     {14:0.6f}    {15:0.6f}    {16:0.6f}    {17:0.6f}    {18:0.6f}    {19:0.8f}   {20:0.8f}    {21:0.8f}".format(l_list[i], nu_nl_list[i], 
				a2_list[i]-2*a2_err_list[i], a2_list[i]-a2_err_list[i], a2_list[i], a2_list[i]+a2_err_list[i], a2_list[i]+2*a2_err_list[i], 
				a2_true_list[i], 
				a4_list[i]-2*a4_err_list[i], a4_list[i]-a4_err_list[i], a4_list[i], a4_list[i]+a4_err_list[i], a4_list[i]+2*a4_err_list[i],
				a4_true_list[i], 
				a6_list[i]-2*a6_err_list[i], a6_list[i]-a6_err_list[i], a6_list[i], a6_list[i]+a6_err_list[i], a6_list[i]+2*a6_err_list[i],
				a6_true_list[i], 
				epsilon_nl_list[i], epsilon_nl_true_list[i])+"\n")	
	f.close()

def reduce_simfile(simfile, fit_type, err=None, outputfile=None):
	# Take a file created by do_simfile() and reduce it into observables according to the fit_type argument.
	#	simfile : Name of the input file with frequencies corresponding to the Anl true values
	#	fit_type: Control the type of outputs,
	# 		If fit_type = 'mean_nl' : Compute the average of the aj coefficients
	# 		If fit_type = ''  : Compute the aj(n)=aj_0 + aj_1*nu(n,l) coefficients ==> two parameters (averaged over l)
	# 		If fit_type = 'full_fit'  : Compute the aj(n,l)= aj_0l + aj_1l*nu(n,l) coefficients ==> 2*lmax parameters (no averaging)
	#  err: relative Uncertainty on the output. 
	#		If None (default) : Exit with error message
	#		If >0 : The uncertainty is interpreted as fix error value in nHz: eg. 10^-3 nHz
	#     If <0 : The uncertainty is not calculated and is set to 0.
	en, l, nu_nl_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs=read_obsfiles(simfile, read_a4=True, read_a6=True)
	if err == None:
		print('err == None is not implemented in reduce_simfile()')
		exit()
	if np.sum(err) >= 0:
		mean_err_a2=err[0]
		mean_err_a4=err[1]
		mean_err_a6=err[2]
	else:
		mean_err_a2=0
		mean_err_a4=0
		mean_err_a6=0
	if fit_type == 'full_fit': # Here, all of the data we use for the posterior are the results of a 1st order polynomial fit to get a polynomial description of aj(nu,l)
		a2_obs_data=[]
		a4_obs_data=[]
		a6_obs_data=[]
		a2_sig_obs_data=[]
		a4_sig_obs_data=[]
		a6_sig_obs_data=[]
		for el in range(1, np.max(l)+1):
			posl=np.where(np.asarray(l) == el)
			a2_tmp=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a2_obs,posl).flatten(), 1)
			a2_sig_tmp=np.repeat(mean_err_a2, len(a2_tmp))
			if	el >= 2:
				a4_tmp=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a4_obs,posl).flatten(), 1)
				a4_sig_tmp=np.repeat(mean_err_a4, len(a4_tmp))
			if	el >= 3:
				a6_tmp=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a6_obs,posl).flatten(), 1)
				a6_tmp=np.repeat(mean_err_a6, len(a6_tmp))
			a2_obs_data.append(a2_tmp)
			a4_obs_data.append(a4_tmp)
			a6_obs_data.append(a6_tmp)
			a2_sig_obs_data.append(a2_sig_tmp)
			a4_sig_obs_data.append(a4_sig_tmp)
			a6_sig_obs_data.append(a6_sig_tmp)	
	if fit_type == 'mean_l': # Here, all of the data we use for the posterior are the results of a 1st order polynomial fit to get a polynomial description of <aj(nu)>_l
		a2_obs_data=np.polyfit(nu_nl_obs, a2_obs, 1)
		a2_sig_obs_data=mean_err_a2 #np.repeat(mean_err_a2, len(a2_obs_data))
		posl=np.where(np.asarray(l) >= 2) # a4 makes sense only for l>=2, we need to filter out the values to select only l>=2 before the fit
		a4_obs_data=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a4_obs, posl).flatten(), 1)
		a4_sig_obs_data=mean_err_a4#np.repeat(mean_err_a4, len(a4_obs_data))
		posl=np.where(np.asarray(l) >= 3) # a6 makes sense only for l>=3, we need to filter out the values to select only l>=3 before the fit
		a6_obs_data=np.polyfit(np.take(nu_nl_obs, posl).flatten(), np.take(a6_obs, posl).flatten(), 1)
		a6_sig_obs_data=mean_err_a6#np.repeat(mean_err_a6, len(a6_obs_data))
	if fit_type == 'mean_nu_l': # Here, all of the data we use for the posterior are the results of the mean of <aj>_ln = cte
		a2_obs_data=np.mean(a2_obs)
		a2_sig_obs_data=mean_err_a2
		posl=np.where(np.asarray(l) >= 2) # a4 makes sense only for l>=2, we need to filter out the values to select only l>=2 before the fit
		a4_obs_data=np.mean(np.take(a4_obs, posl).flatten())
		a4_sig_obs_data=mean_err_a4
		posl=np.where(np.asarray(l) >= 3) # a4 makes sense only for l>=3, we need to filter out the values to select only l>=3 before the fit
		a6_obs_data=np.mean(np.take(a6_obs, posl).flatten())
		a6_sig_obs_data=mean_err_a6
	if outputfile != None:
		f=open(outputfile, 'w')
		f.write("# REDUCED Table of SIMULATED aj coefficient using a linear fit of the aj(nu,l) coefficients"+"\n")
		f.write("# Created using fit_a2sig.py : reduce_simfile()"+" Version 16 Dec 2021\n")
		f.write("# Original file was: " + simfile + "\n")
		f.write("! l = ")
		for el in l:
			f.write(" {0:1d}".format(el))			
		f.write("\n")
		f.write("! nu_nl_obs = ")
		for nu in nu_nl_obs:
			f.write(" {0:10.6f}".format(nu))
		f.write("\n")
		if fit_type == 'full_fit':
			f.write('# Model: aj(nu,l) = aj_0(l) + aj_1(l) * nu(n,l)')
			f.write("# Col(0):l, Col(1):a2_0(l), Col(2):a2_1(l), Col(3):a4_0(l), Col(4):a4_1(l), Col(5):a6_0(l), Col(6):a6_1(l), Col(7-8):err_a2, Col(9-10):err_a4, Col(11-12):err_a6\n")		
			for el in range(1, np.max(l)+1):
				f.write("{0:1d}  	 {1:0.6f}   {2:0.6f}   {3:0.6f}   {4:0.6f}   {5:0.6f}   {6:0.6f}   {7:0.6f}   {8:0.6f}   {9:0.6f}   {10:0.6f}   {11:0.6f}   {12:0.6f}\n".format(el, 
					a2_obs_data[el][0], a2_obs_data[el][1], a4_obs_data[el][0], a4_obs_data[el][1], a6_obs_data[el][0], a6_obs_data[el][1], 
					a2_sig_obs_data[el][0], a2_sig_obs_data[el][1],  a4_sig_obs_data[el][0], a4_sig_obs_data[el][1], a6_sig_obs_data[el][0], a6_sig_obs_data[el][1]))	
		if fit_type == 'mean_l':
			f.write("# Col(0):a2_0, Col(1):a2_1, Col(2):a4_0, Col(3):a4_1, Col(4):a6_0, Col(5):a6_1, Col(6-7):err_a2, Col(8-9):err_a4, Col(10-11):err_a6\n")		
			f.write('# Model: aj(nu) = aj_0 + aj_1 * nu(n,l)')
			f.write("{0:0.6f}   {1:0.6f}   {2:0.6f}   {3:0.6f}   {4:0.6f}   {5:0.6f}   {6:0.6f}   {7:0.6f}   {8:0.6f}   {9:0.6f}   {10:0.6f}   {11:0.6f}\n".format(
					a2_obs_data[0], a2_obs_data[1], a4_obs_data[0], a4_obs_data[1], a6_obs_data[0], a6_obs_data[1], 
					a2_sig_obs_data[0], a2_sig_obs_data[1],  a4_sig_obs_data[0], a4_sig_obs_data[1], a6_sig_obs_data[0], a6_sig_obs_data[1]))
		if fit_type == 'mean_nu_l':
			f.write('# Model: aj(nu,l) = constant = aj')
			f.write("# Col(0):a2, Col(1):a4, Col(2):a6, Col(3):err_a2, Col(4):err_a4, Col(5):err_a6\n")		
			f.write("{0:0.6f}   {1:0.6f}   {2:0.6f}   {3:0.6f}   {4:0.6f}   {5:0.6f}\n".format(
					a2_obs_data, a4_obs_data, a6_obs_data, a2_sig_obs_data, a4_sig_obs_data, a6_sig_obs_data))	
		f.close()		
	return l, nu_nl_obs, [a2_obs_data, a4_obs_data, a6_obs_data], [a2_sig_obs_data, a4_sig_obs_data, a6_sig_obs_data]


def test_do_simfile(err_type='medium', do_a4=False, do_a6=False, ftype='gate', index=1, lmax=2):
		if err_type != 'tiny' and err_type != 'small' and err_type != 'medium':
			print("Error in test_do_simfile: Please use either tiny, small or medium for the err_type argument" )
			exit()
		dir_out='/Users/obenomar/tmp/test_a2AR/acoefs_checks_theta/data/Simulations/' + ftype
		Dnu=85
		epsi0=0.25
		N0=8
		Nmax=14
		a1=0
		epsilon_nl=np.array([-1e-3, 0])
		# --- case of simu_tinyerrors_epsicte_2.txt ---
		#theta0=np.pi/2 - np.pi/6
		#delta=np.pi/8
		# --- case of simu_tinyerrors_epsicte_3.txt ---
		#theta0=np.pi/2  # Equatorial zone of activity
		#delta=np.pi/8
		# --- case of simu_tinyerrors_epsicte_3.txt ---
		theta0=0  # Polar zone of activity
		delta=np.pi/8
		
		if err_type == 'medium':
			fileout=dir_out + '/simu_mediumerrors_epsicte_' + str(index) + '.txt'
			relerr_a2=[0.02, 0.1] # 5nHz + 10% error
			relerr_a4=None
			if do_a4 == True:
				relerr_a4=[0.02, 0.1]
			if do_a6 == True:
				relerr_a4=[0.02, 0.1]
		if err_type == 'small':
			fileout=dir_out + '/simu_smallerrors_epsicte_' + str(index) + '.txt'
			relerr_a2=[0.005, 0.0] #
			relerr_a4=None
			if do_a4 == True:
				relerr_a4=[0.005, 0.0]
			if do_a6 == True:
				relerr_a6=[0.005, 0.0]
		if err_type == 'tiny':
			fileout=dir_out + '/simu_tinyerrors_epsicte_' + str(index) + '.txt'
			relerr_a2=[0.001, 0.0] #
			relerr_a4=None
			if do_a4 == True:
				relerr_a4=[0.0005, 0.0]
			if do_a6 == True:
				relerr_a6=[0.00025, 0.0]
		do_simfile(fileout, Dnu, epsi0, N0, Nmax, a1, epsilon_nl,  theta0, delta, ftype, do_a4=do_a4, do_a6=do_a6, 
			relerr_a2=relerr_a2, relerr_a4=relerr_a4, relerr_a6=relerr_a6, relerr_epsilon_nl=None, lmax=lmax)
		en, el, nu_nl_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs=read_obsfiles(fileout, read_a4=do_a4, read_a6=do_a6)
		#print(a6_obs, sig_a6_obs)
		#print('---')
		#exit()
		Dnu_obs=conditional_resize(Dnu, len(a2_obs))
		a1_obs=conditional_resize(a1, len(a2_obs))
		do_a2_model_plot(el, nu_nl_obs, Dnu, a1_obs, a2_obs, sig_a2_obs, None, ftype, fileout=fileout + '_a2.jpg') # The None is [epsi_nl0, epsi_nl1, theta0, delta] 
		if do_a4 == True:
			do_model_plot(el, nu_nl_obs, Dnu, a1_obs, a4_obs, sig_a4_obs, None, None, ftype, fileout=fileout, aj=4) # The two none are theta0, delta0
		if do_a6 == True:
			do_model_plot(el, nu_nl_obs, Dnu, a1_obs, a6_obs, sig_a6_obs, None, None, ftype, fileout=fileout, aj=6) # The two none are theta0, delta0

# The main function that will compute the statistical criteria for the maximisation procedure
#def do_stats(constants, variables):
def do_stats(variables,l, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype):
	#l, epsilon_nl0, epsilon_nl1, theta0, delta = variables
	#
	# Compute the priors
	P=priors_model(nu_nl_obs, epsilon_nl0, epsilon_nl1, theta0, delta)
	if np.isinf(P):
		return -np.inf
	if np.isnan(P):
		print("---- GOT A NaN in Prior ----")
		print("P = ", P)
		print("      epsilon_nl0 = ", epsilon_nl0)
		print("      epsilon_nl1 = ", epsilon_nl1)
		print("      theta0      = ", theta0)
		print('      delta       = ', delta)
		print('Imposing -infinity to the Posterior in order to reject the solution')
		return -np.inf
	#
	epsilon_nl0, epsilon_nl1, theta0, delta = variables
	a2_nl_mod=[]
	# Given the variables of the model, get a2 of the model a2_mod at the observed frequencies nu_nl_obs of each l and m modes
	for i in range(len(nu_nl_obs)):
		epsilon_nl=epsilon_nl0 + epsilon_nl1*nu_nl_obs[i]*1e-3 # linear term for epsilon_nl the 1e-3 is here for avoiding round off errors
		a2_mod=a2_model_cpp(nu_nl_obs[i], Dnu_obs[i], a1_obs[i], epsilon_nl, theta0, delta, ftype, l[i])
		a2_nl_mod.append(float(a2_mod)*1e3) #  convert a2 in nHz, because we assume here that nu_nl is in microHz
	# Compare the observed and theoretical a2 using the least square method
	L=likelihood_gauss(a2_nl_mod, a2_obs, sig_a2_obs)
	## Add priors
	#P=priors_model(nu_nl_obs, epsilon_nl0, epsilon_nl1, theta0, delta)
	if np.isnan(L):
		print("---- GOT A NaN on L ---")
		print("L = ", L, "     P = ", P)
		print("      epsilon_nl0 = ", epsilon_nl0)
		print("      epsilon_nl1 = ", epsilon_nl1)
		print("      theta0      = ", theta0)
		print('      delta       = ', delta)
		print('Imposing -infinity to the Posterior in order to reject the solution')
		Posterior=-np.inf
	#Posterior=P
	Posterior=L+P
	return Posterior

def do_stats_aj(variables, l, a1_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs,nu_nl_obs, Dnu_obs, do_a46, data_type, ftype, relax, var_init):
	return do_stats_ongrid_for_observations(variables, l, a1_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs,nu_nl_obs, Dnu_obs, 
		interpolator_l1=None, interpolator_l2=None, interpolator_l3=None, do_a4=do_a46[0], do_a6=do_a46[1], data_type=data_type, ftype=ftype, relax=relax, var_init=var_init)


def do_stats_ongrid(variables, l, a1_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs,nu_nl_obs, Dnu_obs, 
		interpolator_l1, interpolator_l2, interpolator_l3, do_a4=False, do_a6=False, fit_acoefs=-1):
	# Main function that handle creating a Posterior using observables for a_j , j=[2,4,6] and grids for Alm(theta,delta)
	# saved in the interpolator functions for l1, l2, l3.
	# fit_acoefs: Controls whether we compare directly the a2 coefficients or if we compare a linear fit 
	# 		(which was shown to be extremely accurate for describing Alm). 
	# 		If set to -1 (default), the full set of a coefficients is used for comparing individual aj coefficients (no fit)
	# 		If set to 0: It will fit the acoefficients with a l dependence (one fit per each l and per aj). 
	# 		If set to 1: It will fit the acoefficients without l dependence (one fit per each aj)
	#		If set to 2: It will compute use the mean of aj (fit with a constant) to compute the posterior
	a1_obs=np.asarray(a1_obs)
	a2_obs=np.asarray(a2_obs)
	Dnu_obs=np.asarray(Dnu_obs)
	sig_a2_obs=np.asarray(sig_a2_obs)
	a4_obs=np.asarray(a4_obs)
	sig_a4_obs=np.asarray(sig_a4_obs)
	a6_obs=np.asarray(a6_obs)
	sig_a6_obs=np.asarray(sig_a6_obs)
	#
	epsilon_nl0, epsilon_nl1, theta0, delta = variables
	#
	# Compute the priors
	P=priors_model(nu_nl_obs, epsilon_nl0, epsilon_nl1, theta0, delta)
	if np.isinf(P):
		print("------ Infinity in prior ----")
		print("      nu_nl_obs   = ", nu_nl_obs)
		print("      epsilon_nl0 = ", epsilon_nl0)
		print("      epsilon_nl1 = ", epsilon_nl1)
		print("      theta0      = ", theta0)
		print('      delta       = ', delta)
		print("Debug Exit")
		exit()
		return -np.inf
	if np.isnan(P):
		print("---- GOT A NaN in Prior ----")
		print("P = ", P)
		print("      epsilon_nl0 = ", epsilon_nl0)
		print("      epsilon_nl1 = ", epsilon_nl1)
		print("      theta0      = ", theta0)
		print('      delta       = ', delta)
		print('Imposing -infinity to the Posterior in order to reject the solution')
		return -np.inf
	#
	epsilon_nl0, epsilon_nl1, theta0, delta = variables
	a2_nl_mod=[]
	a4_nl_mod=[]
	a6_nl_mod=[]	
	a2_mod_data=[]
	a4_mod_data=[]
	a6_mod_data=[]
	a2_obs_data=[]
	a4_obs_data=[]
	a6_obs_data=[]
	a2_sig_obs_data=[]
	a4_sig_obs_data=[]
	a6_sig_obs_data=[]
	# Given the variables of the model, get a2 of the model a2_mod at the observed frequencies nu_nl_obs of each l and m modes
	for i in range(len(nu_nl_obs)):
		epsilon_nl=epsilon_nl0 + epsilon_nl1*nu_nl_obs[i]*1e-3 # linear term for epsilon_nl the 1e-3 is here for avoiding round off errors	
		acoefs=a_model_interpol(nu_nl_obs[i], Dnu_obs[i], a1_obs[i], epsilon_nl, theta0, delta, l[i], interpolator_l1, interpolator_l2, interpolator_l3)
		a2_nl_mod.append(float(acoefs[1])*1e3) #  convert a2 in nHz, because we assume here that nu_nl is in microHz
		a4_nl_mod.append(float(acoefs[3])*1e3) #  convert a4 in nHz, because we assume here that nu_nl is in microHz
		a6_nl_mod.append(float(acoefs[5])*1e3) #  convert a6 in nHz, because we assume here that nu_nl is in microHz
	#
	if fit_acoefs == -1: # Default: All of the data we use for the posterior are the raw a-coefficients and their errors
		a2_mod_data=a2_nl_mod
		a2_obs_data=a2_obs
		a2_sig_obs_data=sig_a2_obs
		a4_mod_data=a4_nl_mod
		a4_obs_data=a4_obs
		a4_sig_obs_data=sig_a4_obs
		a6_mod_data=a6_nl_mod
		a6_obs_data=a6_obs
		a6_sig_obs_data=sig_a6_obs
	if fit_acoefs == 0: # Here, all of the data we use for the posterior are the results of a 1st order polynomial fit to get a polynomial description of aj(nu,l)
		L_a2=0
		L_a4=0
		L_a6=0
		for el in range(1, np.max(l)+1):
			posl=np.where(np.asarray(l) == el)
			a2_obs_data, cov_a2=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a2_obs,posl).flatten(), 1, w=1/np.take(sig_a2_obs,posl).flatten()**2, cov=True)
			a2_sig_obs_data=np.sqrt(np.diag(cov_a2))
			a2_mod_data=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a2_nl_mod,posl).flatten(), 1)
			L_a2=L_a2 + likelihood_gauss(a2_mod_data, a2_obs_data, a2_sig_obs_data)
			if	do_a4 == True and el >= 2:
				a4_obs_data, cov_a4=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a4_obs,posl).flatten(), 1, w=1/np.take(sig_a4_obs,posl).flatten()**2, cov=True)
				#a4_obs_data, cov_a4=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a4_obs,posl).flatten(), 1, cov=True)
				a4_sig_obs_data=np.sqrt(np.diag(cov_a4))
				a4_mod_data=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a4_nl_mod,posl).flatten(), 1)
				L_a4=L_a4 + likelihood_gauss(a4_mod_data, a4_obs_data, a4_sig_obs_data)
			if	do_a6 == True and el >= 3:
				a6_obs_data, cov_a6=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a6_obs,posl).flatten(), 1, w=1/np.take(sig_a6_obs,posl).flatten()**2, cov=True)
				#a6_obs_data, cov_a6=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a6_obs,posl).flatten(), 1, cov=True)
				a6_sig_obs_data=np.sqrt(np.diag(cov_a6))
				a6_mod_data=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a6_nl_mod,posl).flatten(), 1)
				L_a6=L_a6 + likelihood_gauss(a6_mod_data, a6_obs_data, a6_sig_obs_data)
	if fit_acoefs == 1: # Here, all of the data we use for the posterior are the results of a 1st order polynomial fit to get a polynomial description of <aj(nu)>_l
		a2_obs_data, cov_a2=np.polyfit(nu_nl_obs, a2_obs, 1, w=1/sig_a2_obs**2, cov=True)
		a2_sig_obs_data=np.sqrt(np.diag(cov_a2))
		a2_mod_data=np.polyfit(nu_nl_obs, a2_nl_mod, 1)
		if	do_a4 == True:
			posl=np.where(np.asarray(l) >= 2) # a4 makes sense only for l>=2, we need to filter out the values to select only l>=2 before the fit
			a4_obs_data, cov_a4=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a4_obs, posl).flatten(), 1, w=1/np.take(sig_a4_obs, posl).flatten()**2, cov=True)
			a4_sig_obs_data=np.sqrt(np.diag(cov_a4))
			a4_mod_data=np.polyfit(np.take(nu_nl_obs, posl).flatten(), np.take(a4_nl_mod, posl).flatten(), 1)
		if do_a6 == True:
			posl=np.where(np.asarray(l) >= 3) # a6 makes sense only for l>=3, we need to filter out the values to select only l>=3 before the fit
			a6_obs_data, cov_a6=np.polyfit(np.take(nu_nl_obs, posl).flatten(), np.take(a6_obs, posl).flatten(), 1, w=1/np.take(sig_a6_obs, posl).flatten()**2, cov=True)
			a6_sig_obs_data=np.sqrt(np.diag(cov_a6))
			a6_mod_data=np.polyfit(np.take(nu_nl_obs, posl).flatten(), np.take(a6_nl_mod, posl).flatten(), 1)
	if fit_acoefs == 2: # Here, all of the data we use for the posterior are the results of the mean of <aj>_ln = cte
		a2_obs_data=np.mean(a2_obs)
		a2_sig_obs_data=np.sqrt(np.sum(sig_a2_obs**2)/len(sig_a2_obs))
		a2_mod_data=np.mean(a2_nl_mod)
		if do_a4 == True:
			posl=np.where(np.asarray(l) >= 2) # a4 makes sense only for l>=2, we need to filter out the values to select only l>=2 before the fit
			a4_obs_data=np.mean(np.take(a4_obs, posl).flatten())
			a4_sig_obs_data=np.sqrt(np.sum(np.take(sig_a4_obs, posl).flatten()**2)/len(np.take(sig_a4_obs, posl).flatten()))
			a4_mod_data=np.mean(np.take(a4_nl_mod, posl).flatten())
		if do_a6 == True:
			posl=np.where(np.asarray(l) >= 3) # a4 makes sense only for l>=3, we need to filter out the values to select only l>=3 before the fit
			a6_obs_data=np.mean(np.take(a6_obs, posl).flatten())
			a6_sig_obs_data=np.sqrt(np.sum(np.take(sig_a6_obs, posl).flatten()**2)/len(np.take(sig_a6_obs, posl).flatten()))
			a6_mod_data=np.mean(np.take(a6_nl_mod, posl).flatten())
	#
	if fit_acoefs != 0: # Likelihood computation for scenarii -1, 1, 2
		# Compare the observed and theoretical a2 using the least square method
		L_a2=likelihood_gauss(a2_mod_data, a2_obs_data, a2_sig_obs_data)
		L_a4=0
		L_a6=0
		if do_a4 == True:
			L_a4=likelihood_gauss(a4_mod_data, a4_obs_data, a4_sig_obs_data)
		if do_a6 == True:
			L_a6=likelihood_gauss(a6_mod_data, a6_obs_data, a6_sig_obs_data)
	# Compute the total likelihood at the end
	L=L_a2+L_a4+L_a6
	#
	if np.isnan(L):
		print("---- GOT A NaN on L = L_a2 + L_a4 + L_a6---")
		print("L = ", L, "     P = ", P)
		print("      epsilon_nl0 = ", epsilon_nl0)
		print("      epsilon_nl1 = ", epsilon_nl1)
		print("      theta0      = ", theta0)
		print('      delta       = ', delta)
		print('             L_a2 = ', L_a2)
		print('             L_a4 = ', L_a4)
		print('             L_a6 = ', L_a6)	
		print('Imposing -infinity to the Posterior in order to reject the solution')
		Posterior=-np.inf
	Posterior=L+P
	#Posterior=P
	return Posterior

def do_stats_ongrid_for_observations(variables, l, a1_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs,nu_nl_obs, Dnu_obs, 
		interpolator_l1=None, interpolator_l2=None, interpolator_l3=None, do_a4=False, do_a6=False, data_type='mean_nu_l', ftype=None, relax=None, var_init=None):
	# Main function that handle creating a Posterior using observables for a_j , j=[2,4,6] and grids for Alm(theta,delta)
	# saved in the interpolator functions for l1, l2, l3.
	# DIFFERS FROM do_stats_ongrid() by the fact that it is more adapted to analyse real results from a Power Spectrum fit: 
	#		In do_stats_ongrid(), the input file was supposed to contain **ALL** of the aj coefficient. Then the user could choose to 
	#			reduce the amount of information by performing linear fit or mean.
	#			Therefore here, this is the model that define the logic
	#     In do_stats_ongrid_for_observations(), the input is directly either a data set aj(nu,l) or aj(nu) or aj
	#			this is specified by the data_type arguments which allows the code to know how to proceed with the Alm grid
	#			Therefore here, this is the observation that define the logic
	# data_type: Controls whether the observations are made of aj(nu,l) or aj(nu) or aj=cte coefficients
	#		If set to "mean_nu_l" (default): It will compute use the mean of aj (fit with a constant) to compute the posterior
	# 		If set to mean_l: It will fit the acoefficients without l dependence (one fit per each aj)
	# 		If set to full_fit: It will fit the acoefficients with a l dependence (one fit per each l and per aj). 
	if relax == None:
		relax=np.repeat(True, len(variables))
	else:
		if var_init == None:
			print('Error in do_stats_ongrid_for_observations(): ')
			print('        When relax is specified, you also need to provide the initial vector of variables')
			print('        This vector is used to fix the variables that are not relaxed to their initial values')
			exit()
	if interpolator_l1 == None or interpolator_l2 == None or interpolator_l3 == None:
		if ftype != None:
			do_interpol = False
		else:
			print('Error in do_stats_ongrid_for_observations():')
			print('       You cannot set interpolators variables and ftype to None')
			print('       Please specify either ftype = gate/gaussian or all of the interpolators (l1, l2, l3)')
			exit()
	else:
		do_interpol = True
	Dnu_obs=np.asarray(Dnu_obs)
	a1_obs_data=np.asarray(a1_obs)
	a2_obs_data=np.asarray(a2_obs)
	a2_sig_obs_data=np.asarray(sig_a2_obs)
	a4_obs_data=np.asarray(a4_obs)
	a4_sig_obs_data=np.asarray(sig_a4_obs)
	a6_obs_data=np.asarray(a6_obs)
	a6_sig_obs_data=np.asarray(sig_a6_obs)
	# Whenever necessary, bypass the variable specified by the algorithm and use the initial one
	for i in range(len(variables)):
		if relax[i] == False:
			variables[i]=var_init[i]

	epsilon_nl0, epsilon_nl1, theta0, delta = variables
	#
	# Compute the priors
	P=priors_model(nu_nl_obs, epsilon_nl0, epsilon_nl1, theta0, delta)
	if np.isinf(P):
		print("------ Infinity in prior ----")
		print("      nu_nl_obs   = ", nu_nl_obs)
		print("      epsilon_nl0 = ", epsilon_nl0)
		print("      epsilon_nl1 = ", epsilon_nl1)
		print("      theta0      = ", theta0)
		print('      delta       = ', delta)
		print("Debug Exit")
		exit()
		return -np.inf
	if np.isnan(P):
		print("---- GOT A NaN in Prior ----")
		print("P = ", P)
		print("      nu_nl_obs   = ", nu_nl_obs)
		print("      epsilon_nl0 = ", epsilon_nl0)
		print("      epsilon_nl1 = ", epsilon_nl1)
		print("      theta0      = ", theta0)
		print('      delta       = ', delta)
		print('Imposing -infinity to the Posterior in order to reject the solution')
		return -np.inf
	#
	epsilon_nl0, epsilon_nl1, theta0, delta = variables
	a2_nl_mod=[]
	a4_nl_mod=[]
	a6_nl_mod=[]	
	a2_mod_data=[]
	a4_mod_data=[]
	a6_mod_data=[]
	# Given the variables of the model, get aj of the model aj_mod at the observed frequencies nu_nl_obs of each l and m modes
	for i in range(len(nu_nl_obs)):
		epsilon_nl=epsilon_nl0 + epsilon_nl1*nu_nl_obs[i]*1e-3 # linear term for epsilon_nl the 1e-3 is here for avoiding round off errors	
		#print("            {}   {}   {}   {}   {}".format(i, nu_nl_obs[i], Dnu_obs[i], a1_obs[i], l[i]))
		if do_interpol == True:
			acoefs=a_model_interpol(nu_nl_obs[i], Dnu_obs[i], a1_obs[i], epsilon_nl, theta0, delta, l[i], interpolator_l1, interpolator_l2, interpolator_l3)
		else:
			acoefs=a_model_cpp(nu_nl_obs[i], Dnu_obs[i], a1_obs[i], epsilon_nl, theta0, delta0, ftype, l[i])
		a2_nl_mod.append(float(acoefs[1])*1e3) #  convert a2 in nHz, because we assume here that nu_nl is in microHz
		a4_nl_mod.append(float(acoefs[3])*1e3) #  convert a4 in nHz, because we assume here that nu_nl is in microHz
		a6_nl_mod.append(float(acoefs[5])*1e3) #  convert a6 in nHz, because we assume here that nu_nl is in microHz
	#
	if data_type == "mean_nu_l": # Here, all of the data we use for the posterior are the results of the mean of <aj>_ln = cte
		a2_mod_data=np.mean(a2_nl_mod)
		if do_a4 == True:
			posl=np.where(np.asarray(l) >= 2) # a4 makes sense only for l>=2, we need to filter out the values to select only l>=2 before the fit
			a4_mod_data=np.mean(np.take(a4_nl_mod, posl).flatten())
		if do_a6 == True:
			posl=np.where(np.asarray(l) >= 3) # a6 makes sense only for l>=3, we need to filter out the values to select only l>=3 before the fit
			a6_mod_data=np.mean(np.take(a6_nl_mod, posl).flatten())
	#
	if data_type == "mean_l": # Here, all of the data we use for the posterior are the results of a 1st order polynomial fit to get a polynomial description of <aj(nu)>_l
		a2_mod_data=np.polyfit(nu_nl_obs, a2_nl_mod, 1)
		if	do_a4 == True:
			posl=np.where(np.asarray(l) >= 2) # a4 makes sense only for l>=2, we need to filter out the values to select only l>=2 before the fit
			a4_mod_data=np.polyfit(np.take(nu_nl_obs, posl).flatten(), np.take(a4_nl_mod, posl).flatten(), 1)
		if do_a6 == True:
			posl=np.where(np.asarray(l) >= 3) # a6 makes sense only for l>=3, we need to filter out the values to select only l>=3 before the fit
			a6_mod_data=np.polyfit(np.take(nu_nl_obs, posl).flatten(), np.take(a6_nl_mod, posl).flatten(), 1)
	#
	if data_type == "full_fit": # Here, all of the data we use for the posterior are the results of a 1st order polynomial fit to get a polynomial description of aj(nu,l)
		L_a2=0
		L_a4=0
		L_a6=0
		for el in range(1, np.max(l)+1):
			posl=np.where(np.asarray(l) == el)
			a2_mod_data=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a2_nl_mod,posl).flatten(), 1)
			L_a2=L_a2 + likelihood_gauss(a2_mod_data, a2_obs_data, a2_sig_obs_data)
			if	do_a4 == True and el >= 2:
				a4_mod_data=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a4_nl_mod,posl).flatten(), 1)
				L_a4=L_a4 + likelihood_gauss(a4_mod_data, a4_obs_data, a4_sig_obs_data)
			if	do_a6 == True and el >= 3:
				a6_mod_data=np.polyfit(np.take(nu_nl_obs,posl).flatten(), np.take(a6_nl_mod,posl).flatten(), 1)
				L_a6=L_a6 + likelihood_gauss(a6_mod_data, a6_obs_data, a6_sig_obs_data)
			#print(" el = ", el)
			#print("a2_obs_data = ", a2_obs_data)
			#print("a4_obs_data = ", a4_obs_data)
			#print("a6_obs_data = ", a6_obs_data)
			#print("a2_sig_obs_data = ", a2_sig_obs_data)
			#print("a4_sig_obs_data = ", a4_sig_obs_data)
			#print("a6_sig_obs_data = ", a6_sig_obs_data)
			#print("a2_mod_data = ", a2_mod_data)
			#print("a4_mod_data = ", a4_mod_data)
			#print("a6_mod_data = ", a6_mod_data)
			#print("L_a2 =  ", L_a2)
			#print("L_a4 =  ", L_a4)
			#print("L_a6 =  ", L_a6)
			#print("-------")
	#
	if data_type != "full_fit": # Likelihood computation for scenarii -1, 1, 2
		# Compare the observed and theoretical a2 using the least square method
		L_a2=likelihood_gauss(a2_mod_data, a2_obs_data, a2_sig_obs_data)
		L_a4=0
		L_a6=0
		#print("a2_mod_data, a2_obs_data, a2_sig_obs_data : ", a2_mod_data, a2_obs_data, a2_sig_obs_data)
		if do_a4 == True:
			L_a4=likelihood_gauss(a4_mod_data, a4_obs_data, a4_sig_obs_data)
			#print("a4_mod_data, a4_obs_data, a4_sig_obs_data : ", a4_mod_data, a4_obs_data, a4_sig_obs_data)
		if do_a6 == True:
			L_a6=likelihood_gauss(a6_mod_data, a6_obs_data, a6_sig_obs_data)
			#print("a6_mod_data, a6_obs_data, a6_sig_obs_data : ", a6_mod_data, a6_obs_data, a6_sig_obs_data)
	# Compute the total likelihood at the end
	L=L_a2+L_a4+L_a6
	#
	if np.isnan(L):
		print("---- GOT A NaN on L = L_a2 + L_a4 + L_a6---")
		print("L = ", L, "     P = ", P)
		print("      epsilon_nl0 = ", epsilon_nl0)
		print("      epsilon_nl1 = ", epsilon_nl1)
		print("      theta0      = ", theta0)
		print('      delta       = ', delta)
		print('             L_a2 = ', L_a2)
		print('             L_a4 = ', L_a4)
		print('             L_a6 = ', L_a6)	
		print('Imposing -infinity to the Posterior in order to reject the solution')
		Posterior=-np.inf
	Posterior=L+P
	#Posterior=P
	return Posterior

def do_minimise(constants, variables_init):
	l, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype=constants
	epsilon_nl0_init, epsilon_nl1_init, theta0_init, delta_init = variables_init
	nll = lambda *args: -do_stats(*args)
	initial = np.array([epsilon_nl0_init, epsilon_nl1_init, theta0_init, delta_init])
	soln = minimize(nll, initial, args=(l, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype), method="Powell", 
		options={'xtol': 0.00001, 'ftol': 0.00001})
	#outputs_ml, log_proba_ml = soln.x
	return soln

def do_minimise_aj(constants, variables_init, relax, do_a46=[False, False]):
	# used_aj (Boolean): [use_a2, use_a4, use_a6]
	l, a1_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs,nu_nl_obs, Dnu_obs, ftype, data_type=constants
	if do_a46[0] == False:
		a4_obs=[]
		sig_a4_obs=[]
	if do_a46[1] == False:
		a6_obs=[]
		sig_a6_obs=[]
	epsilon_nl0_init, epsilon_nl1_init, theta0_init, delta_init = variables_init
	nll = lambda *args: -do_stats_aj(*args)
	#variables, l, a1_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs,nu_nl_obs, Dnu_obs, do_a46, data_type, ftype
	initial = np.array([epsilon_nl0_init, epsilon_nl1_init, theta0_init, delta_init])
	soln = minimize(nll, initial, args=(l, a1_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs, nu_nl_obs, Dnu_obs, do_a46, data_type, ftype, relax, variables_init), method="Powell", 
		options={'xtol': 0.00001, 'ftol': 0.00001})
	#outputs_ml, log_proba_ml = soln.x
	return soln


def do_emcee(constants, variables_init, nwalkers=100, niter=5000):
	#l, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype=constants
	a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype=constants
	#epsilon_nl0_init, epsilon_nl1_init, theta0_init, delta_init = variables_init

	init_vars = variables_init + 1e-4 * np.random.randn(nwalkers, len(variables_init))
	nwalkers, ndim = init_vars.shape
	with Pool() as pool:
		sampler = emcee.EnsembleSampler(
		    nwalkers, ndim, do_stats, pool=pool, args=(l, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype))
		sampler.run_mcmc(init_vars, niter, progress=True)
	return sampler

def get_errors(hessian):
	# From the hessian, compute the errors (symetrical)
	# see eg. /volume1/homes/dataonly/Pro/Collaborations/PGaulme/Programs/Preset-Analysis-v7.7/Powell_Harvey_fit_v6.1/hessian_calculus_s1_old.pro
	# for a more complete example on how to transform the hessian into errors in more complex situation (eg. asymetrical errors)
	inverse = np.linalg.inv(hessian)
	n_params=len(hessian[0,:])
	errors=np.zeros(n_params)
	for i in range(n_params):
			errors[k] = np.sqrt(hess[k,k])
	return errors

def do_a2_model_plot(el, nu_nl_obs, Dnu_obs, a1_obs, a2_obs, sig_a2_obs, med_params, ftype, fileout='model_plot.jpg'):
	#med_params[2]=np.arcsin(med_params[2])
		# The plot of the fit with some randomly chosen fits to represent the variance
	fig, ax = plt.subplots()
	a2_nl_mod_best_cpp=[]
	try:
		for i in range(len(nu_nl_obs)):
			epsilon_nl=med_params[0] + med_params[1]*nu_nl_obs[i]*1e-3 # linear term for epsilon_nl the 1e-3 is here to avoid round off errors on the params
			a2_mod_cpp=a2_model_cpp(nu_nl_obs[i], Dnu_obs[i], a1_obs[i], epsilon_nl, med_params[2], med_params[3], ftype, el[i])
			a2_nl_mod_best_cpp.append(float(a2_mod_cpp)*1e3)
		ax.plot(nu_nl_obs, a2_nl_mod_best_cpp, "ro")#, '--', color='blue')
	except:
		print("Warning: No parameters provided. The plot will only show the data")
	ax.errorbar(nu_nl_obs, a2_obs, yerr=sig_a2_obs, fmt=".k", capsize=0)
	fig.savefig(fileout)

def do_model_plot(el, nu_nl_obs, Dnu_obs, a1_obs, aj_obs, sig_aj_obs, theta0, delta0, ftype, fileout='model_plot', aj=2):
	fig, ax = plt.subplots()
	aj_nl_mod_best_cpp=[]
	try:
		for i in range(len(nu_nl_obs)):
			epsilon_nl=med_params[0] + med_params[1]*nu_nl_obs[i]*1e-3 # linear term for epsilon_nl the 1e-3 is here to avoid round off errors on the params
			acoefs=a_model_cpp(nu_nl_obs[i], Dnu_obs[i], a1_obs[i], epsilon_nl, theta0, delta0, ftype, el[i])
			aj_nl_mod_best_cpp.append(float(acoefs[j-1])*1e3)
		ax.plot(nu_nl_obs, aj_nl_mod_best_cpp, "ro")#, '--', color='blue')
	except:
		print("Warning: No parameters provided. The plot will only show the data")
	#print("aj_obs :", aj_obs )
	#print("sig_aj_obs :", sig_aj_obs )
	ax.errorbar(nu_nl_obs, aj_obs, yerr=sig_aj_obs, fmt=".k", capsize=0)
	fig.savefig(fileout + '_a'+str(aj)+ '.jpg')

def conditional_resize(x, req_length):
	try:
		length=len(x)
		if length != req_length:
			y=np.repeat(x[0],req_length)
		else:
			y=x
	except:
		y=np.repeat(x,req_length)
	return y

def do_posterior_map_preset():
	#
	dir_core='/Users/obenomar/tmp/test_a2AR/acoefs_checks_theta/'
	#
	# To change only if you change the observables:
	# ------- Case of intermediate activity zone: theta0=60 deg, delta=22 deg -------
	#obsfile=dir_core+'data/Simulations/gate/simu_tinyerrors_epsicte_2.txt' # This files contains a2,a4 and a6 with tiny uncertainties ==> Used to test the accuracy of the approximation in aj(nu,l)
	#dir_posteriors=dir_core + 'grids_posterior/gate/theta0_1.04_delta0_0.39/effects_of_acoefs_modeling/'
	# ------- Case of Equatorial activity zone: theta0=90 deg, delta=22 deg -------
	#obsfile=dir_core+'data/Simulations/gate/simu_tinyerrors_epsicte_3.txt' # This files contains a2,a4 and a6 with tiny uncertainties ==> Used to test the accuracy of the approximation in aj(nu,l)
	#dir_posteriors=dir_core + 'grids_posterior/gate/theta0_1.57_delta0_0.39/effects_of_acoefs_modeling/'
	# ------- Case of Polar activity zone: theta0=0 deg, delta=22 deg -------
	obsfile=dir_core+'data/Simulations/gate/simu_tinyerrors_epsicte_4.txt' # This files contains a2,a4 and a6 with tiny uncertainties ==> Used to test the accuracy of the approximation in aj(nu,l)
	dir_posteriors=dir_core + 'grids_posterior/gate/theta0_0.00_delta0_0.39/effects_of_acoefs_modeling/'
	#
	Dnu_obs=85 # As per set in the simulations
	a1_obs=0   # As per set in the simulations
	epsilon_nl0=-0.001 # As per set in the simulations
	epsilon_nl1=0
	#
	# To change only if you change the grids of Alm:
	dir_grids=dir_core+"/grids/gate/800pts/"
	Almgridfiles=[dir_grids + 'grid_Alm_l1_thetapi_div2_deltapi_div4_800pts.npz', dir_grids + 'grid_Alm_l2_thetapi_div2_deltapi_div4_800pts.npz', dir_grids + 'grid_Alm_l3_thetapi_div2_deltapi_div4_800pts.npz']
	#
	# To change in function of the tested scenario:
	
	# ----------------- fit with polynomial accounting for nu and l ------------------------
	# --------- This should be equivalent to fitting directly individual a-coefficients ----
	# --------------------------------------------------------------------------------------
	# --- Best case Scenario With 1st order polynoms--- # 
	do_a4=True # We use a4
	do_a6=True # We use a6
	fit_acoefs=0 # We fit full polynomial on nu and l
	posterior_outfile=dir_posteriors + 'nu_and_l_dependence/posterior_800pts_theta2pi_div3_tinyerrors_a2a4a6fit.npz'
	do_posterior_map(Almgridfiles, obsfile, Dnu_obs, a1_obs, epsilon_nl0, epsilon_nl1, posterior_outfile=posterior_outfile, do_a4=do_a4, do_a6=do_a6, fit_acoefs=fit_acoefs)

	# --- removing a6 case Scenario With 1st order polynoms--- # 
	do_a4=True # We use a4
	do_a6=False # We DO NOT use a6
	fit_acoefs=0 # We fit full polynomial on nu and l
	posterior_outfile=dir_posteriors + 'nu_and_l_dependence/posterior_800pts_theta2pi_div3_tinyerrors_a2a4fit.npz'
	do_posterior_map(Almgridfiles, obsfile, Dnu_obs, a1_obs, epsilon_nl0, epsilon_nl1, posterior_outfile=posterior_outfile, do_a4=do_a4, do_a6=do_a6, fit_acoefs=fit_acoefs)

	# --- removing a6 case Scenario With 1st order polynoms--- # 
	do_a4=False # We use a4
	do_a6=False # We DO NOT use a6
	fit_acoefs=0 # We fit full polynomial on nu and l
	posterior_outfile=dir_posteriors + 'nu_and_l_dependence/posterior_800pts_theta2pi_div3_tinyerrors_a2fit.npz'
	do_posterior_map(Almgridfiles, obsfile, Dnu_obs, a1_obs, epsilon_nl0, epsilon_nl1, posterior_outfile=posterior_outfile, do_a4=do_a4, do_a6=do_a6, fit_acoefs=fit_acoefs)
	#
	# ----------------- fit with polynomial with nu dependence only (l depedence dropped) ------------------
	# --------------- This is to test whether we can still constrain the activity that way  ----------------
	# ------------- It is also similar to the initial fit I made with MCMC that show trends ----------------
	# ------------------------------------------------------------------------------------------------------
	# --- Best case Scenario With 1st order polynoms--- # 
	do_a4=True # We use a4
	do_a6=True # We use a6
	fit_acoefs=1 # We fit polynomial on nu only
	posterior_outfile=dir_posteriors + 'nu_dependence_only/posterior_800pts_theta2pi_div3_tinyerrors_a2a4a6fit.npz'
	do_posterior_map(Almgridfiles, obsfile, Dnu_obs, a1_obs, epsilon_nl0, epsilon_nl1, posterior_outfile=posterior_outfile, do_a4=do_a4, do_a6=do_a6, fit_acoefs=fit_acoefs)

	# --- removing a6 case Scenario With 1st order polynoms--- # 
	do_a4=True # We use a4
	do_a6=False # We DO NOT use a6
	fit_acoefs=1 # We fit polynomial on nu
	posterior_outfile=dir_posteriors + 'nu_dependence_only/posterior_800pts_theta2pi_div3_tinyerrors_a2a4fit.npz'
	do_posterior_map(Almgridfiles, obsfile, Dnu_obs, a1_obs, epsilon_nl0, epsilon_nl1, posterior_outfile=posterior_outfile, do_a4=do_a4, do_a6=do_a6, fit_acoefs=fit_acoefs)

	# --- removing a6 case Scenario With 1st order polynoms--- # 
	do_a4=False # We use a4
	do_a6=False # We DO NOT use a6
	fit_acoefs=1 # We fit polynomial on nu
	posterior_outfile=dir_posteriors + 'nu_dependence_only/posterior_800pts_theta2pi_div3_tinyerrors_a2fit.npz'
	do_posterior_map(Almgridfiles, obsfile, Dnu_obs, a1_obs, epsilon_nl0, epsilon_nl1, posterior_outfile=posterior_outfile, do_a4=do_a4, do_a6=do_a6, fit_acoefs=fit_acoefs)

	# -------------- fit with polynomial with mean values only (nu and l depedence dropped) ----------------
	# --------------- This is to test whether we can still constrain the activity that way  ----------------
	# ------------------------------------------------------------------------------------------------------
	# --- Best case Scenario With 1st order polynoms--- # 
	do_a4=True # We use a4
	do_a6=True # We use a6
	fit_acoefs=2 # Use of the mean only
	posterior_outfile=dir_posteriors + 'mean_only/posterior_800pts_theta2pi_div3_tinyerrors_a2a4a6fit.npz'
	do_posterior_map(Almgridfiles, obsfile, Dnu_obs, a1_obs, epsilon_nl0, epsilon_nl1, posterior_outfile=posterior_outfile, do_a4=do_a4, do_a6=do_a6, fit_acoefs=fit_acoefs)

	# --- removing a6 case Scenario With 1st order polynoms--- # 
	do_a4=True # We use a4
	do_a6=False # We DO NOT use a6
	fit_acoefs=2 # We fit polynomial on nu
	posterior_outfile=dir_posteriors + 'mean_only/posterior_800pts_theta2pi_div3_tinyerrors_a2a4fit.npz'
	do_posterior_map(Almgridfiles, obsfile, Dnu_obs, a1_obs, epsilon_nl0, epsilon_nl1, posterior_outfile=posterior_outfile, do_a4=do_a4, do_a6=do_a6, fit_acoefs=fit_acoefs)

	# --- removing a6 case Scenario With 1st order polynoms--- # 
	do_a4=False # We use a4
	do_a6=False # We DO NOT use a6
	fit_acoefs=2 # Use of the mean only
	posterior_outfile=dir_posteriors + 'mean_only/posterior_800pts_theta2pi_div3_tinyerrors_a2fit.npz'

def do_posterior_map_bias_grid(theta0_list, delta_list, Dnu_obs=85, epsi0=0, N0=10, Nmax=20, a1_obs=0, epsilon_nl=[-0.001, 0], fit_type='mean_nu_l', filter_type='gate', update_grid=False, verbose=True, do_a4=True, do_a6=True, err=[1e-1, 1e-1, 1e-1]):
	# Function that make a grid Posterior in function of theta0 and delta, as per provided by the user
	# It (1) Generate an emsemble of artificial data for each (theta0_list[i],delta_list[i])  and for the provided epsilion_nl. The content is saved in the dir_obs + '/raw_ajnl'
	#    (2) In function of the fit_type, it reduces those data to the requested type, providing tiny gaussian errors on the coefficients: 
	#					(a) 'mean_nu_l': data are saved in dir_obs + '/mean_aj' ==> A single value of aj per j
	#					(b) 'mean_l': data are saved in dir_obs + '/aj_n'  ==> Two parameters per j such that it is a linear function aj(nu) = aj_0 + aj_1 * nu   
	#					(c) 'full_fit': data are save in dir_obs + '/ajnl' ==> l*2 parameters aj(nu) = aj_0l + aj_1l * nu
	#        The case of a fit of the raw_ajnl is not considered here because this is not what we do observationally
	#    (3) It fits those data made in (2) using the gaussian likelihood function and generate a posterior distribution for each values in theta0_list and in delta_list
	#			All combinations are made so we get Ntheta0*Ndelta outputs classified as follow:
	#					dir_posterior '\'	  theta0_list[0]   ----> delta_list[0] = File posteriors + 2 images for output in theta0 and in delta with the expected input
	#											  					    ----> delta_list[1] = 
	#												  				    ....
	#																    ----> delta_list[N-1]
	#												    ....
	#										'\' theta0_list[N-1]  ----> delta_list[0]
	#												  				    ....
	#																    ----> delta_list[N-1]
	dir_core='/Users/obenomar/tmp/test_a2AR/acoefs_checks_theta/'
	if filter_type == 'gate':
		# Main output directory for the results. Subdirectories will be created there
		#dir_posteriors=dir_core + '/data/Simulations/' + filter_type + '/15-Dec-2021_gridtest_theta0_delta/grids_posterior/' 
		dir_posteriors=dir_core + '/data/Simulations/' + filter_type + '/Sun/22-Dec-2021/grids_posterior/' 
		# To change only if you change the grids of Alm:
		dir_grids=dir_core+"/grids/gate/800pts/"	
		Almgridfiles=[dir_grids + 'grid_Alm_l1_thetapi_div2_deltapi_div4_800pts.npz', dir_grids + 'grid_Alm_l2_thetapi_div2_deltapi_div4_800pts.npz', dir_grids + 'grid_Alm_l3_thetapi_div2_deltapi_div4_800pts.npz']
	else:
		print("filter_type != gate is not implemented yet")
		exit()
	# Create the directories necessary for the observation files, if they do not exist yet
	#os.mkdir(dir_obs + 'raw_ajnl')
	if fit_type == 'mean_nu_l':
		dir_fit_type='mean_aj'
	if fit_type == 'mean_l':
		dir_fit_type='aj_n'
	if fit_type == 'full_fit':
		dir_fit_type='ajnl'
	#
	dir_res=dir_posteriors + '/epsilon_nl' + str(epsilon_nl[0]) + '_' + str(epsilon_nl[1]) + '/' + dir_fit_type
	combi_file=dir_res + '/combinations.txt'
	# Create a first ensemble of directories that encode the settings
	combi_exist=os.path.exists(combi_file)
	if combi_exist == False:
		data_table, combi_file, dir_raw, status=make_dir_tree_bias_grid(dir_posteriors, theta0_list, delta_list, epsilon_nl, dir_fit_type, combi_file=combi_file, debug=False)
		print("  Combinations written in :", combi_file)
		print("  Please use this file if you want to read the structure of the grid or if you want to update the table if it was unfinished")
	else:
		if update_grid == False:
			print('Error: A combination file was found but restoration/continuation of a grid was not requested.')
			print("       Please check your parameters. ")
			print("       Combination File: ", combi_file)
			print("       The program will stop now")
			exit()
		else:
			data_table, combi_file, dir_raw, status=make_dir_tree_bias_grid(dir_posteriors, theta0_list, delta_list, epsilon_nl, dir_fit_type, combi_file=combi_file, debug=False)
			print(' Warning WARNING WARNING WARNING')
			print("  At the moment the restoration of a grid is not functional")
			print("  We basically overwrite the existing combination file")
			print("  Later implementation should read the existing combination file")
			print("  Pursuing...")

	for d in data_table:
		theta0=d[0]
		delta0=d[1]
		outdir=d[2]
		rawobsfile=dir_raw + '/' + str(theta0) + '_' + str(delta0) + '_raw.data'
		obsfile=outdir + '/' + str(theta0) + '_' + str(delta0) + '_' + dir_fit_type + '.data'
		print(' (theta0, delta0) = ({} , {})'.format(theta0, delta0))
		if os.path.exists(rawobsfile) == False: # Do the raw data only if it does not exist 
			do_simfile(rawobsfile, Dnu_obs, epsi0, N0, Nmax, a1_obs, epsilon_nl,  theta0, delta0, filter_type, do_a4=True, do_a6=True, 
				relerr_a2=None, relerr_a4=None, relerr_a6=None, relerr_epsilon_nl=None, lmax=3)
		else:
			if verbose == True:
				print("   Warning: Data found for the raw observation file:", rawobsfile)
				print("            The existing file will be used. Please check if this is the expected behavior")
		if os.path.exists(obsfile) == False: # Do the reduced grid data only if it does not exist 
			el, nu_nl_obs, aj, err_aj=reduce_simfile(rawobsfile, fit_type, err=err, outputfile=obsfile)
			# Perform the posterior estimation and do a basic plotting
			posterior_outfile=outdir + '/posteriors.npz'
			print('aj=', aj)
			print('err_aj =', err_aj)
			do_posterior_map_for_observation(Almgridfiles, el, nu_nl_obs, aj, err_aj, Dnu_obs, a1_obs, epsilon_nl[0], epsilon_nl[1], posterior_outfile=posterior_outfile, do_a4=do_a4, do_a6=do_a6, data_type=fit_type)
		else:
			print("   Grid data found... Not going to calculate it")

def make_dir_tree_bias_grid(dir_posteriors, theta0_list, delta_list, epsilon_nl, dir_fit_type, combi_file=None, debug=False):
	dir_posteriors_root=dir_posteriors  +  '/epsilon_nl' + str(epsilon_nl[0]) + '_' + str(epsilon_nl[1]) +'/'
	exist=[False, False, False, False, False] # [exist_espilon_nl, exist_dir_fit_type, exist_raw_ajnl, exist_theta0, exist_delta]
	#
	print('Main directory for input/outputs :')
	print('  ', dir_posteriors_root)
	try:
		os.mkdir(dir_posteriors + '/epsilon_nl' + str(epsilon_nl[0]) + '_' + str(epsilon_nl[1]))
	except FileExistsError:
		exist[0] = True 
	try:
		dir_res=dir_posteriors + '/epsilon_nl' + str(epsilon_nl[0]) + '_' + str(epsilon_nl[1]) + '/' + dir_fit_type
		os.mkdir(dir_res )
	except FileExistsError:
		exist[1] = True
		if debug == True:
			print('Warning: Directory for epsilon_nl =', epsilon_nl, ' already exist in ', dir_fit_type)
	#
	if combi_file == None:
		combi_file=dir_res + '/combinations.txt'
	#
	# Directories for RAW observations:
	try:
		dir_raw=dir_posteriors_root + '/raw_ajnl'
		os.mkdir(dir_raw)
	except FileExistsError:
		exist[2]=True
		if debug == True:
			print("WARNING: Directory for the RAW observations exist:")
			print("        ", dir_posteriors_root + '/raw_ajnl')
	#
	data=""
	data_table=[]
	for theta0 in theta0_list:
		if debug == True:
			print('theta0:', theta0, '...')
		# Ensure that the directories are available for outputs
		dir_theta0=dir_res + '/theta' + str(theta0)
		try:
			os.mkdir(dir_theta0)
		except:
			exist[3]=True
			if debug == True:
				print('			Warning: directory for ', theta0, 'already exist in ', dir_res)

		for delta0 in delta_list:
			if debug == True:
				print('    delta0: ', delta0)
			outdir=dir_theta0 + '/delta' + str(delta0)
			try:
				os.mkdir(outdir)
			except:
				exist[4]=True
				if debug == True:
					print('			Warning: directory for ', delta0, 'already exist in ', dir_theta0)
			#
			data=data+"{}   {}   {}\n".format(theta0, delta0, outdir)
			data_table.append([theta0, delta0, outdir])
	header='# File containing a summary of all combinations made for a specific test\n'
	header=header+'epsilon_nl={} {}\n'.format(epsilon_nl[0], epsilon_nl[1])
	header=header+'dir_raw='+dir_raw+'\n'
	header=header+'# theta0    /   delta     / dir_data\n'
	f=open(combi_file, 'w')
	f.write(header)
	f.write(data)
	f.close()
	return data_table, combi_file, dir_raw, exist
#
def test_make_dir_tree_bias_grid():
	dir_posteriors='/Users/obenomar/tmp2/dir_post/'
	dir_fit_type='mean_nl'
	epsilon_nl=[-.001,0]
	theta0_list=[0, 25, 50]
	delta_list=[5, 20]
	data_table, combi_file, dir_raw, exist=make_dir_tree_bias_grid(dir_posteriors, theta0_list, delta_list, epsilon_nl, dir_fit_type, combi_file=None, debug=True)
	print('exist =', exist)
	print('dir_raw =', dir_raw)
	print("combi_file =", combi_file)
	print("data_table:")
	print("  # theta0  / delta /  dir_data")
	for d in data_table:
		print(d)

def test_do_posterior_map_bias_grid(update_grid=False):
	theta0_list=np.asarray([0, 25, 50, 75])*np.pi/180.
	delta_list=np.asarray([10, 20, 30])*np.pi/180.
	Dnu_obs=135.1 #85
	epsi0=0.3
	N0=10
	Nmax=20
	a1_obs=0. 
	epsilon_nl=np.asarray([-0.001, 0])
	fit_type='mean_nu_l'
	err=[15, 20, 1]
	#fit_type='mean_l'
	#err=[[0.1,0.5], [0.2, 0.4], [0.3,0.6]]
	filter_type='gate'
	do_a4=True
	do_a6=False
	do_posterior_map_bias_grid(theta0_list, delta_list, Dnu_obs=Dnu_obs, epsi0=epsi0, N0=N0, Nmax=Nmax, 
		a1_obs=a1_obs, epsilon_nl=epsilon_nl, fit_type=fit_type, filter_type=filter_type, update_grid=update_grid, do_a4=do_a4, do_a6=do_a6, err=err)

def do_posterior_map_for_observation(Almgridfiles, el , nu_nl_obs, aj, err_aj, Dnu_obs, a1_obs, epsilon_nl0, epsilon_nl1, posterior_outfile='posterior_grid.npz', do_a4=False, do_a6=False, data_type='mean_nu_l'):
	# One grid for each l in principle. It is up to the user to make sure to have them in the increasing l order: l=1, 2, 3
	Alm_grid_l1=np.load(Almgridfiles[0])# Note: ftype = 'gauss' or 'gate' depends on the Alm grid file. No need to specify it
	Alm_grid_l2=np.load(Almgridfiles[1])# Note: ftype = 'gauss' or 'gate' depends on the Alm grid file. No need to specify it
	if do_a6 == True or len(Almgridfiles) == 3: # We still initialise the l=3 grid if the file is provided by the user
		if len(Almgridfiles) >= 3:
			Alm_grid_l3=np.load(Almgridfiles[2])# Note: ftype = 'gauss' or 'gate' depends on the Alm grid file. No need to specify it
		else:
			print("Error in do_posterior_map() : do_a6 = True but Almgridfiles has a size <3 and is therefore missing the grid file for l=3")
			print("The program will exit now")
			exit()
	Ndelta=len(Alm_grid_l1['delta'])
	Ntheta=len(Alm_grid_l1['theta'])
	#
	if data_type == 'mean_nu_l' or data_type == 'mean_l':
		a2_obs=np.asarray([aj[0]])
		sig_a2_obs=np.asarray([err_aj[0]])
		a4_obs=np.asarray([aj[1]])
		sig_a4_obs=np.asarray([err_aj[2]])
		a6_obs=np.asarray([aj[2]])
		sig_a6_obs=np.asarray([err_aj[2]])		
	else:
		print('aj =', aj)
		print('err_aj = ', err_aj)
		print("data_type != mean_nu_l is not properly implemented in do_posterior_map_for_observation() yet")
		print("It is required to (1) evaluate the structure of the input aj and err_aj. Then (2) adjust a2_obs, a4_obs, a6_obs, sig_a2_obs, etc...")
		print("The program will stop now")
		exit()
	Dnu_obs=conditional_resize(Dnu_obs, len(nu_nl_obs))
	a1_obs=conditional_resize(a1_obs, len(nu_nl_obs))
	#labels = ["epsilon_nl0", "epsilon_nl1", "theta0", "delta"]
	#
	# Initialise the interpolator once
	print("  - Initializing the interpolators for l=1,2,3...")
	funcs_l1=[]
	l=1
	for m in range(-l,l+1):
		Alm_flat=[]
		for j in range(Ndelta):
			Alm_flat.append(Alm_grid_l1['Alm'][l+m,:,j])
		funcs_l1.append(interpolate.interp2d(Alm_grid_l1['theta'], Alm_grid_l1['delta'], Alm_flat, kind='cubic'))
	funcs_l2=[]
	l=2
	for m in range(-l,l+1):
		Alm_flat=[]
		for j in range(Ndelta):
			Alm_flat.append(Alm_grid_l2['Alm'][l+m,:,j])
		funcs_l2.append(interpolate.interp2d(Alm_grid_l2['theta'], Alm_grid_l2['delta'], Alm_flat, kind='cubic'))
	funcs_l3=[]
	l=3
	if do_a6 == True or len(Almgridfiles) == 3:
		for m in range(-l,l+1):
			Alm_flat=[]
			for j in range(Ndelta):
				Alm_flat.append(Alm_grid_l3['Alm'][l+m,:,j])
			funcs_l3.append(interpolate.interp2d(Alm_grid_l3['theta'], Alm_grid_l3['delta'], Alm_flat, kind='cubic'))
	#
	# Compute the statistics on the grid of Alm
	tot=0 # Linear flat inded for {Ntheta x Ndelta} space
	i=0 # Index in theta
	j=0 # Index in delta
	Posterior=np.zeros((Ntheta,Ndelta))
	print('Number of data point on the theta axis:', Ntheta)
	print('Number of data point on the delta axis:', Ndelta)
	print('Resolution in theta: ', Alm_grid_l1['resol_theta']) 
	print('Resolution in delta: ', Alm_grid_l1['resol_delta']) 
	print('Theta range: ', '[', np.min(Alm_grid_l1['theta']), np.max(Alm_grid_l1['theta']), ']')
	print('Delta range: ', '[', np.min(Alm_grid_l1['delta']), np.max(Alm_grid_l1['delta']), ']')
	print("Ndelta =", Ndelta)
	print("Ntheta =", Ntheta)
	print("np.shape(Posterior) =", np.shape(Posterior))
	#exit()
	for theta0 in Alm_grid_l1['theta']:
		j=0
		print('theta0 =', theta0,  '   index:', str(i+1),   '/', str(Ntheta),'    (timestamp: ',str(time.time()),')')
		print('           index : [ 1 ,', Ndelta, ']')
		for delta0 in Alm_grid_l1['delta']:
			variables=epsilon_nl0, epsilon_nl1, theta0, delta0
			P=do_stats_ongrid_for_observations(variables, el, a1_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs,nu_nl_obs, Dnu_obs, 
				interpolator_l1=funcs_l1, interpolator_l2=funcs_l2, interpolator_l3=funcs_l3, do_a4=do_a4, do_a6=do_a6, data_type=data_type)
			#print("Posterior: ", P)
			Posterior[i,j]=P
			j=j+1
		i=i+1
	np.savez(posterior_outfile, theta=Alm_grid_l1['theta'], delta=Alm_grid_l1['delta'], Posterior=Posterior, 
		epsilon_nl0=epsilon_nl0, epsilon_nl1=epsilon_nl1, a1_obs=a1_obs, Dnu_obs=Dnu_obs, nu_nl_obs=nu_nl_obs, el=el, a2_obs=a2_obs, sig_a2_obs=sig_a2_obs, a4_obs=a4_obs, sig_a4_obs=sig_a4_obs, a6_obs=a6_obs, sig_a6_obs=sig_a6_obs,
		do_a4=do_a4, do_a6=do_a6, data_type=data_type, resol_theta=Alm_grid_l1['resol_theta'], resol_delta=Alm_grid_l1['resol_delta'])
	plot_posterior_map_2(Alm_grid_l1['theta'], Alm_grid_l1['delta'], Posterior, posterior_outfile, truncate=None)
	#
	print('Grid done')


def do_posterior_map(Almgridfiles, obsfile, Dnu_obs, a1_obs, epsilon_nl0, epsilon_nl1, posterior_outfile='posterior_grid.npz', do_a4=False, do_a6=False, fit_acoefs=-1):
	# One grid for each l in principle. It is up to the user to make sure to have them in the increasing l order: l=1, 2, 3
	Alm_grid_l1=np.load(Almgridfiles[0])# Note: ftype = 'gauss' or 'gate' depends on the Alm grid file. No need to specify it
	Alm_grid_l2=np.load(Almgridfiles[1])# Note: ftype = 'gauss' or 'gate' depends on the Alm grid file. No need to specify it
	if do_a6 == True or len(Almgridfiles) == 3: # We still initialise the l=3 grid if the file is provided by the user
		if len(Almgridfiles) >= 3:
			Alm_grid_l3=np.load(Almgridfiles[2])# Note: ftype = 'gauss' or 'gate' depends on the Alm grid file. No need to specify it
		else:
			print("Error in do_posterior_map() : do_a6 = True but Almgridfiles has a size <3 and is therefore missing the grid file for l=3")
			print("The program will exit now")
			exit()
	Ndelta=len(Alm_grid_l1['delta'])
	Ntheta=len(Alm_grid_l1['theta'])
	#
	a4_obs=[]
	sig_a4_obs=[]
	en, el, nu_nl_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs=read_obsfiles(obsfile, read_a4=do_a4, read_a6=do_a6)
	Dnu_obs=conditional_resize(Dnu_obs, len(a2_obs))
	a1_obs=conditional_resize(a1_obs, len(a2_obs))
	#labels = ["epsilon_nl0", "epsilon_nl1", "theta0", "delta"]
	#
	# Initialise the interpolator once
	funcs_l1=[]
	l=1
	for m in range(-l,l+1):
		Alm_flat=[]
		for j in range(Ndelta):
			Alm_flat.append(Alm_grid_l1['Alm'][l+m,:,j])
		funcs_l1.append(interpolate.interp2d(Alm_grid_l1['theta'], Alm_grid_l1['delta'], Alm_flat, kind='cubic'))
	funcs_l2=[]
	l=2
	for m in range(-l,l+1):
		Alm_flat=[]
		for j in range(Ndelta):
			Alm_flat.append(Alm_grid_l2['Alm'][l+m,:,j])
		funcs_l2.append(interpolate.interp2d(Alm_grid_l2['theta'], Alm_grid_l2['delta'], Alm_flat, kind='cubic'))
	funcs_l3=[]
	l=3
	if do_a6 == True or len(Almgridfiles) == 3:
		for m in range(-l,l+1):
			Alm_flat=[]
			for j in range(Ndelta):
				Alm_flat.append(Alm_grid_l3['Alm'][l+m,:,j])
			funcs_l3.append(interpolate.interp2d(Alm_grid_l3['theta'], Alm_grid_l3['delta'], Alm_flat, kind='cubic'))
	#
	# Compute the statistics on the grid of Alm
	tot=0 # Linear flat inded for {Ntheta x Ndelta} space
	i=0 # Index in theta
	j=0 # Index in delta
	Posterior=np.zeros((Ntheta,Ndelta))
	print('Number of data point on the theta axis:', Ntheta)
	print('Number of data point on the delta axis:', Ndelta)
	print('Resolution in theta: ', Alm_grid_l1['resol_theta']) 
	print('Resolution in delta: ', Alm_grid_l1['resol_delta']) 
	print('Theta range: ', '[', np.min(Alm_grid_l1['theta']), np.max(Alm_grid_l1['theta']), ']')
	print('Delta range: ', '[', np.min(Alm_grid_l1['delta']), np.max(Alm_grid_l1['delta']), ']')
	print("Ndelta =", Ndelta)
	print("Ntheta =", Ntheta)
	print("np.shape(Posterior) =", np.shape(Posterior))
	#exit()
	for theta0 in Alm_grid_l1['theta']:
		j=0
		print('theta0 =', theta0,  '   index:', str(i+1),   '/', str(Ntheta),'    (timestamp: ',str(time.time()),')')
		print('           index : [ 1 ,', Ndelta, ']')
		for delta0 in Alm_grid_l1['delta']:
			variables=epsilon_nl0, epsilon_nl1, theta0, delta0
			P=do_stats_ongrid(variables, el, a1_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs, nu_nl_obs, Dnu_obs, funcs_l1, funcs_l2, funcs_l3, do_a4=do_a4, do_a6=do_a6,fit_acoefs=fit_acoefs)
			#print("Posterior: ", P)
			Posterior[i,j]=P
			j=j+1
		i=i+1
	np.savez(posterior_outfile, theta=Alm_grid_l1['theta'], delta=Alm_grid_l1['delta'], Posterior=Posterior, 
		epsilon_nl0=epsilon_nl0, epsilon_nl1=epsilon_nl1, a1_obs=a1_obs, Dnu_obs=Dnu_obs, nu_nl_obs=nu_nl_obs, 
		en=en, el=el, a2_obs=a2_obs, sig_a2_obs=sig_a2_obs, a4_obs=a4_obs, sig_a4_obs=sig_a4_obs, a6_obs=a6_obs, sig_a6_obs=sig_a6_obs,
		do_a4=do_a4, do_a6=do_a6, fit_acoefs=fit_acoefs, resol_theta=Alm_grid_l1['resol_theta'], resol_delta=Alm_grid_l1['resol_delta'])
	plot_posterior_map_2(Alm_grid_l1['theta'], Alm_grid_l1['delta'], Posterior, posterior_outfile, truncate=None)
	#
	print('Grid done')

#def plot_posterior_map(theta, delta, Posterior, posterior_outfile, truncate=None): 	
#	fig, ax = plt.subplots(2,1)
#	ax.set_title("Posterior")
#	ax.set_xlabel('delta (rad)')
#	ax.set_ylabel('theta (rad)')
#	if truncate == None:
#		c = ax.pcolormesh(delta, theta,Posterior, vmin=np.min(Posterior), vmax=np.max(Posterior), shading='auto')
#	else:
#		pos=np.where(Posterior <= truncate)
#		Posterior[pos]=truncate
#		c = ax.pcolormesh(delta, theta,Posterior, vmin=np.min(Posterior), vmax=np.max(Posterior), shading='auto')		
#	fig.colorbar(c, ax=ax)
##	if theta1 != None and delta1 != None:
##		ax.plot(theta1,delta1,marker='o',size=10)
#	Ptheta, Pdelta=compute_marginal_distrib(log_p_2d)
#	fig.savefig(posterior_outfile + '.jpg')


def plot_posterior_map_2(theta, delta, Posterior, posterior_outfile, truncate=None): 	
	gs = gridspec.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1])
	ax = plt.subplot(gs[0,1])
	ax_left = plt.subplot(gs[0,0], sharey=ax)
	ax_bottom = plt.subplot(gs[1,1], sharex=ax)
	ax_corner = plt.subplot(gs[1,0])

	#fig, ax = plt.subplots(2,1)
	ax.set_title("Posterior")
	ax_bottom.set_xlabel('delta (rad)')
	ax_left.set_ylabel('theta (rad)')
	if truncate == None:
		c = ax.pcolormesh(delta, theta,Posterior, vmin=np.min(Posterior), vmax=np.max(Posterior), shading='auto')
	else:
		pos=np.where(Posterior <= truncate)
		Post=Posterior
		Post[pos]=truncate
		c = ax.pcolormesh(delta, theta,Post, vmin=np.min(Post), vmax=np.max(Post), shading='auto')		
	#
	#plt.colorbar(c, ax=ax) # Messing with plot 
	#plt.colorbar(c, ax=ax_corner, orientation='horizontal')
	Ptheta, Pdelta=compute_marginal_distrib(Posterior)
	ax_bottom.plot(delta, Pdelta)
	ax_left.plot(Ptheta, theta)
	plt.savefig(posterior_outfile + '.jpg')

def compute_marginal_distrib(log_p_2d, normalise=True):
	# Project a 2D log probability into its two axis in order to create a Marginalized posterior distribution
	Nx=len(log_p_2d[:,0])
	Ny=len(log_p_2d[0,:])
	Px=np.zeros(Nx)
	Py=np.zeros(Ny)
	for i in range(Nx):
		yvals=np.sort(log_p_2d[i,:]) # The array to sum is arranged using sort first. We will sum it by decreasing order of values (regressive order of sort then)
		p=np.exp(yvals[Ny-1])
		for j in range(Ny-1):
			p=p+np.exp(yvals[Ny-j-1])
		Px[i]=p
	for j in range(Ny):
		yvals=np.sort(log_p_2d[:,j]) # The array to sum is arranged using sort first. We will sum it by decreasing order of values (regressive order of sort then)
		p=np.exp(yvals[Nx-1])
		for i in range(Nx-1):
			p=p+np.exp(yvals[Nx-i-1])
		Py[j]=p
	if normalise == True:
		Px=Px/np.sum(Px)
		Py=Py/np.sum(Py)
	return Px, Py

def do_minimise_aj_main(Dnu):
	file='/Users/obenomar/tmp/test_a2AR/tmp/results_postMCMC/19992002_incfix_fast/aj_raw.txt'
	data_type='mean_nu_l' # This depends on the fit that was made: mean_nu_l is average aj coefficients 
	en, el, nu_nl_obs, a1_obs, sig_a1_obs, a2_obs, sig_a2_obs,a3_obs, sig_a3_obs,a4_obs, sig_a4_obs, a5_obs, sig_a5_obs, a6_obs, sig_a6_obs=read_mcmcobs(file)
	Dnu_obs=np.repeat(Dnu,len(a2_obs))
	#
	ftype='gate' 
	labels = ["epsilon_nl0", "epsilon_nl1", "theta0", "delta"]
	constants=el, a1_obs, a2_obs, sig_a2_obs,a4_obs, sig_a4_obs, a6_obs, sig_a6_obs, nu_nl_obs, Dnu_obs, ftype, data_type
	#variables    
	epsilon_nl0_init=-1e-3
	epsilon_nl1_init=0. # no slope initially
	theta0_init=np.pi/2
	delta_init=np.pi/8
	variables_init=epsilon_nl0_init, epsilon_nl1_init, theta0_init, delta_init
	relax         =[True           ,    False        ,    True    ,    True   ]
	#do_aj_model_plot(el, nu_nl_obs, Dnu_obs, a1_obs, a2_obs, sig_a2_obs, variables_init, ftype, fileout='model_plot_init')
	sols=do_minimise_aj(constants, variables_init, relax, do_a46=[True, False])
	print(sols.x)
	#do_aj_model_plot(el, nu_nl_obs, Dnu_obs, a1_obs, a2_obs, sig_a2_obs, variables_init, ftype, fileout='model_plot_powell')
	
def test_do_minimise():
	os.environ["OMP_NUM_THREADS"] = "4"
	#
	t0=time.time()
	#constants
	#file='/Users/obenomar/tmp/test_a2AR/data/kplr008379927_kasoc-psd_slc_v2_1111/a2stats_n0.txt'
	#file='/home/obenomar/data/kplr008379927_kasoc-psd_slc_v2_1111/a2stats_n0.txt'
	file='/Users/obenomar/tmp/test_a2AR/data/Simulations/simu_smallerrors_epsicte_1.txt'
	#file='/Users/obenomar/Work/Github/Data/Simulations/simu_smallerrors_epsicte_1.txt'
	en, el, nu_nl_obs, aj_obs, sig_aj_obs=read_obsfiles(file)
	nu_l0=[2324.5102  , 2443.2154 ,  2563.6517 ,  2684.0427  , 2804.5845 ,  2924.5659 , 3044.9774]
	x=np.linspace(0,len(nu_l0)-1,  len(nu_l0))
	coefs=np.polyfit(x, nu_l0, 1)
	Dnu_obs=np.repeat(coefs[0],len(a2_obs))
	#
	ftype='gauss' 
	a1_obs=np.repeat(0., len(a2_obs))
	labels = ["epsilon_nl0", "epsilon_nl1", "theta0", "delta"]
	#labels = ["epsilon_nl0", "epsilon_nl1", "sin(theta0)", "delta"]
	constants=el, a1_obs, aj_obs, sig_aj_obs, nu_nl_obs, Dnu_obs, ftype
	#variables    
	epsilon_nl0_init=-1e-3
	epsilon_nl1_init=0. # no slope initially
	theta0_init=np.pi/2
	delta_init=np.pi/8
	variables_init=epsilon_nl0_init, epsilon_nl1_init, theta0_init, delta_init
	do_a2_model_plot(el, nu_nl_obs, Dnu_obs, a1_obs, a2_obs, sig_a2_obs, variables_init, ftype, fileout='model_plot_init.jpg')
	#sols=do_minimise(constants, variables_init)
	#t1=time.time()
	#print(sols.x)
	#variables_init_emcee=sols.x
	#variables_init_emcee=[ 3.12720862e-04, -4.08410550e-03 , 2.13381881e+00 , 1.21726515e-01]
	variables_init_emcee=variables_init
	do_a2_model_plot(el, nu_nl_obs, Dnu_obs, a1_obs, a2_obs, sig_a2_obs, variables_init_emcee, ftype, fileout='model_plot_minimise.jpg')
	niter=20000
	nwalkers=10
	burnin=10000
	ndim=len(variables_init_emcee)
	t0=time.time()
	sampler=do_emcee(constants, variables_init_emcee, nwalkers=nwalkers, niter=niter)
	t1=time.time()
	#tau = sampler.get_autocorr_time()
	#
	tau=[800]
	print("Autocorrelation time (in steps):", tau)
	if 2*np.mean(tau) < niter:
		flat_samples = sampler.get_chain(discard=burnin, thin=3*nwalkers, flat=True)
		log_posterior = sampler.get_log_prob(discard=burnin, flat=True, thin=3*nwalkers)
		log_prior = sampler.get_blobs(discard=burnin, flat=True, thin=3*nwalkers)
	else:
		flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
		log_posterior = sampler.get_log_prob(discard=0, flat=True, thin=nwalkers)
		log_prior= sampler.get_blobs(discard=0, flat=True, thin=nwalkers)
	np.save('samples.npy', flat_samples)
	np.save('logposterior.npy', log_posterior)
	np.save('logprior.npy', log_prior)
	#
	# Saving the likelihood graph
	fig, ax = plt.subplots()
	ax.plot(log_posterior)
	ax.set_xlim(0, len(log_posterior))
	ax.set_xlabel("step number");
	fig.savefig('likelihood.jpg')
	#
	# Evaluate uncertainties using the samples
	errors=np.zeros((2,ndim))
	med=np.zeros(ndim)
	for i in range(ndim):
		stats = np.percentile(flat_samples[:, i], [16, 50, 84])
		print(' stats[',i,'] = ', stats)
		errors[0,i]=stats[1] - stats[0]
		errors[1,i]=stats[2] - stats[1]
		med[i]=stats[1]
	#
	# Show summary on the statistics
	print(labels[0], ' =', med[0], '  (-  ', errors[0, 0], '  ,  +', errors[1,0], ')')
	print(labels[1],' =', med[1], '  (-  ', errors[0, 1], '  ,  +', errors[1,1], ')')
	print(labels[2],'=', med[2], '  (-  ', errors[0, 2], '  ,  +', errors[1,2], ')')
	print(labels[3],' =', med[3], '  (-  ', errors[0, 3], '  ,  +', errors[1,3], ')')
	print('Computation time (min): ', (t1-t0)/60)
	#
	# Samples for each parameter
	fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
	samples = sampler.get_chain()
	for i in range(ndim):
		ax = axes[i]
		ax.plot(samples[:, :, i], "k", alpha=0.3)
		ax.set_xlim(0, len(samples))
		ax.set_ylabel(labels[i])
	#
	axes[-1].set_xlabel("step number");
	fig.savefig('params_samples.jpg')
	#
	fig = corner.corner(flat_samples, labels=labels, truths=None);
	#
	fig.savefig('params_pdfs.jpg')
	do_a2_model_plot(el, nu_nl_obs, Dnu_obs, a1_obs, a2_obs, sig_a2_obs, med, ftype, fileout='model_plot_emcee.jpg')
#test_do_minimise()
#print("End")


#do_posterior_map(['grid_Alm_l1.npz','grid_Alm_l2.npz'], 
#	'/Users/obenomar/tmp/test_a2AR/data/Simulations/simu_smallerrors_epsicte_1.txt', 
#	85., 0, -1e-3, 0.0, posterior_outfile='posterior_grid.npz')