import numpy as np
from scipy import interpolate
from fit_a2sig import Qlm
from fit_a2sig import eta0_fct
from acoefs import eval_acoefs
from termcolor import colored
from eval_aj_range import load_Alm_grids
from eval_aj_range import numax_from_stello2009
from eval_aj_range import Dnu_from_stello2009
import matplotlib.pyplot as plt


def grid_aj_mean(numax=None, Dnu=None, Mass=None, Radius=None, Teff=None, epsilon_nl=5*1e-4, a1=1000, dir_core='/Users/obenomar/tmp/test_a2AR/'):
	'''
	Make a full grid of aj E [1,6] coefficients for a given star of Mass=Mass, Radius=Radius and Teff=Teff 
	and given an activity level epsilon_nl and a rotation a1
	If one wants a purely observational approach, it can provides directly numax (instead of M, R, Teff) and/or Dnu
	If only numax is provided, Dnu is determined using scaling relation with numax from Stello+2009. Same if only Dnu is provided
	This is usefull to set the range of simulations for aj and for setting priors on aj odds coefficient while fitting
	Note: This function **only** considers the mean value of aj over l and at numax
	'''
	R_sun=6.96342e5 #in km
	M_sun=1.98855e30 #in kg
	if numax != None and Dnu == None:
		print('Calculating Dnu(numax)...')
		Dnu=Dnu_from_stello2009(numax)
		print('Dnu = ', numax)
		eta0=eta0_fct(Dnu=Dnu)
	#
	if numax == None and Dnu != None:
		print('Calculating numax(Dnu)...')
		numax=numax_from_stello2009(Dnu)
		eta0=eta0_fct(Dnu=Dnu)
	#
	if numax != None and Dnu != None:
		print('Using numax and Dnu...')
		eta0=eta0_fct(Dnu=Dnu)
	#	
	if numax == None and Dnu == None and Mass != None and Radius != None and Teff !=None:
		print('Using M,R,Teff...')
		numax=numax_fct(Mass=Mass, Radius=Radius, Teff=Teff)
		Volume=np.pi*(Radius*R_sun)**3 * 4./3
		rho=Mass*M_sun/Volume * 1e-12 # Conversion into g/cm-3
		#print('rho =', rho)
		eta0=eta0_fct(rho=rho)
	if numax == None and Dnu == None and (Mass == None or Radius == None or Teff == None):
		print('Error in eval_aj_mean_range: You need to provide (M, R, Teff) or numax and/or Dnu')
		print('                             Please read the description of the function before using it')
		exit()
	
	print('numax =', numax)
	lmax=3 #
	jmax=2*lmax 
	dir_grids=dir_core+"/grids/gate/0.25deg_resol/" # High precision grid
	gridfiles=['grid_Alm_1.npz', 'grid_Alm_2.npz', 'grid_Alm_3.npz'] # Fine grids
	#
	theta, delta, Alm= load_Alm_grids(dir_grids, gridfiles, lmax=lmax)
	#
	aj_grid=np.zeros((jmax, len(theta), len(delta))) # The grid contain
	for j in range(len(theta)):
		for k in range(len(delta)):
			aj=np.zeros(6)
			for l in range(1,lmax+1):
				nu_nlm=[]
				for m in range(-l, l+1):
					# perturvation from AR		
					dnu_AR=numax*epsilon_nl*Alm[l-1][m+l][j,k] # All Alm(theta, delta) are in [j,k]
					# perturbation from CF
					dnu_CF=eta0*numax * (a1*1e-9)**2 * Qlm(l,m)
					nu_nlm.append(numax + dnu_AR + dnu_CF)
					#print('(j,k) = (', j, ',', k,')  theta =', theta[j]*180./np.pi, "  delta =", delta[k]*180./np.pi, " dnu_AR =", dnu_AR*1000)
				a=eval_acoefs(l, nu_nlm)
				# Averaging over l
				for o in range(len(aj)):
					if o < 2:
						aj[o]=aj[o] + a[o]/lmax
					if o >=2 and o < 4:
						if l>=2:
							aj[o]=aj[o] + a[o]/(lmax-1)
					if o >=3 and o <6:
						if l>=3:
							aj[o]=aj[o] + a[o]/(lmax-2)
			aj_grid[:,j,k]=aj*1000 # aj converted into nHz
	return theta, delta, aj_grid 

def aj_mean_from_theta_delta(theta_star, delta_star, numax=2150., Dnu=103.11, a1=1000, epsilon_nl=5*1e-4, dir_core='/Users/obenomar/tmp/test_a2AR/'):
	'''
		This function gives the value of aj_mean for an interval of a given set of star parameters and for theta,delta
	'''
	if theta_star > np.pi/2 or theta_stars < 0 or delta_star > np.pi/4 or delta_star < 0:
		print('theta and delta must be in radians, but inputs are outside expected boundaries (theta: [0, pi/2], delta: [0, pi/4]')
		print('Cannot proceed. The program will exit now')
		exit()
	theta_grid, delta_grid, aj_grid=grid_aj_mean(numax=numax, Dnu=Dnu, Mass=None, Radius=None, Teff=None, epsilon_nl=epsilon_nl, a1=a1, dir_core=dir_core)
	# Initialise the interpolator once
	Ndelta=len(delta_grid)
	Ntheta=len(theta_grid)
	print("  - Initializing the interpolators for l=1,2,3...")
	ajgrid_flat=[]
	for j in range(Ndelta):
		ajgrid_flat.append(aj_grid[1][:,j])
	func_a2=interpolate.interp2d(theta_grid, delta_grid, ajgrid_flat, kind='cubic')
	ajgrid_flat=[]
	for j in range(Ndelta):
		ajgrid_flat.append(aj_grid[3][:,j])
	func_a4=interpolate.interp2d(theta_grid, delta_grid, ajgrid_flat, kind='cubic')
	ajgrid_flat=[]
	for j in range(Ndelta):
		ajgrid_flat.append(aj_grid[5][:,j])
	func_a6=interpolate.interp2d(theta_grid, delta_grid, ajgrid_flat, kind='cubic')

	a2=func_a2(theta_star, delta_star)
	a4=func_a4(theta_star, delta_star)
	a6=func_a6(theta_star, delta_star)
	print('<a2>l = ', a2)
	print('<a4>l = ', a4)
	print('<a6>l = ', a6)
	return a2,a4,a6

def show_aj_fct_theta_delta(numax=2150., Dnu=103.11, a1=1000., epsilon_nl=5*1e-4, dir_core='/Users/obenomar/tmp/test_a2AR/', file_out='aj_theta_delta'):
	'''
		Function that plots the a2, a4, a6 coefficient for a serie of theta and delta values. This allows a visualisation
		of the potential uniqueness of a solution in the aj parameter space and thus serves as a basis to explain why a2 
		and a4 are enough to ensure the uniqueness/gaussianity of the solution when Alm is made using a 'gate' function.
		Note that other functions (e.g. 'gauss' filter) may not have the same aj coefficients in function of theta and delta.
		Information about the default numax and Dnu: These are those from 16 Cyg A
		for which l=0 modes are taken as: 1391.62917, 1494.98697, 1598.65899, 1700.89058 ,1802.28949,1904.58551,2007.55442,2110.93592,2214.21018,2317.33199,2420.92615,2525.18818,2629.41124,2734.00654
	'''
	theta, delta, aj_grid=grid_aj_mean(numax=numax, Dnu=Dnu, Mass=None, Radius=None, Teff=None, epsilon_nl=epsilon_nl, a1=a1, dir_core=dir_core)

	fig_a2, ax_a2 = plt.subplots()
	fig_a4, ax_a4 = plt.subplots()
	fig_a6, ax_a6 = plt.subplots()
	ax_a2.set_ylabel('$a_2$ (nHz)')
	ax_a2.set_xlabel(r'$\theta$ ($^\circ$)')
	ax_a4.set_ylabel('$a_4$ (nHz)')
	ax_a4.set_xlabel(r'$\theta$ ($^\circ$)')
	ax_a6.set_ylabel('$a_6$ (nHz)')
	ax_a6.set_xlabel(r'$\theta$ ($^\circ$)')
	ax_a2.axhline(y=0, color='gray', linestyle='--')
	ax_a4.axhline(y=0, color='gray', linestyle='--')
	ax_a6.axhline(y=0, color='gray', linestyle='--')
	ax_a2.set_xlim([-1, 91])
	ax_a4.set_xlim([-1, 91])
	ax_a6.set_xlim([-1, 91])
	ax_a2.set_xticks(np.arange(0, 100, 10))
	ax_a4.set_xticks(np.arange(0, 100, 10))
	ax_a6.set_xticks(np.arange(0, 100, 10))
	# Show all aj(theta) at a given delta = delta_show
	# a2 is in [1], a4 is in [3] and a6 is in [5]
	#print(delta)
	delta_show_list=[5, 10, 20, 40]
	labels=[r'$\delta=5$ deg', r'$\delta=10$ deg', r'$\delta=20$ deg', r'$\delta=40$ deg']
	tol=theta[1] - theta[0]
	lines=['-', '--', '-.', ':']
	for i in range(len(delta_show_list)): 
		delta_show=delta_show_list[i]*np.pi/180.
		pos_delta=np.where(np.bitwise_and(delta >= delta_show - tol, delta <= delta_show + tol))[0]
		pos_delta=pos_delta[0]
		print('pos_delta = ', pos_delta, '  delta =', delta[pos_delta]*180./np.pi)
		ax_a2.plot(theta*180./np.pi, aj_grid[1][:, pos_delta], color='black', linestyle=lines[i], label=labels[i])
		ax_a4.plot(theta*180./np.pi, aj_grid[3][:, pos_delta], color='black', linestyle=lines[i], label=labels[i])
		ax_a6.plot(theta*180./np.pi, aj_grid[5][:, pos_delta], color='black', linestyle=lines[i], label=labels[i])
	ax_a2.legend(loc="upper left")
	ax_a4.legend(loc="upper left")
	ax_a6.legend(loc="upper left")
	fig_a2.savefig(file_out + '_a2.jpg', dpi=300)
	fig_a4.savefig(file_out + '_a4.jpg', dpi=300)
	fig_a6.savefig(file_out + '_a6.jpg', dpi=300)
	plt.close('all')


#show_aj_fct_theta_delta(numax=2150., Dnu=103.11, a1=0., epsilon_nl=5*1e-4, dir_core='/Users/obenomar/tmp/test_a2AR/')

