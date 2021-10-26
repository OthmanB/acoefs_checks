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
import os
import corner
import seaborn as sb

# Definition of the Gaussian likelihood
def likelihood_gauss(xm, xobs, sigobs):
	return np.sum(-(np.array(xm)-np.array(xobs))**2/np.array(sigobs)**2)

# Definition of a uniform prior in log space
def prior_uniform(x, xmin, xmax):
	if x > xmin and x < xmax:
		p=-np.log(np.abs(xmax-xmin))
	else:
		p=-np.inf
	return p

# Read files that contain observed a2 coefficients
def read_obsfiles(file):
	f=open(file, 'r')
	txt=f.read()
	f.close()
	txt=txt.split('\n')
	en=[]
	el=[]
	nu_nl=[]
	a2=[]
	sig_a2=[]
	skip=0
	#print(txt)
	#print('----')
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
				a2.append(float(s[4]))
				em=float(s[4]) - float(s[3])
				ep=float(s[5]) - float(s[4])
				sig_a2.append(np.sqrt(em**2 + ep**2)/2)
	return en, el, nu_nl, a2, sig_a2

def Qlm(l,m):
    Qlm=(l*(l+1) - 3*m**2)/((2*l - 1)*(2*l + 3))
    return Qlm

def nu_CF(nu_nl, Dnu, a1, l, m):
   Dnu_sun=135.1
   numax_sun=3150.
   R_sun=6.96342e5 #in km
   M_sun=1.98855e30 #in kg
   Dnl=0.75
   G=6.667e-8
   rho_sun=M_sun*1e3/(4*np.pi*(R_sun*1e5)**3/3) #in g.cm-3
   rho=(Dnu/Dnu_sun)**2 * rho_sun
   eta0=(4./3.)*np.pi*Dnl/(rho*G)
   return eta0*nu_nl * (a1*1e-6)**2 * Qlm(l,m)

def nu_AR(nu_nl, epsilon_nl, theta0, delta, ftype, l, m):
	return nu_nl*epsilon_nl*Alm(l,m, theta0=theta0, delta=delta, ftype=ftype)

def nu_AR_cpp(nu_nl, epsilon_nl, theta0, delta, ftype, l, m):
	nus=nu_nl*epsilon_nl*Alm_cpp(l, theta0=theta0, delta=delta, ftype=ftype)
	return nus[m+l]

def a2_CF(nu_nl, Dnu, a1, l):
	nu_nlm=[]
	for m in range(-l, l+1):
		perturb=nu_CF(nu_nl, Dnu, a1, l, m)
		nu_nlm.append(nu_nl + perturb)
	acoefs=eval_acoefs(l, nu_nlm)
	#print(nu_nlm)
	return acoefs[1] # returns only a2

def a2_AR(nu_nl, epsilon_nl, theta0, delta, l, ftype):
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
		nu_nlm.append(nu_nl + m*a1 + perturb_CF + perturb_AR)
	#print(nu_nlm)
	acoefs=eval_acoefs(l, nu_nlm)
	return acoefs[1] # returns only a2 

# Compute the a2 coefficient for the theoretical model and provided key parameters of that model
# Use Alm_cpp instead of Alm in python... much faster. Refer to test_convergence.py to see the accuracy
def a2_model_cpp(nu_nl, Dnu, a1, epsilon_nl, theta0, delta, ftype, l):
	nu_nlm=[]
	el, em, Alm=Alm_cpp(l, theta0=theta0, delta=delta, ftype=ftype) # Array of m E [-l,l]
	for m in range(-l, l+1):	
		perturb_CF=nu_CF(nu_nl, Dnu, a1, l, m)
		perturb_AR=nu_nl*epsilon_nl*Alm[m+l]
		nu_nlm.append(nu_nl + m*a1 + perturb_CF + perturb_AR) # We put only a1 and a2 + AR perturbation in a4, a6
	#print("      freq=", nu_nlm,  "     perturb_AR=", perturb_AR, "      perturb_CF", perturb_CF)
	acoefs=eval_acoefs(l, nu_nlm)
	#print("     acoefs : ", acoefs)
	return acoefs[1] # returns only a2 

def priors_model(nu_nl_obs, epsilon_nl, theta0, delta):
	# Reject out or range solutions for theta0
	pena=prior_uniform(theta0, 0, np.pi)
	# Reject absurd negative solutions and large 'spots' that exceed a pi/4 stellar coverage
	pena=pena+prior_uniform(delta, 0, np.pi/4)
	# impose the negativity of the epsilon coefficient, as it is for the Sun
	pena=pena+prior_uniform(epsilon_nl, -0.01, 0.001)
	return pena

# The main function that will compute the statistical criteria for the maximisation procedure
#def do_stats(constants, variables):
def do_stats(variables, l, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype):
	#l, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype = constants
	epsilon_nl, theta0, delta = variables
	a2_nl_mod=[]
	# Given the variables of the model, get a2 of the model a2_mod at the observed frequencies nu_nl_obs of each l and m modes
	for i in range(len(nu_nl_obs)):
		a2_mod=a2_model_cpp(nu_nl_obs[i], Dnu_obs[i], a1_obs[i], epsilon_nl, theta0, delta, ftype, l[i])
		a2_nl_mod.append(float(a2_mod)) # No Need to convert if using the CPP command
	#print(a2_nl_mod)
	#print(a2_obs)
	# Compare the observed and theoretical a2 using the least square method
	L=likelihood_gauss(a2_nl_mod, a2_obs, sig_a2_obs)
	# Add priors
	P=priors_model(nu_nl_obs, epsilon_nl, theta0, delta)
	Posterior=L+P
	#print("L:", L, "      P:", P,   "L+P =", L+P, "       <a2_nl_mod>:", np.mean(a2_nl_mod),  "    <a2_obs>:", np.mean(a2_obs))
	if np.isnan(Posterior):
		print("---- GOT A NaN ----")
		print("L = ", L, "     P = ", P)
		print("      epsilon_nl  = ", epsilon_nl)
		print("      theta0      = ", theta0)
		print('      delta       = ', delta)
		print('Imposing -infinity to the Posterior in order to reject the solution')
		Posterior=-np.inf
	return Posterior

def do_minimise(constants, variables_init):
	l, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype=constants
	epsilon_nl_init, theta0_init, delta_init = variables_init
	nll = lambda *args: -do_stats(*args)
	initial = np.array([epsilon_nl_init, theta0_init, delta_init])
	soln = minimize(nll, initial, args=(l, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype), method="Powell", 
		options={'xtol': 0.00001, 'ftol': 0.00001})
	return soln

def do_emcee(constants, variables_init, nwalkers=100, niter=5000):
	l, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype=constants
	#epsilon_nl0_init, epsilon_nl1_init, theta0_init, delta_init = variables_init

	init_vars = variables_init + 1e-6 * np.random.randn(32, len(variables_init))
	nwalkers, ndim = init_vars.shape
	with Pool() as pool:
		sampler = emcee.EnsembleSampler(
		    nwalkers, ndim, do_stats, pool=pool, args=(l, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype)
		)
		sampler.run_mcmc(init_vars, niter, progress=True, store = True)
	return sampler

def load_samples_emcee(filename, burnin=None, thin=None, pdfs_outfile=None, labels=None):
	nwalkers=50
	ndim=4
	sampler = emcee.EnsembleSampler(
		    nwalkers, ndim, do_stats)
	reader = emcee.backends.HDFBackend(filename)
	#
	tau = reader.get_autocorr_time()
	if burnin == None:
		burnin = int(2 * np.max(tau))
	if thin == None:
		thin = int(0.5 * np.min(tau))
	samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
	#
	print("Max acorr: {0}".format(np.max(tau)))
	print("burn-in: {0}".format(burnin))
	print("thin: {0}".format(thin))
	print("flat chain shape: {0}".format(samples.shape))

	if pdfs_outfile !=None:
			# Saving the pdf:
			if labels == None:
				figure=corner.corner(samples);
			else:
				figure=corner.corner(samples, labels=labels);
			figure.savefig(pdfs_outfile)
	return samples

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

def show_model(med_params, outfile='model_plot.jpg', ftype='gauss', data_file=None, en=None, el=None, nu_nl_obs=None, a2_obs=None, sig_a2_obs=None):
	#
	if data_file != None:
		en, el, nu_nl_obs, a2_obs, sig_a2_obs=read_obsfiles(data_file)
	else:
		if (nu_nl_obs == None) or (a2_obs == None) or (sig_a2_obs == None) or (en == None) or (el == None):
			print("Error: You must either provide a data_file or all the outputs from it (en, el, nu_nl_obs, a2_obs, sig_a2_obs")
			print("       The program will exit now")
			exit()
	#
	a1_obs=np.repeat(1.2, len(a2_obs))
	nu_l0=[2324.5102  , 2443.2154 ,  2563.6517 ,  2684.0427  , 2804.5845 ,  2924.5659 , 3044.9774]
	x=np.linspace(0,len(nu_l0)-1,  len(nu_l0))
	coefs=np.polyfit(x, nu_l0, 1)
	Dnu_obs=np.repeat(coefs[0],len(a2_obs))
	#
	#
	# The plot of the fit with some randomly chosen fits to represent the variance
	fig, ax = plt.subplots()
	a2_nl_mod_best_cpp=[]
	a2_CF_mod=[]
	epsilon_nl_mean=0.
	for i in range(len(nu_nl_obs)):
		a2_mod_cpp=a2_model_cpp(nu_nl_obs[i], Dnu_obs[i], a1_obs[i], med_params[0], med_params[1], med_params[2], ftype, el[i])
		a2_nl_mod_best_cpp.append(float(a2_mod_cpp))
	ax.errorbar(nu_nl_obs, a2_obs, yerr=sig_a2_obs, fmt=".k", capsize=0)
	ax.plot(nu_nl_obs, a2_nl_mod_best_cpp, "ro")#, '--', color='blue')
	fig.savefig(outfile)

def check_results(obs_file, npy_posterior_file, npy_sample_file):
	npy_sample_file='emcee_chains.npy'
	npy_posterior_file='emcee_posterior.npy'
	obs_file='/Users/obenomar/tmp/test_a2AR/data/kplr008379927_kasoc-psd_slc_v2_1111/a2stats_n0.txt'

	posterior=np.load(npy_posterior_file)
	samples=np.load(npy_sample_file)
	#
	en, el, nu_nl_obs, a2_obs, sig_a2_obs=read_obsfiles(obs_file)
	for i in range(len(a2_obs)):
		a2_obs[i]=a2_obs[i]*1e-3
		sig_a2_obs[i]=sig_a2_obs[i]*1e-3
	#sig_a2_obs=float(sig_a2_obs)*1e-3
	nu_l0=[2324.5102  , 2443.2154 ,  2563.6517 ,  2684.0427  , 2804.5845 ,  2924.5659 , 3044.9774]
	x=np.linspace(0,len(nu_l0)-1,  len(nu_l0))
	coefs=np.polyfit(x, nu_l0, 1)
	Dnu_obs=np.repeat(coefs[0],len(a2_obs))
	#
	ftype='gauss' 
	a1_obs=np.repeat(1.2, len(a2_obs))
	#constants=el, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype
	#
	print('# ------- Evaluate difference in modeling for the posterior ------- #')
	Nsamples=len(samples[:,0])
	ndim=len(samples[0,:])
	burnin=Nsamples/2.
	N0=5000
	N1=5005
	labels=['epsilon_nl0_out', 'epsilon_nl1_out', 'theta0_out','delta_out']
	for n in range(N0,N1):
		params=samples[n,:]
		log_proba=do_stats(params, el, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype)
		print("[", str(n), "]")
		print("    params   : ", params)
		print("    posterior: ", posterior[n], "      log_proba: ", log_proba,   "   posterior - log_proba:", posterior[n]-log_proba)
	print("---------------------------------------------------------------------")
	print("# ----------------   Highest Likelihood result   ------------------- ")
	pos_top=np.where(posterior == np.max(posterior))
	log_proba_top=posterior[pos_top]
	params_top=samples[pos_top,:]
	print("Variables for highest likelihood model:")
	print("    Recurrence: ", len(log_proba_top), "  times")
	print("    params    : ", np.ravel(params_top[:,0]))
	print("    log_proba : ", log_proba_top[0])
	show_model(np.ravel(params_top[:,0]), outfile='model_plot_TOP.jpg', data_file=None, en=en, el=el, nu_nl_obs=nu_nl_obs, a2_obs=a2_obs, sig_a2_obs=sig_a2_obs)

	print("# ---------- Evaluate uncertainties using the samples ---------------")
	errors=np.zeros((2,ndim))
	med=np.zeros(ndim)
	for i in range(ndim):
		stats = np.percentile(samples[int(burnin):, i], [16, 50, 84])
		print(' stats[',i,'] = ', stats)
		errors[0,i]=stats[1] - stats[0]
		errors[1,i]=stats[2] - stats[1]
		med[i]=stats[1]
	#
	# Show summary on the statistics
	print(labels[0],' =', med[0], '  (-  ', errors[0, 0], '  ,  +', errors[1,0], ')')
	print(labels[1],' =', med[1], '  (-  ', errors[0, 1], '  ,  +', errors[1,1], ')')
	print(labels[2],'     =', med[2], '  (-  ', errors[0, 2], '  ,  +', errors[1,2], ')')
	print(labels[3],'      =', med[3], '  (-  ', errors[0, 3], '  ,  +', errors[1,3], ')')
	log_proba_med=do_stats(med, el, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype)

	print(" log_proba(med): ", log_proba_med)
	print(" log_proba(top): ", log_proba_top[0])
	print(" Difference    : ", log_proba_top[0] - log_proba_med)
	print("#  ------------------- The plot of the median fit ---------------------")
	show_model(med, outfile='model_plot.jpg', data_file=None, en=en, el=el, nu_nl_obs=nu_nl_obs, a2_obs=a2_obs, sig_a2_obs=sig_a2_obs)
	# 
	print("# ------------- Correlation map ------------")
	skip=50
	reduced_samples=[]
	for r in samples[int(burnin):,]:
		reduced_samples.append(r)
	reduced_samples=np.asarray(reduced_samples)
	figure=corner.corner(reduced_samples, labels=labels);
	figure.savefig("params_pdfs_reduced.jpg")

def test_do_minimise():
	os.environ["OMP_NUM_THREADS"] = "6"
	#
	t0=time.time()
	#constants
	file='/Users/obenomar/tmp/test_a2AR/data/kplr008379927_kasoc-psd_slc_v2_1111/a2stats_n0.txt'
	#file='/home/obenomar/data/kplr008379927_kasoc-psd_slc_v2_1111/a2stats_n0.txt'
	en, el, nu_nl_obs, a2_obs, sig_a2_obs=read_obsfiles(file)
	for i in range(len(a2_obs)):
		a2_obs[i]=a2_obs[i]*1e-3
		sig_a2_obs[i]=sig_a2_obs[i]*1e-3
	print("# ------------ List of observables ------------ ")
	print("Dnu:", Dnu_obs)
	print("a1 :", a1_obs)
	print("n    l    nu_nl,   a2    err(a2)")
	for i in range(len(a2_obs)):
		print("{:.0f}   {:.0f}   {:.2f}   {:.3f}   {:.3f}".format(en[i] , el[i], nu_nl_obs[i], a2_obs[i], sig_a2_obs[i]))
	print("---------------------------------")
	#exit()
	#sig_a2_obs=float(sig_a2_obs)*1e-3
	nu_l0=[2324.5102  , 2443.2154 ,  2563.6517 ,  2684.0427  , 2804.5845 ,  2924.5659 , 3044.9774]
	x=np.linspace(0,len(nu_l0)-1,  len(nu_l0))
	coefs=np.polyfit(x, nu_l0, 1)
	Dnu_obs=np.repeat(coefs[0],len(a2_obs))
	#
	ftype='gauss' 
	a1_obs=np.repeat(1.2, len(a2_obs))
	constants=el, a1_obs, a2_obs, sig_a2_obs, nu_nl_obs, Dnu_obs, ftype
	#variables    
	epsilon_nl_init=-1e-2
	theta0_init=np.pi/2
	delta_init=np.pi/8
	variables_init=epsilon_nl_init, theta0_init, delta_init
	labels=["epsilon_nl","theta0_out","delta_out"]
	sols=do_minimise(constants, variables_init)
	t1=time.time()
	print("Result of minimisation: ", sols.x)
	variables_init_emcee=sols.x
	show_model(sols.x, outfile='model_plot_minimise.jpg', data_file=None, en=en, el=el, nu_nl_obs=nu_nl_obs, a2_obs=a2_obs, sig_a2_obs=sig_a2_obs)
	#
	labels = ["epsilon_nl", "theta0_out", "delta_out"]
	niter=1000
	nwalkers=10
	burnin=400
	thin=10
	ndim=len(variables_init_emcee)
	sampler=do_emcee(constants, variables_init_emcee, nwalkers=nwalkers, niter=niter)
	t1=time.time()
	#tau = sampler.get_autocorr_time()
	tau=[100.]
	#
	print("Autocorrelation time (in steps):", tau)
	if np.mean(tau) < niter:
		flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
		flat_posterior = sampler.get_log_prob(discard=burnin,thin=thin, flat=True)
	else:
		flat_samples = sampler.get_chain(discard=0, thin=thin, flat=True)
		flat_posterior = sampler.get_log_prob(discard=0,thin=thin, flat=True)
	#
	# saving samples with numpy:
	np.save('emcee_chains.npy', flat_samples)
	np.save('emcee_posterior.npy', flat_posterior)
	#
	# Saving the likelihood:
	fig, ax = plt.subplots()
	ax.plot(flat_posterior)
	ax.set_xlabel('iteration')
	ax.set_ylabel('log_likelihood')
	fig.savefig('log_likelihood.jpg')
	#
	# Saving the pdf:
	figure=corner.corner(flat_samples, labels=labels);
	figure.savefig("params_pdfs.jpg")
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
	for i in range(len(labels)):
		print(labels[i],' =',  med[i], '  (-  ', errors[0, i], '  ,  +', errors[1,i], ')')
	#
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
		ax.yaxis.set_label_coords(-0.1, 0.5)
	#
	axes[-1].set_xlabel("step number");
	fig.savefig('params_samples.jpg')
	#
	# The plot of the fit with some randomly chosen fits to represent the variance
	show_model(med, outfile='model_plot_mcmc.jpg', data_file=None, en=en, el=el, nu_nl_obs=nu_nl_obs, a2_obs=a2_obs, sig_a2_obs=sig_a2_obs)
	#
#test_do_minimise()
print("End")
