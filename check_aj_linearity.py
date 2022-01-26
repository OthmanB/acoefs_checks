# This is a small program that demonstrate the linearity of the aj solutions in terms of n and l (monotonically increase/decrease with frequency)

from fit_a2sig import read_obsfiles
import matplotlib.pyplot as plt
import os
import numpy as np
from termcolor import colored

def get_files(dir_data, extension='.data', fullpath=True):
	files=[]
	for file in os.listdir(dir_data):
		# check the extension of files
		if file.endswith(extension):
			if fullpath == True:
				files.append(os.path.join(root, file))
			else:
				files.append(file)
	return files

def check_aj_linearity(dir_data='/Users/obenomar/tmp/test_a2AR/acoefs_checks_theta/tmp/bias_analysis/19992002_incfix_fast/gate/grids_posterior/epsilon_nl0.0008_0.0/raw_ajnl/', do_plots=False):
	''' 
		This is the main function that gives you the level of agreement in term of linearity
		It scan the 'dir_data' to find .data files (created by fit_a2sig.do_simfile(), read all of them and gives the max delta
		between the data points and their linear fit
		Note that to perform the linearity check, do_simfile() must be used with its default relative error setup (set to None).
		This is to ensure that the created data files do not add any random gaussian error, which obviously bias the evaluation
		of the linearity (there is a column for true inputs, but it is not read by read_obsfiles().
		Furthermore, you may need to set a1=0 otherwise it will incorporate the centrifugal force, which may introduce departure to linearity
	''' 
	polorder=1
	#
	#files=get_files(dir_data)
	files_only=get_files(dir_data, fullpath=False)
	print('Search in dir :', dir_data)
	for f in files_only:
		print(' ---------------- ')
		print('  file:', f)
		file=dir_data + '/' + f
		en, el, nu_nl_obs, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs=read_obsfiles(file, read_a4=True, read_a6=True)
		el=np.array(el)
		nu_nl_obs=np.array(nu_nl_obs)
		a2_obs=np.array(a2_obs)
		#sig_a2_obs=np.array(sig_a2_obs)
		a4_obs=np.array(a4_obs)
		#sig_a4_obs=np.array(sig_a4_obs)
		a6_obs=np.array(a6_obs)
		#sig_a6_obs=np.array(sig_a6_obs)
		cols=['', 'red', 'orange', 'blue']
		if do_plots == True:
			fig_a2, ax_a2 = plt.subplots()
			fig_a4, ax_a4 = plt.subplots()
			fig_a6, ax_a6 = plt.subplots()
			ax_a2.set_title('file :' + str(f))
			ax_a2.set_ylabel('a2 (nHz)')
			ax_a2.set_ylabel('Frequency (microHz)')
			ax_a4.set_title('file : ' + str(f))
			ax_a4.set_ylabel('a4 (nHz)')
			ax_a4.set_ylabel('Frequency (microHz)')
			ax_a6.set_title('file : '+ str(f))
			ax_a6.set_ylabel('a6 (nHz)')
			ax_a6.set_ylabel('Frequency (microHz)')
		#
		for l in range(1, np.max(el)+1):
			print(colored('     l =' + str(l), 'cyan'))
			pos=np.array(np.where(np.array(el) == l)).flatten()
			nu=nu_nl_obs[pos]
			a2=a2_obs[pos]
			a4=a4_obs[pos]
			a6=a6_obs[pos]
			a2_pol=np.polyfit(nu, a2, polorder)
			a2_fit=np.polyval(a2_pol, nu)
			a2_delta=np.max(a2-a2_fit)/np.mean(a2) * 100
			if np.isfinite(a2_delta):
				print(colored('         maxdelta_a2 (%):'+str(a2_delta), 'blue'))
			else:
				print(colored('         maxdelta_a2 (%):'+str(a2_delta), 'red'))
			#
			if l>=2:
				a4_pol=np.polyfit(nu, a4, polorder)
				a4_fit=np.polyval(a4_pol, nu)
				a4_delta=np.max(a4-a4_fit)/np.mean(a4) * 100
				if np.isfinite(a4_delta):
					print(colored('         maxdelta_a4 (%): {}'.format(a4_delta), 'blue'))
				else:
					print(colored('         maxdelta_a4 (%): {}'.format(a4_delta), 'red'))
			#
			if l>=3:
				a6_pol=np.polyfit(nu, a6, polorder)
				a6_fit=np.polyval(a6_pol, nu)
				a6_delta=np.max(a6-a6_fit)/np.mean(a6) * 100
				if np.isfinite(a2_delta):
					print(colored('         maxdelta_a6 (%): {}'.format(a6_delta), 'blue'))
				else:
					print(colored('         maxdelta_a6 (%): {}'.format(a6_delta), 'red'))
			#
			#
			if do_plots == True:
				ax_a2.plot(nu, a2, color=cols[l])
				ax_a2.plot(nu, a2_fit, color='black', linestyle='--', marker='D')
				#
				if l>=2:
					ax_a4.plot(nu, a4, color=cols[l])
					ax_a4.plot(nu, a4_fit, color='black', linestyle='--', marker='D')
				#
				if l >=3:
					ax_a6.plot(nu, a6, color=cols[l])
					ax_a6.plot(nu, a6_fit, color='black', linestyle='--', marker='D')
			fig_a2.savefig(file + '_a2.jpg', dpi=300)
			fig_a4.savefig(file + '_a4.jpg', dpi=300)
			fig_a6.savefig(file + '_a6.jpg', dpi=300)
			plt.close('all')
