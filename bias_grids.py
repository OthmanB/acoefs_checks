# Function that visualize a bias map using the function do_posterior_map_bias_grid inside fit_a2sig.py
# It basically read the outputs created by do_posterior_map_bias_grid() and create a visualisation of 
# the bias in function bias(theta0_obs, theta0_true)=pdf(theta_obs) - theta0_true 
# over a grid theta0 values.
# The same is made for delta: bias(delta_obs, delta_true)=pdf(delta_obs) - delta_true 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fit_a2sig import compute_marginal_distrib
#from fit_a2sig import *  


def read_combi(combi_file, format_data=True):
	# Read a combination file created for making bias_grids
	# combi_file: name of the file
	# format_data : If True (default), then return a 2D numpy array for theta0/delta and list for dir_data. Otherwise return a list of lists (corresponding to elements of each lines)
	try:
		f=open(combi_file)
		txt=f.read()
		f.close()
	except:
		print("Error in read_combi(): Could not open/read the combination file!")
		print("Cannot proceed")
		exit()

	data_raw=[]
	comments=[]
	txt=txt.split('\n')
	for t in txt:
		s=t.split()
		if s != '' and s !=[]:
			#print(s[0])
			if s[0] == '#' or s[0] == [] or s[0] == '':
				comments.append(s)
			else:
				get_equal=t.split('=')
				if len(get_equal) != 1: # Case we found some keyword, caracterised by the = symbol
					recognized_key=False
					if get_equal[0] == 'epsilon_nl':
						epsilon_nl=np.asarray(get_equal[1].split())
						recognized_key=True
					if get_equal[0] == 'dir_raw':
						dir_raw=get_equal[1]
						recognized_key=True						
					if recognized_key == False:
						print("Error: keyword get_equal[0] in read_combi() was not recognized")
						print("       Read file: ", combi_file)
						print("       Debug required")
						print("       The program will stop now")
						exit()
				else: # If it is not a comment with # and if it is not a keyword with =, then it must be a 2D array of data
					data_raw.append(t.split())
	if format_data == True:
		Nlines=len(data_raw)
		Ncols=len(data_raw[0])
		data=np.zeros((Nlines,Ncols))
		dir_data=[]
		for i in range(Nlines):
			for j in range(Ncols-1): # Whatever the number of columns, the last one MUST always be containing the path to the data... 
				data[i,j]=np.float(data_raw[i][j])
			dir_data.append(data_raw[i][-1])
		return epsilon_nl, dir_raw, dir_data, data, comments
	else:
		return epsilon_nl, dir_raw, [], data_raw, comments

def read_posteriors(posterior_file, theta0_range=[0, np.pi/2], delta_range=[0, np.pi/4], epsilon_range=[-1,0], return_espilon=False):
	# Expected content of a posterior (npz) file is:
		#theta, delta, Posterior, epsilon_nl0, epsilon_nl1, a1_obs, Dnu_obs, nu_nl_obs, el, a2_obs, sig_a2_obs, a4_obs, sig_a4_obs, a6_obs, sig_a6_obs,
		#do_a4, do_a6, data_type, resol_theta, resol_delta
	# Please see do_posterior_map_for_observation() in fit_a2sig for further details
	data=np.load(posterior_file)
	theta0=data['theta']
	delta=data['delta']
	try:
		epsilon=data['epsilon']
	except:
		epsilon=None
		print("Warning: epsilon no found in the posterior file")
	posterior=data['Posterior']
	pos_theta=np.where(np.bitwise_and(theta0 >= theta0_range[0], theta0 <= theta0_range[1]))
	pos_delta=np.where(np.bitwise_and(delta >= delta_range[0], delta <= delta_range[1]))
	#print("--- BEFORE ---")
	#print("len(theta0) =", len(theta0))
	#print("len(delta) =", len(delta))
	#print("len(posterior[0,:] =", len(posterior[0,:]))
	#print("len(posterior[:,0] =", len(posterior[:,0]))
	theta0=theta0[pos_theta]
	delta=delta[pos_delta]
	posterior=posterior[np.min(pos_theta):np.max(pos_theta)+1,np.min(pos_delta):np.max(pos_delta)+1]
	#print("--- AFTER ---")
	#print("len(theta0) =", len(theta0))
	#print("len(delta) =", len(delta))
	#print("len(posterior[0,:] =", len(posterior[0,:]))
	#print("len(posterior[:,0] =", len(posterior[:,0]))
	if return_espilon == True:
		return epsilon, theta0, delta, posterior
	else:
		return theta0, delta, posterior

def show_bias(combi_file, file_out='bias_map', plot_type='multiple_pdf'):
	filename='posteriors.npz'
	#
	print("1. Reading the combination file...")
	epsilon_nl, dir_raw, dirs_data, data_true, comments=read_combi(combi_file, format_data=True)
	print("2. Reading posterior files and organizing data...")
	#results_list=[]
	print("   - getting x and y axis and the bias map...")
	theta_true=np.unique(data_true[:,0])
	delta_true=np.unique(data_true[:,1])
	print("   -  Scanning directories to create the bias maps...")
	if plot_type == 'multiple_pdf':
		fig_theta, axes_theta = plt.subplots(nrows=len(delta_true), ncols=len(theta_true), figsize=(10, 7), sharex=True)
		fig_delta, axes_delta = plt.subplots(nrows=len(theta_true), ncols=len(delta_true), figsize=(10, 7), sharex=True)
	for i in range(len(dirs_data)):
		posterior_file=dirs_data[i] + '/' + filename
		theta0, delta, posterior=read_posteriors(posterior_file, theta0_range=[0, np.pi/2], delta_range=[0, np.pi/4])
		Ptheta, Pdelta=compute_marginal_distrib(posterior)
		#dico={theta_obs: theta0, delta_obs: delta, Ptheta: Ptheta, Pdelta:Pdelta, theta_true:data_true[i,0], delta_true:data_true[i,1]}
		#results_list.append(dico)
		if i == 0:
			Ntheta_pdf=len(theta0)
			Ndelta_pdf=len(delta)
			theta_obs=np.zeros(  (len(dirs_data), Ntheta_pdf)  )
			delta_obs=np.zeros(  (len(dirs_data), Ndelta_pdf)  )
			bias_map_theta=np.zeros( (len(theta_true), len(delta_true)*Ntheta_pdf))
			bias_map_delta=np.zeros( (len(theta_true)*Ndelta_pdf, len(delta_true)))
			#Ptheta_obs=np.zeros(  (len(dirs_data), len(theta0)) )
			#Pdelta_obs=np.zeros(  (len(dirs_data), len(delta)) )
#		print('len(theta_obs[:,0]) =', len(theta_obs[:,0]) )
#		print('len(theta_obs[0,:]) =', len(theta_obs[0,:]) )
#		print("#")
#		print('len(delta_obs[:,0]) =', len(delta_obs[:,0]) )
#		print('len(delta_obs[0,:]) =', len(delta_obs[0,:]) )
#		print('len(Ptheta) =', len(Ptheta) )
#		print('len(Pdelta) =', len(Pdelta) )
#		print('len(bias_map_theta[:,0]) =', len(bias_map_theta[:,0]) )
#		print('len(bias_map_theta[0,:]) =', len(bias_map_theta[0,:]) )
		theta_obs[i,:]=theta0
		delta_obs[i,:]=delta
		#Ptheta_obs[i,:]=Ptheta
		#Pdelta_obs[i,:]=Pdelta
		i0=np.argwhere(theta_true == data_true[i,0])[0][0]
		j0=np.argwhere(delta_true == data_true[i,1])[0][0]
		print(data_true[i,0])
		print(i0, j0)
		#exit()
		if Ntheta_pdf == len(Ptheta):
			#print('j0 first position index: ', j0*len(Ptheta) )
			#print('j0 last position index:  ', j0*len(Ptheta)+Ntheta_pdf)
			#print('len(bias_map_theta[0,:])=' , len(bias_map_theta[0,:]))
			bias_map_theta[i0,j0*len(Ptheta):j0*len(Ptheta)+Ntheta_pdf]=Ptheta
		else:
			print("Error: Size mismatch between Ptheta and theta0...Exiting")
			exit()
		if plot_type == 'multiple_pdf':
			axes_theta[j0,i0].plot(theta0*180./np.pi, Ptheta)
			axes_theta[j0,i0].set_yticks([])
			axes_theta[j0,i0].axvline(x=theta_true[i0]*180./np.pi, color='red', linestyle='--')
			if j0 == len(delta_true)-1:
				axes_theta[j0,i0].set_xlabel(r'$\theta$')
			if i0 == 0:
				axes_theta[j0,i0].set_ylabel(r'$P(\theta_{obs}|\delta_{true} =$' + "{:0.0f}".format(delta_true[j0]*180./np.pi) + ')')
			#
			axes_delta[i0,j0].plot(delta*180./np.pi, Pdelta)
			axes_delta[i0,j0].set_yticks([])
			axes_delta[i0,j0].axvline(x=delta_true[j0]*180./np.pi, color='red', linestyle='--')
			if i0 == len(theta_true)-1:
				axes_delta[i0,j0].set_xlabel(r'$\delta$')
			if j0 == 0:
				axes_delta[i0,j0].set_ylabel(r'$P(\delta_{obs}|\theta_{true} =$' + "{:0.1f}".format(theta_true[i0]*180./np.pi) + ')')

	#theta_plot={x_axis:theta_true, y_axis:theta_obs-theta_true, matrix:bias_map_theta}
	#delta_plot={x_axis:delta_true, y_axis:delta_obs-delta_true, matrix:bias_map_delta}
	print("3. Plots...")
	ndim=2
	print('theta_true =', theta_true)
	print('delta_true =', delta_true)
	if plot_type == 'multiple_pdf':
		fig_theta.savefig(file_out +'_theta.jpg')
		fig_delta.savefig(file_out +'_delta.jpg')
	if plot_type == 'single_map':
		print("Not Working yet... Please code here for single_plot case")
		exit()
		ax=sns.heatmap(bias_map_theta, xticklabels=theta_true)#, yticklabels=theta_obs[0,:])
		ax.set_xticks(ax.get_xticks()[::3])
		ax.set_xticklabels(theta_true[::3])
		#ax.set_yticks(ax.get_yticks()[::3])
		#ax.set_yticklabels(ylabels[::3])
		plt.savefig(file_out +'_theta.jpg')
		#sns.heatmap(bias_map_delta,annot=True)
		#plt.savefig(file_out +'_delta.jpg')
	if plot_type == 'multiple_map':
		for j in range(len(delta_true)):
			ax=sns.heatmap(bias_map_theta[:,j*len(Ptheta):j*len(Ptheta)+Ntheta_pdf], xticklabels=theta_true)#, yticklabels=theta_obs[0,:])
			#ax.set_xticks(ax.get_xticks()[::3])
			#ax.set_xticklabels(theta_true[::3])
			plt.savefig(file_out +'_theta_' + str(j)+'.jpg')
	#if plot_type == 'multiple_pdf':
	#	print("len(delta_true) = ", len(delta_true))
	#	fig, axes = plt.subplots(nrows=len(delta_true), ncols=len(theta_true), figsize=(10, 7), sharex=True)
	#	for i in range(len(delta_true)):
	#		for j in range(len(theta_true)):
	#			print('i,j =', i, j)
	#			axes[i,j].plot(theta_obs, bias_map_theta[i,j*len(Ptheta):j*len(Ptheta)+Ntheta_pdf])
	#			#axes[i,j].set_xlabel('\theta_{true}')
	#			#axes[i,j].set_ylabel('P(\theta_{obs}| \delta_{true})')
	#			#axes[i,j].title('delta ='+ str(delta_true[i]))
	#	plt.savefig(file_out +'_theta.jpg')
	exit()
