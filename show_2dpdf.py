import numpy as np
import matplotlib.pyplot as plt

def read_mcmc_posteriors(file):
	data=np.load(file)
	Nparams=len(data[0,:])
	Nsamples=len(data[:,0])
	labels = ["epsilon_nl0", "epsilon_nl1", "theta0", "delta"]
	return data, labels, Nparams, Nsamples	

def filter_posteriors(data, labels, Nparams, Nsamples, burnin, skip_epsilon_nl1=False, keep_positive=False, keep_epsilon=None):
	Nburnin=int(burnin*Nsamples) # USER CAN REMOVE A FRACTION OF THE SAMPLES AT THE BEGINING
	if burnin > 1:
		print("Error in filtering posteriors: You specified a value of the burnin parameteter exceeding 1")
		print("                               This should be a fraction of Nsamples and must be < 1")
		exit()
	# -- Section to skip epsilon_nl1 --
	if skip_epsilon_nl1 == True:
		data1=np.zeros((Nsamples-Nburnin, Nparams-1))
		data1[:,0]=data[Nburnin:,0]
		data1[:,1]=data[Nburnin:,2]
		data1[:,2]=data[Nburnin:,3]
		Nparams=len(data1[0,:])	
	else:
		data1=np.zeros((Nsamples-Nburnin, Nparams))
		for i in range(Nparams):
			data1[:,i]=data[Nburnin:,i]
	labels=["epsilon", "theta", "delta"]
	# -- Section to remove positive values of epsilon_nl0 -- (usefull for my trial on KIC 8379927)
	if keep_epsilon != None:
		posok=np.where(np.bitwise_and(data1[:,0] >= keep_epsilon[0], data1[:,0] <= keep_epsilon[1]))
		data2=np.zeros((len(posok[0]), Nparams))
		for i in range(Nparams):
			data2[:,i]=data1[posok,i]
	else:
		if keep_positive == True:
			posok=np.where(data1[:,0] <= 0)
			data2=np.zeros((len(posok[0]), Nparams))
			for i in range(Nparams):
				data2[:,i]=data1[posok,i]
		else:
			data2=data1
	return data2, labels, Nparams, Nsamples

def show_pdf(file_mcmc='/Users/obenomar/tmp/test_a2AR/tmp/results_activity/gate_8379927_a2a4_fast/samples.npy', file_out='plots', burnin=0, keep_positive=False, keep_epsilon=None):
	#file_mcmc='/Users/obenomar/tmp/test_a2AR/tmp/results_activity/gate_8379927_a2a4_fast/samples.npy'
	#file_mcmc='/Users/obenomar/tmp/test_a2AR/tmp/results_activity/gate_Sun_a2a4_19992002/samples.npy'
	#file_mcmc='/Users/obenomar/tmp/test_a2AR/tmp/results_activity/gate_Sun_a2a4_20062009/samples.npy'
	
	data, labels, Nparams, Nsamples=read_mcmc_posteriors(file_mcmc)
	if Nparams == 4:
		data, labels, Nparams,Nsamples=filter_posteriors(data, labels, Nparams,Nsamples, burnin, skip_epsilon_nl1=True, keep_positive=keep_positive, keep_epsilon=keep_epsilon)
	else:
		data, labels, Nparams,Nsamples=filter_posteriors(data, labels, Nparams,Nsamples, burnin, skip_epsilon_nl1=False, keep_positive=keep_positive, keep_epsilon=keep_epsilon)
	#
	# Evaluate uncertainties using the samples
	errors=np.zeros((2,Nparams))
	med=np.zeros(Nparams)
	for i in range(Nparams):
		stats = np.percentile(data[:, i], [16, 50, 84])
		print(' stats[',i,'] = ', stats)
		if labels[i] == 'epsilon':
			errors[0,i]=stats[1] - stats[0]
			errors[1,i]=stats[2] - stats[1]
			med[i]=stats[1]
		else:
			errors[0,i]=(stats[1] - stats[0])*180./np.pi
			errors[1,i]=(stats[2] - stats[1])*180./np.pi
			med[i]=stats[1]*180./np.pi
	#
	# Show summary on the statistics
	string='#param   median   err_m     err_p\n'
	for i in range(Nparams):
		print(labels[i], ' =', med[i], '  (-  ', errors[0, i], '  ,  +', errors[1,i], ')')
		string=string + '  {}  {:0.6f}      {:0.6f}      {:3.6f}\n'.format(labels[i], med[i], errors[0,i], errors[1,i])
	fsum=open(file_out+'_summary.txt', 'w')
	fsum.write(string)
	fsum.close()
	#
	bins=30#75
	for i in range(Nparams):
		fig_1d, ax = plt.subplots(1, figsize=(12, 6))
		if labels[i] == "epsilon":
			ax.hist(data[:,i],bins=5*bins)
		else:
			ax.hist(data[:,i]*180./np.pi,bins=bins)
		ax.set_xlabel(labels[i])
		ax.set_ylabel("PDF")
		ax.axvline(x=med[i], color='green', linestyle='-')
		ax.axvline(x=med[i]-errors[0,i], color='red', linestyle='--')
		ax.axvline(x=med[i]+errors[1,i], color='red', linestyle='--')
		fig_1d.savefig(file_out+'_pdf_'+ str(i) + '.jpg')

	fig_2d, axes = plt.subplots(Nparams+1, Nparams+1, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
	fig_2d.subplots_adjust(hspace=0.3, wspace=0.05)
	#
	for i in range(0, Nparams):
		for j in range(1, Nparams+1):
			if j <= i:
				#bin_x=1./(( max(data[:,i-1]) - min(data[:,i-1]) ) / 30)
				#bin_y=1./(( max(data[:,j-1]) - min(data[:,j-1]) ) / 30)
				#print("bin_x =", bin_x)
				#print("bin_y =", bin_y)
				axes[i,j].hist2d(data[:,i], data[:,j-1], bins=20, cmap=plt.cm.jet)
	#
	fig_2d.savefig(file_out)
	exit()