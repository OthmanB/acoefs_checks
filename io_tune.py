'''
	Contains all functions handling the input and outputs formats and entries
'''
import numpy as np
import os

def format_ID(ID, Ndigits=9):
	'''
		Small function that ensure that the ID number is on Ndigits digits
		If this is not the case, we fill with 0 before the existing ID
		With Kepler data, Ndigits=9 is fine. For TESS, Ndigits=10 shoud be OK (need check)
	'''
	NID=len(ID)
	delta=8-NID+1 # We want to format using 8 digits
	New_ID=ID
	for d in range(delta):
		New_ID='0' + New_ID
	return New_ID

def format_tab2D(txt, dtype=np.str, init_val=''):
	'''
		Takes a 1D array of string with each line having multiple entry
		and change it into a 2D array
		txt: The 1D input array
		dtype: Type of the output
		delimiter: The delimiter used into each 1D line of the array
		init_val: The value used to initialize the array. Critical if one deals with non-square entries
			      Used only for numbers and not string
	'''
	Nlines=len(txt)
	Ncols=-1
	# The number of columns is the max number of cols so that we can handle non-square entries
	for t in txt:
		if Ncols < len(t.split()):
				Ncols=len(t.split())
				#print(t.split())
	#print('Ncols=', Ncols)
	#print('Nlines=', Nlines)
	if dtype != np.str and dtype != str:
		out=np.zeros((Nlines, Ncols), dtype=dtype) + np.float(init_val)
	else:
		out=[]
	for i in range(Nlines):
		k=txt[i].split()
		#print('len(k)=', len(k))
		#print(k)
		if dtype != np.str and dtype != str:
			for j in range(len(k)):
				out[i,j]=k[j]
		else:
			out.append(k)	
	return out

def list_files(startpath, verbose=False):
	'''
		Function that list all content (subdirectories and files)
		of a given directory in a structured manner
		Taken from: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
		startpath: root path where to start the directory/file scan
		Return:
			out: Full path of the directories on the form of a list
			out_type: Type of the retrieved information: 'Directory' or 'File'
	'''
	out=[]
	out_type=[]
	for root, dirs, files in os.walk(startpath):
		level = root.replace(startpath, '').count(os.sep)
		indent = ' ' * 4 * (level)
		if os.path.basename(root) != '': # To avoid the '/' to be counted as directories
			if verbose == True:
				print('{}{}/'.format(indent, os.path.basename(root)))
			#out.append(os.path.basename(root))
			out.append(root)
			out_type.append('Directory')
		subindent = ' ' * 4 * (level + 1)
		for f in files:
			if verbose == True:
				print('{}{}'.format(subindent, f))
			out.append(os.path.join(root,f))	
			out_type.append('File')
	return out, out_type

def make_subdir_group(parent_dirs, child_dirs, verbose=False):
	'''
		This function generate subdirectories into a parent directory using list entires. 
		If you specifically want to generate a deep linear directory tree such as /A/B/AA/BB/CC/
		please consider using instead make_subdir_lintree()

		parent_dirs: A list of root directories considered as parent for the dirnames created dirs
		child_dirs: Names of the directories that will be created into the parent_dirs[k]

		Note: if parent_dirs is a 1D list, then dirnames is also 1D.
		      However if parent_dirs is has N times a 1D list, then dirnames must by of dimension N
		      eg. parent_dirs=['/home/me/dirA', '/home/me/dirB'] you must have
				  child_dirs=[['subdirA', 'subdirB', ...], ['subdirC', 'subdirD', ...]]
				  This gives: 
				  		- /home/me/dirA/subdirA
						  /home/me/dirA/subdirB
						  ...
						
						- /home/me/dirB/subdirC
						  /home/me/dirB/subdirD
						  ...
			  Other example: parent_dirs=['/home/me/', '/home/me/A', 'home/me/A/B']
			  				 child_dirs=[['subdir1', 'subdir2'], ['subdir3', 'subdir4'], ['subdir5', 'subdir6']]
			  	  This gives:
			  	  		- /home/me/subdir1
			  	  		  /home/me/subdir2

			  	  	    - /home/me/A/subdir3
			  	  	      /home/me/A/subdir4

			  	  	    - /home/me/A/B/subdir5
			  	  	      /home/me/A/B/subdir6
	'''
	Nparentdirs=len(parent_dirs)
	Nchilddirs=len(child_dirs)
	if Nparentdirs != Nchilddirs:
		print('Error: The len of parent_dirs must be the same as the len of child_dirs. Please check the documentation')
		print('       Cannot continue. Exiting now')
		exit()
	for i in range(Nparentdirs):
		parent=parent_dirs[i]
		if verbose == True:
			print('[1] parent=', parent)
		for child in child_dirs[i]:
			if verbose == True:
				print('           [2] ', child)
			if os.path.isdir(parent + '/' + child) == False:
				if verbose == True:
					print('              Directory', parent+ '/' + child, ' does not exist ... creating it')
				os.makedirs(os.path.join(parent, child))
			else:
				if verbose == True:
					print('               Directory ', parent+ '/' + child, ' Exist. NOTHING TO DO')

def make_subdir_lintree(rootdir, subdir_seq, verbose=False):
	'''
		This function generates a linear sequence of subdirectory within a root directory.
		The output will therefore be of the type: [rootdir]/subdir/subsubdir/subsubsubdir/...
		rootdir: directory at which you want to create the linear sequence of subdirs
		subdir_seq: A vector of directory names. The directories will be created in the specified order
					in which their appear in subdir_seq
	'''
	subdir=rootdir
	for sdir in subdir_seq:
		subdir=os.path.join(subdir, sdir)
		if os.path.isdir(subdir) == False:
			if verbose == True:
				print('           Directory ', subdir, ' does not exist ... creating it')
			os.makedirs(os.path.join(subdir))
		else:
			if verbose == True:
				print('           Directory ', subdir, ' exist. NOTHING TO DO')

def target_subdir_lintree(rootdir, subdir_seq, list_content=False, verbose=False):
	'''
		Compose the path to reach a specified subdirectory and evaluate its content
		If the directory exists, it returns a (1) list with the content and another 
		with the type of the content (Directory or File).
		Otherwise, return an empty string
		rootdir: The root directory at which we start to build the subdirectory sequence
		subdir_seq: A vector of directory names. The string defining the full path
					of the final subdirecotry will be created in the specified order
					in which their appear in subdir_seq
	'''
	subdir=rootdir
	for sdir in subdir_seq:
		subdir=os.path.join(subdir, sdir)
		if os.path.isdir(subdir) == False:
			if verbose == True:
				print('    Directory ', subdir, 'Does not exist')
			return ''
		else:
			if list_content == True:
				out, out_type=list_files(subdir)

	if list_content == True:
		return subdir, out, out_type
	else:
		return subdir
		
def read_MCMCfile(file):
	'''
		Function that reads a .MCMC file
	'''
	f=open(file, 'r')
	txt=f.read()
	txt=txt.splitlines()
	Ntxt=len(txt)
	i=0
	# ---- Extract the common block ----
	while i < Ntxt:
		l=txt[i].strip()
		if l[0] == '#' and l[1:].split('=')[0].strip() == 'KIC':
			KIC=format_ID(l.split('=')[1].strip())
		if l[0] == '!':
			if l[1] != '!':
				Dnu=np.float(l.split('!')[1])
			else:
				C0=np.float(l.split('!!')[1])
		if l[0] =='*':
			n_range=np.array(l.split()[1:], dtype=np.float)
			i0=i+1
		i=i+1
	# ----- Read the successive blocks delimited by '#' -----
	i0=i0+1
	i=i0
	char=''
	while char != '#':
		char=txt[i].strip()[0]
		i=i+1
	i=i-1
	type_vals=txt[i0:i]
	type_vals=format_tab2D(type_vals, dtype=np.str, init_val='')

	extra_priors=txt[i+2:i+2+6]
	extra_priors=np.array(extra_priors, dtype=np.float)
	i=i+2+7

	i0=i
	char=''
	while char != '#':
		char=txt[i].strip()[0]
		i=i+1
	eigen_vals=txt[i0:i-1]
	eigen_vals=format_tab2D(eigen_vals, dtype=np.float, init_val=-999)

	noise_guess=txt[i:i+4]
	noise_guess=format_tab2D(noise_guess, dtype=np.float, init_val=-999)

	i=i+5
	noise_fit=txt[i+1:i+11]
	noise_fit=format_tab2D(noise_fit, dtype=np.float, init_val=-999)

	'''
	print('KIC=', KIC)
	print('Dnu=', Dnu)
	print('C0=', C0)
	print('n_range=', n_range)
	print('extra_priors:')
	print(extra_priors)
	print('type_vals:')
	print(type_vals)
	print('eigen_vals:')
	print(eigen_vals)
	print('noise_guess:')
	print(noise_guess)
	print('noise_fit:')
	print(noise_fit)
	'''
	return KIC, Dnu, C0, n_range, extra_priors, type_vals, eigen_vals, noise_guess, noise_fit
