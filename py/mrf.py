import numpy as np
import matplotlib.pyplot as plt
from os import path
import sys
import cv2
from tqdm import tqdm
import pickle
from scipy import stats

PATH_IMG = '/Users/mac/Desktop/python/pivision/img/'
img0_name = 'Iobservee.png' #'IoriginaleBW.png'

def isargv_empty():
	if len(sys.argv)>2:
		return sys.argv[1], sys.argv[2]
	else:
		return None,None

def ngbhood_4(i,j,h,w):
	return (np.mod(i-1,h), j), (i, np.mod(j+1,w)), (np.mod(i+1,h), j), (i, np.mod(j-1,w))

def local_cdf_4(i,j,h,w,rand_field,state_s,edge_weights, vert_weights):
	U_s = vert_weights[state_s]
	for ngb in ngbhood_4(i,j,h,w):
		state_ngb = rand_field[ngb[0], ngb[1]]
		V_c = edge_weights[state_s, state_ngb]
		U_s+=V_c
	return U_s

def local_data_attach(i, j, state_s, img, mu, scale):
	return np.log(scale[state_s])+(img[i,j]-mu[state_s])**2/(2*scale[state_s]**2)

def generate_vert_and_edges_4(h,w):
	for i in range(h):
		for j in range(w):
			temp_list = [(i,j)]
			if j+1<w:
				temp_list+=[(i,j+1)]
			if i+1<h:
				temp_list+=[(i+1,j)]
			yield temp_list 

def global_energy_4(rand_field, edges_weights_, vert_weights_):

	U_global = 0
	hrf, wrf = rand_field.shape
	for elem in generate_vert_and_edges_4(hrf,wrf):
		state_s = rand_field[elem[0]]
		U_global += vert_weights_[state_s]
		for vert in elem[1:]:
			U_global += edges_weights_[state_s,rand_field[vert]]

	return U_global

def Gibbs_sampling(rand_field, n_iter, model_dict, png_filename=None):

	hrf, wrf = rand_field.shape
	n_states = model_dict['n_states']
	weights_edges_ = model_dict['edges_weights']
	weights_vert_ = model_dict['vert_weights']

	if n_states==2:
		cmap_ = 'gist_yarg'
		plttitle = 'MRF: Ising model'
	else:
		cmap_ = 'Blues'
		plttitle = 'MRF: potts model'

	if png_filename is not None:
		rf_img = rf * np.floor(255/(n_states-1))
		plt.subplot(161)
		plt.suptitle(plttitle)
		plt.title('initial')
		plt.xticks([], [])
		plt.yticks([], [])
		plt.imshow(rf_img,cmap=cmap_)
		n_subplots=1

	U_global=[]
	for k in tqdm(range(n_iter)):
		i_s = np.random.randint(0,hrf)
		j_s = np.random.randint(0,wrf)

		V = np.empty(n_states)
		for i, state in enumerate(range(n_states)):
			V[i] = local_cdf_4(i_s,j_s,hrf,wrf,rand_field,state,weights_edges_,weights_vert_)

		proba = np.exp(-V)
		proba /= np.sum(proba)

		new_state = np.random.choice(list(range(n_states)),1,p=proba)[0]

		old_state = rand_field[i_s,j_s] 
		rf[i_s,j_s] = new_state
		if png_filename is not None and (k==np.floor(0.2*n_iter) or k==np.floor(0.4*n_iter) or k==np.floor(0.6*n_iter) or k==np.floor(0.8*n_iter)):
			rf_img = rf * np.floor(255/(n_states-1))
			plt.subplot(161+n_subplots)
			plt.title(str(k))
			plt.xticks([], [])
			plt.yticks([], [])
			plt.imshow(rf_img,cmap=cmap_)
			n_subplots+=1

		# compute global energy
		#U_global += [global_energy_4(rand_field, weights_edges_, weights_vert_)]

	if png_filename is not None:
		rf_img = rf * np.floor(255/(n_states-1))
		plt.subplot(161+n_subplots)
		plt.title('final')
		plt.xticks([], [])
		plt.yticks([], [])
		plt.imshow(rf_img,cmap=cmap_)
		#plt.show()
		plt.savefig(png_filename)
	return U_global

def Metropolis_sampling(rand_field, n_iter, model_dict, png_filename=None):

	hrf, wrf = rand_field.shape
	n_states = model_dict['n_states']
	weights_edges_ = model_dict['edges_weights']
	weights_vert_ = model_dict['vert_weights']

	if n_states==2:
		cmap_ = 'gist_yarg'
		plttitle = 'MRF: Ising model'
	else:
		cmap_ = 'Blues'
		plttitle = 'MRF: potts model'

	if png_filename is not None:
		rf_img = rf * np.floor(255/(n_states-1))
		plt.subplot(161)
		plt.suptitle(plttitle)
		plt.title('initial')
		plt.xticks([], [])
		plt.yticks([], [])
		plt.imshow(rf_img,cmap=cmap_)
		n_subplots=1

	U_global = []

	for k in tqdm(range(n_iter)):
		i_s = np.random.randint(0,hrf)
		j_s = np.random.randint(0,wrf)

		curr_state = rand_field[i_s,j_s]
		curr_local_U = local_cdf_4(i_s,j_s,hrf,wrf,rand_field,curr_state,weights_edges_,weights_vert_)

		temp_state = np.random.randint(0,n_states)
		temp_local_U = local_cdf_4(i_s,j_s,hrf,wrf,rand_field,temp_state,weights_edges_,weights_vert_)

		delta_local_U = temp_local_U - curr_local_U
		if delta_local_U < 0:
			rand_field[i_s,j_s] = temp_state
		else:
			if np.random.rand() < np.exp(-delta_local_U):
				rand_field[i_s,j_s] = temp_state

		if png_filename is not None and (k==np.floor(0.2*n_iter) or k==np.floor(0.4*n_iter) or k==np.floor(0.6*n_iter) or k==np.floor(0.8*n_iter)):
			rf_img = rf * np.floor(255/(n_states-1))
			plt.subplot(161+n_subplots)
			plt.title(str(k))
			plt.xticks([], [])
			plt.yticks([], [])
			plt.imshow(rf_img,cmap=cmap_)
			n_subplots+=1

		# compute global energy
		#U_global += [global_energy_4(rand_field, weights_edges_, weights_vert_)]

	if png_filename is not None:
		rf_img = rf * np.floor(255/(n_states-1))
		plt.subplot(161+n_subplots)
		plt.title('final')
		plt.xticks([], [])
		plt.yticks([], [])
		plt.imshow(rf_img,cmap=cmap_)
		#plt.show()
		plt.savefig(png_filename)

	return U_global

def ICM_sampling(rand_field, n_iter, model_dict, png_filename=None):

	hrf, wrf = rand_field.shape
	n_states = model_dict['n_states']
	weights_edges_ = model_dict['edges_weights']
	weights_vert_ = model_dict['vert_weights']

	if n_states==2:
		cmap_ = 'gist_yarg'
		plttitle = 'MRF: Ising model'
	else:
		cmap_ = 'Blues'
		plttitle = 'MRF: potts model'

	if png_filename is not None:
		rf_img = rf * np.floor(255/(n_states-1))
		plt.subplot(161)
		plt.suptitle(plttitle)
		plt.title('initial')
		plt.xticks([], [])
		plt.yticks([], [])
		plt.imshow(rf_img,cmap=cmap_)
		n_subplots=1

	U_global=[]
	for k in tqdm(range(n_iter)):
		i_s = np.random.randint(0,hrf)
		j_s = np.random.randint(0,wrf)

		V = np.empty(n_states)
		for i, state in enumerate(range(n_states)):
			V[i] = local_cdf_4(i_s,j_s,hrf,wrf,rand_field,state,weights_edges_,weights_vert_)

		# choose state which min local energy
		new_state = np.random.choice(np.flatnonzero(V == V.min())) # np.argmin(V)
		rf[i_s,j_s] = new_state

		if png_filename is not None and (k==np.floor(0.2*n_iter) or k==np.floor(0.4*n_iter) or k==np.floor(0.6*n_iter) or k==np.floor(0.8*n_iter)):
			rf_img = rf * np.floor(255/(n_states-1))
			plt.subplot(161+n_subplots)
			plt.title(str(k))
			plt.xticks([], [])
			plt.yticks([], [])
			plt.imshow(rf_img,cmap=cmap_)
			n_subplots+=1

		# compute global energy
		#U_global += [global_energy_4(rand_field, weights_edges_, weights_vert_)]

	if png_filename is not None:
		rf_img = rf * np.floor(255/(n_states-1))
		plt.subplot(161+n_subplots)
		plt.title('final')
		plt.xticks([], [])
		plt.yticks([], [])
		plt.imshow(rf_img,cmap=cmap_)
		#plt.show()
		plt.savefig(png_filename)
	return U_global

def grey_img_seg_by_mrf(img, rand_field, mu, scale, n_iter, model_dict, png_filename=None):

	hrf, wrf = rand_field.shape
	n_states = model_dict['n_states']
	weights_edges_ = model_dict['edges_weights']
	weights_vert_ = model_dict['vert_weights']

	if n_states==2:
		cmap_ = 'gray'#'gist_yarg'
		plttitle = 'MRF: Ising model'
	else:
		cmap_ = 'Blues'
		plttitle = 'MRF: potts model'

	if png_filename is not None:
		rf_img = rf * np.floor(255/(n_states-1))
		plt.subplot(161)
		plt.suptitle(plttitle)
		plt.title('initial')
		plt.xticks([], [])
		plt.yticks([], [])
		plt.imshow(rf_img,cmap=cmap_)
		n_subplots=1

	Err_global=[]
	for k in tqdm(range(n_iter)):
		i_s = np.random.randint(0,hrf)
		j_s = np.random.randint(0,wrf)

		V = np.empty(n_states)
		for i, state in enumerate(range(n_states)):
			# local prior energy
			V[i] = local_cdf_4(i_s,j_s,hrf,wrf,rand_field,state,weights_edges_,weights_vert_)
			# local data attach
			#print('local data attach: ',local_data_attach(i_s, j_s, state, img, mu, scale))
			V[i] += local_data_attach(i_s, j_s, state, img, mu, scale)

		# choose state which min local energy
		new_state = np.random.choice(np.flatnonzero(V == V.min())) # np.argmin(V)
		rf[i_s,j_s] = int(new_state)
		if png_filename is not None and (k==np.floor(0.2*n_iter) or k==np.floor(0.4*n_iter) or k==np.floor(0.6*n_iter) or k==np.floor(0.8*n_iter)):
			rf_img = rf * int(np.floor(255/(n_states-1)))
			plt.subplot(161+n_subplots)
			plt.title(str(k))
			plt.xticks([], [])
			plt.yticks([], [])
			plt.imshow(rf_img, cmap=cmap_)
			n_subplots+=1

		# compute global error 
		#emp_img = rf * np.floor(255/(n_states-1))
		#Err_global += [np.linalg.norm(true_img-emp_img,ord='fro')]

	if png_filename is not None:
		rf_img = rf * np.floor(255/(n_states-1))
		plt.subplot(161+n_subplots)
		plt.title('final')
		plt.xticks([], [])
		plt.yticks([], [])
		plt.imshow(rf_img,cmap=cmap_)
		#plt.show()
		plt.savefig(png_filename)

	return Err_global

# ------------------------------------------------------------------------------------------------------------------------------------------------------


model_choice, sampling_choice = isargv_empty()

if model_choice=='ising':

	# Ising model
	alpha_00_11 = 0.
	alpha_01_10 = 1.
	Ising_model = {}
	Ising_model['n_states'] = 2
	Ising_model['edges_weights'] = np.array([[alpha_00_11,alpha_01_10],[alpha_01_10,alpha_00_11]])
	Ising_model['vert_weights'] = np.zeros(Ising_model['n_states'])

	hrf = 50
	wrf = 50
	rf = np.uint8(np.floor(Ising_model['n_states']*np.random.rand(hrf,wrf)))
	#rf = np.uint8(np.floor(Ising_model['n_states']*np.zeros((hrf,wrf))))
	rf_img = rf * np.floor(255/(Ising_model['n_states']-1))

	png_filename_ = 'ising_gibbs_1.png'#'potts_with_hierarch_mix.png'

	if sampling_choice == 'gibbs':
		# Gibbs sampling
		rf = Gibbs_sampling(rf, 17500, Ising_model, png_filename=png_filename_)

	elif sampling_choice == 'metropolis' or sampling_choice == 'metro':
		# Metropolis sampling
		U_global = Metropolis_sampling(rf, 17500, Ising_model, png_filename=png_filename_)

	elif sampling_choice == 'icm':
		# ICM sampling
		U_global = ICM_sampling(rf, 17500, Ising_model, png_filename=png_filename_)

	plt.imread(png_filename_)
	plt.show()


if model_choice=='potts':

	# Potts model
	alpha_00_11_22 = 0.25
	alpha_01_10 = 0.75
	alpha_02_20 = 2.5
	alpha_12_21 = 0.75
	Potts_model = {}
	Potts_model['n_states'] = 3
	Potts_model['edges_weights'] = np.array([[alpha_00_11_22,alpha_01_10,alpha_02_20],[alpha_01_10,alpha_00_11_22,alpha_12_21],[alpha_02_20,alpha_12_21,alpha_00_11_22]])
	Potts_model['vert_weights'] = np.array([1., 2., 1.]) #np.zeros(Potts_model['n_states'])

	hrf = 75
	wrf = 75
	rf = np.uint8(np.floor(Potts_model['n_states']*np.random.rand(hrf,wrf)))
	rf_img = rf * np.floor(255/(Potts_model['n_states']-1))

	png_filename_ = 'potts_test.png'#'potts_with_hierarch_mix.png'

	if sampling_choice == 'gibbs':
		# Gibbs sampling
		U_global = Gibbs_sampling(rf, 17500, Potts_model, png_filename=png_filename_)

	elif sampling_choice == 'metropolis' or sampling_choice == 'metro':
		# Metropolis sampling
		U_global = Metropolis_sampling(rf, 17500, Potts_model, png_filename=png_filename_)

	elif sampling_choice == 'icm':
		# ICM sampling
		U_global = ICM_sampling(rf, 17500, Potts_model, png_filename=png_filename_)

	plt.imread(png_filename_)
	plt.show()



if path.exists(PATH_IMG+img0_name):
	img0 = cv2.imread(PATH_IMG+img0_name)
	cv2.imshow('img',img0)
	cv2.waitKey(0)

if path.exists(PATH_IMG+'IoriginaleBW.png'):
	true_img = cv2.imread(PATH_IMG+'IoriginaleBW.png')
	cv2.imshow('img original',true_img)
	cv2.waitKey(0)


"""
mean_emp0 = 97.19183673469388
std_emp0 = 22.384121893720472

mean_emp1 = 163.88734693877552
std_emp1 = 22.667462140699424

patch_0 = img0[50:120,175:245].flatten()
#plt.hist(img0[50:120,175:245].flatten(),40)
#plt.show()
print('mean: ',np.mean(patch_0),'  |  std: ',np.std(patch_0))

patch_1 = img0[240:310,125:195].flatten()
#plt.hist(img0[240:310,125:195].flatten(),40)
#plt.show()
print('mean: ',np.mean(patch_1),'  |  std: ',np.std(patch_1))

n_samples = np.prod(patch_0.shape)
print('n_samples: ', n_samples)
print('student quantile: ',stats.t.ppf(q=0.95, df=n_samples-1))

q_t95 = stats.t.ppf(q=0.95, df=n_samples-1)
conf_interv_0 = lambda alpha: (mean_emp0 - stats.t.ppf(q=alpha, df=n_samples-1) *std_emp0/np.sqrt(n_samples-1), mean_emp0 + stats.t.ppf(q=alpha, df=n_samples-1)*std_emp0/np.sqrt(n_samples-1))
conf_interv_1 = lambda alpha: (mean_emp1 - stats.t.ppf(q=alpha, df=n_samples-1) *std_emp1/np.sqrt(n_samples-1), mean_emp1 + stats.t.ppf(q=alpha, df=n_samples-1)*std_emp1/np.sqrt(n_samples-1))

print('conf_interv: ', conf_interv_0(0.95))
print('conf_interv: ', conf_interv_0(0.97))
print('conf_interv: ', conf_interv_0(0.99))

def test(y_s):
	lb0, ub0 = conf_interv_0(0.95)
	stats.norm(y_s,)
"""


# Ising model
alpha_00_11 = 0.
alpha_01_10 = 2.
Ising_model = {}
Ising_model['n_states'] = 2
Ising_model['edges_weights'] = np.array([[alpha_00_11,alpha_01_10],[alpha_01_10,alpha_00_11]])
Ising_model['vert_weights'] = np.zeros(Ising_model['n_states'])

hrf = img0.shape[0]
wrf = img0.shape[1]
rf = np.uint8(np.floor(Ising_model['n_states']*np.random.rand(hrf,wrf)))
#rf = np.uint8(np.floor(Ising_model['n_states']*np.zeros((hrf,wrf))))
rf_img = rf * int(np.floor(255/(Ising_model['n_states']-1)))

mu = (97.2, 163.9)
scale = (22.4, 22.7) 
png_filename_ = 'test_with_ising.png'#'img_seg_without_ising_test.png'#'img_seg_without_ising_test.png'
Err_global = grey_img_seg_by_mrf(img0[:,:,0], true_img[:,:,0], rf, mu, scale, 1000000, Ising_model, png_filename=png_filename_)

plt.imread(png_filename_)
plt.show()

plt.plot(list(range(1000000)),Err_global)
plt.show()










