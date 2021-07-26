import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from gridlod import util
import algorithms

root= 'data/'

#exp1
err1_low = sio.loadmat(root+'exp1_lowreg_errs.mat')
err1_high = sio.loadmat(root+'exp1_highreg_errs.mat')
NList = [4,8,16,32,64]
H = [1./ii for ii in NList]
deltatList = [1./4., 1./8., 1./16., 1./32., 1./64.]
relenergy_space1_low = err1_low['energyspat'][0]
relenergy_time1_low = err1_low['energytime'][0]
relenergy_space1_high = err1_high['energyspat'][0]
relenergy_time1_high = err1_high['energytime'][0]

fig = plt.figure(figsize=(11,7))
ax1 = fig.add_subplot(1,2,1)
ax1.loglog(H, relenergy_space1_low, '-o', color='blue', label='error $f_1$')
ax1.loglog(H, relenergy_space1_high, '-o', color='green', label='error $f_2$')
ax1.loglog(H, 0.5*np.array(H), '--', color='black', label='$O(H)$', dashes=(1.5, 5.),
    dash_capstyle='round')
ax1.loglog(H, 1.5*np.array(H)**2, '--', color='black', label='$O(H^2)$')
ax1.legend(loc='lower right')
ax1.set_xlabel('H')
ax1.set_ylabel('relative energy error')

ax2 = fig.add_subplot(1,2,2)
ax2.loglog(deltatList, relenergy_time1_low, '-o', color='blue', label='error $f_1$')
ax2.loglog(deltatList, relenergy_time1_high, '-o', color='green', label='error $f_2$')
ax2.loglog(deltatList, 15*np.array(deltatList)**2, '--', color='black', label=r'$O(\tau^2)$')
ax2.legend(loc='lower right')
ax2.set_xlabel(r'$\tau$')
ax2.set_ylabel('relative energy error')
plt.show()

#exp2a
err2a = sio.loadmat(root+'exp2a_errs.mat')
NList = [4,8,16,32,64]
H = [1./ii for ii in NList]
relenergy_2a = err2a['energy'][0]

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.loglog(H, relenergy_2a, '-o', color='black', label='error $a_1$')
ax1.loglog(H, np.array(H)**2, '--', color='black', label='$O(H^2)$')
ax1.legend(loc='lower right')
ax1.set_xlabel('H')
ax1.set_ylabel('relative energy error')
plt.show()


#plot coefficient
NFine = np.array([128,128])
xtFine = util.tCoordinates(NFine)
Nepsilon=np.array([32,32])
amult = lambda x: algorithms.build_inclusions_defect_2d(x,Nepsilon,1.,10.)
a = amult(xtFine)
a_Grid = a.reshape(NFine, order='C')

fig = plt.figure()
ax2 = fig.add_subplot(1,1,1)
ax2.imshow(a_Grid, origin='lower_left',extent=(0,1,0,1))
plt.show()


#exp2b - spatial
err2b_local_space = sio.loadmat(root+'exp2b_local_errs_space.mat')
err2b_add_space = sio.loadmat(root+'exp2b_add_errs_space.mat')
NList = err2b_local_space['NList'][0]
H = [1./ii for ii in NList]
relenergy_space2b_local = err2b_local_space['energy'][0]
relenergy_space2b_add = err2b_add_space['energy'][0]

plt.loglog(H, relenergy_space2b_local, '-o', color='blue', label='error $a_2$')
plt.loglog(H, relenergy_space2b_add, '-o', color='green', label='error $a_3$')
plt.loglog(H, 1.5*np.array(H)**2, '--', color='black', label='$O(H^2)$')
plt.loglog(H, 0.25*np.array(H), '--', color='black', label='$O(H)$', dashes=(1.5, 5.),
    dash_capstyle='round')
plt.legend(loc='lower right')
plt.xlabel('H')
plt.xticks(ticks=[1e-1], labels=['$10^{-1}$'])
plt.ylabel('relative energy error')
plt.show()

#exp2b -tolerance
err2b_local_tol = sio.loadmat('exp2b_local_errs_tol.mat')
err2b_add_tol = sio.loadmat('exp2b_add_errs_tol.mat')
tolList = err2b_local_tol['tolList'][0]
relenergy_tol2b_local = err2b_local_tol['energy'][0]
abstol_local = err2b_local_tol['maxtol'][0]
update_local = err2b_local_tol['avupdate'][0]

print('For model local, at tolerance factor 0.5, we have (maximal) tolerance {} and {}% updates on average every step'
      .format(abstol_local[6], update_local[6]))

relenergy_tol2b_add = err2b_add_tol['energy'][0]
abstol_add = err2b_add_tol['maxtol'][0]
update_add = err2b_add_tol['avupdate'][0]

print('For model add, at tolerance factor 0.5, we have (maximal) tolerance {} and {}% updates on average every step'
      .format(abstol_add[6], update_add[6]))

plt.semilogy(tolList, relenergy_tol2b_local, '-o', color='blue', label='error $a_2$')
plt.semilogy(tolList, abstol_local,'--', color='blue', label='maximal tolerance $a_2$')
plt.semilogy(tolList, relenergy_tol2b_add, '-o', color='green', label='error $a_3$')
plt.semilogy(tolList, abstol_add,'--', color='green', label='maximal tolerance $a_3$')
plt.xlabel('tolerance factor')
plt.ylabel('')
plt.legend()
plt.show()
