import numpy as np
import scipy.io as sio

from gridlod.world import World
from gridlod import fem, util
import timestepping

NFine = np.array([512,512])
NpFine = np.prod(NFine+1)
T= 1.
deltatref = 1./128.
NTimeref = int(T/deltatref)
dim = np.size(NFine)
tol = -0.1 #to get 100% updates

boundaryConditions = np.array([[0, 0], [0, 0]])

xpFine = util.pCoordinates(NFine)

#inspired by Tan, Hoang
eps = 128
afunc = lambda x,t: (3+np.sin(2*np.pi*eps*x[:,0])+np.sin(2*np.pi*t))*(3+np.sin(2*np.pi*eps*x[:,1])+np.sin(2*np.pi*t))
u0 = np.zeros_like(xpFine[:,0])
v0 = np.zeros_like(xpFine[:,0])

modelList=['lowreg', 'highreg']
ffunc_high = lambda x, t: 20*t*(x[:,0]-x[:,0]**2)*(x[:,1]-x[:,1]**2)+230*t**2*(x[:,0]-x[:,0]**2+x[:,1]-x[:,1]**2)
def ffunc_low(x,t):
    dom = (x[:,0]<0.4).astype(float)
    one = np.ones_like(x[:,0])
    return (one +9*dom)*(20*t+230*t**2)
ffuncList = {'lowreg': ffunc_low, 'highreg': ffunc_high}

NList = [4,8,16,32,64]
kList = [1,2,2,3,3]
deltatList = [1./4., 1./8., 1./16., 1./32., 1./64.]

for model in modelList:
    relL2u_spat = np.zeros(len(NList))
    relH1u_spat = np.zeros(len(NList))
    relL2v_spat = np.zeros(len(NList))
    relenergy_spat = np.zeros(len(NList))
    relL2u_time = np.zeros(len(deltatList))
    relH1u_time = np.zeros(len(deltatList))
    relL2v_time = np.zeros(len(deltatList))
    relenergy_time = np.zeros(len(deltatList))

    ffunc = ffuncList[model]

    for ii in range(len(NList)):
        NCoarse = np.array([NList[ii],NList[ii]])
        k = kList[ii]
        deltat = deltatref #spatial error - consider finest time step
        NTime = int(T/deltat)

        NCoarseElement = NFine // NCoarse
        world = World(NCoarse, NCoarseElement, boundaryConditions)
        xpCoarse = util.pCoordinates(NCoarse)

        if ii == 0: #only compute refsol once
            #reference solution
            outRef = timestepping.implicitMidpoint_fem(world, ffunc, afunc, u0, v0, deltatref, NTimeref, coarse=False, storev=True)
            uRef = outRef[0]
            vRef = outRef[1]
            MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
            AoneFull = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine) #could also include coeff a (e.g. at end time) here
            uRef_l2norm = np.sqrt(np.dot(MFull*uRef, uRef))
            uRef_h1norm = np.sqrt(np.dot(AoneFull*uRef, uRef))
            vRef_l2norm = np.sqrt(np.dot(MFull*vRef, vRef))
            vRef_h1norm = np.sqrt(np.dot(AoneFull*vRef, vRef))

            print('reference solution computed')


        #LOD solution
        outadap = timestepping.implicitMidpoint_lod(world, ffunc, afunc, u0, v0, deltat, NTime, k, tol, scale=True, storev=True)
        uLodadap = outadap[0]
        vLodadap = outadap[1]
        l2erru = np.sqrt(np.dot(MFull*(uRef-uLodadap), uRef-uLodadap))

        print('LOD for Nc = {}, deltat = {}'.format(NCoarse[0], deltat))

        print('absolute L2 error in u: {}'.format(l2erru))
        print('relative L2 error in u: {}'.format(l2erru/uRef_l2norm))
        h1erru = np.sqrt(np.dot(AoneFull*(uRef-uLodadap), uRef-uLodadap))
        print('absolute H1 error in u: {}'.format(h1erru))
        print('relative H1 error in u: {}'.format(h1erru/uRef_h1norm))

        l2errv = np.sqrt(np.dot(MFull*(vRef-vLodadap), vRef-vLodadap))
        print('absolute L2 error in v: {}'.format(l2errv))
        print('relative L2 error in v: {}'.format(l2errv/vRef_l2norm))
        h1errv = np.sqrt(np.dot(AoneFull*(vRef-vLodadap), vRef-vLodadap))
        print('absolute H1 error in v: {}'.format(h1errv))
        print('relative H1 error in v: {}'.format(h1errv/vRef_h1norm))

        print('relative error in energy norm: {}'.format(np.sqrt(h1erru**2+l2errv**2)/np.sqrt(uRef_h1norm**2+vRef_l2norm)))

        relL2u_spat[ii] = l2erru/uRef_l2norm
        relH1u_spat[ii] = h1erru/uRef_h1norm
        relL2v_spat[ii] = l2errv/vRef_l2norm
        relenergy_spat[ii] = np.sqrt(h1erru**2+l2errv**2)/np.sqrt(uRef_h1norm**2+vRef_l2norm)

    for jj in range(len(deltatList)): #for finest spatial grid do time error analysis
        deltat = deltatList[jj]
        NTime = int(T / deltat)
        # LOD solution
        outadap = timestepping.implicitMidpoint_lod(world, ffunc, afunc, u0, v0, deltat, NTime, k, tol, scale=True,
                                                    storev=True)
        uLodadap = outadap[0]
        vLodadap = outadap[1]

        print('LOD for Nc = {}, deltat = {}'.format(NCoarse[0], deltat))

        l2erru = np.sqrt(np.dot(MFull * (uRef - uLodadap), uRef - uLodadap))
        print('absolute L2 error in u: {}'.format(l2erru))
        print('relative L2 error in u: {}'.format(l2erru / uRef_l2norm))
        h1erru = np.sqrt(np.dot(AoneFull * (uRef - uLodadap), uRef - uLodadap))
        print('absolute H1 error in u: {}'.format(h1erru))
        print('relative H1 error in u: {}'.format(h1erru / uRef_h1norm))

        l2errv = np.sqrt(np.dot(MFull * (vRef - vLodadap), vRef - vLodadap))
        print('absolute L2 error in v: {}'.format(l2errv))
        print('relative L2 error in v: {}'.format(l2errv / vRef_l2norm))
        h1errv = np.sqrt(np.dot(AoneFull * (vRef - vLodadap), vRef - vLodadap))
        print('absolute H1 error in v: {}'.format(h1errv))
        print('relative H1 error in v: {}'.format(h1errv / vRef_h1norm))

        print('relative error in energy norm: {}'.format(
            np.sqrt(h1erru ** 2 + l2errv ** 2) / np.sqrt(uRef_h1norm ** 2 + vRef_l2norm)))

        relL2u_time[jj] = l2erru / uRef_l2norm
        relH1u_time[jj] = h1erru / uRef_h1norm
        relL2v_time[jj] = l2errv / vRef_l2norm
        relenergy_time[jj] = np.sqrt(h1erru ** 2 + l2errv ** 2) / np.sqrt(uRef_h1norm ** 2 + vRef_l2norm)

    sio.savemat('exp1_'+model+'_errs.mat', {'L2uspat': relL2u_spat, 'H1uspat': relH1u_spat, 'L2vspat': relL2v_spat, 'energyspat': relenergy_spat,
                         'L2utime': relL2u_time, 'H1utime': relH1u_time, 'L2vtime': relL2v_time, 'energytime': relenergy_time})
