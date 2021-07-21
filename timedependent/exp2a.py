import numpy as np
import scipy.io as sio

from gridlod.world import World
from gridlod import algorithms, fem, util
import timestepping

NFine = np.array([512,512])
NpFine = np.prod(NFine+1)
T= 1.
deltatref = 1./128.
NTimeref = int(T/deltatref)
dim = np.size(NFine)
tol = -0.1

boundaryConditions = np.array([[0, 0], [0, 0]])

xpFine = util.pCoordinates(NFine)

#inspired by Ammari, Hiltunen - uniformly in time modulated, discontinuous in space
b = 0.5
om = 9.
Nepsilon=np.array([128,128])
amult = lambda x: algorithms.build_inclusions_defect_2d(x,Nepsilon,1.,10.)
afunc = lambda x, t: amult(x) * (1+b*np.cos(om*t))
u0 = np.zeros_like(xpFine[:,0])
v0 = np.zeros_like(xpFine[:,0])
ffunc = lambda x, t: np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])*(5*t+50*t**2)

NList = [4,8,16,32,64]
kList = [1,2,2,3,3]
deltat = 1./128.
NTime = int(T/deltat)

relL2u = np.zeros(len(NList))
relH1u = np.zeros(len(NList))
relL2v = np.zeros(len(NList))
relenergy = np.zeros(len(NList))

for ii in range(len(NList)):
    NCoarse = np.array([NList[ii],NList[ii]])
    k = kList[ii]

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

    relL2u[ii] = l2erru/uRef_l2norm
    relH1u[ii] = h1erru/uRef_h1norm
    relL2v[ii] = l2errv/vRef_l2norm
    relenergy[ii] = np.sqrt(h1erru**2+l2errv**2)/np.sqrt(uRef_h1norm**2+vRef_l2norm)


sio.savemat('exp2a_errs.mat', {'NList': NList, 'L2u': relL2u, 'H1u': relH1u, 'L2vs': relL2v, 'energy': relenergy})
