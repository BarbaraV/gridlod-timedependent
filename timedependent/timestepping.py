import numpy as np
import scipy.sparse as sparse

from gridlod import coef, fem, interp, lod, pglod, util
import algorithms


def implicitMidpoint_fem(world, f, a, u0, v0, deltat, nsteps, coarse=False, storeall=False, storev=False):
    #solve hyperbolic system with implicit Midpoint rule using finite element method on coarse or fine grid (switched according to
    #variable coarse, default: fine grid). With storeall you can decide whether to return only the value at final time
    #(default) or the whole time trajectory

    if coarse:
        M = fem.assemblePatchMatrix(world.NWorldCoarse, world.MLocCoarse)
        basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
        xp = util.pCoordinates(world.NWorldCoarse)
        xt = util.tCoordinates(world.NWorldCoarse)
    else:
        M = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
        xp = util.pCoordinates(world.NWorldFine)
        xt = util.tCoordinates(world.NWorldFine)


    u = np.copy(u0)
    v = np.copy(v0)
    if storeall:
        uList = []
        if storev:
            vList = []
        if coarse:
            uList.append(basis*u0)
            if storev:
                vList.append(basis*v0)
        else:
            uList.append(u0)
            if storev:
                vList.append(v0)

    if coarse:
        fixed = util.boundarypIndexMap(world.NWorldCoarse, world.boundaryConditions == 0)
        free = np.setdiff1d(np.arange(world.NpCoarse), fixed)
    else:
        fixed = util.boundarypIndexMap(world.NWorldFine, world.boundaryConditions == 0)
        free = np.setdiff1d(np.arange(world.NpFine), fixed)

    for nt in range(nsteps):
        fn = f(xp, (nt+0.5)*deltat).flatten()
        vFull = np.zeros_like(v)
        if coarse:
            K = fem.assemblePatchMatrix(world.NWorldCoarse, world.ALocCoarse, a(xt, (nt + 0.5) * deltat))
        else:
            K = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine, a(xt, (nt + 0.5)*deltat))
        S = M + deltat**2/4*K
        b = M * deltat/2*fn - deltat/2*K*u + M*v
        SFree = S[free][:, free]
        bFree = b[free]
        vFree = sparse.linalg.spsolve(SFree, bFree)
        vFull[free] = vFree
        uFull = deltat* vFull + u
        vFulln = 2*vFull - v
        v = np.copy(vFulln)
        u = np.copy(uFull)
        if storeall:
            if coarse:
                uList.append(basis*u)
                if storev:
                    vList.append(basis*v)
            else:
                uList.append(u)
                if storev:
                    vList.append(v)

    if storeall:
        if storev:
            out = [uList, vList]
        else:
            out = uList
    elif coarse:
        if storev:
            out=[basis*u, basis*v]
        else:
            out = basis*u
    else:
        if storev:
            out = [u, v]
        else:
            out = u

    return out


def implicitMidpoint_lod(world, f, a, u0, v0, deltat, nsteps, k, tol, use_anew=False,
                         scale=False, storeall=False, storev=False, outdict = None):
    #solve hyperbolic system with implicit midpoint rule using adaptive PG-LOD.
    # With storeall you can decide whether to return
    # only the value at final time (default) or the whole time trajectory
    #u0,v0 is assumed to be in the fine FE space
    #with tol=0 multiscale space is updated in every step
    #usage of use_anew and scale explained in algorithms.py

    #non.zero initial values are not yet tested!

    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)

    basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    xp = util.pCoordinates(world.NWorldFine)
    xt = util.tCoordinates(world.NWorldFine)

    elementsupdatedList = []
    tolList = []

    for nt in range(nsteps):
        fn = f(xp, (nt+0.5)*deltat).flatten()
        aNew = a(xt, (nt+0.5)*deltat)

        if nt == 0:
            if use_anew:
                patchT,correctorsRef,KmsijRef,muTPrimeRef,MmsijRef,aRef = algorithms.initialComputations_correctors(world,aNew,k)
            else:
                patchT,KmsijRef,muTPrimeRef,MmsijRef,aRef = algorithms.initialComputations_nocorrectors(world,aNew,k)
                correctorsRef=None
            K = pglod.assembleMsStiffnessMatrix(world,patchT,KmsijRef)
            M = pglod.assembleMsStiffnessMatrix(world, patchT, MmsijRef)

            uFull = sparse.linalg.spsolve(M, basis.T * MFull * u0)
            vFull = sparse.linalg.spsolve(M, basis.T * MFull * v0)
            if storeall:
                uList = []
                uList.append(u0)
                if storev:
                    vList = []
                    vList.append(v0)

        else:
            K, M, E_vh, quantitiesNew = algorithms.compute_Stiffness_adaptive(world,aNew,aRef,KmsijRef,muTPrimeRef,MmsijRef,
                                                                           patchT,tol,use_anew,correctorsRef,scale)
            aRef = quantitiesNew['a']
            KmsijRef = quantitiesNew['Kmsij']
            muTPrimeRef = quantitiesNew['muTPrime']
            MmsijRef = quantitiesNew['Mmsij']
            elementsupdatedList.append(quantitiesNew['updates'])
            tolList.append(quantitiesNew['tol'])
            if use_anew:
                correctorsRef = quantitiesNew['correctors']

        SFull = M + deltat**2/ 4 * K #MCoarse
        bFull = basis.T * MFull * deltat / 2 * fn - deltat / 2 * K * uFull + M * vFull #MCoarse*vFull
        vNew, _ = pglod.solve(world, SFull, bFull, world.boundaryConditions)
        uFulln = deltat * vNew + uFull
        vFulln = 2*vNew-vFull
        uFull = np.copy(uFulln)
        vFull = np.copy(vFulln)

        def computeCorrec(TInd):
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patchT[TInd], world.boundaryConditions)
            aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aNew)

            correctorsList = lod.computeBasisCorrectors(patchT[TInd], IPatch, aPatch)
            return correctorsList

        if storeall:
            correcT = list(map(computeCorrec, range(world.NtCoarse)))
            modifiedbasis = basis - pglod.assembleBasisCorrectors(world, patchT, correcT)
            uList.append(modifiedbasis*uFull)
            if storev:
                vList.append(modifiedbasis*vFull)

    if storeall:
        if storev:
            out = [uList,vList]
        else:
            out = uList
    else:
        correcT = list(map(computeCorrec, range(world.NtCoarse)))
        modifiedbasis = basis - pglod.assembleBasisCorrectors(world, patchT, correcT)
        if storev:
            out = [modifiedbasis*uFull, modifiedbasis*vFull]
        else:
            out = modifiedbasis*uFull

    elementsupdatedList = np.array(elementsupdatedList)
    tolList = np.array(tolList)
    print('{}% updates on average every step, minimum {}% updates, maximum {}% updates, over all time steps {}% updates'.
          format(np.mean(elementsupdatedList/world.NtCoarse*100), np.min(elementsupdatedList/world.NtCoarse*100),
                 np.max(elementsupdatedList/world.NtCoarse*100),
                 np.sum(elementsupdatedList)/(world.NtCoarse*len(elementsupdatedList))*100))
    print('maximal tolerance over all time steps {}'.format(np.max(tolList)))

    if outdict is not None:
        outdict['maxtol'].append(np.max(tolList))
        outdict['avupdate'].append(np.mean(elementsupdatedList/world.NtCoarse*100))

    return out
