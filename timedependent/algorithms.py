import numpy as np
import scipy.sparse as sparse
import scipy

from gridlod import coef, fem, interp, lod, pglod, util
from gridlod.world import Patch

def scaleCoefficient(aFine):

    aMean = None

    if aFine.ndim == 1:
        aMean = np.mean(aFine, axis=0)
    elif aFine.ndim == 3:
        aMean = np.mean(np.trace(aFine, axis1=1, axis2=2))
    else:
        NotImplementedError('only scalar- and matrix-valued coefficients supported')

    return aFine/aMean

def averageCoefficient(aFine):

    aMean = None

    if aFine.ndim == 1:
        aMean = np.mean(aFine, axis=0)
    elif aFine.ndim == 3:
        aMean = np.mean(np.trace(aFine, axis1=1, axis2=2))
    else:
        NotImplementedError('only scalar- and matrix-valued coefficients supported')

    return aMean

def assembleMsStiffnesMatrix_scaling(world, patchT, KmsijT, factorT):
    '''Compute the multiscale Petrov-Galerkin stiffness matrix given
    Kmsij and a scaling factor (associated with the coefficient) for each coarse element.

    '''
    NWorldCoarse = world.NWorldCoarse

    NtCoarse = np.prod(world.NWorldCoarse)
    NpCoarse = np.prod(world.NWorldCoarse+1)

    TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
    TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)

    cols = []
    rows = []
    data = []
    for TInd in range(NtCoarse):
        Kmsij = factorT[TInd]*KmsijT[TInd]
        patch = patchT[TInd]

        NPatchCoarse = patch.NPatchCoarse

        patchpIndexMap = util.lowerLeftpIndexMap(NPatchCoarse, NWorldCoarse)
        patchpStartIndex = util.convertpCoordIndexToLinearIndex(NWorldCoarse, patch.iPatchWorldCoarse)

        colsT = TpStartIndices[TInd] + TpIndexMap
        rowsT = patchpStartIndex + patchpIndexMap
        dataT = Kmsij.flatten()

        cols.extend(np.tile(colsT, np.size(rowsT)))
        rows.extend(np.repeat(rowsT, np.size(colsT)))
        data.extend(dataT)

    Kms = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

    return Kms


def compute_Stiffness_adaptive(world, aNew, aRef, KmsijRef, muTPrimeRef, MmsijRef, patchT, tol_rel,
                               use_anew=False, correctorsRef=None, scale=False):
    #compute PG-LOD stiffness matrix given reference quantitities and new coefficient based on evaluation of error
    #indicator and tolerance (relative w.r.t. maximal error indicator value). Two variants implemented: re-using old
    #bilinear forms where possible, does not require storage of correctors (use_anew=False, default); and only re-using
    #old correctors where possible, but assembling stiffness matrix with current coefficient. With scale you can switch
    #between scaled and unscaled version of the error indicator (default: unscaled).

    if use_anew:
        assert(correctorsRef is not None)

    #compute error indicator for a single element
    def computeIndicator(TInd):
        aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aNew)  # true coefficient
        E_vh = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], muTPrimeRef[TInd], aRef[TInd], aPatch)

        return E_vh

    #compute scaled error indicator for a single element
    def computeIndicatorScaled(TInd):
        aPatchFactor = averageCoefficient(coef.localizeCoefficient(patchT[TInd], aNew)) \
                       / averageCoefficient(aRef[TInd])
        aPatch = lambda: scaleCoefficient(
            coef.localizeCoefficient(patchT[TInd], aNew))  # true coefficient of current iteration
        rPatch = lambda: scaleCoefficient(aRef[TInd])  # 'reference' coefficient from previous iteration, already localized!

        E_vh = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], muTPrimeRef[TInd], rPatch, aPatch)

        return E_vh, aPatchFactor

    #update correctors for a single element
    def UpdateCorrectors(TInd):
        rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aNew)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patchT[TInd], world.boundaryConditions)

        correctorsList = lod.computeBasisCorrectors(patchT[TInd], IPatch, rPatch)
        csi = computeBasisCoarseQuantities(patchT[TInd], correctorsList, rPatch)

        return patchT[TInd], correctorsList, csi.Kmsij, csi.muTPrime, csi.Mmsij, rPatch()

    def UpdateElements(tol, E, a_old, Kmsij_old, Mmsij_old, use_anew, correctorsRef, scale=False, aFactorRef=None):
        if scale:
            assert(aFactorRef is not None)

        Elements_to_be_updated = []
        for (i, eps) in E.items():
            if eps > tol:
                Elements_to_be_updated.append(i)

        KmsijT_list = list(np.copy(Kmsij_old))
        MmsijT_list = list(np.copy(Mmsij_old))

        if use_anew:
            for TInd in np.setdiff1d(range(world.NtCoarse), Elements_to_be_updated):
                rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aNew)
                csi = lod.computeBasisCoarseQuantities(patchT[TInd], correctorsRef[TInd], rPatch)
                KmsijT_list[TInd] = csi.Kmsij

        if np.size(Elements_to_be_updated) != 0:
            aT_List = list(np.copy(a_old))
            muTPrimeList = list(np.copy(muTPrimeRef))
            if use_anew:
                correctors_List = list(np.copy(correctorsRef))
            if scale and not use_anew:
                aFactor_List = list(np.copy(aFactorRef))
            patchT_irrelevant, correctorsListTNew, KmsijTNew, muTPrimeNew, MmsijTNew, aPatchNew \
                = zip(*map(UpdateCorrectors, Elements_to_be_updated))

            i = 0
            for T in Elements_to_be_updated:
                KmsijT_list[T] = np.copy(KmsijTNew[i])
                MmsijT_list[T] = np.copy(MmsijTNew[i])
                aT_List[T] = np.copy(aPatchNew[i])
                if use_anew:
                    correctors_List[T] = correctorsListTNew[i]
                    muTPrimeList[T] = muTPrimeNew[i]
                if scale and not use_anew:
                    aFactor_List[T] = 1.
                i += 1

            KmsijT = tuple(KmsijT_list)
            MmsijT = tuple(MmsijT_list)
            if use_anew:
                correctorsT = tuple(correctors_List)
                quantitiesNew = {'Kmsij': KmsijT, 'muTPrime': muTPrimeList, 'Mmsij': MmsijT, 'a': aT_List, 'correctors': correctorsT, 'updates': np.size(Elements_to_be_updated), 'tol': tol}
            elif scale:
                quantitiesNew = {'Kmsij': KmsijT, 'muTPrime': muTPrimeList, 'Mmsij': MmsijT, 'a': aT_List, 'aFactor': aFactor_List, 'updates': np.size(Elements_to_be_updated), 'tol': tol}
            else:
                quantitiesNew = {'Kmsij': KmsijT, 'muTPrime': muTPrimeList, 'Mmsij': MmsijT, 'a': aT_List, 'updates': np.size(Elements_to_be_updated), 'tol': tol}
        else:
            KmsijT = tuple(KmsijT_list)
            MmsijT = tuple(MmsijT_list)
            if use_anew:
                quantitiesNew = {'Kmsij': KmsijT, 'muTPrime': muTPrimeRef, 'Mmsij': MmsijT, 'a': aRef, 'correctors': correctorsRef, 'updates': np.size(Elements_to_be_updated), 'tol': tol}
            elif scale:
                quantitiesNew = {'Kmsij': KmsijT, 'muTPrime': muTPrimeRef, 'Mmsij': MmsijT, 'a': aRef, 'aFactor': aFactorRef, 'updates': np.size(Elements_to_be_updated), 'tol': tol}
            else:
                quantitiesNew = {'Kmsij': KmsijT, 'muTPrime': muTPrimeRef, 'Mmsij': MmsijT, 'a': aRef, 'updates': np.size(Elements_to_be_updated), 'tol': tol}
        return quantitiesNew

    #compute error indicators
    if scale:
        E_vh, aFactorList = zip(*map(computeIndicatorScaled, range(world.NtCoarse)))
    else:
        E_vh = list(map(computeIndicator, range(world.NtCoarse)))
        aFactorList = None
    E = {i: E_vh[i] for i in range(np.size(E_vh)) if E_vh[i] > 1e-14}
    tol = np.min(E_vh)+tol_rel*(np.max(E_vh)-np.min(E_vh))

    # loop over elements with possible recomputation of correctors
    quantitiesNew = UpdateElements(tol, E, aRef, KmsijRef, MmsijRef, use_anew, correctorsRef, scale, aFactorList)

    #assembly of matrix
    if scale:
        KFull = assembleMsStiffnesMatrix_scaling(world, patchT, quantitiesNew['Kmsij'], quantitiesNew['aFactor'])
    else:
        KFull = pglod.assembleMsStiffnessMatrix(world, patchT, quantitiesNew['Kmsij'])
    MFull = pglod.assembleMsStiffnessMatrix(world, patchT, quantitiesNew['Mmsij'])

    return KFull, MFull, E_vh, quantitiesNew


def initialComputations_correctors(world, a, k):

    def computeKmsij(TInd):
        patch = Patch(world, k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, world.boundaryConditions)
        aPatch = coef.localizeCoefficient(patch, a)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        return patch, correctorsList, csi.Kmsij, csi.muTPrime, csi.Mmsij, aPatch

    patchT, correctorsListT, KmsijT, muTPrimeList, MmsijT, aT = zip(*map(computeKmsij, range(world.NtCoarse)))
    return patchT, correctorsListT, KmsijT, muTPrimeList, MmsijT, aT

def initialComputations_nocorrectors(world, a, k):

    def computeKmsij(TInd):
        patch = Patch(world, k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, world.boundaryConditions)
        aPatch = coef.localizeCoefficient(patch, a)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        return patch, csi.Kmsij, csi.muTPrime, csi.Mmsij, aPatch

    patchT, KmsijT, muTPrimeList, MmsijT, aT = zip(*map(computeKmsij, range(world.NtCoarse)))
    return patchT, KmsijT, muTPrimeList, MmsijT, aT

#================================================================
#adaption from lod
def computeBasisCoarseQuantities(patch, correctorsList, aPatch):
    ''' Compute the coarse quantities for the basis and its correctors

    Compute the tensors (T is implcit by the patch definition) and
    return them in a CoarseScaleInformation object:

       KTij   = (A \nabla lambda_j, \nabla lambda_i)_{T}
       KmsTij = (A \nabla (lambda_j - corrector_j), \nabla lambda_i)_{U_k(T)}
       MTij   = (lambda_j, lambda_i)_{T}
       KmsTij = (lambda_j - corrector_j, lambda_i)_{U_k(T)}
       muTT'  = max_{\alpha_j} || A \alpha_j (\chi_T \nabla lambda_j - \nabla corrector_j) ||^2_T' / || A \alpha_j \nabla lambda_j ||^2_T

       Auxiliary quantities are computed, but not saved, e.g.
         LTT'ij = (A \nabla (chi_T - Q_T)lambda_j, \nabla (chi_T - Q_T) lambda_i)_{T'}

    '''

    lambdasList = list(patch.world.localBasis.T)

    NPatchCoarse = patch.NPatchCoarse
    NTPrime = np.prod(NPatchCoarse)
    NpPatchCoarse = np.prod(NPatchCoarse + 1)
    numLambdas = len(lambdasList)

    TInd = util.convertpCoordIndexToLinearIndex(patch.NPatchCoarse - 1, patch.iElementPatchCoarse)

    Kmsij = np.zeros((NpPatchCoarse, numLambdas))
    Mmsij = np.zeros((NpPatchCoarse, numLambdas))
    LTPrimeij = np.zeros((NTPrime, numLambdas, numLambdas))
    Kij = np.zeros((numLambdas, numLambdas))
    Mij = np.zeros((numLambdas, numLambdas))

    def accumulate(TPrimeInd, TPrimei, P, Q, KTPrime, MTPrime, _MTPrimeij, BTPrimeij, CTPrimeij):
        if TPrimeInd == TInd:
            Kij[:] = np.dot(P.T, KTPrime * P)
            Mij[:] = np.dot(P.T, MTPrime * P)
            LTPrimeij[TPrimeInd] = CTPrimeij \
                                   - BTPrimeij \
                                   - BTPrimeij.T \
                                   + Kij

            Kmsij[TPrimei, :] += Kij - BTPrimeij
            Mmsij[TPrimei, :] += Mij - _MTPrimeij
        else:
            LTPrimeij[TPrimeInd] = CTPrimeij
            Kmsij[TPrimei, :] += -BTPrimeij
            Mmsij[TPrimei, :] += -_MTPrimeij

    performTPrimeLoop(patch, lambdasList, correctorsList, aPatch, accumulate)

    muTPrime = np.zeros(NTPrime)
    cutRows = 0
    while np.linalg.cond(Kij[cutRows:, cutRows:]) > 1e8:
        cutRows = cutRows + 1
    for TPrimeInd in np.arange(NTPrime):
        # Solve eigenvalue problem LTPrimeij x = mu_TPrime Mij x
        eigenvalues = scipy.linalg.eigvals(LTPrimeij[TPrimeInd][cutRows:, cutRows:], Kij[cutRows:, cutRows:])
        muTPrime[TPrimeInd] = np.max(np.real(eigenvalues))

    return CoarseScaleInformation(Kij, Kmsij, muTPrime, Mij, Mmsij)


def performTPrimeLoop(patch, lambdasList, correctorsList, aPatch, accumulate):
    while callable(aPatch):
        aPatch = aPatch()

    world = patch.world
    NCoarseElement = world.NCoarseElement
    NPatchCoarse = patch.NPatchCoarse
    NPatchFine = NPatchCoarse * NCoarseElement

    NTPrime = np.prod(NPatchCoarse)
    NpPatchCoarse = np.prod(NPatchCoarse + 1)

    d = np.size(NPatchCoarse)

    assert (aPatch.ndim == 1 or aPatch.ndim == 3)

    if aPatch.ndim == 1:
        ALocFine = world.ALocFine
    elif aPatch.ndim == 3:
        ALocFine = world.ALocMatrixFine
    MLocFine = world.MLocFine

    lambdas = np.column_stack(lambdasList)
    numLambdas = len(lambdasList)

    TPrimeCoarsepStartIndices = util.lowerLeftpIndexMap(NPatchCoarse - 1, NPatchCoarse)
    TPrimeCoarsepIndexMap = util.lowerLeftpIndexMap(np.ones_like(NPatchCoarse), NPatchCoarse)

    TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse - 1, NPatchFine - 1, NCoarseElement)
    TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement - 1, NPatchFine - 1)

    TPrimeFinepStartIndices = util.pIndexMap(NPatchCoarse - 1, NPatchFine, NCoarseElement)
    TPrimeFinepIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)

    TInd = util.convertpCoordIndexToLinearIndex(NPatchCoarse - 1, patch.iElementPatchCoarse)

    QPatch = np.column_stack(correctorsList)

    for (TPrimeInd,
         TPrimeCoarsepStartIndex,
         TPrimeFinetStartIndex,
         TPrimeFinepStartIndex) \
            in zip(np.arange(NTPrime),
                   TPrimeCoarsepStartIndices,
                   TPrimeFinetStartIndices,
                   TPrimeFinepStartIndices):
        aTPrime = aPatch[TPrimeFinetStartIndex + TPrimeFinetIndexMap]
        KTPrime = fem.assemblePatchMatrix(NCoarseElement, ALocFine, aTPrime)
        MTPrime = fem.assemblePatchMatrix(NCoarseElement, MLocFine)
        P = lambdas
        Q = QPatch[TPrimeFinepStartIndex + TPrimeFinepIndexMap, :]
        BTPrimeij = np.dot(P.T, KTPrime * Q)
        CTPrimeij = np.dot(Q.T, KTPrime * Q)
        _MTPrimeij = np.dot(P.T, MTPrime * Q)
        TPrimei = TPrimeCoarsepStartIndex + TPrimeCoarsepIndexMap

        accumulate(TPrimeInd, TPrimei, P, Q, KTPrime, MTPrime, _MTPrimeij, BTPrimeij, CTPrimeij)

class CoarseScaleInformation:
    def __init__(self, Kij, Kmsij, muTPrime, Mij, Mmsij):
        self.Kij = Kij
        self.Kmsij = Kmsij
        self.muTPrime = muTPrime
        self.Mij = Mij
        self.Mmsij = Mmsij

#======================================================
#coefficient building
def build_inclusions_defect_2d(x, Nepsilon, bg, val):
    # builds a fine coefficient which is periodic with periodicity length 1/epsilon.
    # On the unit cell, the coefficient takes the value val inside a rectangle

    incl_bl = [0.25, 0.25]
    incl_tr = [0.75, 0.75]

    Nf = 4* Nepsilon

    aBaseSquare = bg*np.ones(Nf)
    for ii in range(Nepsilon[0]):
        for jj in range(Nepsilon[1]):
            startindexcols = int((ii + incl_bl[0])*4)
            stopindexcols = int((ii + incl_tr[0])*4)
            startindexrows = int((jj + incl_bl[1])*4)
            stopindexrows = int((jj + incl_tr[1])*4)
            aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val

    values = aBaseSquare.flatten()

    index = (x * Nf).astype(int)
    d = np.shape(index)[1]
    assert(d==2)
    flatindex = index[:, 1] * Nf[0] + index[:, 0]

    return values[flatindex]

