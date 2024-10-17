# Most of the implementation in this file is derived from:
# github.com/wernerpe/manipulation/blob/clique_seeding_manual_distribution/book/trajectories/iris_builder.ipynb

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix
from typing import Callable

from pydrake.all import (
    CollisionChecker,
    HPolyhedron,
    VisibilityGraph,
    MaxCliqueSolverBase,
    Hyperellipsoid,
    AffineBall,
    IrisOptions,
    IrisInConfigurationSpace,
    IrisFromCliqueCoverOptions
)


def SampleVisibilityGraph(
        collisionChecker: CollisionChecker,
        sets: list[HPolyhedron],
        numSamplesVisibilityGraph: int,
        randomConfigGenerator: Callable[[], npt.NDArray[np.float64]]
        ) -> tuple[list[npt.NDArray[np.float64]], csc_matrix]:
    
    numSamplesGenerated = 0
    samples = []

    while numSamplesVisibilityGraph > numSamplesGenerated:
        q = randomConfigGenerator()

        if collisionChecker.CheckConfigCollisionFree(q) and not any([set.PointInSet(q) for set in sets]):
            samples.append(q)
            numSamplesGenerated += 1

    return samples, VisibilityGraph(collisionChecker, np.array(samples).T, parallelize=True)



def TruncatedCliqueCover(
        samples: list[npt.NDArray[np.float64]],
        visibilityGraphAdjacencyMatrix: csc_matrix,
        maxCliqueSolver: MaxCliqueSolverBase,
        minCliqueSize: int
        ) -> list[npt.NDArray[np.float64]]:
    
    cliques = []

    currentAdjacencyMatrix = visibilityGraphAdjacencyMatrix.copy()
    currentIndices = np.arange(currentAdjacencyMatrix.shape[0])
    
    while True:
        # Create clique by solving maximum clique problem
        cliqueSamples = maxCliqueSolver.SolveMaxClique(currentAdjacencyMatrix)

        # Get indices (in current and starting arrays) of clique samples
        localIndicesCliqueSamples = np.where(cliqueSamples)[0]
        globalIndicesCliqueSamples = np.array([currentIndices[i] for i in localIndicesCliqueSamples])

        cliques.append(globalIndicesCliqueSamples.reshape(-1))

        # Remove samples that make up the clique
        currentAdjacencyMatrix = currentAdjacencyMatrix[~cliqueSamples][:, ~cliqueSamples]
        currentIndices = currentIndices[~cliqueSamples]

        if currentAdjacencyMatrix.shape[0] == 0 or len(cliques[-1]) < minCliqueSize:
            break

    return [np.array(samples)[c] for c in cliques]



def MinVolumeEllipsoids(
        collisionChecker: CollisionChecker,
        cliqueCover: list[npt.NDArray[np.float64]]
        ) -> list[Hyperellipsoid]:
    
    ellipsoids = []
    # Depending on the points, the solution can be of lower rank
    for clique in cliqueCover:
        affineBall = AffineBall.MinimumVolumeCircumscribedEllipsoid(clique.T)
        U, S, Vt = np.linalg.svd(affineBall.B())

        # Force the ellipsoid to have volume, i.e., at length to small axes
        for i in range(S.shape[0]):
            if S[i] < 1e-4:
                S[i] = 1e-4
        newB = U @ np.diag(S) @ Vt

        ellipsoidCenter = affineBall.center()

        # Move the ellipsoid center if in collision
        if not collisionChecker.CheckConfigCollisionFree(ellipsoidCenter):
            ellipsoidCenter = clique[np.argmin([np.linalg.norm(ellipsoidCenter - node) for node in clique])]

        ellipsoids.append(Hyperellipsoid(AffineBall(newB, ellipsoidCenter)))

    return ellipsoids



def InflatePolytopes(
        collisionChecker: CollisionChecker,
        ellipsoids: list[Hyperellipsoid],
        numOfIrisIterations: int = 1
        ) -> list[HPolyhedron]:
    

    vccIrisOptions = IrisOptions()
    # VCC only requires 1 iteration
    vccIrisOptions.iteration_limit = numOfIrisIterations

    plant = collisionChecker.plant()
    plantContext = collisionChecker.plant_context()

    vccRegions = []
    for ellipsoid in ellipsoids:
        vccIrisOptions.starting_ellipse = ellipsoid
        plant.SetPositions(plantContext, ellipsoid.center())
        vccRegions.append(
            IrisInConfigurationSpace(plant, plantContext, vccIrisOptions)
        )

    return vccRegions



def CheckCoverage(
        collisionChecker: CollisionChecker,
        sets: list[HPolyhedron],
        numSamplesCoverageCheck: int,
        randomConfigGenerator: Callable[[], npt.NDArray[np.float64]]
        ) -> float:

    numSamplesGenerated = 0
    numCoveredPoints = 0

    while numSamplesCoverageCheck > numSamplesGenerated:
        q = randomConfigGenerator()

        if collisionChecker.CheckConfigCollisionFree(q):
            numSamplesGenerated += 1
            if any([set.PointInSet(q) for set in sets]):
                numCoveredPoints += 1 

    return float(numCoveredPoints) / float(numSamplesCoverageCheck)    



def VisibilityCliqueCover(
        collisionChecker: CollisionChecker,
        maxCliqueSolver: MaxCliqueSolverBase,
        options: IrisFromCliqueCoverOptions,
        randomConfigGenerator: Callable[[], npt.NDArray[np.float64]]
        ) -> tuple[list[HPolyhedron], float]:
    

    vccRegions = []

    for _ in range(options.iteration_limit):
        coverage = CheckCoverage(
            collisionChecker,
            vccRegions,
            options.num_points_per_coverage_check,
            randomConfigGenerator
            )
        
        if not (coverage < options.coverage_termination_threshold):
            break

        samples, visibilityGraphAdjacencyMatrix = SampleVisibilityGraph(
            collisionChecker,
            vccRegions, 
            options.num_points_per_visibility_round,
            randomConfigGenerator
            )
        
        cliqueCover = TruncatedCliqueCover(
            samples,
            visibilityGraphAdjacencyMatrix,
            maxCliqueSolver,
            options.minimum_clique_size
            )
        
        ellipsoids = MinVolumeEllipsoids(collisionChecker, cliqueCover)
        newRegions = InflatePolytopes(collisionChecker, ellipsoids)
        vccRegions.extend(newRegions)

    return vccRegions, coverage
