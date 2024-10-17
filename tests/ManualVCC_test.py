import pytest
from ManualVCC import *
from pydrake.all import HPolyhedron


@pytest.mark.parametrize("freeRandomConfigs, expectedVisibilityGraph", [
    ([
        np.array([0, 0]),
        np.array([-np.pi, 0]),
        np.array([np.pi, 0]),
        np.array([np.pi, np.pi])
    ],
    csc_matrix(([True] * 8, ([0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 1, 0, 3, 2])), shape=(4, 4))),
])



def testSampleVisibilityGraph(twoDofPlanarArmWithCylinder, freeRandomConfigs, expectedVisibilityGraph):
    def randomConfigSampler():
        randomConfigSampler.cnt = (randomConfigSampler.cnt + 1) % len(randomConfigSampler.samples)
        return randomConfigSampler.samples[randomConfigSampler.cnt - 1]
    
    randomConfigSampler.cnt = 0
    randomConfigSampler.samples = freeRandomConfigs

    samples, graph = SampleVisibilityGraph(
        twoDofPlanarArmWithCylinder.checker, [], len(freeRandomConfigs), randomConfigSampler
    )

    assert np.array_equal(freeRandomConfigs, samples)
    assert np.array_equal(graph.toarray(), expectedVisibilityGraph.toarray())



@pytest.mark.parametrize("points, expectedCenter, expectedMatrixAtA", [
    ([[
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1, 0])
    ]],
    [np.array([0, 0])], [np.array([[1, 0], [0, 1]])]
    ),
    ([[
        np.array([1, 1]),
        np.array([1, -1]),
        np.array([2, 0]),
        np.array([0, 0])
    ], [
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1, 0])
    ]],
    [np.array([1, 0]), np.array([0, 0])],
    [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]
    ),
])



def testMinVolumeEllipsoids(twoDofPlanarArmWithoutObstacles, points, expectedCenter, expectedMatrixAtA):
    ellipsoids = MinVolumeEllipsoids(
        twoDofPlanarArmWithoutObstacles.checker, np.array(points)
    )

    for i in range(len(points)):
        assert np.allclose(expectedCenter[i], ellipsoids[i].center())
        assert np.allclose(expectedMatrixAtA[i], np.matmul(ellipsoids[i].A().T, ellipsoids[i].A()), atol=1e-5)



@pytest.mark.parametrize("sets, randomConfigs, expectedCoverage", [
    ([], [
        np.array([0, 0]),
        np.array([-np.pi, 0]),
        np.array([np.pi / 4, 0]),
        np.array([np.pi / 4, np.pi / 4])
    ], 0.0),
    ([
        HPolyhedron(
            np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]),
            np.array([1, 1, 1, 1])
        )
    ], [
        np.array([0, 0]),
        np.array([-np.pi, 0]),
        np.array([np.pi / 4, 0]),
        np.array([np.pi / 4, np.pi / 4])
    ], 0.75),
])


def testCheckCoverage(twoDofPlanarArmWithoutObstacles, sets, randomConfigs, expectedCoverage):
    def randomConfigSampler():
        randomConfigSampler.cnt = (randomConfigSampler.cnt + 1) % len(randomConfigSampler.samples)
        return randomConfigSampler.samples[randomConfigSampler.cnt - 1]
    
    randomConfigSampler.cnt = 0
    randomConfigSampler.samples = randomConfigs

    coverage = CheckCoverage(
        twoDofPlanarArmWithoutObstacles.checker, sets, len(randomConfigs), randomConfigSampler 
    )

    assert np.isclose(coverage, expectedCoverage)
    