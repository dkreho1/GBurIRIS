import pytest
import numpy as np
from GeneralizedBur import *
from utils import twoDofForwKin, distancePointFromLine



@pytest.mark.parametrize("numOfSpines, burOrder, expectedShape", [
    (10, 0, (1, 10, 2)),
    (7, 3, (4, 7, 2)),
    (1, 1, (2, 1, 2)),
    (2, 10, (11, 2, 2))
])



def testLayersShapes(twoDofPlanarArmWithCylinders, numOfSpines, burOrder, expectedShape):
    gBur = GeneralizedBur(
        [0, 0],
        GeneralizedBurConfig(numOfSpines=numOfSpines, burOrder=burOrder),
        twoDofPlanarArmWithCylinders,
        None
        )

    assert gBur.layers.shape == expectedShape



@pytest.mark.parametrize("configuration, expectedMinDistance", [
    ([0, 0], np.linalg.norm([-0.8, -0.8]) - 0.4 - 0.1),
    ([np.pi / 6, 0], distancePointFromLine(np.array([-1.2, 0.6]), twoDofForwKin([np.pi / 6]), twoDofForwKin([np.pi / 6, 0])) - 0.4 - 0.1),
    ([np.pi / 6, 2], distancePointFromLine(np.array([-1.2, 0.6]), twoDofForwKin([np.pi / 6]), twoDofForwKin([np.pi / 6, 2])) - 0.4 - 0.1),
    ([np.pi, 2.22], np.linalg.norm(np.array([1.2, -0.6]) - twoDofForwKin([np.pi, 2.22])) - 0.4 - 0.1),
])



def testGetMinDistanceToCollision(twoDofPlanarArmWithCylinders, configuration, expectedMinDistance):
    gBur = GeneralizedBur(configuration, GeneralizedBurConfig(), twoDofPlanarArmWithCylinders, None)
    assert np.isclose(gBur.getMinDistanceToCollision(), expectedMinDistance)



@pytest.mark.parametrize("collisionConfiguration", [
    ([np.pi, -np.pi / 2]),
    ([np.pi, np.pi / 2]),
    ([0, np.pi / 2]),
    ([2.43, 0])
])



def testExceptionGetMinDistanceToCollision(twoDofPlanarArmWithCylinders, collisionConfiguration):
    with pytest.raises(Exception, match="The robot is in collision with an obstacle!"):
        GeneralizedBur(collisionConfiguration, GeneralizedBurConfig(), twoDofPlanarArmWithCylinders, None).getMinDistanceToCollision()



@pytest.mark.parametrize("configuration, expectedPointsOnPlane", [
    ([0, 0], np.meshgrid(-0.5 * np.ones((15)), np.arange(-1.75, 1.8, 0.25), np.arange(-1.75, 1.8, 0.25)))
])



def testApproximateObstaclesWithPlanes(twoDofPlanarArmWithCylinder, configuration, expectedPointsOnPlane):
    gBur = GeneralizedBur(configuration, GeneralizedBurConfig(), twoDofPlanarArmWithCylinder, None)
    gBur.approximateObstaclesWithPlanes()

    x, y, z = expectedPointsOnPlane
    points = np.hstack((
        x.ravel().reshape((x.size, 1)),
        y.ravel().reshape((y.size, 1)),
        z.ravel().reshape((z.size, 1))
        ))
    
    for point in points:
        assert np.isclose(np.dot(gBur.linkObstaclePlanes[0][0], np.append(point, 1)), 0)
    


@pytest.mark.parametrize("configuration, expectedPointsOnPlane2", [
    ([0, 0], np.meshgrid(-1.4 * np.ones((15)), np.arange(-1.75, 1.8, 0.25), np.arange(-1.75, 1.8, 0.25))),
    ([0, np.pi / 2], np.meshgrid(-1.4 * np.ones((15)), np.arange(-1.75, 1.8, 0.25), np.arange(-1.75, 1.8, 0.25))),
    ([0, 3 * np.pi / 4], np.meshgrid(-1.4 * np.ones((15)), np.arange(-1.75, 1.8, 0.25), np.arange(-1.75, 1.8, 0.25))),
    ([np.pi, 0], np.meshgrid(-1.4 * np.ones((15)), np.arange(-1.75, 1.8, 0.25), np.arange(-1.75, 1.8, 0.25))),
    ([np.pi, -np.pi / 2], np.meshgrid(-1.4 * np.ones((15)), np.arange(-1.75, 1.8, 0.25), np.arange(-1.75, 1.8, 0.25))),
    ([np.pi, -3 * np.pi / 4], np.meshgrid(-1.4 * np.ones((15)), np.arange(-1.75, 1.8, 0.25), np.arange(-1.75, 1.8, 0.25)))
])



def testApproximateObstaclesWithPlanes2(twoDofPlanarArmWithWall, configuration, expectedPointsOnPlane2):
    gBur = GeneralizedBur(configuration, GeneralizedBurConfig(), twoDofPlanarArmWithWall, None)
    gBur.approximateObstaclesWithPlanes()

    x, y, z = expectedPointsOnPlane2
    points = np.hstack((
        x.ravel().reshape((x.size, 1)),
        y.ravel().reshape((y.size, 1)),
        z.ravel().reshape((z.size, 1))
        ))
    
    for point in points:
        assert np.isclose(np.dot(gBur.linkObstaclePlanes[0][0], np.append(point, 1)), 0)



@pytest.mark.parametrize("configuration1, configuration2, expectedDistance", [
    ([0, np.pi], [-np.pi / 4, 3 * np.pi / 4], np.abs(-0.5 - twoDofForwKin([-np.pi / 4, 3 * np.pi / 4])[0]) -0.1),
    ([0, np.pi], [-np.pi / 4, np.pi / 2], np.abs(-0.5 - twoDofForwKin([-np.pi / 4, np.pi / 2])[0]) -0.1),
    ([0, np.pi], [-3 * np.pi / 4, -3 * np.pi / 4], np.abs(-0.5 - twoDofForwKin([-3 * np.pi / 4, -3 * np.pi / 4])[0]) -0.1),
    ([0, np.pi], [-3 * np.pi / 4, -np.pi / 2], np.abs(-0.5 - twoDofForwKin([-3 * np.pi / 4, -np.pi / 2])[0]) -0.1),
    ([0, np.pi], [0, 0], np.abs(-0.5 - twoDofForwKin([0, 0])[0]) -0.1)
])



def testGetMinDistanceUnderestimation(twoDofPlanarArmWithCylinder, configuration1, configuration2, expectedDistance):
    gBur = GeneralizedBur(configuration1, GeneralizedBurConfig(), twoDofPlanarArmWithCylinder, None)

    assert np.isclose(gBur.getMinDistanceUnderestimation(configuration2), expectedDistance)



@pytest.mark.parametrize("burCenter, randomConfigs, burOrder, phiTol, expectedBur", [
    ([0, 0], [
        np.array([10, 0]),
        np.array([-10, 0]),
        np.array([0, 10]),
        np.array([0, -10])
    ], 0, 0.2, [[
        np.array([0.76470588, 0]),
        np.array([-0.76470588, 0]),
        np.array([0, 1.44444444]),
        np.array([0, -1.44444444])  
    ]]),
    ([0, 0], [
        np.array([10, 0]),
        np.array([-10, 0]),
        np.array([0, 10]),
        np.array([0, -10])
    ], 1, 0.2, [[
        np.array([0.76470588, 0]),
        np.array([-0.76470588, 0]),
        np.array([0, 1.44444444]),
        np.array([0, -1.44444444])  
    ], [
        np.array([0.87781251, 0]),
        np.array([-1.52941176, 0]),
        np.array([0, 2.00708603]),
        np.array([0, -2.88888888])  
    ]]),
])



def testCalculateBurBur(twoDofPlanarArmWithWall, burCenter, randomConfigs, burOrder, phiTol, expectedBur):
    gBur = GeneralizedBur(
        burCenter,
        GeneralizedBurConfig(numOfSpines=len(randomConfigs), burOrder=burOrder, phiTol=phiTol),
        twoDofPlanarArmWithWall,
        None
        )

    gBur.randomConfigs = randomConfigs

    burSpines, _ =  gBur.calculateBur()

    for i in range(gBur.generalizedBurConfig.burOrder + 1):
        for j in range(gBur.generalizedBurConfig.numOfSpines):
            assert np.allclose(burSpines[i][j], expectedBur[i][j])
