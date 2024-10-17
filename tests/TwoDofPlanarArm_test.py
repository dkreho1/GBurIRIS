import pytest
import numpy as np
from utils import twoDofForwKin



@pytest.mark.parametrize("configuration, expectedLinkPositions", [
    ([0, 0], np.array([np.array([0.0, 0.0]), twoDofForwKin([0]), twoDofForwKin([0, 0])])),
    ([0, np.pi / 2], np.array([np.array([0.0, 0.0]), twoDofForwKin([0]), twoDofForwKin([0, np.pi / 2])])),
    ([0, -np.pi / 2], np.array([np.array([0.0, 0.0]), twoDofForwKin([0]), twoDofForwKin([0, -np.pi / 2])])),
    ([-np.pi / 2, 0], np.array([np.array([0.0, 0.0]), twoDofForwKin([-np.pi / 2]), twoDofForwKin([-np.pi / 2, 0])])),
    ([0, -np.pi / 4], np.array([np.array([0.0, 0.0]), twoDofForwKin([0]), twoDofForwKin([0, -np.pi / 4])])),
    ([0, np.pi / 4], np.array([np.array([0.0, 0.0]), twoDofForwKin([0]), twoDofForwKin([0, np.pi / 4])])),
    ([-np.pi / 4, -np.pi / 4], np.array([np.array([0.0, 0.0]), twoDofForwKin([-np.pi / 4]), twoDofForwKin([-np.pi / 4, -np.pi / 4])])),
    ([np.pi / 3, -np.pi / 6], np.array([np.array([0.0, 0.0]),twoDofForwKin([np.pi / 3]), twoDofForwKin([np.pi / 3, -np.pi / 6])]))
])



def testGetLinkPositions(twoDofPlanarArmWithoutObstacles, configuration, expectedLinkPositions):
    linkPositions = twoDofPlanarArmWithoutObstacles.getLinkPositions(configuration)
    assert linkPositions.shape == expectedLinkPositions.shape
    assert np.allclose(linkPositions, expectedLinkPositions)



@pytest.mark.parametrize("configuration1, configuration2, expectedMaxDisplacement", [
    ([0, 0], [0, 0], 0.0),
    ([0, 0], [0, np.pi / 2], np.linalg.norm(twoDofForwKin([0, 0]) - twoDofForwKin([0, np.pi / 2]))),
    ([0, 0], [0, -np.pi / 2], np.linalg.norm(twoDofForwKin([0, 0]) - twoDofForwKin([0, -np.pi / 2]))),
    ([-np.pi / 4, -np.pi / 4], [0, -np.pi / 2], np.linalg.norm(twoDofForwKin([-np.pi / 4, -np.pi / 4]) - twoDofForwKin([0, -np.pi / 2]))),
    ([-np.pi / 2, 3 * np.pi / 4], [np.pi / 2, -3 * np.pi / 4], np.linalg.norm(twoDofForwKin([-np.pi / 2]) - twoDofForwKin([np.pi / 2]))),
    ([0, -np.pi], [np.pi, -np.pi], np.linalg.norm(twoDofForwKin([0]) - twoDofForwKin([np.pi])))
])



def testGetMaxDisplacement(twoDofPlanarArmWithoutObstacles, configuration1, configuration2, expectedMaxDisplacement):
    maxDisplacement = twoDofPlanarArmWithoutObstacles.getMaxDisplacement(configuration1, configuration2)
    assert np.isclose(maxDisplacement, expectedMaxDisplacement)



@pytest.mark.parametrize("linkNumber, distance, add, expectedDistance", [
    (0, 1, True, 1.1),
    (1, 1, True, 1.1),
    (0, 0, True, 0.1),
    (0, 1, False, 0.9),
    (1, 1, False, 0.9),
    (0, 0, False, -0.1)
])



def testCompensateForLinkGeometry(twoDofPlanarArmWithoutObstacles, linkNumber, distance, add, expectedDistance):
    assert np.isclose(twoDofPlanarArmWithoutObstacles.compensateForLinkGeometry(linkNumber, distance, add), expectedDistance)



@pytest.mark.parametrize("distances, add, expectedDistance", [
    ([0, 0], True, [0.1, 0.1]),
    ([1, 4], True, [1.1, 4.1]),
    ([0, 0], False, [-0.1, -0.1]),
    ([1, 4], False, [0.9, 3.9])
])



def testCompensateForLinkGeometries(twoDofPlanarArmWithoutObstacles, distances, add, expectedDistance):
    assert np.allclose(twoDofPlanarArmWithoutObstacles.compensateForLinkGeometries(distances, add), expectedDistance)



@pytest.mark.parametrize("configuration, expectedRadii", [
    ([0, 0], [np.linalg.norm(twoDofForwKin([0, 0])) + 0.1, 0.9]),
    ([0, np.pi / 2], [np.linalg.norm(twoDofForwKin([0, np.pi / 2])) + 0.1, 0.9]),
    ([np.pi / 4, -np.pi / 2], [np.linalg.norm(twoDofForwKin([np.pi / 4, -np.pi / 2])) + 0.1, 0.9]),
    ([np.pi / 4, -3 * np.pi / 4], [0.9, 0.9]),
    ([0, np.pi], [0.9, 0.9]),
    ([np.pi / 6, np.pi / 3], [np.linalg.norm(twoDofForwKin([np.pi / 6, np.pi / 3])) + 0.1, 0.9]),
])



def testGetEnclosingRadii(twoDofPlanarArmWithoutObstacles, configuration, expectedRadii):
    assert np.allclose(twoDofPlanarArmWithoutObstacles.getEnclosingRadii(configuration), expectedRadii)
