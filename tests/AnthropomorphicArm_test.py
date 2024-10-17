import pytest
import numpy as np
from utils import anthrArmForwKin



@pytest.mark.parametrize("configuration, expectedLinkPositions", [
    ([0, 0, 0], np.array([
        np.array([0.0, 0.0, 0.15]),
        anthrArmForwKin([0]),
        anthrArmForwKin([0, 0]),
        anthrArmForwKin([0, 0, 0])
        ])),
    ([np.pi/2, 0, 0], np.array([
        np.array([0.0, 0.0, 0.15]),
        anthrArmForwKin([np.pi/2]),
        anthrArmForwKin([np.pi/2, 0]),
        anthrArmForwKin([np.pi/2, 0, 0])
        ])),
    ([np.pi/2, np.pi/4, np.pi/4], np.array([
        np.array([0.0, 0.0, 0.15]),
        anthrArmForwKin([np.pi/2]),
        anthrArmForwKin([np.pi/2, np.pi/4]),
        anthrArmForwKin([np.pi/2, np.pi/4, np.pi/4])
        ])),
    ([np.pi/2, np.pi/4, -np.pi/4], np.array([
        np.array([0.0, 0.0, 0.15]),
        anthrArmForwKin([np.pi/2]),
        anthrArmForwKin([np.pi/2, np.pi/4]),
        anthrArmForwKin([np.pi/2, np.pi/4, -np.pi/4])
        ]))
])



def testGetLinkPositions(anthropomorphicArmWithoutObstacles, configuration, expectedLinkPositions):
    linkPositions = anthropomorphicArmWithoutObstacles.getLinkPositions(configuration)
    assert linkPositions.shape == expectedLinkPositions.shape
    assert np.allclose(linkPositions, expectedLinkPositions)



@pytest.mark.parametrize("configuration1, configuration2, expectedMaxDisplacement", [
    ([0, 0, 0], [0, 0, 0], 0.0),
    ([0, 0, 0], [np.pi/2, 0, 0], np.linalg.norm(anthrArmForwKin([0, 0, 0]) - anthrArmForwKin([np.pi/2, 0, 0]))),
    ([0, 0, 0], [np.pi/2, np.pi/4, np.pi/4], np.linalg.norm(anthrArmForwKin([0, 0, 0]) - anthrArmForwKin([np.pi/2, np.pi/4, np.pi/4]))),
    ([0, 0, 0], [np.pi/2, np.pi/4, -np.pi/4], np.linalg.norm(anthrArmForwKin([0, 0, 0]) - anthrArmForwKin([np.pi/2, np.pi/4, -np.pi/4]))),
    ([0, 0, np.pi/2], [0, np.pi/2, -np.pi/2], np.linalg.norm(anthrArmForwKin([0, 0]) - anthrArmForwKin([0, np.pi/2])))
])



def testGetMaxDisplacement(anthropomorphicArmWithoutObstacles, configuration1, configuration2, expectedMaxDisplacement):
    maxDisplacement = anthropomorphicArmWithoutObstacles.getMaxDisplacement(configuration1, configuration2)
    assert np.isclose(maxDisplacement, expectedMaxDisplacement)



@pytest.mark.parametrize("linkNumber, distance, add, expectedDistance", [
    (0, 1, True, 1.1),
    (1, 1, True, 1.1),
    (2, 1, True, 1.1),
    (0, 0, True, 0.1),
    (0, 1, False, 0.9),
    (1, 1, False, 0.9),
    (2, 1, False, 0.9),
    (0, 0, False, -0.1)
])



def testCompensateForLinkGeometry(anthropomorphicArmWithoutObstacles, linkNumber, distance, add, expectedDistance):
    assert np.isclose(anthropomorphicArmWithoutObstacles.compensateForLinkGeometry(linkNumber, distance, add), expectedDistance)



@pytest.mark.parametrize("distances, add, expectedDistance", [
    ([0, 0, 0], True, [0.1, 0.1, 0.1]),
    ([1, 4, 7], True, [1.1, 4.1, 7.1]),
    ([0, 0, 0], False, [-0.1, -0.1, -0.1]),
    ([1, 4, 7], False, [0.9, 3.9, 6.9])
])



def testCompensateForLinkGeometries(anthropomorphicArmWithoutObstacles, distances, add, expectedDistance):
    assert np.allclose(anthropomorphicArmWithoutObstacles.compensateForLinkGeometries(distances, add), expectedDistance)



@pytest.mark.parametrize("configuration, expectedRadii", [
    ([0, 0, 0], np.array([1.7, 1.7, 0.9])),
    ([0, np.pi/2, 0], np.array([0.1, 1.7, 0.9])),
    ([np.pi/2, np.pi/4, np.pi/4], np.array([
        np.linalg.norm(anthrArmForwKin([np.pi/2, np.pi/4, np.pi/4])[0:2]) + 0.1,
        np.linalg.norm(anthrArmForwKin([np.pi/2]) - anthrArmForwKin([np.pi/2, np.pi/4, np.pi/4])) + 0.1,
        0.9
    ])),
    ([np.pi/2, np.pi/4, -3*np.pi/4], np.array([
        np.linalg.norm(anthrArmForwKin([np.pi/2, np.pi/4, -3*np.pi/4])[0:2]) + 0.1,
        0.9,
        0.9
    ]))
])



def testGetEnclosingRadii(anthropomorphicArmWithoutObstacles, configuration, expectedRadii):
    assert np.allclose(anthropomorphicArmWithoutObstacles.getEnclosingRadii(configuration), expectedRadii)
