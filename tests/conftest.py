import pytest
import os

from pydrake.all import (
    RobotDiagramBuilder,
    Parser,
    SceneGraphCollisionChecker
)

from PlanarArm import PlanarArm
from AnthropomorphicArm import AnthropomorphicArm



assetsFilePath = os.path.join(os.path.dirname(__file__), 'assets')



def twoDofPlanarArm(
        sceneFilePath: str
        ) -> PlanarArm:
    
    builder = RobotDiagramBuilder()
    plant = builder.plant()

    parser = Parser(plant)
    parser.package_map().Add("testAssets", assetsFilePath)
    parser.AddModels(sceneFilePath)

    plant.Finalize()

    diagram = builder.Build()

    checker = SceneGraphCollisionChecker(
        model=diagram,
        robot_model_instances=[plant.GetModelInstanceByName("2dofPlanarArm")],
        edge_step_size=0.1,
    )

    jointChildAndEndEffectorLinks = [
        plant.GetBodyByName("2dofPlanarLink1", plant.GetModelInstanceByName("2dofPlanarArm")),
        plant.GetBodyByName("2dofPlanarLink2", plant.GetModelInstanceByName("2dofPlanarArm")),
        plant.GetBodyByName("2dofPlanarEndEffector", plant.GetModelInstanceByName("2dofPlanarArm"))
    ]

    return PlanarArm(checker, jointChildAndEndEffectorLinks, [0.1] * 2)



@pytest.fixture
def twoDofPlanarArmWithoutObstacles() -> PlanarArm:
    return twoDofPlanarArm(assetsFilePath + "/2dofWithoutObstaclesTestScene.dmd.yaml")



@pytest.fixture
def twoDofPlanarArmWithCylinders() -> PlanarArm:
    return twoDofPlanarArm(assetsFilePath + "/2dofWithCylindersTestScene.dmd.yaml")



@pytest.fixture
def twoDofPlanarArmWithCylinder() -> PlanarArm:
    return twoDofPlanarArm(assetsFilePath + "/2dofWithCylinderTestScene.dmd.yaml")



@pytest.fixture
def twoDofPlanarArmWithWall() -> PlanarArm:
    return twoDofPlanarArm(assetsFilePath + "/2dofWithWallTestScene.dmd.yaml")



def anthropomorphicArm(
        sceneFilePath: str
        ) -> PlanarArm:
    
    builder = RobotDiagramBuilder()
    plant = builder.plant()

    parser = Parser(plant)
    parser.package_map().Add("testAssets", assetsFilePath)
    parser.AddModels(sceneFilePath)

    plant.Finalize()

    diagram = builder.Build()

    checker = SceneGraphCollisionChecker(
        model=diagram,
        robot_model_instances=[plant.GetModelInstanceByName("AnthropomorphicArm")],
        edge_step_size=0.1,
    )

    bodyIndices = plant.GetBodyIndices(plant.GetModelInstanceByName("AnthropomorphicArm"))
    for i in range(len(bodyIndices)):
        for j in range(i+1, len(bodyIndices)):
            checker.SetCollisionFilteredBetween(bodyIndices[i], bodyIndices[j], True)

    jointChildAndEndEffectorLinks = [
        plant.GetBodyByName("AnthropomorphicArmLink1", plant.GetModelInstanceByName("AnthropomorphicArm")),
        plant.GetBodyByName("AnthropomorphicArmLink2", plant.GetModelInstanceByName("AnthropomorphicArm")),
        plant.GetBodyByName("AnthropomorphicArmLink3", plant.GetModelInstanceByName("AnthropomorphicArm")),
        plant.GetBodyByName("AnthropomorphicArmEndEffector", plant.GetModelInstanceByName("AnthropomorphicArm"))
    ]

    return AnthropomorphicArm(checker, jointChildAndEndEffectorLinks, [0.1] * 3)



@pytest.fixture
def anthropomorphicArmWithoutObstacles() -> PlanarArm:
    return anthropomorphicArm(assetsFilePath + "/3dofWithoutObstaclesTestScene.dmd.yaml")
