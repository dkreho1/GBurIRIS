import mcubes as mc
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot
from matplotlib.colors import ListedColormap
from GeneralizedBur import GeneralizedBur

from pydrake.all import (
    Meshcat,
    RobotDiagramBuilder,
    Parser,
    Rgba,
    RobotCollisionType,
    HPolyhedron,
    CollisionChecker,
    RigidBody,
    MultibodyPlant
)



def RgbaToHex(
        color: Rgba
        ) -> str:
    return '#%02x%02x%02x' % (int(color.r() * 255), int(color.g() * 255), int(color.b() * 255))



def getMaxRobotLinkDisplacement(
        checker: CollisionChecker,
        robotLinkBodies: list[RigidBody],
        q1: list[float] | npt.NDArray[np.float64],
        q2: list[float] | npt.NDArray[np.float64]
        ) -> float:

    plant = checker.plant()
    plantContext = checker.plant_context()

    plant.SetPositions(plantContext, q1)    
    positionLinksConfig1 = np.array([ plant.EvalBodyPoseInWorld(plantContext, linkBody).translation()
                                     for linkBody in robotLinkBodies ])

    plant.SetPositions(plantContext, q2)    
    positionLinksConfig2 = np.array([ plant.EvalBodyPoseInWorld(plantContext, linkBody).translation()
                                     for linkBody in robotLinkBodies ])

    return np.linalg.norm(np.array(positionLinksConfig1) - np.array(positionLinksConfig2), axis=1).max()



def visualize3dConfigSpace(
        meshcat: Meshcat,
        collisionChecker: CollisionChecker,
        numOfSamples: int,
        plotColor: Rgba = Rgba(0.9, 0.5, 0.5, 1)
        ) -> None:

    plant = collisionChecker.plant()

    vertices, triangles = mc.marching_cubes_func(
        tuple(plant.GetPositionLowerLimits()),
        tuple(plant.GetPositionUpperLimits()),
        numOfSamples,
        numOfSamples,
        numOfSamples,
        lambda q0, q1, q2: collisionChecker.CheckConfigCollisionFree([q0, q1, q2]),
        0.5
    )

    meshcat.SetTriangleMesh(
        "collisionConstraint",
        vertices.T,
        triangles.T,
        plotColor
    )



def visualize2dConfigSpace(
        plt: matplotlib.pyplot,
        collisionChecker: CollisionChecker,
        numOfSamples: int,
        plotColor: Rgba = Rgba(0.9, 0.5, 0.5, 1)
        ) -> None:
    
    plant = collisionChecker.plant()

    qLowerBounds = plant.GetPositionLowerLimits()
    qUpperBounds = plant.GetPositionUpperLimits()

    q0s = np.linspace(qLowerBounds[0], qUpperBounds[0], numOfSamples)
    q1s = np.linspace(qLowerBounds[1], qUpperBounds[1], numOfSamples)

    configSpace = [ [ collisionChecker.CheckConfigCollisionFree([q0, q1]) for q0 in q0s ] for q1 in q1s[::-1] ]
            
    cmap = ListedColormap([RgbaToHex(plotColor), 'none'])
    plt.imshow(configSpace, cmap=cmap, alpha=plotColor.a())
    plt.xticks([])
    plt.yticks([])



def visualize2dCompleteBubble(
        plt: matplotlib.pyplot,
        collisionChecker: CollisionChecker,
        linkBodies: list[RigidBody],
        configuration: list[float] | npt.NDArray[np.float64],
        numOfSamples: int,
        plotColor: Rgba = Rgba(0.9, 0.5, 0.5, 1)
        ) -> None:
    
    qLowerBounds = collisionChecker.plant().GetPositionLowerLimits()
    qUpperBounds = collisionChecker.plant().GetPositionUpperLimits()
    
    q0s = np.linspace(qLowerBounds[0], qUpperBounds[0], numOfSamples)
    q1s = np.linspace(qLowerBounds[1], qUpperBounds[1], numOfSamples)

    robotClearance = collisionChecker.CalcRobotClearance(configuration, 1000)
    minDistance = robotClearance.distances()[np.array(robotClearance.collision_types())
                                             == RobotCollisionType.kEnvironmentCollision].min()

    completeBubble = [ [ getMaxRobotLinkDisplacement(
        collisionChecker,
        linkBodies,
        configuration,
        np.array([q0, q1])
        ) < minDistance for q0 in q0s ] for q1 in q1s[::-1] ]
            
    cmap = ListedColormap(['none', RgbaToHex(plotColor)])
    plt.imshow(completeBubble, cmap=cmap, alpha=plotColor.a())
    plt.xticks([])
    plt.yticks([])



def visualize2dIrisPolytope(
        plt: matplotlib.pyplot,
        plant: MultibodyPlant,
        region: HPolyhedron,
        numOfSamples: int,
        plotColor: Rgba = Rgba(0.9, 0.5, 0.5, 0.5)
        ) -> None:
    
    qLowerBounds = plant.GetPositionLowerLimits()
    qUpperBounds = plant.GetPositionUpperLimits()
    
    q0s = np.linspace(qLowerBounds[0], qUpperBounds[0], numOfSamples)
    q1s = np.linspace(qLowerBounds[1], qUpperBounds[1], numOfSamples)
    
    polytope = [ [ region.PointInSet([q0, q1], 0.0) for q0 in q0s] for q1 in q1s[::-1] ]
            
    cmap = ListedColormap(['none', RgbaToHex(plotColor)])
    plt.imshow(polytope, cmap=cmap, alpha=plotColor.a())
    plt.xticks([])
    plt.yticks([])



def visualize3dIrisPolytope(
        meshcat: Meshcat,
        collisionChecker: CollisionChecker,
        region: HPolyhedron,
        numOfSamples: int,
        plotColor: Rgba = Rgba(0, 0, 1, 1)
        ) -> None:

    plant = collisionChecker.plant()

    vertices, triangles = mc.marching_cubes_func(
        tuple(plant.GetPositionLowerLimits()),
        tuple(plant.GetPositionUpperLimits()),
        numOfSamples,
        numOfSamples,
        numOfSamples,
        lambda q0, q1, q2: region.PointInSet([q0, q1, q2]),
        0.5
    )

    meshcat.SetTriangleMesh(
        str(region),
        vertices.T,
        triangles.T,
        plotColor
    )


        
def visualize2dGeneralizedBur(
        plt: matplotlib.pyplot,
        collisionChecker: CollisionChecker,
        bur: GeneralizedBur,
        numOfSamples: int,
        plotColors: list[Rgba]
        ) -> None:
    
    if bur.generalizedBurConfig.burOrder + 1 != len(plotColors):
        raise Exception("Number of bur layers and colors do not match!")

    qLowerBounds = collisionChecker.plant().GetPositionLowerLimits()
    qUpperBounds = collisionChecker.plant().GetPositionUpperLimits()

    prevBurLayerInPixels = None

    for j, layer in enumerate(bur.layers[::-1]):
        burLayerInPixels = np.zeros((bur.generalizedBurConfig.numOfSpines, 2))
        for i in range(layer.shape[0]):
            burLayerInPixels[i, :] = [(layer[i][0] - qLowerBounds[0]) / (qUpperBounds[0] - qLowerBounds[0]) * numOfSamples,
                                      (layer[i][1] - qUpperBounds[1]) / (qLowerBounds[1] - qUpperBounds[1]) * numOfSamples]



        if prevBurLayerInPixels is not None:
            for i in range(bur.generalizedBurConfig.numOfSpines):
                plt.plot(
                    [prevBurLayerInPixels[i][0], burLayerInPixels[i][0]],
                    [prevBurLayerInPixels[i][1], burLayerInPixels[i][1]],
                    color=(plotColors[::-1][j - 1].r(), plotColors[::-1][j - 1].g(), plotColors[::-1][j - 1].b()),
                    alpha=plotColors[::-1][j - 1].a()
                    )
        
        plt.plot(burLayerInPixels[:, 0], burLayerInPixels[:, 1],
                '.',
                color=(plotColors[::-1][j].r(), plotColors[::-1][j].g(), plotColors[::-1][j].b()),
                alpha=plotColors[::-1][j].a()
                )
        
        prevBurLayerInPixels = burLayerInPixels

    burCenterInPixels = [(bur.qCenter[0] - qLowerBounds[0]) / (qUpperBounds[0] - qLowerBounds[0]) * numOfSamples,
                         (bur.qCenter[1] - qUpperBounds[1]) / (qLowerBounds[1] - qUpperBounds[1]) * numOfSamples]
    plt.plot(burCenterInPixels[0], burCenterInPixels[1],
            '.',
            color=(plotColors[0].r(), plotColors[0].g(), plotColors[0].b()),
            alpha=plotColors[0].a()
            )

    for i in range(bur.generalizedBurConfig.numOfSpines):
        plt.plot(
            [burCenterInPixels[0], prevBurLayerInPixels[i][0]],
            [burCenterInPixels[1], prevBurLayerInPixels[i][1]],
            color=(plotColors[0].r(), plotColors[0].g(), plotColors[0].b()),
            alpha=plotColors[0].a()
            )
        