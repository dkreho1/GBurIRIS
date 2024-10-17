import numpy as np
import numpy.typing as npt
from typing import Callable
from dataclasses import dataclass

from pydrake.all import (
    RobotCollisionType,
    SignedDistancePair
)

from Robot import Robot



@dataclass
class GeneralizedBurConfig:
    numOfSpines: int = 7
    burOrder: int = 3
    minDistanceTol: float = 1e-5
    phiTol: float = 0.1



class GeneralizedBur:
    def __init__(
            self,
            qCenter: list[float] | npt.NDArray[np.float64],
            generalizedBurConfig: GeneralizedBurConfig,
            robot: Robot,
            randomConfigGenerator: Callable[[], npt.NDArray[np.float64]]
            ) -> None:

        self.qCenter = qCenter

        self.generalizedBurConfig = generalizedBurConfig

        self.robot = robot
        self.robot.plant.SetPositions(self.robot.plantContext, self.qCenter)

        self.randomConfigGenerator = randomConfigGenerator
        self.randomConfigs = None

        self.layers = np.zeros((generalizedBurConfig.burOrder + 1, generalizedBurConfig.numOfSpines, len(qCenter)))
        
        self.minDistance = None
        self.linkObstacleDistancePairs = None
        self.linkObstaclePlanes = None



    def getMinDistanceToCollision(
            self
            ) -> float:
        
        if self.minDistance is not None:
            return self.minDistance
        
        if self.linkObstacleDistancePairs is None and self.linkObstaclePlanes is None:
            self.approximateObstaclesWithPlanes()

        self.minDistance = np.inf

        for _, _, _, _, distance in self.linkObstacleDistancePairs:
            if distance < self.minDistance:
                self.minDistance = distance

        # robotClearance = self.robot.checker.CalcRobotClearance(self.qCenter, self.influenceDistance)

        # minDistance = robotClearance.distances()[np.array(robotClearance.collision_types())
        #                                          == RobotCollisionType.kEnvironmentCollision].min()

        if self.minDistance < 0:
            raise Exception("The robot is in collision with an obstacle!")
        
        return self.minDistance



    def approximateObstaclesWithPlanes(
            self
            ) -> None:

        if self.linkObstacleDistancePairs is not None and self.linkObstaclePlanes is not None:
            return
        
        self.linkObstacleDistancePairs = []
        self.linkObstaclePlanes = []

        plant = self.robot.plant
        plantContext = self.robot.plantContext    
    
        plant.SetPositions(plantContext, self.qCenter)

        queryObject = self.robot.checker.model_context().GetQueryObject()
        distancePairs: list[SignedDistancePair] = queryObject.ComputeSignedDistancePairwiseClosestPoints()
        inspector = queryObject.inspector()

 
        for i in range(len(distancePairs)):
            bodyA = plant.GetBodyFromFrameId(inspector.GetFrameId(distancePairs[i].id_A))            
            bodyB = plant.GetBodyFromFrameId(inspector.GetFrameId(distancePairs[i].id_B))

            if self.robot.checker.IsPartOfRobot(bodyA) and self.robot.checker.IsPartOfRobot(bodyB):
                continue

            pointOnA = np.matmul(
                bodyA.EvalPoseInWorld(plantContext).GetAsMatrix4(),
                np.matmul(inspector.GetPoseInFrame(distancePairs[i].id_A).GetAsMatrix4(), np.append(distancePairs[i].p_ACa, 1))
                )[0:3]
            
            pointOnB = np.matmul(
                bodyB.EvalPoseInWorld(plantContext).GetAsMatrix4(),
                np.matmul(inspector.GetPoseInFrame(distancePairs[i].id_B).GetAsMatrix4(), np.append(distancePairs[i].p_BCb, 1))
                )[0:3]


            exists = False

            for j in range(len(self.linkObstacleDistancePairs)):
                if self.linkObstacleDistancePairs[j][0] == bodyA.index() and self.linkObstacleDistancePairs[j][1] == bodyB.index():
                    exists = True
                    
                    if self.linkObstacleDistancePairs[j][4] > distancePairs[i].distance:
                        self.linkObstacleDistancePairs[j] = [bodyA.index(), bodyB.index(), pointOnA, pointOnB, distancePairs[i].distance]

                    break
                
            if not exists:
                self.linkObstacleDistancePairs.append([bodyA.index(), bodyB.index(), pointOnA, pointOnB, distancePairs[i].distance])


        for linkObstacleDistancePair in self.linkObstacleDistancePairs:
            normalVector = linkObstacleDistancePair[3] - linkObstacleDistancePair[2]
            obstaclePlane = np.append(normalVector, - np.dot(normalVector, linkObstacleDistancePair[3]))
            self.linkObstaclePlanes.append((obstaclePlane, normalVector))



    def getMinDistanceUnderestimation(
            self,
            q: list[float] | npt.NDArray[np.float64]
            ) -> float:
        
        if self.linkObstacleDistancePairs is None and self.linkObstaclePlanes is None:
            self.approximateObstaclesWithPlanes()

        minDistance = np.inf
        minBodyIndex = int(self.linkObstacleDistancePairs[0][0])

        linkPositions = self.robot.getLinkPositions(q)

        for linkObstacleDistancePair, linkObstaclePlane in zip(self.linkObstacleDistancePairs, self.linkObstaclePlanes):
            linkNumber = int(linkObstacleDistancePair[0]) - minBodyIndex

            proximalLinkPoint = linkPositions[linkNumber]
            distalLinkPoint = linkPositions[linkNumber + 1]


            if len(proximalLinkPoint) == 2:
                proximalToPlaneHelper = np.dot(linkObstaclePlane[0], np.append(proximalLinkPoint, [0, 1]))
                distalToPlaneHelper = np.dot(linkObstaclePlane[0], np.append(distalLinkPoint, [0, 1]))
            else : 
                proximalToPlaneHelper = np.dot(linkObstaclePlane[0], np.append(proximalLinkPoint, [1]))
                distalToPlaneHelper = np.dot(linkObstaclePlane[0], np.append(distalLinkPoint, [1]))


            proximalToPlaneDistance = self.robot.compensateForLinkGeometry(
                linkNumber,
                np.abs(proximalToPlaneHelper) / np.linalg.norm(linkObstaclePlane[1]),
                False
            )

            distalToPlaneDistance = self.robot.compensateForLinkGeometry(
                linkNumber,
                np.abs(distalToPlaneHelper) / np.linalg.norm(linkObstaclePlane[1]),
                False
            )


            if proximalToPlaneHelper < 0 and proximalToPlaneDistance > 0 and proximalToPlaneDistance < minDistance:
                minDistance = proximalToPlaneDistance

            if distalToPlaneHelper < 0 and distalToPlaneDistance > 0 and distalToPlaneDistance < minDistance:
                minDistance = distalToPlaneDistance



        if np.isinf(minDistance):
            raise Exception("Min distance belaj")
        
        return minDistance



    def evaluatePhiFunction(
            self,
            distance: float,
            t: float,
            qe: list[float] | npt.NDArray[np.float64],
            startingPoint: list[float] | npt.NDArray[np.float64]
            ) -> float:
        
        return distance - self.robot.getMaxDisplacement(startingPoint, startingPoint + t * (qe - startingPoint)) 
    
    

    def calculateBur(
            self
            ) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]]:


        qLowerBounds = self.robot.plant.GetPositionLowerLimits()
        qUpperBounds = self.robot.plant.GetPositionUpperLimits()


        if self.randomConfigs is None:
            maxDistanceConfigSpace = (qUpperBounds - qLowerBounds).max()

            self.randomConfigs = []

            while self.generalizedBurConfig.numOfSpines > len(self.randomConfigs):
                q = self.randomConfigGenerator()

                unitVec = (q - self.qCenter) / np.linalg.norm(q - self.qCenter) 

                self.randomConfigs.append(self.qCenter + unitVec * 2 * maxDistanceConfigSpace)  


        initMinDistance = self.getMinDistanceToCollision()


        for j, qe in enumerate(self.randomConfigs):
            startingPoint = self.qCenter
            minDistance = initMinDistance
            
            for i in range(self.generalizedBurConfig.burOrder + 1):
                tk = 0
                qk = startingPoint

                while minDistance > self.generalizedBurConfig.minDistanceTol and \
                    self.evaluatePhiFunction(minDistance, tk, qe, startingPoint) >= self.generalizedBurConfig.phiTol * minDistance:

                    radii = self.robot.getEnclosingRadii(qk)

                    prevTk = tk
                    tk = tk + (self.evaluatePhiFunction(minDistance, tk, qe, startingPoint) / np.dot(radii, np.abs(qe - qk))) * (1 - tk)
                    qk = startingPoint + tk * (qe - startingPoint)
                    
                    if not (all(qk >= qLowerBounds) and all(qk <= qUpperBounds)):
                        tk = prevTk
                        qk = startingPoint + tk * (qe - startingPoint)
                        break

                self.layers[i, j, :] = qk
                startingPoint = qk
                minDistance = self.getMinDistanceUnderestimation(qk)


        return self.layers, self.randomConfigs
    