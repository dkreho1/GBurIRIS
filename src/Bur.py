import numpy as np
import numpy.typing as npt
from typing import Callable

from pydrake.all import (
    RobotCollisionType
)

from Robot import Robot



class Bur:
    def __init__(
            self,
            qCenter: list[float] | npt.NDArray[np.float64],
            numOfSpines: int,
            robot: Robot,
            randomConfigGenerator: Callable[[], npt.NDArray[np.float64]],
            influenceDistance: float = 1000
            ) -> None:
        
        self.qCenter = qCenter
        self.influenceDistance = influenceDistance
        self.numOfSpines = numOfSpines

        self.robot = robot
        self.robot.plant.SetPositions(self.robot.plantContext, self.qCenter)

        self.randomConfigGenerator = randomConfigGenerator

        self.minDistance = self.getMinDistanceToCollision()

        self.spines = []
        self.randomConfigs = []



    def getMinDistanceToCollision(
            self
            ) -> float:
        
        robotClearance = self.robot.checker.CalcRobotClearance(self.qCenter, self.influenceDistance)

        minDistance = robotClearance.distances()[np.array(robotClearance.collision_types())
                                                 == RobotCollisionType.kEnvironmentCollision].min()

        if minDistance < 0:
            raise Exception("Collision!")
        
        return minDistance



    def evaluatePhiFunction(
            self,
            t: float,
            qe: list[float] | npt.NDArray[np.float64]
            ) -> float:
        
        return self.minDistance - self.robot.getMaxDisplacement(self.qCenter, self.qCenter + t * (qe - self.qCenter)) 
    
    

    def calculateBur(
            self,
            phiTolerance: float = 0.1
            ) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]]:
        

        qLowerBounds = self.robot.plant.GetPositionLowerLimits()
        qUpperBounds = self.robot.plant.GetPositionUpperLimits()

        numSamplesGenerated = 0
        
        while self.numOfSpines > numSamplesGenerated:
            q = self.randomConfigGenerator()

            if not (all(q >= qLowerBounds) and all(q <= qUpperBounds)):
                self.randomConfigs.append(q)
                numSamplesGenerated += 1
        # while self.numOfSpines > numSamplesGenerated:
        #     q = np.random.uniform(qLowerBounds, qUpperBounds)

        #     if self.robot.getMaxDisplacement(self.qCenter, q) > self.minDistance:
        #         self.randomConfigs.append(q)
        #         numSamplesGenerated += 1

        # for qe in self.randomConfigs:
        #     tk = 0
        #     while self.evaluatePhiFunction(tk, qe) >= phiTolerance * self.minDistance:
        #         qk = self.qCenter + tk * (qe - self.qCenter)
        #         radii = self.robot.getEnclosingRadii(qk)

        #         tk = tk + (self.evaluatePhiFunction(tk, qe) / np.dot(radii, np.abs(qe - qk))) * (1 - tk)
                
        #     self.spines.append(self.qCenter + tk * (qe - self.qCenter))


        for qe in self.randomConfigs:      
            tk = 0
            qk = self.qCenter
            
            while self.evaluatePhiFunction(tk, qe) >= phiTolerance * self.minDistance:
                radii = self.robot.getEnclosingRadii(qk)

                prevTk = tk
                tk = tk + (self.evaluatePhiFunction(tk, qe) / np.dot(radii, np.abs(qe - qk))) * (1 - tk)
                qk = self.qCenter + tk * (qe - self.qCenter)
                
                if not (all(qk >= qLowerBounds) and all(qk <= qUpperBounds)):
                    tk = prevTk
                    qk = self.qCenter + tk * (qe - self.qCenter)
                    break

            self.spines.append(qk)

        return self.spines, self.randomConfigs
