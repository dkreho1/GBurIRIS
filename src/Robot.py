from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import overload

from pydrake.all import (
    CollisionChecker,
    RigidBody
)



class Robot(ABC):
    def __init__(
            self,
            checker: CollisionChecker,
            jointChildAndEndEffectorLinks: list[RigidBody],
            linkGeometryCompensation: list[float] | npt.NDArray[np.float64]
            ) -> None:
        
        self.checker = checker
        self.plant = self.checker.plant()
        self.plantContext = self.checker.plant_context()

        if len(jointChildAndEndEffectorLinks) != len(linkGeometryCompensation) + 1:
            raise Exception("Not matching")

        self.jointChildAndEndEffectorLinks = jointChildAndEndEffectorLinks
        self.linkGeometryCompensation = np.array(linkGeometryCompensation)



    @abstractmethod
    def getEnclosingRadii(
            self,
            qk: list[float] | npt.NDArray[np.float64]
            ) -> npt.NDArray[np.float64]:
        pass



    @abstractmethod
    def getMaxDisplacement(
            self,
            q1: list[float] | npt.NDArray[np.float64],
            q2: list[float] | npt.NDArray[np.float64]
            ) -> float:
        pass



    @abstractmethod
    def getLinkPositions(
            self,
            q: list[float] | npt.NDArray[np.float64]
            ) -> npt.NDArray[np.float64]:
        pass



    @abstractmethod
    def compensateForLinkGeometry(
            self,
            linkNumber: int,
            distance: float,
            add: bool = True
            ) -> float | npt.NDArray[np.float64]:
        pass



    @abstractmethod
    def compensateForLinkGeometries(
            self,
            distance: list[float] | npt.NDArray[np.float64],
            add: bool = True
            ) -> float | npt.NDArray[np.float64]:
        pass