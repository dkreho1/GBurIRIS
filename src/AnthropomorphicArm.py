from Robot import Robot
import numpy as np
import numpy.typing as npt



class AnthropomorphicArm(Robot):
    def getLinkPositions(
            self,
            q: list[float] | npt.NDArray[np.float64]
            ) -> npt.NDArray[np.float64]:
        
        previousConfig = self.plant.GetPositions(self.plantContext)
    
        self.plant.SetPositions(self.plantContext, q)
       
        positionLinks = np.array([ self.plant.EvalBodyPoseInWorld(self.plantContext, linkBody).translation()
                                  for linkBody in self.jointChildAndEndEffectorLinks ])
    
        self.plant.SetPositions(self.plantContext, previousConfig)

        return positionLinks



    def compensateForLinkGeometry(
            self,
            linkNumber: int,
            distance: float,
            add: bool = True
            ) -> float | npt.NDArray[np.float64]:
        
        compensation = self.linkGeometryCompensation[linkNumber]

        if not add:
            compensation = -compensation

        return distance + compensation



    def compensateForLinkGeometries(
            self,
            distance: list[float] | npt.NDArray[np.float64],
            add: bool = True
            ) -> float | npt.NDArray[np.float64]:
        
        compensation = self.linkGeometryCompensation

        if not add:
            compensation = -compensation

        return np.array(distance) + compensation



    def getEnclosingRadii(
            self,
            qk: list[float] | npt.NDArray[np.float64]
            ) -> npt.NDArray[np.float64]:
           
        positionLinks = self.getLinkPositions(qk)
    
        maxRadius1 = 0
        for i in range(1, len(positionLinks)):
            currentRadius = np.linalg.norm(positionLinks[i][0:2] - positionLinks[0][0:2])
            if maxRadius1 < currentRadius:
                maxRadius1 = currentRadius

        positionLinks = positionLinks[1:]

        radii = np.append([maxRadius1], np.triu(np.linalg.norm(positionLinks - positionLinks[:, np.newaxis, :], axis=2)).max(axis=1)[:len(qk)-1])

        return self.compensateForLinkGeometries(radii)



    def getMaxDisplacement(
            self,
            q1: list[float] | npt.NDArray[np.float64],
            q2: list[float] | npt.NDArray[np.float64]
            ) -> float:
       
        previousConfig = self.plant.GetPositions(self.plantContext)

        self.plant.SetPositions(self.plantContext, q1)    
        positionLinksConfig1 = np.array([ self.plant.EvalBodyPoseInWorld(self.plantContext, linkBody).translation()
                                         for linkBody in self.jointChildAndEndEffectorLinks[1:] ])
    
        self.plant.SetPositions(self.plantContext, q2)    
        positionLinksConfig2 = np.array([ self.plant.EvalBodyPoseInWorld(self.plantContext, linkBody).translation()
                                         for linkBody in self.jointChildAndEndEffectorLinks[1:] ])
           
        self.plant.SetPositions(self.plantContext, previousConfig)
    
        return np.linalg.norm(np.array(positionLinksConfig1) - np.array(positionLinksConfig2), axis=1).max()