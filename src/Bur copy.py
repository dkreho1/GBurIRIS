import numpy as np
import numpy.typing as npt

from pydrake.all import (
    CollisionChecker,
    RigidBody,
    RobotCollisionType
)



class Bur:
    def __init__(
            self,
            qCenter: list[float] | npt.NDArray[np.float64],
            numOfSpines: int,
            checker: CollisionChecker,
            robotBaseLink: RigidBody,
            robotLinkBodies: list[RigidBody],
            influenceDistance: float = 1000
            ) -> None:
        
        self.qCenter = qCenter

        self.checker = checker
        self.plant = self.checker.plant()
        self.plantContext = self.checker.plant_context()

        self.influenceDistance = influenceDistance
        self.numOfSpines = numOfSpines
        self.robotBaseLink = robotBaseLink
        self.robotLinkBodies = robotLinkBodies

        self.plant.SetPositions(self.plantContext, self.qCenter)

        self.minDistance = self.getMinDistanceToCollision()

        self.spines = []



    def getMinDistanceToCollision(
            self
            ) -> None:
        
        robotClearance = self.checker.CalcRobotClearance(self.qCenter, self.influenceDistance)

        minDistance = robotClearance.distances()[np.array(robotClearance.collision_types())
                                                 == RobotCollisionType.kEnvironmentCollision].min()

        if minDistance < 0:
            raise Exception("Collision!")
        
        return minDistance



    def getMaxDisplacement(
            self,
            q1: list[float] | npt.NDArray[np.float64],
            q2: list[float] | npt.NDArray[np.float64]
            ) -> float:
        
        self.plant.SetPositions(self.plantContext, q1)    
        positionLinksConfig1 = np.array([ self.plant.EvalBodyPoseInWorld(self.plantContext, linkBody).translation()
                                         for linkBody in self.robotLinkBodies ])

        self.plant.SetPositions(self.plantContext, q2)    
        positionLinksConfig2 = np.array([ self.plant.EvalBodyPoseInWorld(self.plantContext, linkBody).translation()
                                         for linkBody in self.robotLinkBodies ])
        
        self.plant.SetPositions(self.plantContext, self.qCenter)

        return np.linalg.norm(np.array(positionLinksConfig1) - np.array(positionLinksConfig2), axis=1).max()



    def getEnclosingRadii(
            self,
            qk: list[float] | npt.NDArray[np.float64]
            ) -> list[float]:
        
        self.plant.SetPositions(self.plantContext, qk)

        radii = []
        
        positionLinks = [ self.plant.EvalBodyPoseInWorld(self.plantContext, self.robotBaseLink).translation() ]
        positionLinks.extend([ self.plant.EvalBodyPoseInWorld(self.plantContext, linkBody).translation()
                              for linkBody in self.robotLinkBodies ])

        positionLinks = np.array(positionLinks)

        radii = np.triu(np.linalg.norm(positionLinks - positionLinks[:, np.newaxis, :], axis=2)).max(axis=1)[:positionLinks.shape[0]-1]

        print(radii)
        radii = []

        numOfDof = len(qk)
        for i in range(numOfDof):
            maxDistance = 0
            for j in range(i + 1, numOfDof + 1):
                distance = np.linalg.norm(positionLinks[i] - positionLinks[j])
                if distance > maxDistance:
                    maxDistance = distance

            radii.append(maxDistance)

        self.plant.SetPositions(self.plantContext, self.qCenter)

        print(radii)
        return radii
    


    def evaluatePhiFunction(
            self,
            t: float,
            qe: list[float] | npt.NDArray[np.float64]
            ) -> float:
        
        return self.minDistance - self.getMaxDisplacement(self.qCenter, self.qCenter + t * (qe - self.qCenter)) 
    
    

    def calculateBur(
            self,
            phiTolerance: float = 0.1,
            seed: int = None
            ) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]]:
        
        if seed is not None:
            np.random.seed(seed)

        plant = self.checker.plant()

        qLowerBounds = plant.GetPositionLowerLimits()
        qUpperBounds = plant.GetPositionUpperLimits()

        numSamplesGenerated = 0
        randomConfigs = []

        while self.numOfSpines > numSamplesGenerated:
            q = np.random.uniform(qLowerBounds, qUpperBounds)

            if self.getMaxDisplacement(self.qCenter, q) > self.minDistance:
                randomConfigs.append(q)
                numSamplesGenerated += 1

        for qe in randomConfigs:
            tk = 0
            while self.evaluatePhiFunction(tk, qe) >= phiTolerance * self.minDistance:
                qk = self.qCenter + tk * (qe - self.qCenter)
                radii = self.getEnclosingRadii(qk)

                tk = tk + (self.evaluatePhiFunction(tk, qe) / np.dot(radii, np.abs(qe - qk))) * (1 - tk)
                
            self.spines.append(self.qCenter + tk * (qe - self.qCenter))

        return self.spines, randomConfigs





from pydrake.all import (
    RobotDiagramBuilder,
    Parser,
    SceneGraphCollisionChecker,
)



builder = RobotDiagramBuilder()
plant = builder.plant()
Parser(plant).AddModels("/home/dzenan/Desktop/MasterThesis/2dofWithCylinders.dmd.yaml")

diagram = builder.Build()

diagramContext = diagram.CreateDefaultContext()
plantContext = plant.GetMyMutableContextFromRoot(diagramContext)


collisionChecker = SceneGraphCollisionChecker(
    model=diagram,
    robot_model_instances=[plant.GetModelInstanceByName("2dofIIWA")],
    # edge_step_size sets the discretization for the visibility graph
    edge_step_size=0.1,
)

bur = Bur([0, 0], 20, collisionChecker,
          plant.GetBodyByName("iiwa_twoDOF_link_2", plant.GetModelInstanceByName("2dofIIWA")),
          [ plant.GetBodyByName("iiwa_twoDOF_link_4", plant.GetModelInstanceByName("2dofIIWA")),
            plant.GetBodyByName("iiwa_twoDOF_link_7", plant.GetModelInstanceByName("2dofIIWA"))
           ])

print(bur.getMaxDisplacement([0, 0], [1.57, 0]))
bur.getEnclosingRadii([1.57, 1.57])

# A = np.array([np.array([1, 2, 5]), np.array([4, 5, 6]), np.array([7, 8, 9]), np.array([10, 11, 12])])
# print()
# print(A)
# print(A - A[:, np.newaxis, :])
# print(np.linalg.norm(A - A[:, np.newaxis, :], axis=2))
# print(np.linalg.norm(A - A[:, np.newaxis, :], axis=2)[np.triu_indices(A.shape[0], 1)])

A = np.random.uniform(size=(5, 3))
import time

t0 = time.time()
np.triu(np.linalg.norm(A - A[:, np.newaxis, :], axis=2)).max(axis=1)[:A.shape[0]-1]
t1 = time.time()
print(t1-t0)

t0 = time.time()
radii = []
numOfDof = A.shape[0]-1
for i in range(numOfDof):
    maxDistance = 0
    for j in range(i + 1, numOfDof + 1):
        distance = np.linalg.norm(A[i] - A[j])
        if distance > maxDistance:
            maxDistance = distance

    radii.append(maxDistance)
t1 = time.time()
print(t1-t0)
