import numpy.typing as npt
from typing import Callable
from dataclasses import dataclass

from Robot import Robot

from GeneralizedBur import *

from ManualVCC import (
    CheckCoverage,
    MinVolumeEllipsoids,
    InflatePolytopes
)

from pydrake.all import (
    HPolyhedron,
    CollisionChecker
)



irisCenterMarginErrorStr = "The current center of the IRIS region is within options.configuration_space_margin of being infeasible." \
    + "  Check your sample point and/or any additional constraints you've passed in via the options." \
    + " The configuration space surrounding the sample point must have an interior."



@dataclass
class GBurIRISConfig:
    numOfSpines: int = 7
    burOrder: int = 4
    minDistanceTol: float = 1e-5
    phiTol: float = 0.1
    numPointsCoverageCheck: int = 5000
    coverage: float = 0.7
    numOfIter: int = 100
    numOfIterIRIS: int = 1
    ignoreDeltaExceptionFromIRISNP: bool = True



def GBurIRIS(
        checker: CollisionChecker,
        robot: Robot,
        configs: GBurIRISConfig,
        randomGenerator: Callable[[], npt.NDArray[np.float64]],
        ) -> tuple[list[HPolyhedron], float, list[npt.NDArray[np.float64]]]:
    
    regions = []
    burs = []

    for _ in range(configs.numOfIter):

        coverage = CheckCoverage(
            checker,
            regions,
            configs.numPointsCoverageCheck,
            randomGenerator
            )

        if not (coverage < configs.coverage):
            break

        while True:
            burCenter = randomGenerator()

            if checker.CheckConfigCollisionFree(burCenter) and not any([set.PointInSet(burCenter) for set in regions]):
                break


        bur = GeneralizedBur(
            burCenter,
            GeneralizedBurConfig(
                configs.numOfSpines,
                configs.burOrder,
                configs.minDistanceTol,
                configs.phiTol
                ),
            robot,
            randomGenerator
            )
        

        if bur.getMinDistanceToCollision() < configs.minDistanceTol:
            continue

        bur.calculateBur()

        burs.append(bur)

        ellipsoids = MinVolumeEllipsoids(
            checker,
            [bur.layers[-1].reshape((bur.generalizedBurConfig.numOfSpines, len(bur.qCenter)))]
            )

        try:
            regions.extend(InflatePolytopes(checker, ellipsoids, configs.numOfIterIRIS))
        except RuntimeError as err:
            if not configs.ignoreDeltaExceptionFromIRISNP:
                raise err

            if str(err) == irisCenterMarginErrorStr:
                del burs[-1]
                continue
            else:
                raise err

    return regions, coverage, burs