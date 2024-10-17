import numpy as np
import numpy.typing as npt



def twoDofForwKin(
        config: list[float] | npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
    
    # The used model has the manipulator align with the +y axis
    return np.array([-0.8 * np.sin(np.cumsum(config)).sum(), 0.8 * np.cos(np.cumsum(config)).sum()])



def distancePointFromLine(
        point: npt.NDArray[np.float64],
        linePoint1: npt.NDArray[np.float64],
        linePoint2: npt.NDArray[np.float64]
        ) -> float:
    
    return np.linalg.norm(np.cross(linePoint2 - linePoint1, linePoint1 - point)) / np.linalg.norm(linePoint2 - linePoint1)



def anthrArmForwKin(
        config: list[float] | npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
    
    # The used model has the manipulator align with the +y axis

    if len(config) == 1:
        return np.array([0, 0, 0.2])
    

    if len(config) == 2:
        return np.array([
            0.8 * np.sin(-config[0]) * np.cos(config[1]),
            0.8 * np.cos(-config[0]) * np.cos(config[1]),
            0.2 + 0.8 * np.sin(config[1])
            ])   


    return np.array([
        0.8 * np.sin(-config[0]) * (np.cos(config[1]) + np.cos(config[1] + config[2])),
        0.8 * np.cos(-config[0]) * (np.cos(config[1]) + np.cos(config[1] + config[2])),
        0.2 + 0.8 * (np.sin(config[1]) + np.sin(config[1] + config[2]))
        ])
