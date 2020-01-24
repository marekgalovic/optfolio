import numpy as np

from optfolio.nsga2 import non_dominated_fronts


def test_non_dominated_fronts():
    points = np.asarray([
        [r * np.cos(phi), r * np.sin(phi)]
        for r in (np.arange(5) + 1)
        for phi in np.linspace(np.pi/2, np.pi, 10)
    ])

    fronts, _ = non_dominated_fronts(points[:,1], points[:,0], np.zeros(len(points), dtype=np.float32)) 

    for front_id, i in enumerate(reversed(range(5))):
        expected_ids = np.arange(i * 10, (i+1) * 10)
        front_ids = np.argwhere(fronts == front_id).reshape((-1,))

        assert np.all(front_ids == expected_ids)
