"""Check the anatomical orientation of foot and lower-leg meshes."""

import numpy as np
import pytest

from ppds import dataset


@pytest.mark.parametrize("subject", ["s001", "s003"])
@pytest.mark.parametrize("side", ["l", "r"])
def test_foot_is_below_the_lower_leg(source, subject, side):
    vertices, _faces = dataset.scan(subject, side)
    x, z = vertices[:, 0], vertices[:, 2]
    bottom, top = np.quantile(z, [0.005, 0.995])
    height = top - bottom
    lower = (z >= bottom) & (z <= bottom + 0.25 * height)
    upper = (z <= top) & (z >= top - 0.25 * height)
    foot_span = np.ptp(np.quantile(x[lower], [0.005, 0.995]))
    leg_span = np.ptp(np.quantile(x[upper], [0.005, 0.995]))
    # The foot projects along native x; its lower envelope is much longer than
    # the shank cross-section. An upside-down scan reverses this relationship.
    assert foot_span > 1.5 * leg_span, (subject, side, foot_span, leg_span)
