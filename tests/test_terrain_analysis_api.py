"""P1.3 Terrain Analysis API tests -- proxy to test_api_contracts.py Section 16.

The checklist gate ``python -m pytest tests/test_terrain_analysis_api.py -v -x``
expects this file to exist.  Canonical tests live in
``tests/test_api_contracts.py::TestTerrainAnalysisApi``.
"""

import pytest

from tests.test_api_contracts import TestTerrainAnalysisApi  # noqa: F401

pytestmark = pytest.mark.apple_metal_proxy
