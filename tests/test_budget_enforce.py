"""CENSOR: enforce-by-default host-visible memory budget.

A single 600 MiB host-visible allocation exceeds the 512 MiB budget even with an
otherwise-empty ledger, so it must raise a named ``MemoryBudgetExceeded`` and the
process must survive to keep using the API.
"""

import pytest

import forge3d as f3d


@pytest.mark.apple_metal_physical
def test_600mib_host_visible_raises_named_budget_error():
    if not f3d.has_gpu():
        pytest.skip("no GPU adapter")

    from forge3d._forge3d import request_host_visible_allocation_for_test

    # Ensure the enforce policy is active (it is the default, but be explicit so
    # the test is order-independent within the session).
    f3d.mem.set_budget_policy("enforce")

    with pytest.raises(f3d.MemoryBudgetExceeded) as ei:
        request_host_visible_allocation_for_test(600 * 1024 * 1024, "budget-test-blob")

    msg = str(ei.value)
    assert "budget-test-blob" in msg
    assert "top consumers" in msg

    # Process did not abort: we are still here and can keep using the API.
    assert f3d.mem.memory_metrics()["budget_policy"] == "enforce"
