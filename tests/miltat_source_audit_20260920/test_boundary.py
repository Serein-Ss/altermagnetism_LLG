import pytest
from tests.miltat_source_audit_20260920.boundary import centered_crop


def test_centered_cells():
    assert centered_crop((200,116),(64,38)) == (slice(68,132),slice(39,77))
    assert centered_crop((200,116),(200,116)) == (slice(0,200),slice(0,116))
    with pytest.raises(ValueError): centered_crop((200,116),(201,116))
    with pytest.raises(ValueError): centered_crop((200,116),(63,38))
