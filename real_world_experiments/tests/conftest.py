import pytest

from native_fixtures import native_fixture


@pytest.fixture
def native_final_cell(tmp_path):
    cfg, key, root, _ = native_fixture(tmp_path)
    return cfg, key, root
