import pytest
from unittest.mock import patch
from passos_magicos.data.setup import main

# ==============================================================================
# UNIT TESTS
# ==============================================================================


def test_setup_handles_exceptions(caplog):
    """Tests if the script gracefully catches and logs OS errors (like permission denied)."""

    # We force os.makedirs to simulate a crash (e.g., lack of admin privileges)
    with patch("os.makedirs", side_effect=PermissionError("Access Denied")):
        main()

        # The script should catch the error and log it, rather than blowing up the app
        assert "Failed to create directories" in caplog.text
        assert "Access Denied" in caplog.text
