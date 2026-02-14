import pandas as pd
import numpy as np
import pytest

from passos_magicos.data.constants import FeatureNames as FN
from passos_magicos.data.make_silver import _transform_2022


@pytest.fixture
def mock_df_2022():
    """Provides a mock dataframe representing raw 2022 data after renaming."""
    return pd.DataFrame(
        {
            FN.RA: ["RA-1", "RA-2"],
            FN.INDE: [7.5, 8.0],
            FN.IAN: [5.0, 7.0],
            FN.IDA: [6.0, 8.0],
            FN.IEG: [8.0, 9.0],
            FN.IAA: [7.0, 8.0],
            FN.IPS: [6.0, 7.0],
            FN.IPV: [7.0, 8.0],
            # IPP is missing to reflect 2022 missing data.
        }
    )


def test_transform_2022_ipp_calculation(mock_df_2022):
    """Tests if the IPP mathematical inference for 2022 is correct."""

    # Run the transformation
    df_transformed = _transform_2022(mock_df_2022)

    # Check if the IPP column was created
    assert FN.IPP in df_transformed.columns

    # Manual calculation for the first row (RA-1):
    # INDE = 7.5
    # deductions = (5.0*0.1) + (6.0*0.2) + (8.0*0.2) + (7.0*0.1) + (6.0*0.1) + (7.0*0.2)
    # deductions = 0.5 + 1.2 + 1.6 + 0.7 + 0.6 + 1.4 = 6.0
    # IPP = (7.5 - 6.0) / 0.1 = 15.0

    expected_ipp_ra1 = 15.0
    actual_ipp_ra1 = df_transformed.loc[df_transformed[FN.RA] == "RA-1", FN.IPP].values[
        0
    ]

    assert np.isclose(
        actual_ipp_ra1, expected_ipp_ra1
    ), f"Expected {expected_ipp_ra1}, got {actual_ipp_ra1}"


def test_transform_2023_pass_through():
    """Tests if 2023 data passes through without unintended modifications."""
    from passos_magicos.data.make_silver import _transform_2023

    df_mock = pd.DataFrame({FN.RA: ["RA-1"], FN.IPP: [8.5]})
    df_result = _transform_2023(df_mock)

    pd.testing.assert_frame_equal(df_mock, df_result)
