from __future__ import annotations

import pandas as pd


class PSMProcessor:
    @staticmethod
    def filter_psms(df: pd.DataFrame, fdr_threshold: float) -> pd.DataFrame:
        """Filter TT PSMs using the existing Q-value column."""
        working = df.copy()
        working['Score'] = pd.to_numeric(working['Score'], errors='coerce').fillna(0)
        working.loc[working['Score'] < 0, 'Score'] = 0
        working['Target_Decoy'] = pd.to_numeric(working['Target_Decoy'], errors='coerce').astype('Int64')
        working['Q-value'] = pd.to_numeric(working['Q-value'], errors='coerce')
        working = working.sort_values('Score', ascending=False)
        filtered = working[working['Q-value'] <= fdr_threshold]
        return filtered[filtered['Target_Decoy'] == 2].copy()
