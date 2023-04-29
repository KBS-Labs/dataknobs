import numpy as np
import os
import pandas as pd
import tempfile
import dataknobs.structures.record_store as record_store


def test_basics():
    RECS = [
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 4, 'b': 5, 'c': 6},
        {'a': 7, 'b': 8, 'c': 9},
    ]

    with tempfile.TemporaryDirectory(
            suffix='.record_store', prefix='test-'
    ) as recdir:
        tsv_fpath = os.path.join(recdir, 'basics_test.tsv')
        rs1 = record_store.RecordStore(tsv_fpath)
        rs1.add_rec(RECS[0])
        assert len(rs1.records) == 1
        assert len(rs1.df) == 1
        rs1.add_rec(RECS[1])
        assert len(rs1.records) == 2
        assert len(rs1.df) == 2
        rs1.save()

        rs1.add_rec(RECS[2])
        assert len(rs1.records) == 3
        assert len(rs1.df) == 3

        rs1.restore()
        assert len(rs1.records) == 2
        assert len(rs1.df) == 2

        rs1.add_rec(RECS[2])
        assert len(rs1.records) == 3
        assert len(rs1.df) == 3
        rs1.save()

        rs2 = record_store.RecordStore(tsv_fpath)
        assert np.all(rs2.df == rs1.df)
        assert rs2.records == rs1.records

        tsv_fpath2 = os.path.join(recdir, 'basics_test2.tsv')
        rs3 = record_store.RecordStore(
            tsv_fpath2,
            df=pd.DataFrame(
                [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
                columns=['a', 'b', 'c']
            )
        )
        assert np.all(rs3.df == rs1.df)
        assert rs3.records == rs1.records
