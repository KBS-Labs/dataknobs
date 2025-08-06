import os
import re
import tempfile

import dataknobs_utils.file_utils as dk_futils


def test_basics():
    # Set up:
    #     root_dir/
    #         file1.txt w/lines: 'a', 'b', 'c'
    #         file2.txt.gz w/lines: 'd', 'e', 'f'
    #         dir1/
    #             file3.txt.gz w/lines: 'g', 'h', 'i'
    #             file4.txt w/lines: 'j', 'k', 'l'
    root_dir = tempfile.TemporaryDirectory(suffix=".filepath_generator", prefix="dk")
    dk_futils.write_lines("file1.txt", ["a", "b", "c"], rootdir=root_dir.name)
    dk_futils.write_lines("file2.txt.gz", ["d", "e", "f"], rootdir=root_dir.name)
    dir1 = os.path.join(root_dir.name, "dir1")
    os.makedirs(dir1)
    dk_futils.write_lines("file3.txt.gz", ["g", "h", "i"], rootdir=dir1)
    dk_futils.write_lines("file4.txt", ["j", "k", "l"], rootdir=dir1)

    # Test filepath_generator and fileline_generator, files_only
    filenames = list()
    for fpath in dk_futils.filepath_generator(root_dir.name):
        fnum_match = re.match(r"^.*file(\d).*$", fpath)
        fnum = int(fnum_match.group(1))
        expect_line = chr(ord("a") + (fnum - 1) * 3)
        fname = os.path.basename(fpath)
        fdir = os.path.dirname(fpath)
        filenames.append(fname)
        for line in dk_futils.fileline_generator(fname, rootdir=fdir):
            assert expect_line == line
            expect_line = chr(ord(expect_line) + 1)
    assert sorted(filenames) == ["file1.txt", "file2.txt.gz", "file3.txt.gz", "file4.txt"]

    # Test filepath_generator, no descend, with directories
    filenames = [
        os.path.basename(x)
        for x in dk_futils.filepath_generator(
            root_dir.name,
            descend=False,
            files_only=False,
        )
    ]
    assert sorted(filenames) == [
        "dir1",
        "file1.txt",
        "file2.txt.gz",
    ]


def test_is_gzip_file(test_utils_dir):
    not_gzip_count = 0
    gzip_count = 0
    for fpath in dk_futils.filepath_generator(
        test_utils_dir,
        descend=False,
        files_only=True,
    ):
        is_gzip = dk_futils.is_gzip_file(fpath)
        assert is_gzip == fpath.endswith(".gz")
        if is_gzip:
            gzip_count += 1
        else:
            not_gzip_count += 1
    assert gzip_count > 0
    assert not_gzip_count > 0
