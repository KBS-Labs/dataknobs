import os
import tempfile
from pathlib import Path

from dataknobs_utils import resource_utils


def test_get_nltk_wordnet():
    resources = dict()

    def downloader(resource, download_dir=None):
        resources[resource] = download_dir

    # remember environ
    datadir = os.environ.get("DATADIR", None)

    with tempfile.TemporaryDirectory() as tempdir:
        # override environ
        cur_datadir = str(Path(tempdir) / "data")
        os.environ["DATADIR"] = cur_datadir
        os.makedirs(cur_datadir, exist_ok=True)

        # check active datadir
        assert resource_utils.active_datadir() == cur_datadir

        # "download" nltk resources
        resource_utils.get_nltk_wordnet(downloader=downloader)

        # Check "downloaded"
        assert Path(resources["wordnet"]).name == "nltk_resources"
        assert Path(resources["omw-1.4"]).name == "nltk_resources"

    # restore environ
    if datadir:
        os.environ["DATADIR"] = datadir
