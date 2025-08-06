import os
import tempfile

from dataknobs_utils import resource_utils


def test_get_nltk_wordnet():
    resources = dict()

    def downloader(resource, download_dir=None):
        resources[resource] = download_dir

    # remember environ
    datadir = os.environ.get("DATADIR", None)

    with tempfile.TemporaryDirectory() as tempdir:
        # override environ
        cur_datadir = os.path.join(tempdir, "data")
        os.environ["DATADIR"] = cur_datadir
        os.makedirs(cur_datadir, exist_ok=True)

        # check active datadir
        assert resource_utils.active_datadir() == cur_datadir

        # "download" nltk resources
        nltk_wn = resource_utils.get_nltk_wordnet(downloader=downloader)

        # Check "downloaded"
        assert os.path.basename(resources["wordnet"]) == "nltk_resources"
        assert os.path.basename(resources["omw-1.4"]) == "nltk_resources"

    # restore environ
    if datadir:
        os.environ["DATADIR"] = datadir
