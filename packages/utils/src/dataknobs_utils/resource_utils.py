"""General data utilities for working with 3rd party resources"""

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict

import nltk

# NLTK Wordnet Resources
NLTK_RESOURCES_PATH = "opt/nltk_resources"
NLTK_RESOURCES = {
    "wordnet": str(Path("corpora") / "wordnet.zip"),
    "omw-1.4": str(Path("corpora") / "omw-1.4.zip"),
    "wordnet_ic": str(Path("corpora") / "wordnet_ic.zip"),
}


# Runtime cache using mutable container to avoid global statement
_CACHE = {
    "datadir": None,
    "nltk_resources_dir": None,
    "nltk_wn": None,
}


def active_datadir() -> str | None:
    """Get the active data directory from available locations.

    Searches for an existing data directory in the following order:
    1. DATADIR environment variable
    2. $HOME/data directory
    3. /data directory

    Note:
        An active data directory must exist to get a non-None result.

    Returns:
        str | None: Path to the active data directory, or None if not found.
    """
    if _CACHE["datadir"] is None:
        _CACHE["datadir"] = os.environ.get("DATADIR", os.environ.get("HOME", "") + "/data")
        if not os.path.exists(_CACHE["datadir"]) and os.path.exists("/data"):
            _CACHE["datadir"] = "/data"
    return _CACHE["datadir"]


def download_nltk_resources(
    resources: Dict[str, str] | None,
    resources_dir: str | None = None,
    verbose: bool = False,
    downloader: Callable = nltk.download,
) -> None:
    """Download NLTK resources that don't yet exist locally.

    Args:
        resources: Dictionary mapping resource names to their relative paths
            (relative to resources_dir).
        resources_dir: Root directory for resources. If None, uses the default
            from get_nltk_resources_dir(). Defaults to None.
        verbose: If True, prints download status messages. Defaults to False.
        downloader: Callable for downloading resources. Defaults to nltk.download.
    """
    if resources is not None:
        if resources_dir is None:
            resources_dir = get_nltk_resources_dir()
        if resources_dir is None:
            return  # Can't download without a resources directory
        for resource, relpath in resources.items():
            respath = str(Path(resources_dir) / relpath)
            if not os.path.exists(respath):
                if verbose:
                    print(f"NOTE: {respath} does not exist. Downloading...", file=sys.stderr)
                downloader(resource, download_dir=resources_dir)


def get_nltk_resources_dir(
    resources: Dict[str, str] | None = None,
    verbose: bool = False,
    downloader: Callable = nltk.download,
) -> str | None:
    """Get the NLTK resources directory and optionally download resources.

    Determines the NLTK resources directory from:
    1. NLTK_DATA environment variable
    2. active_datadir()/NLTK_RESOURCES_PATH

    If resources are specified, downloads any missing resources to the directory.

    Note:
        An active DATADIR must exist to get a non-None result.

    Args:
        resources: Optional dictionary mapping resource names to their relative
            paths (relative to NLTK resources dir) to ensure are downloaded.
        verbose: If True, prints status messages. Defaults to False.
        downloader: Callable for downloading resources. Defaults to nltk.download.

    Returns:
        str | None: Path to NLTK resources directory, or None if not found.
    """
    if _CACHE["nltk_resources_dir"] is None:
        _CACHE["nltk_resources_dir"] = os.environ.get("NLTK_DATA", None)
        if _CACHE["nltk_resources_dir"] is None:
            datadir = active_datadir()
            if datadir is not None:
                resdir = str(Path(datadir) / NLTK_RESOURCES_PATH)
                _CACHE["nltk_resources_dir"] = resdir
                os.environ["NLTK_DATA"] = resdir
                nltk.data.path.append(resdir)
                if resources is not None:
                    download_nltk_resources(
                        resources,
                        resources_dir=resdir,
                        verbose=verbose,
                        downloader=downloader,
                    )
    return _CACHE["nltk_resources_dir"]


def get_nltk_wordnet(downloader: Callable = nltk.download) -> Any:
    """Get NLTK's WordNet corpus, ensuring resources are downloaded.

    Automatically downloads required WordNet resources if not already present.

    Args:
        downloader: Callable for downloading resources. Defaults to nltk.download.

    Returns:
        nltk.corpus.wordnet: The NLTK WordNet corpus object, or None if
            resources directory cannot be determined.
    """
    # Make sure resources have been downloaded
    if _CACHE["nltk_wn"] is None:
        if get_nltk_resources_dir(resources=NLTK_RESOURCES, downloader=downloader) is not None:
            from nltk.corpus import wordnet as wn

            _CACHE["nltk_wn"] = wn
    return _CACHE["nltk_wn"]
