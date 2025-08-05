'''
General data utilities for working with 3rd party resources
'''
import nltk
import os


# NLTK Wordnet Resources
NLTK_RESOURCES_PATH = 'opt/nltk_resources'
NLTK_RESOURCES = {
    'wordnet': os.path.join('corpora', 'wordnet.zip'),
    'omw-1.4': os.path.join('corpora', 'omw-1.4.zip'),
    'wordnet_ic': os.path.join('corpora', 'wordnet_ic.zip')
}


# Runtime variables
_DATADIR = None
_NLTK_RESOURCES_DIR = None
_NLTK_WN = None


def active_datadir():
    '''
    Get the active data directory as the first that exists from:
        * the DATADIR environment variable
        * the HOME/data directory
        * the /data directory

    NOTE: If an active DATADIR must exist to get a non-None result.

    :return: The active data directory or None
    '''
    global _DATADIR  # pylint: disable-msg=W0603
    if _DATADIR is None:
        _DATADIR = os.environ.get(
            'DATADIR',
            os.environ.get('HOME', '') + '/data'
        )
        if not os.path.exists(_DATADIR) and os.path.exists('/data'):
            _DATADIR = '/data'
    return _DATADIR


def download_nltk_resources(resources, resources_dir=None, verbose=False, downloader=nltk.download):
    '''
    Download the nltk resources that don't yet exist.
    :resources_dir: The resources (root) directory to download to, or
        the default if None.
    :resources: A dictionary of resource names to download mapped to
        the relative (to the resoures dir) path of the resource
    :verbose: True to print status
    '''
    if resources is not None:
        if resources_dir is None:
            resources_dir = get_nltk_resources_dir()
        for resource, relpath in resources.items():
            respath = os.path.join(resources_dir, relpath)
            if not os.path.exists(respath):
                if verbose:
                    print(f'NOTE: {respath} does not exist. Downloading...')
                downloader(resource, download_dir=resources_dir)


def get_nltk_resources_dir(resources=None, verbose=False, downloader=nltk.download):
    '''
    Get the NLTK resources directory, optionally downloading resources, from:
        * the NLTK_DATA environment variable
        * the active_datadir()/NLTK_RESOURCES_PATH

    NOTE: If an active DATADIR must exist to get a non-None result.

    :param resources: A dictionary identifying the resources and relative (to
        the NLTK resources directory) paths to ensure are downloaded
    :verbose: True to print status
    :return: The NLTK resources directory path or None
    '''
    global _NLTK_RESOURCES_DIR  # pylint: disable-msg=W0603
    if _NLTK_RESOURCES_DIR is None:
        _NLTK_RESOURCES_DIR = os.environ.get('NLTK_DATA', None)
        if _NLTK_RESOURCES_DIR is None:
            datadir = active_datadir()
            if datadir is not None:
                resdir = os.path.join(datadir, NLTK_RESOURCES_PATH)
                _NLTK_RESOURCES_DIR = resdir
                os.environ['NLTK_DATA'] = resdir
                nltk.data.path.append(resdir)
                if resources is not None:
                    download_nltk_resources(
                        resources, resources_dir=resdir, verbose=verbose,
                        downloader=downloader,
                    )
    return _NLTK_RESOURCES_DIR


def get_nltk_wordnet(downloader=nltk.download):
    '''
    Get a handle on NLTK's wordnet object, ensuring resources have
    been downloaded.
    :return: nltk.corpus.wordnet
    '''
    # Make sure resources have been downloaded
    global _NLTK_WN  # pylint: disable-msg=W0603
    if _NLTK_WN is None:
        if get_nltk_resources_dir(resources=NLTK_RESOURCES, downloader=downloader) is not None:
            from nltk.corpus import wordnet as wn
            _NLTK_WN = wn
    return _NLTK_WN
