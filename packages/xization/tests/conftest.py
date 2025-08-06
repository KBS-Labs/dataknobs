import json
import os
from typing import Dict, List

import pytest


def resources_path(package: str) -> str:
    """Compose the path to a package's resources.
    :param package: The name of the resource's package (subdir).
    :param filename: The name of the file under the package.
    :return: The file path to the package's resources.
    """
    dir_path = os.path.join(os.path.dirname(__file__), package)
    return os.path.join(dir_path, "resources")


def resource(package: str, filename: str) -> str:
    """Compose the path to a resource.
    :param package: The name of the resource's package (subdir).
    :param filename: The name of the file under the package.
    :return: The file path to the resource.
    """
    return os.path.join(resources_path(package), filename)


def resource_as_text(package: str, filename: str) -> str:
    """Load a resource's contents as a single text string.
    :param package: The name of the resource's package (subdir).
    :param filename: The name of the file under the package.
    :return: The resource contents.
    """
    path = resource(package, filename)
    with open(path, encoding="utf-8") as infile:
        return infile.read()


def resource_as_list(
    package: str, filename: str, ignore_comments: str = "#", ignore_empties: str = True
) -> List[str]:
    """Read each file line and add to a list.
    :param package: The package (subdirectory)
    :param filename: The filename (within the package directory)
    :param ignore_comments: If non-null, skip lines beginning with this value
    :param ignore_empties: True to skip empty lines
    """
    result = list()
    path = resource(package, filename)
    with open(path, encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not ignore_empties or line:
                if not ignore_comments or not line.startswith(ignore_comments):
                    result.append(line)
    return result


def resource_as_json(package: str, filename: str) -> Dict:
    """Load a resource's contents as json
    :param package: The name of the resource's package (subdir).
    :param filename: The name of the file under the package.
    :return: The resource contents.
    """
    return json.loads(resource_as_text(package, filename))


TEST_JSON_001 = "test-001.json"
TEST_JSON_002 = "test-002.json"
TEST_JSON_003 = "test-003.json"


@pytest.fixture
def test_utils_dir() -> str:
    return resources_path("utils")


@pytest.fixture
def test_json_001() -> Dict:
    return resource_as_text("utils", TEST_JSON_001)


@pytest.fixture
def test_json_002() -> Dict:
    return resource_as_text("utils", TEST_JSON_002)


@pytest.fixture
def test_json_003() -> Dict:
    return resource_as_text("utils", TEST_JSON_003)
