"""System and environment utility functions.

Provides utilities for loading environment variables, network discovery,
and system configuration management.
"""

import os
import re
import socket
from pathlib import Path
from typing import Any, Dict, Union

from dotenv import dotenv_values  # type: ignore[import-not-found]


def load_project_vars(
    pvname: str = ".project_vars",
    include_dot_env: bool = True,
    set_environ: bool = False,
    start_path: Path | None = None,
) -> Dict[str, str] | None:
    """Load project variables from the closest configuration file.

    Walks up the directory tree from the current working directory to find
    the closest project variables file and optionally merges with .env settings.

    Notes:
        - Project variable files can be checked into the repo (public)
        - .env files should not be checked in (contain secrets)
        - .env variables will override project variables

    Args:
        pvname: Name of the project variables file. Defaults to ".project_vars".
        include_dot_env: If True, also loads and merges the closest .env file.
            Defaults to True.
        set_environ: If True, also sets loaded variables in os.environ.
            Only sets variables not already in environment. Defaults to False.
        start_path: Directory to start searching from. Defaults to cwd.

    Returns:
        Dict[str, str] | None: Dictionary of configuration variables, or None
            if no configuration file is found.
    """
    config = None
    path = Path(start_path) if start_path else Path.cwd()
    while not os.path.exists(path.joinpath(pvname)) and path.parent != path:
        # Walk up the parents to find the closest project variables file
        path = path.parent
    pvpath = path.joinpath(pvname).absolute()
    if os.path.exists(pvpath):
        config = dotenv_values(pvpath)
    if include_dot_env and pvname != ".env":
        cfg = load_project_vars(
            pvname=".env",
            include_dot_env=False,
            set_environ=False,
            start_path=start_path,
        )
        if cfg is not None:
            if config is not None:
                config.update(cfg)
            else:
                config = cfg

    # Optionally set in os.environ
    if set_environ and config:
        for key, value in config.items():
            if key not in os.environ:
                os.environ[key] = value

    return config


class MySubnet:
    """Collect and store information about the current process's subnet.

    Discovers and caches information about the local machine and other hosts
    on the same subnet using nmap network scanning.

    Attributes:
        my_hostname: The local machine's hostname.
        my_ip: The local machine's IP address.
        all_ips: Dictionary mapping hostnames to IP addresses for all active
            hosts on the subnet (cached).
    """

    def __init__(self) -> None:
        self._my_hostname: str | None = None
        self._my_ip: str | None = None
        self._subnet_ips: Dict[str, str] | None = None

    def rescan(self) -> None:
        """Clear cached subnet information to force a fresh scan.

        Call this method to invalidate the cached subnet hosts data. The next
        access to all_ips will trigger a new nmap scan.
        """
        self._subnet_ips = None

    @property
    def my_hostname(self) -> str:
        """Get my hostname"""
        if self._my_hostname is None:
            self._my_hostname = socket.gethostname()
        return self._my_hostname

    @property
    def my_ip(self) -> str:
        """Get my IP address"""
        if self._my_ip is None:
            self._my_ip = socket.gethostbyname(self.my_hostname)
        return self._my_ip

    @property
    def all_ips(self) -> Dict[str, str]:
        """Get a dictionary mapping hostname to ip_address of all hosts that
        are "up" on the currently running process's subnet.

        NOTE: These are cached on first call but will be recomputed after
              a call to self.rescan().
        """
        if self._subnet_ips is None:

            def is_up(scan_data: Any) -> bool:
                if isinstance(scan_data, dict):
                    state_info = scan_data.get("state", {})
                    state_value = (
                        state_info.get("state", "down") if isinstance(state_info, dict) else "down"
                    )
                    return bool(state_value == "up")
                return False

            def get_hostname(scan_data: Any) -> str | None:
                if isinstance(scan_data, dict):
                    hn = scan_data.get("hostname", [])
                    return hn[0].get("name", None) if len(hn) > 0 else None
                return None

            import nmap3  # type: ignore[import-not-found]

            nmap = nmap3.NmapHostDiscovery()
            subnet = ".".join(self.my_ip.split(".")[:3]) + ".*"
            subnet_ips: Dict[str, str] = {}
            for the_ip, scan_data in nmap.nmap_no_portscan(subnet).items():
                if is_up(scan_data):
                    hostname = get_hostname(scan_data)
                    if hostname is not None:
                        subnet_ips[hostname] = the_ip
            self._subnet_ips = subnet_ips
        return self._subnet_ips if self._subnet_ips is not None else {}

    def get_ips(self, name_re: Union[str, re.Pattern]) -> Dict[str, str]:
        """Get IP addresses of hosts matching a name pattern.

        Args:
            name_re: Regular expression pattern or string to match against
                hostnames.

        Returns:
            Dict[str, str]: Dictionary mapping matching hostnames to their IP
                addresses.
        """
        return {name: ip for name, ip in self.all_ips.items() if re.match(name_re, name)}

    def get_ip(self, name_re: Union[str, re.Pattern]) -> str | None:
        """Get the first IP address matching a hostname pattern.

        Args:
            name_re: Regular expression pattern or string to match against
                hostnames.

        Returns:
            str | None: The first matching IP address, or None if no matches found.
        """
        ips = self.get_ips(name_re)
        if len(ips) > 0:
            return next(iter(ips.values()))
        return None
