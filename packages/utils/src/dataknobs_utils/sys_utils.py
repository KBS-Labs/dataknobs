import os
import re
import socket
from pathlib import Path
from typing import Dict, Union, Optional, Any

import nmap3  # type: ignore[import-not-found]
from dotenv import dotenv_values  # type: ignore[import-not-found]


def load_project_vars(
    pvname: str = ".project_vars", include_dot_env: bool = True
) -> Optional[Dict[str, str]]:
    """Load "closest" project variables.

    :param pvname: The name of the project variables file.
    :param include_dot_env: True to also find and load the closest ".env" file
        NOTEs:
          * project variable files can be checked in to the repo (public)
          * .env files cannot be checked into the repo (secrets)
          * .env vars will supercede project variables
    :return: The project variables config or None
    """
    config = None
    path = Path(os.getcwd())
    while not os.path.exists(path.joinpath(pvname)) and path.parent != path:
        # Walk up the parents to find the closest project variables file
        path = path.parent
    pvpath = path.joinpath(pvname).absolute()
    if os.path.exists(pvpath):
        config = dotenv_values(pvpath)
    if include_dot_env and pvname != ".env":
        cfg = load_project_vars(pvname=".env", include_dot_env=False)
        if cfg is not None:
            if config is not None:
                config.update(cfg)
            else:
                config = cfg
    return config


class MySubnet:
    """Class to collect and store information about the subnet of the current
    running process.
    """

    def __init__(self) -> None:
        self._my_hostname: Optional[str] = None
        self._my_ip: Optional[str] = None
        self._subnet_ips: Optional[Dict[str, str]] = None

    def rescan(self) -> None:
        """Rescan subnet hosts."""
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
                    state_value = state_info.get("state", "down") if isinstance(state_info, dict) else "down"
                    return bool(state_value == "up")
                return False

            def get_hostname(scan_data: Any) -> Optional[str]:
                if isinstance(scan_data, dict):
                    hn = scan_data.get("hostname", [])
                    return hn[0].get("name", None) if len(hn) > 0 else None
                return None

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
        """Get the IP addresses of the hosts whose names match the regex.
        :param name_re: The name or regex pattern to match
        :return: The dictionary mapping matched names to their IP addresses
        """
        return {name: ip for name, ip in self.all_ips.items() if re.match(name_re, name)}

    def get_ip(self, name_re: Union[str, re.Pattern]) -> Optional[str]:
        """Get the "first" IP address of the hosts whose names match the regex.
        :param name_re: The name or regex pattern to match
        :return: The "first" matching IP address or None
        """
        ips = self.get_ips(name_re)
        if len(ips) > 0:
            return list(ips.values())[0]
        return None
