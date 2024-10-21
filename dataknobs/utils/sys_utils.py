import nmap3
import os
import re
import socket
from dotenv import dotenv_values
from pathlib import Path
from typing import Dict, Union


def load_project_vars(pvname: str = ".project_vars", include_dot_env: bool = True) -> Dict[str, str]:
    '''
    Load "closest" project variables.

    :param pvname: The name of the project variables file.
    :param include_dot_env: True to also find and load the closest ".env" file
        NOTEs:
          * project variable files can be checked in to the repo (public)
          * .env files cannot be checked into the repo (secrets)
          * .env vars will supercede project variables
    :return: The project variables config or None
    '''
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
    '''
    Class to collect and store information about the subnet of the current
    running process.
    '''

    def __init__(self):
        self._my_hostname = None
        self._my_ip = None
        self._subnet_ips = None

    def rescan(self):
        '''
        Rescan subnet hosts.
        '''
        self._subnet_ips = None

    @property
    def my_hostname(self) -> str:
        ''' Get my hostname '''
        if self._my_hostname is None:
            self._my_hostname = socket.gethostname()
        return self._my_hostname

    @property
    def my_ip(self) -> str:
        ''' Get my IP address '''
        if self._my_ip is None:
            self._my_ip = socket.gethostbyname(self.my_hostname)
        return self._my_ip

    @property
    def all_ips(self) -> Dict[str, str]:
        '''
        Get a dictionary mapping hostname to ip_address of all hosts that
        are "up" on the currently running process's subnet.

        NOTE: These are cached on first call but will be recomputed after
              a call to self.rescan().
        '''
        if self._subnet_ips is None:
            def is_up(scan_data):
                if isinstance(scan_data, dict):
                    return scan_data.get("state", {}).get("state", "down") == "up"
                return None
            def get_hostname(scan_data):
                if isinstance(scan_data, dict):
                    hn = scan_data.get("hostname", [])
                    return hn[0].get("name", None) if len(hn) > 0 else None
                return None
            nmap =  nmap3.NmapHostDiscovery()
            subnet = '.'.join(self.my_ip.split('.')[:3]) + ".*"
            self._subnet_ips = {
                get_hostname(scan_data): the_ip
                for the_ip, scan_data in nmap.nmap_no_portscan(subnet).items()
                if is_up(scan_data) and get_hostname(scan_data) is not None
            }
        return self._subnet_ips

    def get_ips(self, name_re: Union[str, re.Pattern]) -> Dict[str, str]:
        '''
        Get the IP addresses of the hosts whose names match the regex.
        :param name_re: The name or regex pattern to match
        :return: The dictionary mapping matched names to their IP addresses
        '''
        return {
            name: ip
            for name, ip in self.all_ips.items()
            if re.match(name_re, name)
        }

    def get_ip(self, name_re: Union[str, re.Pattern]) -> str:
        '''
        Get the "first" IP address of the hosts whose names match the regex.
        :param name_re: The name or regex pattern to match
        :return: The "first" matching IP address or None
        '''
        ips = self.get_ips(name_re)
        if len(ips) > 0:
            return list(ips.values())[0]
        return None
