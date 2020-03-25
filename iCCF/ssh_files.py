from contextlib import closing
from paramiko import SSHClient, AuthenticationException
from astropy.io import fits

from .config import get_config, get_config_file

def get_user_host_msg(USER=None, HOST=None):
    """
    Try to get USER and HOST from the user's configuration file. If not found,
    print a message letting them know they can save them.

    Arguments
    ---------
    USER: str (optional)
        The username for the SSH connection. Takes precededence over the value
        in the user's configuration
    HOST: str (optional)
        The host for the SSH connection. Takes precededence over the value in
        the user's configuration
    """
    config = get_config()
    USER = USER or config.get('USER')
    HOST = HOST or config.get('HOST')
    if USER is None and HOST is None:
        f = get_config_file()
        msg = "Consider running iCCF.config.add_to_config to store USER and HOST"
        print(msg)
    return USER, HOST


def ssh_fits_open(filename, USER=None, HOST=None, verbose=True):
    """
    A wrapper around fits.open to load remote files in USER@HOST, using SSH
    and SFTP clients.

    Arguments
    ---------
    USER: str (optional)
        The username for the SSH connection
    HOST: str (optional)
        The host for the SSH connection
    verbose: bool (optional, default True)
        Whether to be verbose about the connection
    """
    if USER is None or HOST is None:
        USER, HOST = get_user_host_msg(USER, HOST)
    
    if USER is None or HOST is None:  # still?
        raise ValueError('Need to provide USER and HOST for ssh connection')

    with closing(SSHClient()) as client:
        
        client.load_system_host_keys()
        if verbose:
            print(f'Connecting to "{HOST}" with username "{USER}"')

        try:
            client.connect(HOST, username=USER)
        except AuthenticationException:
            import getpass
            p = getpass.getpass(prompt='Password: ', stream=None) 
            client.connect(HOST, username=USER, password=p)

        with closing(client.open_sftp()) as sftp_client:
            
            with closing(sftp_client.open(filename)) as remote_file:
                if verbose:
                    print(f'Opened {filename} over SFTP')

                hdul = fits.open(remote_file, lazy_load_hdus=False)
                for hdu in hdul:
                    hdu.data

    return hdul