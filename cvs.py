try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None
import sys

from darknet.cli import cliHandler

if __name__ == '__main__':
    cliHandler(sys.argv)