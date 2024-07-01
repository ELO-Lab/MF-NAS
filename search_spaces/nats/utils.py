from collections import namedtuple

def dict2config(xdict, logger):
    assert isinstance(xdict, dict), "invalid type : {:}".format(type(xdict))
    Arguments = namedtuple("Configure", " ".join(xdict.keys()))
    content = Arguments(**xdict)
    if hasattr(logger, "log"):
        logger.log("{:}".format(content))
    return content
