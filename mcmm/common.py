from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type


class Error(Exception):
    """Base class for all exceptions raised by the mcmm module."""


class InvalidOperation(Error):
    """An operation was called on a object that does not support it."""


class InvalidValue(Error):
    """A function was called with an invalid argument."""

