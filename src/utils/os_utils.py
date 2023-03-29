""" Collection of util functions related to os."""
from datetime import date


def get_date_string():
    """ Return date string for today's date in year-month-day format."""
    return str(date.today())
