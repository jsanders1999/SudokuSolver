#!/usr/bin/env python
"""
A class to easily time code.

adapted from https://realpython.com/python-timer/#a-python-timer-class
added functinality to have repeated calls and have the timings from each call stored in a list,
as opposed to having each new result overwrite the previous timing result.
"""

from dataclasses import dataclass, field
import time
from typing import Callable, ClassVar, Dict, Optional
import logging
import numpy as np
import functools

def addLoggingLevel(levelName, levelNum, methodName=None):
	"""
	Comprehensively adds a new logging level to the `logging` module and the
	currently configured logging class.

	`levelName` becomes an attribute of the `logging` module with the value
	`levelNum`. `methodName` becomes a convenience method for both `logging`
	itself and the class returned by `logging.getLoggerClass()` (usually just
	`logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
	used.

	To avoid accidental clobberings of existing attributes, this method will
	raise an `AttributeError` if the level name is already an attribute of the
	`logging` module or if the method name is already present 

	Example
	-------
	>>> addLoggingLevel('TRACE', logging.DEBUG - 5)
	>>> logging.getLogger(__name__).setLevel("TRACE")
	>>> logging.getLogger(__name__).trace('that worked')
	>>> logging.trace('so did this')
	>>> logging.TRACE
	5

	"""
	if not methodName:
		methodName = levelName.lower()

	if hasattr(logging, levelName):
	   raise AttributeError('{} already defined in logging module'.format(levelName))
	if hasattr(logging, methodName):
	   raise AttributeError('{} already defined in logging module'.format(methodName))
	if hasattr(logging.getLoggerClass(), methodName):
	   raise AttributeError('{} already defined in logger class'.format(methodName))

	# This method was inspired by the answers to Stack Overflow post
	# http://stackoverflow.com/q/2183233/2988730, especially
	# http://stackoverflow.com/a/13638084/2988730
	def logForLevel(self, message, *args, **kwargs):
		if self.isEnabledFor(levelNum):
			self._log(levelNum, message, args, **kwargs)
	def logToRoot(message, *args, **kwargs):
		logging.log(levelNum, message, *args, **kwargs)

	logging.addLevelName(levelNum, levelName)
	setattr(logging, levelName, levelNum)
	setattr(logging.getLoggerClass(), methodName, logForLevel)
	setattr(logging, methodName, logToRoot)

addLoggingLevel("TIMER", logging.INFO-5)
addLoggingLevel("MEMORY", logging.INFO-4)


class TimerError(Exception):
	"""A custom exception used to report errors in use of Timer class"""


#TODO: add functionality for calling one timer object multiple times. Save all the results as an array. 
@dataclass
class Timer:
	timers: ClassVar[Dict[str, list[float]]] = {}
	name: Optional[str] = None
	text: str = " {:<30} {:0.6f} seconds"
	logger: Optional[Callable[[str], None]] = logging.timer
	_start_time: Optional[float] = field(default=None, init=False, repr=False)

	#n_timers = 0
	#timer_names = []

	def __post_init__(self) -> None:
		"""Add timer to dict of timers after initialization"""
		if self.name is not None:
			self.timers.setdefault(self.name, [])

	def start(self) -> None:
		"""Start a new timer"""
		if self._start_time is not None:
			raise TimerError(f"Timer is running. Use .stop() to stop it")

		self._start_time = time.perf_counter()

	def stop(self) -> float:
		"""Stop the timer, and report the elapsed time"""
		if self._start_time is None:
			raise TimerError(f"Timer is not running. Use .start() to start it")

		# Calculate elapsed time
		elapsed_time = time.perf_counter() - self._start_time
		self._start_time = None

		# Report elapsed time
		#logging.info("Elapsed time for timer {:}: {:0.6f} seconds".format(self.timer_name, elapsed_time))
		if self.logger:
			self.logger(self.text.format(self.name+" :", elapsed_time))
		if self.name:
			self.timers[self.name] += [elapsed_time]
		#self.time_list += [elapsed_time] #could be a np array. concatenate or append to it.

		return elapsed_time
		
	def __enter__(self):
		"""Start a new timer as a context manager"""
		self.start()
		return self

	def __exit__(self, *exc_info):
		"""Stop the context manager timer"""
		self.stop()

	def __call__(self, func):
		"""Support using Timer as a decorator"""
		@functools.wraps(func)
		def wrapper_timer(*args, **kwargs):
			with self:
				return func(*args, **kwargs)

		return wrapper_timer





