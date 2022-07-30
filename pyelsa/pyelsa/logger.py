from . import *

class Logger(object):
	def setLevel(level: LogLevel):
		logger_pyelsa_io.setLevel(level)
		logger_pyelsa_operators.setLevel(level)
		logger_pyelsa_functionals.setLevel(level)
		logger_pyelsa_problems.setLevel(level)
		logger_pyelsa_proximity_operators.setLevel(level)
		logger_pyelsa_solvers.setLevel(level)
		logger_pyelsa_projectors.setLevel(level)
		logger_pyelsa_projectors_cuda.setLevel(level)
		logger_pyelsa_generators.setLevel(level)

	def enableFileLogging(filename: str):
		logger_pyelsa_io.enableFileLogging(filename)
		logger_pyelsa_operators.enableFileLogging(filename)
		logger_pyelsa_functionals.enableFileLogging(filename)
		logger_pyelsa_problems.enableFileLogging(filename)
		logger_pyelsa_proximity_operators.enableFileLogging(filename)
		logger_pyelsa_solvers.enableFileLogging(filename)
		logger_pyelsa_projectors.enableFileLogging(filename)
		logger_pyelsa_projectors_cuda.enableFileLogging(filename)
		logger_pyelsa_generators.enableFileLogging(filename)

	def flush():
		logger_pyelsa_io.flush()
		logger_pyelsa_operators.flush()
		logger_pyelsa_functionals.flush()
		logger_pyelsa_problems.flush()
		logger_pyelsa_proximity_operators.flush()
		logger_pyelsa_solvers.flush()
		logger_pyelsa_projectors.flush()
		logger_pyelsa_projectors_cuda.flush()
		logger_pyelsa_generators.flush()
