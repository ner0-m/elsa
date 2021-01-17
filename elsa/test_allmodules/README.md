# Integration tests

This is an effort to better tests elsa. But instead of just running unit tests,
this tries to compare different use cases for elsa, time them and compute
some meaningful error values.

So far this can only be checked by looking at the output, but at some point,
this should be readable by a machine and automating the process.

## Want to test something?

The entry point is the `testDrvierImpl` function. It has many template parameters
that most settings can be adjusted the way needed in certain areas.

Most of these knobs can be changed in `SetupHelpers.h`. If you have a new projector or solver,
you can specialize the necessary class, such that the driver can pick the name up and print it.

Similarly if you want to test a new solver, that requires a specific setup, specialize
`SolverSetup`. In the future it should be possible to have the same solver with different
setups, but it's not implemented yet.

Also if you need different setups for the trajectors or phantom look into `SetupHelpers.h`.

For future reference: If you want a different logging, check `LoggingHelpers.h`, we should
at some point make it possible to print a nice format to console and a machine readable format
to file.


