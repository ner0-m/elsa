In elsa, we are using expression templates for fast and efficient mathematical operations while at the same time having intuitive syntax. This technique is also known as lazy-evaluation which means that computations are delayed until the results are actually needed.

Take the example
```cpp
auto expression = dataContainer1 + dataContainer2;
```
where the type of `expression` is `elsa::Expression<operator+, DataContainer<real_t>, DataContainer<real_t>>`.

The `expression` contains the work *to be done* rather than the actual result. Only when assigning to a `DataContainer` or constructing a new DataContainer, the expression gets evaluated automatically
```cpp
DataContainer result = dataContainer1 + dataContainer2;
```
into `result`.

Nesting is easily possible like
```cpp
auto expression = dataContainer1 + dataContainer2 * expression;
```
which will store an expression tree as the type information.

Please note also that the operators/member functions available on an `Expression` type are different from a `DataContainer`. 

An expression has a member function `eval()`. However, calling `eval()` will *not* return a `DataContainer` but depending on the current `DataHandler` the underlying raw data computation result.

If single element-wise access is necessary, it is possible to call `expression.eval()[index]`. Note that this is computational very expensive as the whole expression gets evaluated at every index individually.
