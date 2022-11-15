***********
elsa linalg
***********

.. contents:: Table of Contents


The `linalg` module, is a basic and rudimentary (dense) linear algebra module for elsa. It is not a
complete dense or sparse linear algebra library and it is not the intended use case. The module
should and will only contain as much support as needed to implement certain optimization algorithms,
which require certain linear algebra structures and algorithms.


Vector
======

.. doxygenclass:: elsa::linalg::Vector
   :project: elsa

.. doxygenfunction:: elsa::linalg::operator+(const Vector<data_t>& v, data_t s)
   :project: elsa

.. doxygenfunction:: elsa::linalg::operator-(const Vector<data_t>& v, data_t s)
   :project: elsa

.. doxygenfunction:: elsa::linalg::operator*(const Vector<data_t>& v, data_t s)
   :project: elsa

.. doxygenfunction:: elsa::linalg::operator/(const Vector<data_t>& v, data_t s)
   :project: elsa

.. doxygenfunction:: elsa::linalg::operator+(const Vector<data_t>& x, const Vector<data_t>& y)
   :project: elsa

.. doxygenfunction:: elsa::linalg::operator-(const Vector<data_t>& x, const Vector<data_t>& y)
   :project: elsa

.. doxygenfunction:: elsa::linalg::operator*(const Vector<data_t>& x, const Vector<data_t>& y)
   :project: elsa

.. doxygenfunction:: elsa::linalg::operator/(const Vector<data_t>& x, const Vector<data_t>& y)
   :project: elsa

.. doxygenfunction:: elsa::linalg::normalize
   :project: elsa

.. doxygenfunction:: elsa::linalg::normalized
   :project: elsa

.. doxygenfunction:: elsa::linalg::norm
   :project: elsa

.. doxygenfunction:: elsa::linalg::dot
   :project: elsa

Matrix
======

.. doxygenclass:: elsa::linalg::Matrix
   :project: elsa

.. doxygenfunction:: elsa::linalg::operator*(const Matrix<data_t> &mat, const T &x)
   :project: elsa

.. doxygenclass:: elsa::linalg::RowView
   :project: elsa

.. doxygenclass:: elsa::linalg::ConstRowView
   :project: elsa

.. doxygenclass:: elsa::linalg::ColumnView
   :project: elsa

.. doxygenclass:: elsa::linalg::ConstColumnView
   :project: elsa
