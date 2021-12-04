/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded functions relating to various Eigen3 types
 */

#ifndef OPENKALMAN_EIGEN3_MATRIX_OVERLOADS_HPP
#define OPENKALMAN_EIGEN3_MATRIX_OVERLOADS_HPP

#include <type_traits>
#include <random>

namespace OpenKalman::Eigen3
{
#ifdef __cpp_concepts
  template<eigen_native_general Arg>
  requires (not native_eigen_matrix<Arg>)
#else
  template<typename Arg, std::enable_if_t<eigen_native_general<Arg> and (not native_eigen_matrix<Arg>), int> = 0>
#endif
  constexpr decltype(auto)
  nested_matrix(Arg&& arg)
  {
    if constexpr (eigen_SelfAdjointView<Arg> or eigen_TriangularView<Arg>)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }
    else
    {
      static_assert(eigen_DiagonalMatrix<Arg> or eigen_DiagonalWrapper<Arg>);
      return std::forward<Arg>(arg).diagonal();
    }
  }


  /**
   * Make a native Eigen matrix from a list of coefficients in row-major order.
   * \tparam Scalar The scalar type of the matrix.
   * \tparam rows The number of rows.
   * \tparam columns The number of columns.
   */
#ifdef __cpp_concepts
  template<arithmetic_or_complex Scalar, std::size_t rows, std::size_t columns = 1, std::convertible_to<Scalar> ... Args>
  requires
    (rows == 0 and columns == 0) or
    (rows != 0 and columns != 0 and sizeof...(Args) == rows * columns) or
    (columns == 0 and sizeof...(Args) % rows == 0) or
    (rows == 0 and sizeof...(Args) % columns == 0)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdiv-by-zero"
  template<typename Scalar, std::size_t rows, std::size_t columns = 1, typename ... Args, std::enable_if_t<
    arithmetic_or_complex<Scalar> and (std::is_convertible_v<Args, Scalar> and ...) and
    ((rows == 0 and columns == 0) or
    (rows != 0 and columns != 0 and sizeof...(Args) == rows * columns) or
    (columns == 0 and sizeof...(Args) % rows == 0) or
    (rows == 0 and sizeof...(Args) % columns == 0)), int> = 0>
// See below for pragma pop...
#endif
  inline auto
  make_native_matrix(const Args ... args)
  {
    using M = Eigen3::eigen_matrix_t<Scalar, rows, columns>;
    return MatrixTraits<M>::make(static_cast<const Scalar>(args)...);
  }


  /**
   * \overload
   * \brief Make a native Eigen matrix from a list of coefficients in row-major order.
   */
#ifdef __cpp_concepts
  template<std::size_t rows, std::size_t columns = 1, arithmetic_or_complex ... Args>
  requires
    (rows == 0 and columns == 0) or
    ((rows != 0 and columns != 0 and sizeof...(Args) == rows * columns) or
    (columns == 0 and sizeof...(Args) % rows == 0) or
    (rows == 0 and sizeof...(Args) % columns == 0))
#else
  template<std::size_t rows, std::size_t columns = 1, typename ... Args, std::enable_if_t<
    (arithmetic_or_complex<Args> and ...) and
    ((rows == 0 and columns == 0) or
    ((rows != 0 and columns != 0 and sizeof...(Args) == rows * columns) or
    (columns == 0 and sizeof...(Args) % rows == 0) or
    (rows == 0 and sizeof...(Args) % columns == 0))), int> = 0>
#endif
  inline auto
  make_native_matrix(const Args ... args)
  {
    using Scalar = std::common_type_t<Args...>;
    return make_native_matrix<Scalar, rows, columns>(args...);
  }


  /// Make a native Eigen 1-column vector from a list of coefficients in row-major order.
#ifdef __cpp_concepts
  template<arithmetic_or_complex ... Args>
#else
#pragma GCC diagnostic pop

  template<typename ... Args, std::enable_if_t<(arithmetic_or_complex<Args> and ...), int> = 0>
#endif
  inline auto
  make_native_matrix(const Args ... args)
  {
    using Scalar = std::common_type_t<Args...>;
    return make_native_matrix<Scalar, sizeof...(Args), 1>(args...);
  }


  /**
   * Convert to a self-contained Eigen3 matrix.
   */
#ifdef __cpp_concepts
  template<typename Arg> requires
    eigen_native_general<Arg> or
    eigen_matrix<Arg> or
    eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or
    eigen_diagonal_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_native_general<Arg> or
    eigen_matrix<Arg> or
    eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or
    eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_native_matrix(Arg&& arg) noexcept
  {
    if constexpr (std::is_same_v<std::decay_t<Arg>, native_matrix_t<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return native_matrix_t<Arg> {std::forward<Arg>(arg)};
    }
  }


  /// Convert to function-returnable version of the matrix.
#ifdef __cpp_concepts
  template<typename ... Ts, eigen_native_general Arg>
#else
  template<typename ... Ts, typename Arg, std::enable_if_t<eigen_native_general<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr(self_contained<Arg> or ((sizeof...(Ts) > 0) and ... and std::is_lvalue_reference_v<Ts>))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return make_native_matrix(std::forward<Arg>(arg));
    }
  }


  /// Get element (i, j) of matrix arg
#ifdef __cpp_concepts
  template<eigen_native_general Arg>
  requires element_gettable<Arg, 2>
#else
  template<typename Arg, std::enable_if_t<eigen_native_general<Arg> and element_gettable<Arg, 2>, int> = 0>
#endif
  inline auto
  get_element(const Arg& arg, const std::size_t i, const std::size_t j)
  {
    return arg.coeff(i, j);
  }


  /// Get element (i) of one-column matrix arg
#ifdef __cpp_concepts
  template<eigen_native_general Arg> requires element_gettable<Arg, 1>
#else
  template<typename Arg, std::enable_if_t<eigen_native_general<Arg> and element_gettable<Arg, 1>, int> = 0>
#endif
  inline auto
  get_element(const Arg& arg, const std::size_t i)
  {
    return arg.coeff(i);
  }


  /// Set element (i, j) of matrix arg to s.
#ifdef __cpp_concepts
  template<eigen_native_general Arg, typename Scalar>
  requires element_settable<Arg, 2>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    eigen_native_general<Arg> and element_settable<Arg, 2>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    arg(i, j) = s;
  }


  /// Set element (i) of one-column matrix arg to s.
#ifdef __cpp_concepts
  template<eigen_native_general Arg, typename Scalar>
  requires element_settable<Arg, 1>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    eigen_native_general<Arg> and element_settable<Arg, 1>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    arg(i) = s;
  }


#ifdef __cpp_concepts
  template<eigen_native_general Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native_general<Arg>, int> = 0>
#endif
  constexpr std::size_t
  row_count(Arg&& arg)
  {
    if constexpr (dynamic_rows<Arg>)
      return arg.rows();
    else
      return MatrixTraits<Arg>::rows;
  }


#ifdef __cpp_concepts
  template<eigen_native_general Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native_general<Arg>, int> = 0>
#endif
  constexpr std::size_t
  column_count(Arg&& arg)
  {
    if constexpr (dynamic_columns<Arg>)
      return arg.cols();
    else
      return MatrixTraits<Arg>::columns;
  }


  /// Return row <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<std::size_t...index, native_eigen_matrix Arg, std::convertible_to<const std::size_t>...runtime_index_t>
  requires (sizeof...(index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_rows<Arg> or ((index + ... + 0) < MatrixTraits<Arg>::rows))
#else
  template<size_t...index, typename Arg, typename...runtime_index_t, std::enable_if_t<native_eigen_matrix<Arg> and
    (std::is_convertible_v<runtime_index_t, const std::size_t> and ...) and
    (sizeof...(index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_rows<Arg> or ((index + ... + 0) < MatrixTraits<Arg>::rows)), int> = 0>
#endif
  constexpr decltype(auto)
  row(Arg&& arg, runtime_index_t...i)
  {
    if constexpr (row_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      auto e_i = static_cast<Eigen::Index>((index + ... + (i + ... + 0)));

      if constexpr (dynamic_columns<Arg>)
      {
        Eigen::Index c = column_count(arg);
        return make_self_contained<Arg>(std::forward<Arg>(arg).template block<1, Eigen::Dynamic>(e_i, 0, 1, c));
      }
      else
      {
        constexpr Eigen::Index c = MatrixTraits<Arg>::columns;
        return make_self_contained<Arg>(std::forward<Arg>(arg).template block<1, c>(e_i, 0));
      }
    }
  }


  /// Return column <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<std::size_t...index, native_eigen_matrix Arg, std::convertible_to<const std::size_t>...runtime_index_t>
    requires (sizeof...(index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_columns<Arg> or ((index + ... + 0) < MatrixTraits<Arg>::columns))
#else
  template<std::size_t...index, typename Arg, typename...runtime_index_t, std::enable_if_t<native_eigen_matrix<Arg> and
    (std::is_convertible_v<runtime_index_t, const std::size_t> and ...) and
    (sizeof...(index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_columns<Arg> or ((index + ... + 0) < MatrixTraits<Arg>::columns)), int> = 0>
#endif
  constexpr decltype(auto)
  column(Arg&& arg, runtime_index_t...i)
  {
    if constexpr (column_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      auto e_i = static_cast<Eigen::Index>((index + ... + (i + ... + 0)));

      if constexpr (dynamic_rows<Arg>)
      {
        Eigen::Index r = row_count(arg);
        return make_self_contained<Arg>(std::forward<Arg>(arg).template block<Eigen::Dynamic, 1>(0, e_i, r, 1));
      }
      else
      {
        constexpr Eigen::Index r = MatrixTraits<Arg>::rows;
        return make_self_contained<Arg>(std::forward<Arg>(arg).template block<r, 1>(0, e_i));
      }
    }
  }


#ifdef __cpp_concepts
  template<fixed_coefficients Coefficients, native_eigen_matrix Arg>
  requires (dynamic_rows<Arg> or Coefficients::dimensions == MatrixTraits<Arg>::rows)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<
    fixed_coefficients<Coefficients> and eigen_matrix<Arg> and
    (dynamic_rows<Arg> or Coefficients::dimensions == MatrixTraits<Arg>::rows), int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg) noexcept
  {
    if constexpr (dynamic_rows<Arg>) assert(row_count(arg) == Coefficients::dimensions);

    if constexpr (Coefficients::axes_only)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return ToEuclideanExpr<Coefficients, Arg>(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<dynamic_coefficients Coefficients, native_eigen_matrix Arg>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<
    coefficients<Coefficients> and eigen_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Coefficients&& c, Arg&& arg) noexcept
  {
    assert(row_count(arg) == c.runtime_dimensions);

    if (c.axes_only)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return ToEuclideanExpr<Coefficients, Arg>(std::forward<Coefficients>(c), std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<fixed_coefficients Coefficients, native_eigen_matrix Arg>
  requires (dynamic_rows<Arg> or Coefficients::dimensions == MatrixTraits<Arg>::rows)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<
    fixed_coefficients<Coefficients> and eigen_matrix<Arg> and
    (dynamic_rows<Arg> or Coefficients::dimensions == MatrixTraits<Arg>::rows), int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg) noexcept
  {
    if constexpr (dynamic_rows<Arg>) assert(row_count(arg) == Coefficients::euclidean_dimensions);

    if constexpr (Coefficients::axes_only)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return FromEuclideanExpr<Coefficients, Arg>(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<dynamic_coefficients Coefficients, native_eigen_matrix Arg>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<
    coefficients<Coefficients> and eigen_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Coefficients&& c, Arg&& arg) noexcept
  {
    assert(row_count(arg) == c.runtime_euclidean_dimensions);

    if (c.axes_only)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return FromEuclideanExpr<Coefficients, Arg>(std::forward<Coefficients>(c), std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<fixed_coefficients Coefficients, native_eigen_matrix Arg>
  requires (dynamic_rows<Arg> or Coefficients::dimensions == MatrixTraits<Arg>::rows)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<
    fixed_coefficients<Coefficients> and eigen_matrix<Arg> and
    (dynamic_rows<Arg> or Coefficients::dimensions == MatrixTraits<Arg>::rows), int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg) noexcept
  {
    if constexpr (dynamic_rows<Arg>) assert(row_count(arg) == Coefficients::dimensions);

    if constexpr (Coefficients::axes_only or identity_matrix<Arg> or zero_matrix<Arg>)
    {
      /// \todo: Add functionality to conditionally wrap zero and identity, depending on wrap min and max.
      return std::forward<Arg>(arg);
    }
    else
    {
      return from_euclidean<Coefficients>(to_euclidean<Coefficients>(std::forward<Arg>(arg)));
    }
  }


#ifdef __cpp_concepts
  template<dynamic_coefficients Coefficients, native_eigen_matrix Arg>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<
    coefficients<Coefficients> and eigen_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Coefficients&& c, Arg&& arg) noexcept
  {
    assert(row_count(arg) == c.runtime_dimensions);

    if (c.axes_only or identity_matrix<Arg> or zero_matrix<Arg>)
    {
      /// \todo: Add functionality to conditionally wrap zero and identity, depending on wrap min and max.
      return std::forward<Arg>(arg);
    }
    else
    {
      return from_euclidean<Coefficients>(to_euclidean<Coefficients>(
        std::forward<Coefficients>(c), std::forward<Arg>(arg)));
    }
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg> requires dynamic_columns<Arg> or column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (dynamic_columns<Arg> or column_vector<Arg>), int> = 0>
#endif
  inline decltype(auto)
  to_diagonal(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;

    if constexpr (dynamic_columns<Arg>) assert(column_count(arg) == 1);

    if constexpr (one_by_one_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg>)
    {
      if constexpr (dynamic_rows<Arg>)
      {
        // Wrap in a DiagonalMatrix to preserve knowledge that the zero matrix is diagonal.
        return DiagonalMatrix {ZeroMatrix<Scalar, 0, 1> {row_count(arg)}};
      }
      else
      {
        constexpr auto dim = MatrixTraits<Arg>::rows;
        return ZeroMatrix<Scalar, dim, dim>();
      }
    }
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr auto constant = constant_coefficient_v<Arg>;

      if constexpr (dynamic_rows<Arg>)
      {
        return DiagonalMatrix {ConstantMatrix<Scalar, constant, 0, 1> {row_count(arg)}};
      }
      else
      {
        return DiagonalMatrix {ConstantMatrix<Scalar, constant, MatrixTraits<Arg>::rows, 1> {}};
      }
    }
    else
    {
      if constexpr (dynamic_columns<Arg>) assert(column_count(arg) == 1);
      return DiagonalMatrix {std::forward<Arg>(arg)};
    }
  }


#ifdef __cpp_concepts
  template<eigen_native_general Arg> requires (dynamic_shape<Arg> or square_matrix<Arg>)
#else
  template<typename Arg, std::enable_if_t<
    eigen_native_general<Arg> and (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;

    constexpr std::size_t dim = dynamic_rows<Arg> ? MatrixTraits<Arg>::columns : MatrixTraits<Arg>::rows;

    if constexpr (identity_matrix<Arg>)
    {
      if constexpr (dim == 0)
      {
        auto rows = row_count(arg);
        assert(rows == column_count(arg));
        return ConstantMatrix<Scalar, 1, 0, 1> {rows};
      }
      else
      {
        if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
        return ConstantMatrix<Scalar, 1, dim, 1> {};
      }
    }
    else if constexpr (zero_matrix<Arg>)
    {
      if constexpr (dim == 0)
      {
        auto rows = row_count(arg);
        assert(rows == column_count(arg));
        return ZeroMatrix<Scalar, 0, 1> {rows};
      }
      else
      {
        if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
        return ZeroMatrix<Scalar, dim, 1> {};
      }
    }
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr auto constant = constant_coefficient_v<Arg>;

      if constexpr (dim == 0)
      {
        auto rows = row_count(arg);
        assert(rows == column_count(arg));
        return ConstantMatrix<Scalar, constant, 0, 1> {row_count(arg)};
      }
      else
      {
        if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
        return ConstantMatrix<Scalar, constant, dim, 1> {};
      }
    }
    else if constexpr (eigen_DiagonalWrapper<Arg>)
    {
      // Note: we assume that the nested matrix reference is not dangling.
      return std::forward<Arg>(arg).diagonal();
    }
    else if constexpr (eigen_SelfAdjointView<Arg> or eigen_TriangularView<Arg>)
    {
      // Note: we assume that the nested matrix reference is not dangling.
      return diagonal_of(std::forward<Arg>(arg).nestedExpression());
    }
    else if constexpr (dim == 1)
    {
      if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
      return std::forward<Arg>(arg);
    }
    else if constexpr (self_contained<decltype(std::forward<Arg>(arg).diagonal())> or std::is_lvalue_reference_v<Arg>)
    {
      return std::forward<Arg>(arg).diagonal();
    }
    else
    {
      return eigen_matrix_t<Scalar, dim, 1> {std::forward<Arg>(arg).diagonal()};
    }
  }


  /**
   * \brief Take the transpose of an eigen native matrix
   * \note The result has the same lifetime as arg.
   */
#ifdef __cpp_concepts
  template<eigen_native_general Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native_general<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    if constexpr (self_adjoint_matrix<Arg> and not complex_number<typename MatrixTraits<Arg>::Scalar>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg>)
    {
      return transpose(ZeroMatrix {std::forward<Arg>(arg)});
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return transpose(ConstantMatrix {std::forward<Arg>(arg)});
    }
    else if constexpr (eigen_SelfAdjointView<Arg>)
    {
      return transpose(SelfAdjointMatrix {std::forward<Arg>(arg)});
    }
    else if constexpr (eigen_TriangularView<Arg>)
    {
      return transpose(TriangularMatrix {std::forward<Arg>(arg)});
    }
    else if constexpr (eigen_DiagonalMatrix<Arg> or eigen_DiagonalWrapper<Arg>)
    {
      return DiagonalMatrix {std::forward<Arg>(arg)};
    }
    else
    {
      static_assert(native_eigen_matrix<Arg>);
      return std::forward<Arg>(arg).transpose();
    }
  }


  /**
   * \brief Take the adjoint of an eigen native matrix
   * \note The result has the same lifetime as arg.
   */
#ifdef __cpp_concepts
  template<eigen_native_general Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native_general<Arg>, int> = 0>
#endif
  inline decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    if constexpr (self_adjoint_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg>)
    {
      return adjoint(ZeroMatrix {std::forward<Arg>(arg)});
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return adjoint(ConstantMatrix {std::forward<Arg>(arg)});
    }
    else if constexpr (eigen_SelfAdjointView<Arg>)
    {
      return adjoint(SelfAdjointMatrix {std::forward<Arg>(arg)});
    }
    else if constexpr (eigen_TriangularView<Arg>)
    {
      return adjoint(TriangularMatrix {std::forward<Arg>(arg)});
    }
    else if constexpr (eigen_DiagonalMatrix<Arg> or eigen_DiagonalWrapper<Arg>)
    {
      if constexpr (complex_number<typename MatrixTraits<Arg>::Scalar>)
      {
        return DiagonalMatrix {std::forward<Arg>(arg).diagonal().conjugate()};
      }
      else
      {
        return DiagonalMatrix {std::forward<Arg>(arg)};
      }
    }
    else
    {
      static_assert(native_eigen_matrix<Arg>);

      if constexpr (complex_number<typename MatrixTraits<Arg>::Scalar>)
      {
        return std::forward<Arg>(arg).adjoint();
      }
      else
      {
        return std::forward<Arg>(arg).transpose();
      }
    }
  }


#ifdef __cpp_concepts
  template<eigen_native_general Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_native_general<Arg> and (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  inline auto
  determinant(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;

    if constexpr (constant_matrix<Arg>)
    {
      if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
      return static_cast<Scalar>(0);
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return static_cast<Scalar>(1);
    }
    else if constexpr (one_by_one_matrix<Arg>)
    {
      return get_element(arg, 0, 0);
    }
    else if constexpr (eigen_SelfAdjointView<Arg> or eigen_TriangularView<Arg>)
    {
      return determinant(make_native_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (eigen_DiagonalMatrix<Arg> or eigen_DiagonalWrapper<Arg>)
    {
      return std::forward<Arg>(arg).diagonal().prod();
    }
    else
    {
      if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
      return arg.determinant();
    }
  }


#ifdef __cpp_concepts
  template<eigen_native_general Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native_general<Arg> and (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto
  trace(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;

    if constexpr (zero_matrix<Arg>)
    {
      if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
      return static_cast<Scalar>(0);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
      return static_cast<Scalar>(constant_coefficient_v<Arg> * row_count(arg));
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return static_cast<Scalar>(row_count(arg));
    }
    else if constexpr (one_by_one_matrix<Arg>)
    {
      return get_element(arg, 0, 0);
    }
    else if constexpr (eigen_SelfAdjointView<Arg> or eigen_TriangularView<Arg>)
    {
      return std::forward<Arg>(arg).nestedExpression().trace();
    }
    else if constexpr (eigen_DiagonalMatrix<Arg> or eigen_DiagonalWrapper<Arg>)
    {
      return std::forward<Arg>(arg).diagonal().sum();
    }
    else
    {
      if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
      return arg.trace();
    }
  }


  namespace detail
  {
    template<typename Arg, typename U>
    inline decltype(auto)
    rank_update_diag_impl(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
    {
      using M = eigen_matrix_t<typename MatrixTraits<Arg>::Scalar, 1, 1>;

      if constexpr ((row_vector<Arg> or column_vector<Arg> or row_vector<U>) and
        (dynamic_rows<Arg> or row_vector<Arg>) and
        (dynamic_columns<Arg> or column_vector<Arg>) and
        (dynamic_rows<U> or row_vector<U>) and
        element_settable<Arg, 2>)
      {
        auto e = std::sqrt(trace(arg) * trace(arg) + alpha * trace(u * adjoint(u)));

        if constexpr (std::is_lvalue_reference_v<Arg> and not std::is_const_v<std::remove_reference_t<Arg>>)
        {
          set_element(arg, e, 0, 0);
          return (arg);
        }
        else
        {
          return OpenKalman::make_native_matrix<M>(e);
        }
      }
      else
      {
        auto d = (diagonal_of(arg).array().square() + alpha * diagonal_of(u).array().square()).sqrt().matrix();

        if constexpr (std::is_lvalue_reference_v<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
          writable<Arg>)
        {
          if constexpr (eigen_DiagonalMatrix<Arg>)
          {
            arg.diagonal() = std::move(d);
          }
          else if constexpr (eigen_DiagonalWrapper<Arg>)
          {
            // DiagonalWrapper does not (but maybe should) have a non-const .diagonal() member function.
            const_cast<nested_matrix_t<Arg>&>(arg.diagonal()) = std::move(d);
          }
          else
          {
            arg = to_diagonal(std::move(d));
          }
          return (arg);
        }
        else
        {
          return to_diagonal(std::move(d));
        }
      }
    }


    template<int UpLo, typename Arg, typename U>
    inline Arg&
    rank_update_tri_impl(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;

      for (Eigen::Index i = 0; i < Eigen::Index(MatrixTraits<U>::columns); ++i)
      {
        if (Eigen::internal::llt_inplace<Scalar, UpLo>::rankUpdate(arg, u.col(i), alpha) >= 0)
          throw (std::runtime_error("rank_update_triangular: product is not positive definite"));
      }

      return arg;
    }

  } // namespace detail


  /**
   * \brief Do a rank update on a native Eigen matrix, treating it as a self-adjoint matrix.
   * \details If arg is an lvalue reference and is writable, it will be updated in place. You can tell this has
   * happened if the return value is an lvalue reference.
   * \tparam t Whether to use the upper triangle elements (TriangleType::upper), lower triangle elements
   * (TriangleType::lower) or diagonal elements (TriangleType::diagonal).
   * \tparam Arg The matrix to be rank updated.
   * \tparam U The update vector or matrix.
   * \returns an updated native, writable matrix.
   */
#ifdef __cpp_concepts
  template<TriangleType t, eigen_native_general Arg, typename U>
  requires
    (not eigen_TriangularView<Arg>) and native_eigen_matrix<native_matrix_t<U>> and
    (not eigen_SelfAdjointView<Arg> or requires { t == self_adjoint_triangle_type_of_v<Arg>; }) and
    (dynamic_rows<U> or dynamic_rows<Arg> or MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<TriangleType t, typename Arg, typename U, std::enable_if_t<eigen_native_general<Arg> and
    (not eigen_TriangularView<Arg>) and native_eigen_matrix<native_matrix_t<U>> and
    (not eigen_SelfAdjointView<Arg> or
      (t == TriangleType::upper ? upper_self_adjoint_matrix<Arg> : lower_self_adjoint_matrix<Arg>)) and
    (dynamic_rows<U> or dynamic_rows<Arg> or MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline decltype(auto)
  rank_update_self_adjoint(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    if constexpr (dynamic_rows<Arg> or dynamic_rows<U>) assert(row_count(arg) == row_count(u));

    if constexpr (t == TriangleType::diagonal or (((diagonal_matrix<Arg> and diagonal_matrix<U>) or
        row_vector<Arg> or column_vector<Arg> or row_vector<U>) and
      (dynamic_rows<Arg> or row_vector<Arg>) and
      (dynamic_columns<Arg> or column_vector<Arg>) and
      (dynamic_rows<U> or row_vector<U>)))
    {
      return detail::rank_update_diag_impl(std::forward<Arg>(arg), u, alpha);
    }
    else
    {
      constexpr auto UpLo = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;

      if constexpr (std::is_lvalue_reference_v<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
        writable<Arg>)
      {
        if constexpr (eigen_SelfAdjointView<Arg>)
        {
          return arg.template rankUpdate(u, alpha);
        }
        else
        {
          arg.template selfadjointView<UpLo>().template rankUpdate(u, alpha);
          return arg;
        }
      }
      else if constexpr (writable<Arg>) // arg is a writable rvalue reference
      {
        if constexpr (eigen_SelfAdjointView<Arg>)
        {
          arg.template rankUpdate(u, alpha);
          auto& a = arg.nestedExpression();
          return SelfAdjointMatrix<decltype((a)), t> {a};
        }
        else
        {
          arg.template selfadjointView<UpLo>().template rankUpdate(u, alpha);
          return SelfAdjointMatrix<Arg, t> {std::forward<Arg>(arg)};
        }
      }
      else // arg is not writable and must be copied
      {
        using A = native_matrix_t<Arg>;

        if constexpr (eigen_SelfAdjointView<Arg>)
        {
          A a {arg.nestedExpression()};
          a.template selfadjointView<UpLo>().template rankUpdate(u, alpha);
          return SelfAdjointMatrix<A, t> {std::move(a)};
        }
        else
        {
          A a {std::forward<Arg>(arg)};
          a.template selfadjointView<UpLo>().template rankUpdate(u, alpha);
          return SelfAdjointMatrix<A, t> {std::move(a)};
        }
      }
    }
  }


  /**
   * \brief Do a rank update on a native Eigen matrix, treating it as a triangular matrix.
   * \details If arg is an lvalue reference and is writable, it will be updated in place. You can tell this has
   * happened if the return value is an lvalue reference.
   * \tparam t Whether to use the upper triangle elements (TriangleType::upper), lower triangle elements
   * (TriangleType::lower) or diagonal elements (TriangleType::diagonal).
   * \tparam Arg The matrix to be rank updated.
   * \tparam U The update vector or matrix.
   * \returns an updated native, writable matrix.
   */
#ifdef __cpp_concepts
  template<TriangleType t, eigen_native_general Arg, typename U>
  requires
    (not eigen_SelfAdjointView<Arg>) and native_eigen_matrix<native_matrix_t<U>> and
    (not eigen_TriangularView<Arg> or requires { t == triangle_type_of_v<Arg>; }) and
    (dynamic_rows<U> or dynamic_rows<Arg> or MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<TriangleType t, typename Arg, typename U, std::enable_if_t<eigen_native_general<Arg> and
    (not eigen_SelfAdjointView<Arg>) and native_eigen_matrix<native_matrix_t<U>> and
    (not eigen_TriangularView<Arg> or
      (t == TriangleType::upper ? upper_triangular_matrix<Arg> : lower_triangular_matrix<Arg>)) and
    (dynamic_rows<U> or dynamic_rows<Arg> or MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline decltype(auto)
  rank_update_triangular(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    if constexpr (dynamic_rows<Arg> or dynamic_rows<U>) assert(row_count(arg) == row_count(u));

    if constexpr (t == TriangleType::diagonal or (((diagonal_matrix<Arg> and diagonal_matrix<U>) or
        row_vector<Arg> or column_vector<Arg> or row_vector<U>) and
      (dynamic_rows<Arg> or row_vector<Arg>) and
      (dynamic_columns<Arg> or column_vector<Arg>) and
      (dynamic_rows<U> or row_vector<U>)))
    {
      return detail::rank_update_diag_impl(std::forward<Arg>(arg), u, alpha);
    }
    else
    {
      constexpr auto UpLo = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;

      if constexpr (std::is_lvalue_reference_v<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
        writable<Arg>)
      {
        if constexpr (eigen_TriangularView<Arg>)
        {
          detail::rank_update_tri_impl<UpLo>(arg.nestedExpression(), u, alpha);
          return arg;
        }
        else
        {
          return detail::rank_update_tri_impl<UpLo>(arg, u, alpha);
        }
      }
      else if constexpr (writable<Arg>) // arg is a writable rvalue reference
      {
        if constexpr (eigen_TriangularView<Arg>)
        {
          auto& a = detail::rank_update_tri_impl<UpLo>(arg.nestedExpression(), u, alpha);
          return TriangularMatrix<decltype((a)), t> {a};
        }
        else
        {
          detail::rank_update_tri_impl<UpLo>(arg, u, alpha);
          return TriangularMatrix<Arg, t> {std::forward<Arg>(arg)};
        }
      }
      else // arg is not writable and must be copied
      {
        using A = native_matrix_t<Arg>;
        A a;
        a.template triangularView<UpLo>() = std::forward<Arg>(arg);
        detail::rank_update_tri_impl<UpLo>(a, u, alpha);
        return TriangularMatrix<A, t> {std::move(a)};
      }
    }
  }


  /**
   * \brief Do a rank update on a non-diagonal Eigen self-adjoint matrix.
   * \details If arg is an lvalue reference and is writable, it will be updated in place. You can tell this has
   * happened if the return value is an lvalue reference.
   * \returns a self_adjoint_matrix (which is arg if it is a non-const lvalue reference)
   */
#ifdef __cpp_concepts
  template<self_adjoint_matrix Arg, typename U>
  requires (not diagonal_matrix<Arg>) and eigen_native_general<Arg> and
    native_eigen_matrix<native_matrix_t<U>> and
    (dynamic_rows<U> or dynamic_rows<Arg> or MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<self_adjoint_matrix<Arg> and (not diagonal_matrix<Arg>) and
    eigen_native_general<Arg> and native_eigen_matrix<native_matrix_t<U>> and
    (dynamic_rows<U> or dynamic_rows<Arg> or MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline decltype(auto)
  rank_update(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    constexpr TriangleType t = self_adjoint_triangle_type_of_v<Arg>;

    return rank_update_self_adjoint<t>(std::forward<Arg>(arg), u, alpha);
  }


  /**
   * \brief Do a rank update on an Eigen triangular matrix.
   * \details If arg is an lvalue reference and is writable, it will be updated in place. You can tell this has
   * happened if the return value is an lvalue reference.
   * \returns a triangular_matrix (which is arg if it is a non-const lvalue reference)
   */
#ifdef __cpp_concepts
  template<triangular_matrix Arg, typename U>
  requires eigen_native_general<Arg> and native_eigen_matrix<native_matrix_t<U>> and
    (dynamic_rows<U> or dynamic_rows<Arg> or MatrixTraits<U>::rows == MatrixTraits<Arg>::rows)
#else
  template<typename Arg, typename U, std::enable_if_t<triangular_matrix<Arg> and
    eigen_native_general<Arg> and native_eigen_matrix<native_matrix_t<U>> and
    (dynamic_rows<U> or dynamic_rows<Arg> or MatrixTraits<U>::rows == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline decltype(auto)
  rank_update(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    constexpr TriangleType t = triangle_type_of_v<Arg>;

    return rank_update_triangular<t>(std::forward<Arg>(arg), u, alpha);
  }


  /**
   * \brief Do a rank update on an Eigen native matrix if it could be a 1-by-1 matrix.
   * \details If arg is an lvalue reference and is writable, it will be updated in place. You can tell this has
   * happened if the return value is an lvalue reference.
   */
#ifdef __cpp_concepts
  template<native_eigen_matrix Arg, typename U>
  requires (not self_adjoint_matrix<Arg>) and (not triangular_matrix<Arg>) and native_eigen_matrix<native_matrix_t<U>> and
    (dynamic_rows<U> or dynamic_rows<Arg> or MatrixTraits<U>::rows == MatrixTraits<Arg>::rows) and
    (dynamic_rows<Arg> or row_vector<Arg>) and (dynamic_columns<Arg> or column_vector<Arg>) and
    (dynamic_rows<U> or row_vector<U>)
#else
  template<typename Arg, typename U, std::enable_if_t<native_eigen_matrix<Arg> and
    (not self_adjoint_matrix<Arg>) and (not triangular_matrix<Arg>) and native_eigen_matrix<native_matrix_t<U>> and
    (dynamic_rows<U> or dynamic_rows<Arg> or MatrixTraits<U>::rows == MatrixTraits<Arg>::rows) and
    (dynamic_rows<Arg> or row_vector<Arg>) and (dynamic_columns<Arg> or column_vector<Arg>) and
    (dynamic_rows<U> or row_vector<U>), int> = 0>
#endif
  inline decltype(auto)
  rank_update(Arg&& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    constexpr TriangleType t = TriangleType::diagonal;

    if constexpr (dynamic_rows<Arg>) assert(row_count(arg) == 1);
    if constexpr (dynamic_columns<Arg>) assert(column_count(arg) == 1);

    if constexpr (std::is_lvalue_reference_v<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      writable<Arg>)
    {
      return rank_update_triangular<t>(arg, u, alpha);
    }
    else
    {
      using A = eigen_matrix_t<typename MatrixTraits<Arg>::Scalar, 1, 1>;
      A a {std::forward<Arg>(arg)};
      return rank_update_triangular<t>(a, u, alpha);
    }
  }


  /**
   * \brief Solve the equation AX = B for X. A is an invertible square matrix. (Does not check that A is invertible.)
   * \details Uses the square LU decomposition.
   */
#ifdef __cpp_concepts
  template<native_eigen_matrix A, eigen_matrix B> requires (dynamic_shape<A> or square_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<B> or MatrixTraits<A>::rows == MatrixTraits<B>::rows)
#else
  template<typename A, typename B, std::enable_if_t<native_eigen_matrix<A> and eigen_matrix<B> and
    (dynamic_shape<A> or square_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<B> or MatrixTraits<A>::rows == MatrixTraits<B>::rows), int> = 0>
#endif
  constexpr auto
  solve(A&& a, B&& b)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;

    constexpr std::size_t dim = dynamic_rows<A> ?
      (dynamic_rows<B> ? MatrixTraits<A>::columns: MatrixTraits<B>::rows) : MatrixTraits<A>::rows;

    using M = eigen_matrix_t<Scalar, dim, MatrixTraits<B>::columns>;

    if constexpr (dynamic_shape<A>) assert(row_count(a) == column_count(a));
    if constexpr (dynamic_rows<A> or dynamic_rows<B>) assert(row_count(a) == row_count(b));

    if constexpr (zero_matrix<B>)
    {
      return std::forward<B>(b);
    }
    else if constexpr (zero_matrix<A>)
    {
      if constexpr (dim == 0)
      {
        if constexpr (dynamic_columns<B>)
          return ZeroMatrix<Scalar, 0, 0> {row_count(b), column_count(b)};
        else
          return ZeroMatrix<Scalar, 0, MatrixTraits<B>::columns> {row_count(b)};
      }
      else
      {
        if constexpr (dynamic_columns<B>)
          return ZeroMatrix<Scalar, dim, 0> {column_count(b)};
        else
          return ZeroMatrix<Scalar, dim, MatrixTraits<B>::columns> {};
      }
    }
    else if constexpr (constant_matrix<A>)
    {
      return solve(ConstantMatrix {std::forward<A>(a)}, std::forward<B>(b));
    }
    else if constexpr (dim == 1)
    {
      Scalar s = trace(a);
      if (s == 0)
      {
        if constexpr (dynamic_columns<B>)
          return M {ZeroMatrix<Scalar, dim, 0> {column_count(b)}};
        else
          return M {ZeroMatrix<Scalar, dim, MatrixTraits<B>::columns> {}};
      }
      else
      {
        return M {std::forward<B>(b) / s};
      }
    }
    else
    {
      return Eigen::PartialPivLU<eigen_matrix_t<Scalar, dim, dim>>{a}.solve(b);
    }
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors.
#ifdef __cpp_concepts
  template<eigen_native_general Arg>
#else
  template<typename Arg, std::enable_if_t<native_eigen_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    if constexpr (column_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg>)
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;

      if constexpr (dynamic_rows<Arg>)
        return ZeroMatrix<Scalar, 0, 1> {row_count(arg)};
      else
        return ZeroMatrix<Scalar, MatrixTraits<Arg>::rows, 1> {};
    }
    else if constexpr (constant_matrix<Arg>)
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      constexpr auto constant = constant_coefficient_v<Arg>;

      if constexpr (dynamic_rows<Arg>)
        return ConstantMatrix<Scalar, constant, 0, 1> {row_count(arg)};
      else
        return ConstantMatrix<Scalar, constant, MatrixTraits<Arg>::rows, 1> {};
    }
    else if constexpr (diagonal_matrix<Arg>)
    {
      return diagonal_of(std::forward<Arg>(arg));
    }
    else if constexpr (dynamic_columns<Arg>)
    {
      using N = eigen_matrix_t<typename MatrixTraits<Arg>::Scalar, MatrixTraits<Arg>::rows, 1>;

      if (column_count(arg) == 1)
        return N {std::forward<Arg>(arg)};
      else
        return N {arg.rowwise().sum() / column_count(arg)};
    }
    else
    {
      return make_self_contained(arg.rowwise().sum() / column_count(arg));
    }
  }


  /// Create a row vector by taking the mean of each column in a set of row vectors.
#ifdef __cpp_concepts
  template<eigen_native_general Arg>
#else
  template<typename Arg, std::enable_if_t<native_eigen_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_rows(Arg&& arg) noexcept
  {
    if constexpr (row_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg>)
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;

      if constexpr (dynamic_columns<Arg>)
        return ZeroMatrix<Scalar, 1, 0> {column_count(arg)};
      else
        return ZeroMatrix<Scalar, 1, MatrixTraits<Arg>::columns> {};
    }
    else if constexpr (constant_matrix<Arg>)
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      constexpr auto constant = constant_coefficient_v<Arg>;

      if constexpr (dynamic_columns<Arg>)
        return ConstantMatrix<Scalar, constant, 1, 0> {column_count(arg)};
      else
        return ConstantMatrix<Scalar, constant, 1, MatrixTraits<Arg>::columns> {};
    }
    else if constexpr (diagonal_matrix<Arg>)
    {
      return transpose(diagonal_of(std::forward<Arg>(arg)));
    }
    else if constexpr (dynamic_rows<Arg>)
    {
      using N = eigen_matrix_t<typename MatrixTraits<Arg>::Scalar, 1, MatrixTraits<Arg>::rows>;

      if (row_count(arg) == 1)
        return N {std::forward<Arg>(arg)};
      else
        return N {arg.colwise().sum() / row_count(arg)};
    }
    else
    {
      return make_self_contained(arg.colwise().sum() / row_count(arg));
    }
  }


  namespace detail
  {
    template<typename A>
    constexpr auto
    QR_decomp_impl(A&& a)
    {
      using Scalar = typename MatrixTraits<A>::Scalar;
      constexpr auto rows = MatrixTraits<A>::rows;
      constexpr auto cols = MatrixTraits<A>::columns;
      using MatrixType = Eigen3::eigen_matrix_t<Scalar, rows, cols>;
      using ResultType = Eigen3::eigen_matrix_t<Scalar, cols, cols>;

      Eigen::HouseholderQR<MatrixType> QR {std::forward<A>(a)};

      if constexpr (dynamic_columns<A>)
      {
        auto rt_cols = column_count(a);

        ResultType ret {rt_cols, rt_cols};

        if constexpr (dynamic_rows<A>)
        {
          auto rt_rows = row_count(a);

          if (rt_rows < rt_cols)
            ret << QR.matrixQR().topRows(rt_rows),
              Eigen3::eigen_matrix_t<Scalar, 0, 0>::Zero(rt_cols - rt_rows, rt_cols);
          else
            ret = QR.matrixQR().topRows(rt_cols);
        }
        else
        {
          if (rows < rt_cols)
            ret << QR.matrixQR().template topRows<rows>(),
              Eigen3::eigen_matrix_t<Scalar, 0, 0>::Zero(rt_cols - rows, rt_cols);
          else
            ret = QR.matrixQR().topRows(rt_cols);
        }

        return ret;
      }
      else
      {
        ResultType ret;

        if constexpr (dynamic_rows<A>)
        {
          auto rt_rows = row_count(a);

          if (rt_rows < cols)
            ret << QR.matrixQR().topRows(rt_rows), Eigen3::eigen_matrix_t<Scalar, 0, 0>::Zero(cols - rt_rows, cols);
          else
            ret = QR.matrixQR().template topRows<cols>();
        }
        else
        {
          if constexpr (rows < cols)
            ret << QR.matrixQR().template topRows<rows>(), Eigen3::eigen_matrix_t<Scalar, cols - rows, cols>::Zero();
          else
            ret = QR.matrixQR().template topRows<cols>();
        }

        return ret;
      }
    }
  }


  /**
   * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * Returns L as a lower-triangular matrix.
   */
#ifdef __cpp_concepts
  template<native_eigen_matrix A>
#else
  template<typename A, std::enable_if_t<native_eigen_matrix<A>, int> = 0>
#endif
  constexpr auto
  LQ_decomposition(A&& a)
  {
    if constexpr (lower_triangular_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else
    {
      using Scalar = typename MatrixTraits<A>::Scalar;
      constexpr auto rows = MatrixTraits<A>::rows;
      using ResultType = Eigen3::eigen_matrix_t<Scalar, rows, rows>;
      using TType = typename MatrixTraits<ResultType>::template TriangularMatrixFrom<TriangleType::lower>;

      ResultType ret = adjoint(detail::QR_decomp_impl(adjoint(std::forward<A>(a))));

      return MatrixTraits<TType>::make(std::move(ret));
    }
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<native_eigen_matrix A>
#else
  template<typename A, std::enable_if_t<native_eigen_matrix<A>, int> = 0>
#endif
  constexpr auto
  QR_decomposition(A&& a)
  {
    if constexpr (upper_triangular_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else
    {
      using Scalar = typename MatrixTraits<A>::Scalar;
      constexpr auto cols = MatrixTraits<A>::columns;
      using ResultType = Eigen3::eigen_matrix_t<Scalar, cols, cols>;
      using TType = typename MatrixTraits<ResultType>::template TriangularMatrixFrom<TriangleType::upper>;

      ResultType ret = detail::QR_decomp_impl(std::forward<A>(a));

      return MatrixTraits<TType>::make(std::move(ret));
    }
  }


  /**
   * \brief Concatenate one or more Eigen::MatrixBase objects vertically.
   * \todo Add special cases for all ZeroMatrix and all ConstantMatrix
   */
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires
#else
  template<typename V, typename ... Vs, std::enable_if_t<
#endif
    ((eigen_matrix<V> or eigen_self_adjoint_expr<V> or eigen_triangular_expr<V> or
      eigen_diagonal_expr<V> or from_euclidean_expr<V>) and ... and
    (eigen_matrix<Vs> or eigen_self_adjoint_expr<Vs> or eigen_triangular_expr<Vs> or
      eigen_diagonal_expr<Vs> or from_euclidean_expr<Vs>)) and
    (not (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)) and
    ((dynamic_columns<V> or dynamic_columns<Vs> or MatrixTraits<V>::columns == MatrixTraits<Vs>::columns) and ...)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_vertical(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      using Scalar = typename MatrixTraits<V>::Scalar;

      if constexpr ((dynamic_columns<V> and ... and dynamic_columns<Vs>))
      {
        auto cols = column_count(v);
        assert(((cols == column_count(vs)) and ...));

        if constexpr ((dynamic_rows<V> or ... or dynamic_rows<Vs>))
        {
          auto rows = (row_count(v) + ... + row_count(vs));
          Eigen3::eigen_matrix_t<Scalar, 0, 0> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
        else
        {
          constexpr auto rows = (MatrixTraits<V>::rows + ... + MatrixTraits<Vs>::rows);
          Eigen3::eigen_matrix_t<Scalar, rows, 0> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
      }
      else
      {
        constexpr auto cols = std::max({MatrixTraits<V>::columns, MatrixTraits<Vs>::columns...});

        static_assert(((dynamic_columns<V> or MatrixTraits<V>::columns == cols) and ... and
          (dynamic_columns<Vs> or MatrixTraits<Vs>::columns == cols)));

        if constexpr ((dynamic_rows<V> or ... or dynamic_rows<Vs>))
        {
          auto rows = (row_count(v) + ... + row_count(vs));
          Eigen3::eigen_matrix_t<Scalar, 0, cols> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
        else
        {
          constexpr auto rows = (MatrixTraits<V>::rows + ... + MatrixTraits<Vs>::rows);
          Eigen3::eigen_matrix_t<Scalar, rows, cols> m;
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
      }
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  /**
   * \brief Concatenate one or more Eigen::MatrixBase objects horizontally.
   * \todo Add special cases for all ZeroMatrix and all ConstantMatrix
   */
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires
#else
  template<typename V, typename ... Vs, std::enable_if_t<
#endif
    ((eigen_matrix<V> or eigen_self_adjoint_expr<V> or eigen_triangular_expr<V> or
      eigen_diagonal_expr<V> or from_euclidean_expr<V>) and ... and
    (eigen_matrix<Vs> or eigen_self_adjoint_expr<Vs> or eigen_triangular_expr<Vs> or
      eigen_diagonal_expr<Vs> or from_euclidean_expr<Vs>)) and
    (not (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)) and
    ((dynamic_rows<V> or dynamic_rows<Vs> or MatrixTraits<V>::rows == MatrixTraits<Vs>::rows) and ...)
#ifndef __cpp_concepts
      , int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_horizontal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      using Scalar = typename MatrixTraits<V>::Scalar;

      if constexpr ((dynamic_rows<V> and ... and dynamic_rows<Vs>))
      {
        auto rows = row_count(v);
        assert(((rows == row_count(vs)) and ...));

        if constexpr ((dynamic_columns<V> or ... or dynamic_columns<Vs>))
        {
          auto cols = (column_count(v) + ... + column_count(vs));
          Eigen3::eigen_matrix_t<Scalar, 0, 0> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
        else
        {
          constexpr auto cols = (MatrixTraits<V>::columns + ... + MatrixTraits<Vs>::columns);
          Eigen3::eigen_matrix_t<Scalar, 0, cols> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
      }
      else
      {
        constexpr auto rows = std::max({MatrixTraits<V>::rows, MatrixTraits<Vs>::rows...});

        static_assert(((dynamic_rows<V> or MatrixTraits<V>::rows == rows) and ... and
          (dynamic_rows<Vs> or MatrixTraits<Vs>::rows == rows)));

        if constexpr ((dynamic_columns<V> or ... or dynamic_columns<Vs>))
        {
          auto cols = (column_count(v) + ... + column_count(vs));
          Eigen3::eigen_matrix_t<Scalar, rows, 0> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
        else
        {
          constexpr auto cols = (MatrixTraits<V>::columns + ... + MatrixTraits<Vs>::columns);
          Eigen3::eigen_matrix_t<Scalar, rows, cols> m;
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
      }
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  namespace detail
  {
    // Concatenate one or more Eigen::MatrixBase objects diagonally.
    template<typename M, typename ... Vs, std::size_t ... ints>
    void concatenate_diagonal_impl(M& m, const std::tuple<Vs...>& vs_tup, std::index_sequence<ints...>)
    {
      using Scalar = typename MatrixTraits<M>::Scalar;
      constexpr auto dim = sizeof...(Vs);

      ((m << std::get<0>(vs_tup)), ..., [] (const auto& vs_tup)
      {
        constexpr auto row = (ints + 1) / dim;
        constexpr auto col = (ints + 1) % dim;
        using Vs_row = std::tuple_element_t<row, std::tuple<Vs...>>;
        using Vs_col = std::tuple_element_t<col, std::tuple<Vs...>>;

        if constexpr (row == col)
        {
          return std::get<row>(vs_tup);
        }
        else if constexpr (dynamic_rows<Vs_row>)
        {
          if constexpr (dynamic_columns<Vs_col>)
            return ZeroMatrix<Scalar, 0, 0> {row_count(std::get<row>(vs_tup)), column_count(std::get<col>(vs_tup))};
          else
            return ZeroMatrix<Scalar, 0, MatrixTraits<Vs_col>::columns> {row_count(std::get<row>(vs_tup))};
        }
        else
        {
          if constexpr (dynamic_columns<Vs_col>)
            return ZeroMatrix<Scalar, MatrixTraits<Vs_row>::rows, 0> {column_count(std::get<col>(vs_tup))};
          else
            return ZeroMatrix<Scalar, MatrixTraits<Vs_row>::rows, MatrixTraits<Vs_col>::columns> {};
        }
      }(vs_tup));
    }
  }


  /**
   * \brief Concatenate one or more Eigen::MatrixBase objects diagonally.
   * \todo Add special cases for all ZeroMatrix and all ConstantMatrix
   */
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires
#else
  template<typename V, typename ... Vs, std::enable_if_t<
#endif
    ((eigen_matrix<V> or eigen_self_adjoint_expr<V> or eigen_triangular_expr<V> or
      eigen_diagonal_expr<V> or from_euclidean_expr<V>) and ... and
    (eigen_matrix<Vs> or eigen_self_adjoint_expr<Vs> or eigen_triangular_expr<Vs> or
      eigen_diagonal_expr<Vs> or from_euclidean_expr<Vs>)) and
    (not (diagonal_matrix<V> and ... and diagonal_matrix<Vs>)) and
    (not (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)) and
    (not (eigen_self_adjoint_expr<V> and ... and eigen_self_adjoint_expr<Vs>)) and
    (not (eigen_triangular_expr<V> and ... and eigen_triangular_expr<Vs>))
#ifndef __cpp_concepts
    , int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      using Scalar = typename MatrixTraits<V>::Scalar;
      auto seq = std::make_index_sequence<(sizeof...(vs) + 1) * (sizeof...(vs) + 1) - 1> {};

      if constexpr ((dynamic_rows<V> or ... or dynamic_rows<Vs>))
      {
        if constexpr ((dynamic_columns<V> or ... or dynamic_columns<Vs>))
        {
          auto rows = (row_count(v) + ... + row_count(vs));
          auto cols = (column_count(v) + ... + column_count(vs));
          eigen_matrix_t<Scalar, 0, 0> m {rows, cols};
          detail::concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...), seq);
          return m;
        }
        else
        {
          auto rows = (row_count(v) + ... + row_count(vs));
          constexpr auto cols = (MatrixTraits<V>::columns + ... + MatrixTraits<Vs>::columns);
          eigen_matrix_t<Scalar, 0, cols> m {rows, cols};
          detail::concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...), seq);
          return m;
        }
      }
      else
      {
        if constexpr ((dynamic_columns<V> or ... or dynamic_columns<Vs>))
        {
          constexpr auto rows = (MatrixTraits<V>::rows + ... + MatrixTraits<Vs>::rows);
          auto cols = (column_count(v) + ... + column_count(vs));
          eigen_matrix_t<Scalar, rows, 0> m {rows, cols};
          detail::concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...), seq);
          return m;
        }
        else
        {
          constexpr auto rows = (MatrixTraits<V>::rows + ... + MatrixTraits<Vs>::rows);
          constexpr auto cols = (MatrixTraits<V>::columns + ... + MatrixTraits<Vs>::columns);
          eigen_matrix_t<Scalar, rows, cols> m;
          detail::concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...), seq);
          return m;
        }
      }
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  namespace detail
  {
    /// Make a tuple containing an Eigen matrix (general case).
    template<typename F, typename RC, typename CC, typename Arg>
    auto
    make_split_tuple(Arg&& arg)
    {
      auto val = F::template call<RC, CC>(std::forward<Arg>(arg));
      return std::tuple<const decltype(val)> {std::move(val)};
    }


    /// Make a tuple containing an Eigen::Block.
    template<typename F, typename RC, typename CC, typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    auto
    make_split_tuple(Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& arg)
    {
      auto b = [](auto& arg) {
        using NonConstBlock = Eigen::Block<std::remove_const_t<XprType>, BlockRows, BlockCols, InnerPanel>;

        // A const_cast is necessary, because a const Eigen::Block cannot be inserted into a tuple.
        auto& xpr = const_cast<std::remove_const_t<XprType>&>(arg.nestedExpression());

        if constexpr (BlockRows == Eigen::Dynamic or BlockCols == Eigen::Dynamic)
          return NonConstBlock(xpr, arg.startRow(), arg.startCol(), row_count(arg), column_count(arg));
        else
          return NonConstBlock(xpr, arg.startRow(), arg.startCol());
      } (arg);

      auto val = F::template call<RC, CC>(std::move(b));
      return std::tuple<const decltype(val)> {std::move(val)};
    }

  }


  /**
   * \brief Split a matrix vertically.
   * \tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
   * \tparam euclidean Whether coefficients RC and RCs are transformed to Euclidean space.
   * \tparam RC Coefficients for the first cut.
   * \tparam RCs Coefficients for each of the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<typename F, bool euclidean, typename RC, typename...RCs, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or ((euclidean ? RC::euclidean_dimensions : RC::dimensions) + ... +
    (euclidean ? RCs::euclidean_dimensions : RCs::dimensions)) <= MatrixTraits<Arg>::rows)
#else
  template<typename F, bool euclidean, typename RC, typename...RCs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (dynamic_rows<Arg> or ((euclidean ? RC::euclidean_dimensions : RC::dimensions) + ... +
      (euclidean ? RCs::euclidean_dimensions : RCs::dimensions)) <= MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    // \todo Can g be replaced by make_self_contained?
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg>)
        return std::forward<decltype(m)>(m);
      else
        return make_native_matrix(std::forward<decltype(m)>(m));
    };

    using CC = Axes<MatrixTraits<Arg>::columns>;
    constexpr Eigen::Index dim1 = euclidean ? RC::euclidean_dimensions : RC::dimensions;

    if constexpr (sizeof...(RCs) > 0)
    {
      if constexpr (dynamic_rows<Arg>)
      {
        Eigen::Index dim2 = row_count(arg) - dim1;
        return std::tuple_cat(
          detail::make_split_tuple<F, RC, CC>(g(arg.topRows(dim1))),
          split_vertical<F, euclidean, RCs...>(g(arg.bottomRows(dim2))));
      }
      else
      {
        constexpr Eigen::Index dim2 = MatrixTraits<Arg>::rows - dim1;
        return std::tuple_cat(
          detail::make_split_tuple<F, RC, CC>(g(arg.template topRows<dim1>())),
          split_vertical<F, euclidean, RCs...>(g(arg.template bottomRows<dim2>())));
      }
    }
    else if constexpr (dim1 < MatrixTraits<Arg>::rows)
    {
      if constexpr (dynamic_rows<Arg>)
        return detail::make_split_tuple<F, RC, CC>(g(arg.topRows(dim1)));
      else
        return detail::make_split_tuple<F, RC, CC>(g(arg.template topRows<dim1>()));
    }
    else
    {
      return detail::make_split_tuple<F, RC, CC>(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Split a matrix vertically (case in which there is no split).
   */
#ifdef __cpp_concepts
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false, eigen_matrix Arg>
#else
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false,
    typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return std::tuple {};
  }


  /**
   * \brief Split a matrix vertically.
   * \tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
   * \tparam RCs Coefficients for each of the cuts.
   */
#ifdef __cpp_concepts
  template<typename F, coefficients RC, coefficients...RCs, eigen_matrix Arg>
  requires (not coefficients<F>) and
  (dynamic_rows<Arg> or (RC::dimensions + ... + RCs::dimensions) <= MatrixTraits<Arg>::rows)
#else
  template<typename F, typename RC, typename...RCs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    not coefficients<F> and (coefficients<RC> and ... and coefficients<RCs>) and
      (dynamic_rows<Arg> or (RC::dimensions + ... + RCs::dimensions) <= MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<F, false, RC, RCs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix vertically.
   * \tparam RC Coefficients for the first cut.
   * \tparam RCs Coefficients for the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<coefficients RC, coefficients...RCs, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or (RC::dimensions + ... + RCs::dimensions) <= MatrixTraits<Arg>::rows)
#else
  template<typename RC, typename...RCs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (coefficients<RC> and ... and coefficients<RCs>) and
    (dynamic_rows<Arg> or (RC::dimensions + ... + RCs::dimensions) <= MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<OpenKalman::internal::default_split_function, RC, RCs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix vertically.
   * \tparam cut Number of rows in the first cut.
   * \tparam cuts Numbers of rows in the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or (cut + ... + cuts) <= MatrixTraits<Arg>::rows)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and
      (dynamic_rows<Arg> or (cut + ... + cuts) <= MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix horizontally and invoke function F on each segment, returning a tuple.
   * \tparam CC Coefficients for the first cut.
   * \tparam CCs Coefficients for each of the second and subsequent cuts.
   * \tparam F An object having a static call() method to which the result is applied before creating the tuple.
   */
#ifdef __cpp_concepts
  template<typename F, coefficients CC, coefficients...CCs, eigen_matrix Arg>
  requires (not coefficients<F>) and
    (dynamic_columns<Arg> or (CC::dimensions + ... + CCs::dimensions) <= MatrixTraits<Arg>::columns)
#else
  template<typename F, typename CC, typename...CCs, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and not coefficients<F> and
      (dynamic_columns<Arg> or (CC::dimensions + ... + CCs::dimensions) <= MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    // \todo Can g be replaced by make_self_contained?
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg&&>)
        return std::forward<decltype(m)>(m);
      else
        return make_native_matrix(std::forward<decltype(m)>(m));
    };

    using RC = Axes<MatrixTraits<Arg>::rows>;
    constexpr Eigen::Index dim1 = CC::dimensions;

    if constexpr(sizeof...(CCs) > 0)
    {
      if constexpr (dynamic_columns<Arg>)
      {
        Eigen::Index dim2 = column_count(arg) - dim1;
        return std::tuple_cat(
          detail::make_split_tuple<F, RC, CC>(g(arg.leftCols(dim1))),
          split_horizontal<F, CCs...>(g(arg.rightCols(dim2))));
      }
      else
      {
        constexpr Eigen::Index dim2 = MatrixTraits<Arg>::columns - dim1;
        return std::tuple_cat(
          detail::make_split_tuple<F, RC, CC>(g(arg.template leftCols<dim1>())),
          split_horizontal<F, CCs...>(g(arg.template rightCols<dim2>())));
      }
    }
    else if constexpr (dim1 < MatrixTraits<Arg>::columns)
    {
      if constexpr (dynamic_columns<Arg>)
        return detail::make_split_tuple<F, RC, CC>(g(arg.leftCols(dim1)));
      else
        return detail::make_split_tuple<F, RC, CC>(g(arg.template leftCols<dim1>()));
    }
    else
    {
      return detail::make_split_tuple<F, RC, CC>(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Split a matrix horizontally (case in which there is no split).
   */
#ifdef __cpp_concepts
  template<typename F = OpenKalman::internal::default_split_function, eigen_matrix Arg>
#else
  template<typename F = OpenKalman::internal::default_split_function, typename Arg,
    std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return std::tuple {};
  }


  /**
   * \brief Split a matrix horizontally.
   * \tparam CC Coefficients for the first cut.
   * \tparam CCs Coefficients for the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<coefficients CC, coefficients...CCs, eigen_matrix Arg>
  requires (dynamic_columns<Arg> or (CC::dimensions + ... + CCs::dimensions) <= MatrixTraits<Arg>::columns)
#else
  template<typename CC, typename...CCs, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and coefficients<CC> and
      (dynamic_columns<Arg> or (CC::dimensions + ... + CCs::dimensions) <= MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<OpenKalman::internal::default_split_function, CC, CCs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix horizontally.
   * \tparam cut Number of columns in the first cut.
   * \tparam cuts Numbers of columns in the second and subsequent cuts.
   * \tparam F An object having a static call() method to which the result is applied before creating the tuple.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, eigen_matrix Arg>
  requires (dynamic_columns<Arg> or (cut + ... + cuts) <= MatrixTraits<Arg>::columns)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (dynamic_columns<Arg> or (cut + ... + cuts) <= MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix diagonally and invoke function F on each segment, returning a tuple.
   * \tparam F An object having a static call() method to which the result is applied before creating the tuple.
   * \tparam C Coefficients for the first cut.
   * \tparam Cs Coefficients for each of the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<typename F, bool euclidean, coefficients C, coefficients...Cs, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or
    ((euclidean ? C::euclidean_dimensions : C::dimensions) + ... +
      (euclidean ? Cs::euclidean_dimensions : Cs::dimensions)) <= MatrixTraits<Arg>::rows) and
    (dynamic_columns<Arg> or (C::dimensions + ... + Cs::dimensions) <= MatrixTraits<Arg>::columns)
#else
  template<typename F, bool euclidean, typename C, typename...Cs, typename Arg, std::enable_if_t<
    eigen_matrix<Arg> and (coefficients<C> and ... and coefficients<Cs>) and
    (dynamic_rows<Arg> or
    ((euclidean ? C::euclidean_dimensions : C::dimensions) + ... +
      (euclidean ? Cs::euclidean_dimensions : Cs::dimensions)) <= MatrixTraits<Arg>::rows) and
    (dynamic_columns<Arg> or (C::dimensions + ... + Cs::dimensions) <= MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    // \todo Can g be replaced by make_self_contained?
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg&&>)
        return std::forward<decltype(m)>(m);
      else
        return make_native_matrix(std::forward<decltype(m)>(m));
    };

    constexpr Eigen::Index rdim1 = euclidean ? C::euclidean_dimensions : C::dimensions;
    constexpr Eigen::Index cdim1 = C::dimensions;

    if constexpr(sizeof...(Cs) > 0)
    {
      if constexpr (dynamic_shape<Arg>)
      {
        Eigen::Index rdim2 = row_count(arg) - rdim1;
        Eigen::Index cdim2 = column_count(arg) - cdim1;

        return std::tuple_cat(
          detail::make_split_tuple<F, C, C>(g(arg.topLeftCorner(rdim1, cdim1))),
          split_diagonal<F, euclidean, Cs...>(g(arg.bottomRightCorner(rdim2, cdim2))));
      }
      else
      {
        constexpr Eigen::Index rdim2 = MatrixTraits<Arg>::rows - rdim1;
        constexpr Eigen::Index cdim2 = MatrixTraits<Arg>::columns - cdim1;

        return std::tuple_cat(
          detail::make_split_tuple<F, C, C>(g(arg.template topLeftCorner<rdim1, cdim1>())),
          split_diagonal<F, euclidean, Cs...>(g(arg.template bottomRightCorner<rdim2, cdim2>())));
      }
    }
    else if constexpr(rdim1 < std::decay_t<Arg>::RowsAtCompileTime)
    {
      if constexpr (dynamic_shape<Arg>)
        return detail::make_split_tuple<F, C, C>(g(arg.topLeftCorner(rdim1, cdim1)));
      else
        return detail::make_split_tuple<F, C, C>(g(arg.template topLeftCorner<rdim1, cdim1>()));
    }
    else
    {
      return detail::make_split_tuple<F, C, C>(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Split a matrix diagonally (case in which there is no split).
   */
#ifdef __cpp_concepts
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false, eigen_matrix Arg>
  requires (not coefficients<F>)
#else
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false,
    typename Arg, std::enable_if_t<eigen_matrix<Arg> and not coefficients<F>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return std::tuple {};
  }


  /**
   * \brief Split a matrix diagonally by carving out square blocks along the diagonal.
   * \tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
   * \tparam C Coefficients for the first cut.
   * \tparam Cs Coefficients for each of the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<typename F, coefficients C, coefficients...Cs, eigen_matrix Arg>
  requires (not coefficients<F>) and
    (dynamic_rows<Arg> or (C::dimensions + ... + Cs::dimensions) <= MatrixTraits<Arg>::rows) and
    (dynamic_columns<Arg> or (C::dimensions + ... + Cs::dimensions) <= MatrixTraits<Arg>::columns)
#else
  template<typename F, typename C, typename...Cs, typename Arg, std::enable_if_t<
    eigen_matrix<Arg> and not coefficients<F> and (coefficients<C> and ... and coefficients<Cs>) and
    (dynamic_rows<Arg> or (C::dimensions + ... + Cs::dimensions) <= MatrixTraits<Arg>::rows) and
    (dynamic_columns<Arg> or (C::dimensions + ... + Cs::dimensions) <= MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<F, false, C, Cs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix diagonally by carving out square blocks along the diagonal.
   * \tparam C Coefficients for the first cut.
   * \tparam Cs Coefficients for the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<coefficients C, coefficients...Cs, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or (C::dimensions + ... + Cs::dimensions) <= MatrixTraits<Arg>::rows) and
    (dynamic_columns<Arg> or (C::dimensions + ... + Cs::dimensions) <= MatrixTraits<Arg>::columns)
#else
  template<typename C, typename...Cs, typename Arg, std::enable_if_t<
    eigen_matrix<Arg> and (coefficients<C> and ... and coefficients<Cs>) and
    (dynamic_rows<Arg> or (C::dimensions + ... + Cs::dimensions) <= MatrixTraits<Arg>::rows) and
    (dynamic_columns<Arg> or (C::dimensions + ... + Cs::dimensions) <= MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<OpenKalman::internal::default_split_function, C, Cs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split a matrix diagonally by carving out square blocks along the diagonal.
   * \tparam cut Number of rows and columns in the first cut.
   * \tparam cuts Numbers of rows and columns in the second and subsequent cuts.
   * \tparam F An object having a static call() method to which the result is applied before creating the tuple.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, eigen_matrix Arg>
  requires (dynamic_columns<Arg> or (cut + ... + cuts) <= MatrixTraits<Arg>::columns) and
    (dynamic_rows<Arg> or (cut + ... + cuts) <= MatrixTraits<Arg>::rows)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and
    (dynamic_columns<Arg> or (cut + ... + cuts) <= MatrixTraits<Arg>::columns) and
    (dynamic_rows<Arg> or (cut + ... + cuts) <= MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  // \todo Add functions that return stl-compatible iterators.


  namespace detail
  {
#ifdef __cpp_concepts
    template<typename Function, typename Arg>
    concept col_lvalue_ref_fun = (not std::is_const_v<Arg>) and std::is_lvalue_reference_v<Arg> and
      requires(const Function& f, std::decay_t<decltype(column<0>(std::declval<Arg&&>()))>& col) {
        {col} -> writable;
        requires std::is_void_v<decltype(f(col))> or requires { {f(col)} -> std::same_as<decltype(col)>; };
      };

    template<typename Function, typename Arg>
    concept col_lvalue_ref_fun_ind = (not std::is_const_v<Arg>) and std::is_lvalue_reference_v<Arg> and
      requires(const Function& f, std::decay_t<decltype(column<0>(std::declval<Arg&&>()))>& col, std::size_t i) {
        {col} -> writable;
        requires std::is_void_v<decltype(f(col, i))> or requires { {f(col, i)} -> std::same_as<decltype(col)>; };
      };
#else
    template<typename Function, typename Arg>
    using ColRes = std::invoke_result_t<const Function&, std::decay_t<decltype(column<0>(std::declval<Arg&&>()))>&>;

    template<typename Function, typename Arg>
    static constexpr bool col_lvalue_ref_fun = (not std::is_const_v<Arg>) and std::is_lvalue_reference_v<Arg> and
      writable<decltype(column<0>(std::declval<Arg&&>()))> and
      (std::is_void_v<detail::ColRes<Function, Arg>> or
        std::is_same_v<detail::ColRes<Function, Arg>, decltype(column<0>(std::declval<Arg&&>()))>);

    template<typename Arg, typename Function>
    using ColResI = std::invoke_result_t<
      const Function&, std::decay_t<decltype(column<0>(std::declval<Arg&&>()))>&, std::size_t >;

    template<typename Function, typename Arg>
    static constexpr bool col_lvalue_ref_fun_ind = (not std::is_const_v<Arg>) and std::is_lvalue_reference_v<Arg> and
      writable<decltype(column<0>(std::declval<Arg&&>()))> and
      (std::is_void_v<detail::ColResI<Function, Arg>> or
        std::is_same_v<detail::ColResI<Function, Arg>, decltype(column<0>(std::declval<Arg&&>()))>);
#endif


    template<bool index, typename F, typename Arg, std::size_t ... ints>
    inline decltype(auto) columnwise_impl(const F& f, Arg&& arg, std::index_sequence<ints...>)
    {
      static_assert(not dynamic_columns<Arg>);

      if constexpr ((index and detail::col_lvalue_ref_fun_ind<F, Arg>) or
        (not index and detail::col_lvalue_ref_fun<F, Arg>))
      {
        static_assert(std::is_lvalue_reference_v<Arg>);

        return ([](const F& f, Arg&& arg) -> Arg& {
          decltype(auto) c {column<ints>(arg)};
          static_assert(writable<decltype(c)>);
          if constexpr (index) f(c, ints); else f(c);
          return arg;
        }(f, arg), ...);
      }
      else
      {
        if constexpr (index)
          return concatenate_horizontal(f(column<ints>(std::forward<Arg>(arg)), ints)...);
        else
          return concatenate_horizontal(f(column<ints>(std::forward<Arg>(arg)))...);
      }
    };


    template<bool index, typename F, typename Arg>
    inline decltype(auto) columnwise_impl(const F& f, Arg&& arg)
    {
      static_assert(dynamic_columns<Arg>);

      auto cols = column_count(arg);

      if constexpr ((index and detail::col_lvalue_ref_fun_ind<F, Arg>) or
        (not index and detail::col_lvalue_ref_fun<F, Arg>))
      {
        static_assert(std::is_lvalue_reference_v<Arg>);

        for (std::size_t j = 0; j<cols; j++)
        {
          decltype(auto) c {column(arg, j)};
          static_assert(writable<decltype(c)>);
          if constexpr (index) f(c, j); else f(c);
        }
        return (arg);
      }
      else
      {
        auto res_col0 = [](const F& f, Arg&& arg){
          auto col0 = column(std::forward<Arg>(arg), 0);
          if constexpr (index) return f(col0, 0); else return f(col0);
        }(f, std::forward<Arg>(arg));

        using ResultType = decltype(res_col0);
        using M = eigen_matrix_t<typename MatrixTraits<ResultType>::Scalar, MatrixTraits<ResultType>::rows, 0>;
        M m {row_count(res_col0), cols};

        column(m, 0) = res_col0;

        for (std::size_t j = 1; j<cols; j++)
        {
          if constexpr (index)
            column(m, j) = f(column(std::forward<Arg>(arg), j), j);
          else
            column(m, j) = f(column(std::forward<Arg>(arg), j));
        }
        return m;
      }
    };

  } // namespace detail


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires detail::col_lvalue_ref_fun<Function, Arg> or
    requires(const Function& f, Arg&& arg) {
      {f(column<0>(std::forward<Arg>(arg)))} -> eigen_matrix;
      {f(column<0>(std::forward<Arg>(arg)))} -> column_vector;
    }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
    (detail::col_lvalue_ref_fun<Function, Arg> or column_vector<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column<0>(std::declval<Arg&&>()))>&& >>), int> = 0>
#endif
  inline decltype(auto)
  apply_columnwise(Arg&& arg, const Function& f)
  {
    if constexpr (dynamic_columns<Arg>)
      return detail::columnwise_impl<false>(f, std::forward<Arg>(arg));
    else
      return detail::columnwise_impl<false>(
        f, std::forward<Arg>(arg), std::make_index_sequence<MatrixTraits<Arg>::columns>());
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires detail::col_lvalue_ref_fun_ind<Function, Arg> or
    requires(const Function& f, Arg&& arg, std::size_t i) {
      {f(column<0>(std::forward<Arg>(arg)), i)} -> eigen_matrix;
      {f(column<0>(std::forward<Arg>(arg)), i)} -> column_vector;
    }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
    (detail::col_lvalue_ref_fun_ind<Function, Arg> or column_vector<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column<0>(std::declval<Arg&&>()))>&&, std::size_t>>), int> = 0>
#endif
  inline decltype(auto)
  apply_columnwise(Arg&& arg, const Function& f)
  {
    if constexpr (dynamic_columns<Arg>)
      return detail::columnwise_impl<true>(f, std::forward<Arg>(arg));
    else
      return detail::columnwise_impl<true>(
        f, std::forward<Arg>(arg), std::make_index_sequence<MatrixTraits<Arg>::columns>());
  }


#ifdef __cpp_concepts
  template<std::size_t...count, typename Function, std::convertible_to<std::size_t>...runtime_count>
  requires (sizeof...(count) + sizeof...(runtime_count) == 1) and
    requires(const Function& f) {
      {f()} -> eigen_matrix;
      (requires { {f()} -> dynamic_columns; } or requires {{f()} -> column_vector; });
    }
#else
  template<std::size_t...count, typename Function, typename...runtime_count, std::enable_if_t<
    (sizeof...(count) + sizeof...(runtime_count) == 1) and
    (std::is_convertible_v<std::size_t, runtime_count> and ...) and
    (dynamic_columns<std::invoke_result_t<const Function&>> or column_vector<std::invoke_result_t<const Function&>>) and
    eigen_matrix<std::invoke_result_t<const Function&>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, runtime_count...i)
  {
    auto r = f();
    using R = decltype(r);
    if constexpr (dynamic_columns<R>) assert (column_count(r) == 1);

    if constexpr (sizeof...(count) > 0)
      return Eigen::Replicate<R, 1, count...>(r);
    else
      return Eigen::Replicate<R, 1, Eigen::Dynamic>(r, 1, i...);
  }


  namespace detail
  {
    template<typename Function, std::size_t ... ints>
    inline auto cat_columnwise_impl(const Function& f, std::index_sequence<ints...>)
    {
      return concatenate_horizontal(f(ints)...);
    };
  }


#ifdef __cpp_concepts
  template<std::size_t...count, typename Function, std::convertible_to<std::size_t>...runtime_count>
  requires (sizeof...(count) + sizeof...(runtime_count) == 1) and
    requires(const Function& f, std::size_t i) {
      {f(i)} -> eigen_matrix;
      (requires { {f(i)} -> dynamic_columns; } or requires {{f(i)} -> column_vector; });
    }
#else
  template<std::size_t...count, typename Function, typename...runtime_count, std::enable_if_t<
    (sizeof...(count) + sizeof...(runtime_count) == 1) and
    (std::is_convertible_v<runtime_count, std::size_t> and ...) and
    eigen_matrix<std::invoke_result_t<const Function&, std::size_t>> and
    (dynamic_columns<std::invoke_result_t<const Function&, std::size_t>> or
      column_vector<std::invoke_result_t<const Function&, std::size_t>>), int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, runtime_count...i)
  {
    if constexpr (sizeof...(runtime_count) == 0)
    {
      return detail::cat_columnwise_impl(f, std::make_index_sequence<(count + ... + 0)>());
    }
    else
    {
      using R = std::invoke_result_t<const Function&, std::size_t>;
      using Scalar = typename MatrixTraits<R>::Scalar;
      auto cols = (i + ... + 0);

      if constexpr (dynamic_rows<R>)
      {
        auto r0 = f(0);
        auto rows = row_count(r0);
        eigen_matrix_t<Scalar, 0, 0> m {rows, cols};
        m.col(0) = r0;
        for (std::size_t j = 1; j < cols; j++)
        {
          auto rj = f(j);
          assert(row_count(rj) == rows);
          m.col(j) = rj;
        }
        return m;
      }
      else
      {
        constexpr auto rows = MatrixTraits<R>::rows;
        eigen_matrix_t<Scalar, rows, 0> m {rows, cols};
        for (std::size_t j = 0; j < cols; j++)
        {
          m.col(j) = f(j);
        }
        return m;
      }
    }
  }


  namespace detail
  {
#ifdef __cpp_concepts
    template<typename Function, typename Arg>
    concept row_lvalue_ref_fun = (not std::is_const_v<Arg>) and std::is_lvalue_reference_v<Arg> and
      requires(const Function& f, std::decay_t<decltype(row<0>(std::declval<Arg&&>()))>& row) {
        {row} -> writable;
        requires std::is_void_v<decltype(f(row))> or requires { {f(row)} -> std::same_as<decltype(row)>; };
      };

    template<typename Function, typename Arg>
    concept row_lvalue_ref_fun_ind = (not std::is_const_v<Arg>) and std::is_lvalue_reference_v<Arg> and
      requires(const Function& f, std::decay_t<decltype(row<0>(std::declval<Arg&&>()))>& row, std::size_t i) {
        {row} -> writable;
        requires std::is_void_v<decltype(f(row, i))> or requires { {f(row, i)} -> std::same_as<decltype(row)>; };
      };
#else
    template<typename Arg, typename Function>
    using RowRes = std::invoke_result_t<const Function&, std::decay_t<decltype(row<0>(std::declval<Arg&&>()))>& >;

    template<typename Function, typename Arg>
    static constexpr bool row_lvalue_ref_fun = (not std::is_const_v<Arg>) and std::is_lvalue_reference_v<Arg> and
      writable<decltype(row<0>(std::declval<Arg&&>()))> and
      (std::is_void_v<detail::RowRes<Function, Arg>> or
        std::is_same_v<detail::RowRes<Function, Arg>, decltype(row<0>(std::declval<Arg&&>()))>);

    template<typename Arg, typename Function>
    using RowResI = std::invoke_result_t<
      const Function&, std::decay_t<decltype(row<0>(std::declval<Arg&&>()))>&, std::size_t>;

    template<typename Function, typename Arg>
    static constexpr bool row_lvalue_ref_fun_ind = (not std::is_const_v<Arg>) and std::is_lvalue_reference_v<Arg> and
      writable<decltype(row<0>(std::declval<Arg&&>()))> and
      (std::is_void_v<detail::RowResI<Function, Arg>> or
        std::is_same_v<detail::RowResI<Function, Arg>, decltype(row<0>(std::declval<Arg&&>()))>);
#endif


    template<bool index, typename F, typename Arg, std::size_t ... ints>
    inline decltype(auto) rowwise_impl(const F& f, Arg&& arg, std::index_sequence<ints...>)
    {
      if constexpr ((index and detail::row_lvalue_ref_fun_ind<F, Arg>) or
        (not index and detail::row_lvalue_ref_fun<F, Arg>))
      {
        static_assert(std::is_lvalue_reference_v<Arg>);

        return ([](const F& f, Arg&& arg) -> Arg& {
          decltype(auto) r {row<ints>(arg)};
          static_assert(writable<decltype(r)>);
          if constexpr (index) f(r, ints); else f(r);
          return arg;
        }(f, arg), ...);
      }
      else
      {
        if constexpr (index)
          return concatenate_vertical(f(row<ints>(std::forward<Arg>(arg)), ints)...);
        else
          return concatenate_vertical(f(row<ints>(std::forward<Arg>(arg)))...);
      }
    };


  template<bool index, typename F, typename Arg>
  inline decltype(auto) rowwise_impl(const F& f, Arg&& arg)
  {
    static_assert(dynamic_rows<Arg>);

    auto rows = row_count(arg);

    if constexpr ((index and detail::row_lvalue_ref_fun_ind<F, Arg>) or
      (not index and detail::row_lvalue_ref_fun<F, Arg>))
    {
      static_assert(std::is_lvalue_reference_v<Arg>);

      for (std::size_t i = 0; i<rows; i++)
      {
        decltype(auto) r {row(arg, i)};
        static_assert(writable<decltype(r)>);
        if constexpr (index) f(r, i); else f(r);
      }
      return (arg);
    }
    else
    {
      auto res_row0 = [](const F& f, Arg&& arg){
        auto row0 = row(std::forward<Arg>(arg), 0);
        if constexpr (index) return f(row0, 0); else return f(row0);
      }(f, std::forward<Arg>(arg));

      using ResultType = decltype(res_row0);
      using M = eigen_matrix_t<typename MatrixTraits<ResultType>::Scalar, 0, MatrixTraits<ResultType>::columns>;
      M m {rows, column_count(res_row0)};

      row(m, 0) = res_row0;

      for (std::size_t i = 1; i<rows; i++)
      {
        if constexpr (index)
          row(m, i) = f(row(std::forward<Arg>(arg), i), i);
        else
          row(m, i) = f(row(std::forward<Arg>(arg), i));
      }
      return m;
    }
  };

  } // namespace detail


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires
    detail::row_lvalue_ref_fun<Function, Arg> or
    requires(Arg&& arg, const Function& f) {
      {f(row<0>(std::forward<Arg>(arg)))} -> eigen_matrix;
      {f(row<0>(std::forward<Arg>(arg)))} -> row_vector;
    }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
    (detail::row_lvalue_ref_fun<Function, Arg> or row_vector<std::invoke_result_t<const Function&,
      std::decay_t<decltype(row<0>(std::declval<Arg&&>()))>&& >>), int> = 0>
#endif
  inline decltype(auto)
  apply_rowwise(Arg&& arg, const Function& f)
  {
    if constexpr (dynamic_rows<Arg>)
      return detail::rowwise_impl<false>(f, std::forward<Arg>(arg));
    else
      return detail::rowwise_impl<false>(f, std::forward<Arg>(arg), std::make_index_sequence<MatrixTraits<Arg>::rows>());
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires
    detail::row_lvalue_ref_fun_ind<Function, Arg> or
    requires(const Function& f, Arg&& arg, std::size_t i) {
      {f(row<0>(std::forward<Arg>(arg)), i)} -> eigen_matrix;
      {f(row<0>(std::forward<Arg>(arg)), i)} -> row_vector;
    }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
    (detail::row_lvalue_ref_fun_ind<Function, Arg> or row_vector<std::invoke_result_t<const Function&,
      std::decay_t<decltype(row<0>(std::declval<Arg&&>()))>&&, std::size_t>>), int> = 0>
#endif
  inline decltype(auto)
  apply_rowwise(Arg&& arg, const Function& f)
  {
    if constexpr (dynamic_rows<Arg>)
      return detail::rowwise_impl<true>(f, std::forward<Arg>(arg));
    else
      return detail::rowwise_impl<true>(f, std::forward<Arg>(arg), std::make_index_sequence<MatrixTraits<Arg>::rows>());
  }


#ifdef __cpp_concepts
  template<std::size_t...count, typename Function, std::convertible_to<std::size_t>...runtime_count>
  requires (sizeof...(count) + sizeof...(runtime_count) == 1) and
    requires(const Function& f) {
      {f()} -> eigen_matrix;
      (requires { {f()} -> dynamic_rows; } or requires {{f()} -> row_vector; });
    }
#else
  template<std::size_t...count, typename Function, typename...runtime_count, std::enable_if_t<
    (sizeof...(count) + sizeof...(runtime_count) == 1) and
    (std::is_convertible_v<std::size_t, runtime_count> and ...) and
    (dynamic_rows<std::invoke_result_t<const Function&>> or row_vector<std::invoke_result_t<const Function&>>) and
    eigen_matrix<std::invoke_result_t<const Function&>>, int> = 0>
#endif
  inline auto
  apply_rowwise(const Function& f, runtime_count...i)
  {
    auto c = f();
    using C = decltype(c);
    if constexpr (dynamic_rows<C>) assert (row_count(c) == 1);

    if constexpr (sizeof...(count) > 0)
      return Eigen::Replicate<C, count..., 1>(c);
    else
      return Eigen::Replicate<C, Eigen::Dynamic, 1>(c, i..., 1);
  }


  namespace detail
  {
    template<typename Function, std::size_t ... ints>
    inline auto cat_rowwise_impl(const Function& f, std::index_sequence<ints...>)
    {
      return concatenate_vertical(f(ints)...);
    };
  }


#ifdef __cpp_concepts
  template<std::size_t...count, typename Function, std::convertible_to<std::size_t>...runtime_count>
  requires (sizeof...(count) + sizeof...(runtime_count) == 1) and
    requires(const Function& f, std::size_t i) {
      {f(i)} -> eigen_matrix;
      {f(i)} -> row_vector;
    }
#else
  template<std::size_t...count, typename Function, typename...runtime_count, std::enable_if_t<
    (sizeof...(count) + sizeof...(runtime_count) == 1) and
    (std::is_convertible_v<std::size_t, runtime_count> and ...) and
    eigen_matrix<std::invoke_result_t<const Function&, std::size_t>> and
    row_vector<std::invoke_result_t<const Function&, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_rowwise(const Function& f, runtime_count...i)
  {
    if constexpr (sizeof...(runtime_count) == 0)
    {
      return detail::cat_rowwise_impl(f, std::make_index_sequence<(count + ... + 0)>());
    }
    else
    {
      using R = std::invoke_result_t<const Function&, std::size_t>;
      using Scalar = typename MatrixTraits<R>::Scalar;
      auto rows = (i + ... + 0);

      if constexpr (dynamic_columns<R>)
      {
        auto c0 = f(0);
        auto cols = column_count(c0);
        eigen_matrix_t<Scalar, 0, 0> m {rows, cols};
        m.row(0) = c0;
        for (std::size_t i = 1; i < rows; i++)
        {
          auto ci = f(i);
          assert(column_count(ci) == cols);
          m.row(i) = ci;
        }
        return m;
      }
      else
      {
        constexpr auto cols = MatrixTraits<R>::columns;
        eigen_matrix_t<Scalar, 0, cols> m {rows, cols};
        for (std::size_t i = 0; i < rows; i++)
        {
          m.row(i) = f(i);
        }
        return m;
      }
    }
  }


  ////

#ifdef __cpp_concepts
  template<native_eigen_matrix Arg, typename Function> requires element_settable<Arg, 2> and
    (requires(Function f, typename MatrixTraits<Arg>::Scalar& s) { {f(s)} -> std::same_as<void>; } or
    requires(Function f, typename MatrixTraits<Arg>::Scalar& s) {
      {f(s)} -> std::same_as<typename MatrixTraits<Arg>::Scalar&>;
    }) and
    (not requires(Function f, typename MatrixTraits<Arg>::Scalar& s, std::size_t& i, std::size_t& j) { f(s, i, j); })
#else
  template<typename Arg, typename Function, std::enable_if_t<native_eigen_matrix<Arg> and element_settable<Arg, 2> and
    (std::is_same_v<std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar&>, void> or
    std::is_same_v<std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar&>,
      typename MatrixTraits<Arg>::Scalar&>), int> = 0>
#endif
  inline Arg&
  apply_coefficientwise(Arg& arg, const Function& f)
  {
    for (std::size_t j = 0; j < column_count(arg); j++)
    {
      for (std::size_t i = 0; i < row_count(arg); i++)
      {
        f(arg(i, j));
      }
    }
    return arg;
  }


#ifdef __cpp_concepts
  template<native_eigen_matrix Arg, typename Function> requires element_settable<Arg, 2> and
    (requires(Function f, typename MatrixTraits<Arg>::Scalar& s, std::size_t i, std::size_t j) {
      {f(s, i, j)} -> std::same_as<void>;
    } or
    requires(Function f, typename MatrixTraits<Arg>::Scalar& s, std::size_t i, std::size_t j) {
      {f(s, i, j)} -> std::same_as<typename MatrixTraits<Arg>::Scalar&>;
    })
#else
  template<typename Arg, typename Function, std::enable_if_t<native_eigen_matrix<Arg> and element_settable<Arg, 2> and
    (std::is_same_v<std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar&, std::size_t, std::size_t>,
      void> or
    std::is_same_v<std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar&, std::size_t, std::size_t>,
      typename MatrixTraits<Arg>::Scalar&>), int> = 0>
#endif
  inline Arg&
  apply_coefficientwise(Arg& arg, Function&& f)
  {
    for (std::size_t j = 0; j < column_count(arg); j++)
    {
      for (std::size_t i = 0; i < row_count(arg); i++)
      {
        f(arg(i, j), i, j);
      }
    }
    return arg;
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires
    requires(Function f, typename MatrixTraits<Arg>::Scalar& s) {
      {f(s)} -> std::convertible_to<const typename MatrixTraits<Arg>::Scalar&>;
    } and
    (not requires(Function f, typename MatrixTraits<Arg>::Scalar& s, std::size_t& i, std::size_t& j) {
      {f(s, i, j)} -> std::convertible_to<const typename MatrixTraits<Arg>::Scalar&>;
    })
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
    std::is_convertible_v<std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar&>,
      const typename MatrixTraits<Arg>::Scalar> and
    (not std::is_invocable_v<Function, typename MatrixTraits<Arg>::Scalar&, std::size_t&, std::size_t&>), int> = 0>
#endif
  inline auto
  apply_coefficientwise(Arg&& arg, Function&& f)
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    return make_self_contained<Arg>(arg.unaryExpr([fun = std::forward<Function>(f)] (const Scalar& x) {
      if constexpr (std::is_invocable_v<Function, const Scalar&>) return fun(x);
      else { Scalar xx = x; return fun(xx); }
    }));
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires element_gettable<Arg, 2> and
    requires(Function f, typename MatrixTraits<Arg>::Scalar& s, std::size_t i, std::size_t j) {
      {f(s, i, j)} -> std::convertible_to<const typename MatrixTraits<Arg>::Scalar&>;
    }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and element_gettable<Arg, 2> and
    std::is_convertible_v<std::invoke_result_t<Function,
      typename MatrixTraits<Arg>::Scalar&, std::size_t, std::size_t>,
      const typename MatrixTraits<Arg>::Scalar&>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(Arg&& arg, Function&& f)
  {
    return native_matrix_t<Arg>::NullaryExpr(
      [m = std::forward<Arg>(arg), fun = std::forward<Function>(f)] (Eigen::Index i, Eigen::Index j) {
        std::size_t ix = i, jx = j;
        return fun(get_element(m, ix, jx), ix, jx);
      });
  }


#ifdef __cpp_concepts
  template<std::size_t rows, std::size_t columns, typename Function> requires
    requires { typename Eigen::NumTraits<std::invoke_result_t<Function>>; }
#else
  template<std::size_t rows, std::size_t columns, typename Function,
    std::enable_if_t<std::is_void_v<std::void_t<Eigen::NumTraits<std::invoke_result_t<Function>>>>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(Function&& f)
  {
    using Scalar = std::invoke_result_t<Function>;
    using Mat = eigen_matrix_t<Scalar, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(columns)>;
    if constexpr (std::is_lvalue_reference_v<Function>)
      return native_matrix_t<Mat>::NullaryExpr(std::cref(f));
    else
      return native_matrix_t<Mat>::NullaryExpr([f = std::move(f)] () { return f(); });
  }


#ifdef __cpp_concepts
  template<std::size_t rows, std::size_t columns, typename Function> requires
    requires { typename Eigen::NumTraits<std::invoke_result_t<Function, std::size_t&, std::size_t&>>; }
#else
  template<std::size_t rows, std::size_t columns, typename Function,
    typename = std::void_t<Eigen::NumTraits<std::invoke_result_t<Function, std::size_t&, std::size_t&>>>>
#endif
  inline auto
  apply_coefficientwise(Function&& f)
  {
    using Scalar = std::invoke_result_t<Function, std::size_t, std::size_t>;
    using Mat = eigen_matrix_t<Scalar, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(columns)>;
    if constexpr (std::is_lvalue_reference_v<Function>)
      return native_matrix_t<Mat>::NullaryExpr(std::cref(f));
    else
      return native_matrix_t<Mat>::NullaryExpr([f = std::move(f)] (std::size_t i, std::size_t j) { return f(i, j); });
  }


  namespace detail
  {
    template<typename random_number_engine>
    struct Rnd
    {
      template<typename distribution_type>
      static inline auto
      get(distribution_type& dist)
      {
        if constexpr (std::is_arithmetic_v<distribution_type>)
        {
          return dist;
        }
        else
        {
          static std::random_device rd;
          static random_number_engine rng {rd()};

#ifdef __cpp_concepts
          static_assert(requires { typename distribution_type::result_type; typename distribution_type::param_type; });
#endif
          return dist(rng);
        }
      }

    };


#ifdef __cpp_concepts
  template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct dist_result_type {};

#ifdef __cpp_concepts
    template<typename T> requires requires { typename T::result_type; typename T::param_type; }
    struct dist_result_type<T> { using type = typename T::result_type; };
#else
    template<typename T>
    struct dist_result_type<T, std::enable_if_t<not std::is_arithmetic_v<T>>> { using type = typename T::result_type; };
#endif

#ifdef __cpp_concepts
  template<typename T> requires std::is_arithmetic_v<T>
  struct dist_result_type<T> { using type = T; };
#else
  template<typename T>
    struct dist_result_type<T, std::enable_if_t<std::is_arithmetic_v<T>>> { using type = T; };
#endif



  } // namespace detail


  /**
   * \brief Fill a fixed-shape Eigen matrix with random values selected from a single random distribution.
   * \details The following example constructs a 2-by-2 matrix (m) in which each element is a random value selected
   * based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     Mat m = randomize<Mat>(N {1.0, 0.3}));
   *   \endcode
   * \tparam ReturnType The type of the matrix to be filled.
   * \tparam random_number_engine The random number engine (e.g., std::mt19937).
   * \tparam Dist A distribution (e.g., std::normal_distribution<double>).
   **/
#ifdef __cpp_concepts
  template<eigen_matrix ReturnType, std::uniform_random_bit_generator random_number_engine = std::mt19937,
    typename Dist>
  requires
    (not dynamic_shape<ReturnType>) and
    requires { typename std::decay_t<Dist>::result_type; typename std::decay_t<Dist>::param_type; } and
      (not std::is_const_v<std::remove_reference_t<Dist>>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename Dist,
    std::enable_if_t<eigen_matrix<ReturnType> and (not dynamic_shape<ReturnType>)and
    (not std::is_const_v<std::remove_reference_t<Dist>>), int> = 0>
#endif
  inline auto
  randomize(Dist&& dist)
  {
    using D = struct { mutable Dist value; };
    return ReturnType::NullaryExpr([d = D {std::forward<Dist>(dist)}] {
      return detail::Rnd<random_number_engine>::get(d.value);
    });
  }


  /**
   * \overload
   * \brief Fill a dynamic-shape Eigen matrix with random values selected from a single random distribution.
   * \details The following example constructs two 2-by-2 matrices (m, n, and p) in which each element is a
   * random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     auto m = randomize<Eigen::Matrix<float, 2, Eigen::Dynamic>>(2, 2, std::normal_distribution<float> {1.0, 0.3}));
   *     auto n = randomize<Eigen::Matrix<double, Eigen::Dynamic, 2>>(2, 2, std::normal_distribution<double> {1.0, 0.3}));
   *     auto p = randomize<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(2, 2, std::normal_distribution<double> {1.0, 0.3});
   *   \endcode
   * \tparam ReturnType The type of the matrix to be filled.
   * \tparam random_number_engine The random number engine (e.g., std::mt19937).
   * \param rows Number of rows, decided at runtime. Must match rows of ReturnType if they are fixed.
   * \param columns Number of columns, decided at runtime. Must match columns of ReturnType if they are fixed.
   * \tparam Dist A distribution (type distribution_type).
   **/
#ifdef __cpp_concepts
  template<eigen_matrix ReturnType, std::uniform_random_bit_generator random_number_engine = std::mt19937,
    typename Dist>
  requires
    dynamic_shape<ReturnType> and
    requires { typename std::decay_t<Dist>::result_type; typename std::decay_t<Dist>::param_type; } and
      (not std::is_const_v<std::remove_reference_t<Dist>>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename Dist, std::enable_if_t<
    eigen_matrix<ReturnType> and dynamic_shape<ReturnType> and
    (not std::is_const_v<std::remove_reference_t<Dist>>), int> = 0>
#endif
  inline auto
  randomize(const std::size_t rows, const std::size_t columns, Dist&& dist)
  {
    if constexpr (not dynamic_rows<ReturnType>) assert(rows == MatrixTraits<ReturnType>::rows);
    if constexpr (not dynamic_columns<ReturnType>) assert(columns == MatrixTraits<ReturnType>::columns);
    using D = struct { mutable Dist value; };
    return ReturnType::NullaryExpr(rows, columns, [d = D {std::forward<Dist>(dist)}] {
      return detail::Rnd<random_number_engine>::get(d.value);
    });
  }


  /**
   * \overload
   * \brief Fill a fixed Eigen matrix with random values selected from multiple random distributions.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix m containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 2>>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *
   *  - One distribution for each row. The following code constructs a 3-by-2 (n) or 2-by-2 (o) matrices
   *  in which elements in each row are selected according to the three (n) or two (o) listed distribution
   *  parameters:
   *   \code
   *     auto n = randomize<Eigen::Matrix<double, 3, 2>>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto o = randomize<Eigen::Matrix<double, 2, 2>>(N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *   Note that in the case of o, there is an ambiguity as to whether the listed distributions correspond to rows
   *   or columns. In case of such an ambiguity, this function assumes that the parameters correspond to the rows.
   *
   *  - One distribution for each column. The following code constructs 2-by-3 matrix p
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto p = randomize<Eigen::Matrix<double, 2, 3>>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   *
   * \tparam ReturnType The return type reflecting the size of the matrix to be filled. The actual result will be
   * a fixed shape matrix.
   * \tparam random_number_engine The random number engine.
   * \tparam Dists A set of distributions (e.g., std::normal_distribution<double>) or, alternatively,
   * means (a definite, non-stochastic value).
   **/
#ifdef __cpp_concepts
  template<eigen_matrix ReturnType, std::uniform_random_bit_generator random_number_engine = std::mt19937,
    typename...Dists>
  requires
  (not dynamic_shape<ReturnType>) and (sizeof...(Dists) > 1) and
    (((requires { typename std::decay_t<Dists>::result_type;  typename std::decay_t<Dists>::param_type; } or
      std::is_arithmetic_v<std::decay_t<Dists>>) and ... ))and
      ((not std::is_const_v<std::remove_reference_t<Dists>>) and ...) and
    (MatrixTraits<ReturnType>::rows * MatrixTraits<ReturnType>::columns == sizeof...(Dists) or
      MatrixTraits<ReturnType>::rows == sizeof...(Dists) or MatrixTraits<ReturnType>::columns == sizeof...(Dists))
#else
  template<
    typename ReturnType, typename random_number_engine = std::mt19937, typename...Dists,
    std::enable_if_t<
      eigen_matrix<ReturnType> and (not dynamic_shape<ReturnType>) and (sizeof...(Dists) > 1) and
      ((not std::is_const_v<std::remove_reference_t<Dists>>) and ...) and
      (MatrixTraits<ReturnType>::rows * MatrixTraits<ReturnType>::columns == sizeof...(Dists) or
        MatrixTraits<ReturnType>::rows == sizeof...(Dists) or
        MatrixTraits<ReturnType>::columns == sizeof...(Dists)), int> = 0>
#endif
  inline auto
  randomize(Dists&& ... dists)
  {
    using Scalar = std::common_type_t<typename detail::dist_result_type<Dists>::type...>;
    constexpr std::size_t s = sizeof...(Dists);
    constexpr std::size_t rows = MatrixTraits<ReturnType>::rows;
    constexpr std::size_t cols = MatrixTraits<ReturnType>::columns;

    // One distribution for each element
    if constexpr (rows * cols == s)
    {
      using M = eigen_matrix_t<Scalar, rows, cols>;
      return MatrixTraits<M>::make(detail::Rnd<random_number_engine>::get(dists)...);
    }

    // One distribution for each row
    else if constexpr (rows == s)
    {
      using M = eigen_matrix_t<Scalar, rows, 1>;
      return apply_columnwise<cols>([&] {
        return MatrixTraits<M>::make(detail::Rnd<random_number_engine>::get(dists)...);
      });
    }

    // One distribution for each column
    else
    {
      static_assert(cols == s);
      using M = eigen_matrix_t<Scalar, 1, cols>;
      return apply_rowwise<rows>([&] {
        return MatrixTraits<M>::make(detail::Rnd<random_number_engine>::get(dists)...);
      });
    }

  }


}

#endif //OPENKALMAN_EIGEN3_MATRIX_OVERLOADS_HPP
