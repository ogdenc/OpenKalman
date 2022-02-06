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
 * \brief Overloaded functions for Eigen3 extensions
 */

#ifndef OPENKALMAN_EIGEN3_SPECIAL_MATRIX_OVERLOADS_HPP
#define OPENKALMAN_EIGEN3_SPECIAL_MATRIX_OVERLOADS_HPP

namespace OpenKalman::Eigen3
{
  /// Return row <code>i</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  inline auto
  row(Arg&& arg, const std::size_t i)
  {
    if (i < row_count(arg)) throw std::out_of_range {
      "Index " + std::to_string(i) + "is out of range 0 <= i < " + std::to_string(row_count(arg))};

    return reduce_rows(std::forward<Arg>(arg));
  }


  /// Return row <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, eigen_zero_expr Arg> requires (not dynamic_rows<Arg>) and
    (index < row_extent_of_v<Arg>)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
    eigen_zero_expr<Arg> and (not dynamic_rows<Arg>) and (index < row_extent_of<Arg>::value), int> = 0>
#endif
  inline decltype(auto)
  row(Arg&& arg)
  {
    if constexpr(row_vector<Arg>)
      return std::forward<Arg>(arg);
    else
      return reduce_rows(std::forward<Arg>(arg));
  }


  /// Return row <code>i</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  inline auto
  row(Arg&& arg, const std::size_t i)
  {
    if (i < row_count(arg)) throw std::out_of_range {
      "Index " + std::to_string(i) + "is out of range 0 <= i < " + std::to_string(row_count(arg))};

    return reduce_rows(std::forward<Arg>(arg));
  }


  /// Return row <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, eigen_constant_expr Arg> requires (index < row_extent_of_v<Arg>)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
      eigen_constant_expr<Arg> and (index < row_extent_of<Arg>::value), int> = 0>
#endif
  inline decltype(auto)
  row(Arg&& arg)
  {
    return reduce_rows(std::forward<Arg>(arg));
  }


  /// Return row <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> or
      eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  row(Arg&& arg, const std::size_t index)
  {
    return row(make_native_matrix(std::forward<Arg>(arg)), index);
  }


  /// Return row <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, typename Arg> requires (eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and (index < row_extent_of_v<Arg>)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<(eigen_self_adjoint_expr<Arg> or
      eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and (index < row_extent_of<Arg>::value), int> = 0>
#endif
  inline decltype(auto)
  row(Arg&& arg)
  {
    return row<index>(make_native_matrix(std::forward<Arg>(arg)));
  }


  /// Return column <code>col</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t col)
  {
    if (col < row_count(arg)) throw std::out_of_range {
      "Index " + std::to_string(col) + "is out of range 0 <= c < " + std::to_string(column_count(arg))};

    return reduce_columns(std::forward<Arg>(arg));
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, eigen_zero_expr Arg>
  requires (not dynamic_columns<Arg>) and (index < column_extent_of_v<Arg>)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
    eigen_zero_expr<Arg> and (not dynamic_columns<Arg>) and (index < column_extent_of<Arg>::value), int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    if constexpr(column_vector<Arg>)
      return std::forward<Arg>(arg);
    else
      return reduce_columns(std::forward<Arg>(arg));
  }


/// Return column <code>i</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t i)
  {
    if (i < column_count(arg)) throw std::out_of_range {
      "Index " + std::to_string(i) + "is out of range 0 <= i < " + std::to_string(column_count(arg))};

    return reduce_columns(std::forward<Arg>(arg));
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, eigen_constant_expr Arg> requires (index < column_extent_of_v<Arg>)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
    eigen_constant_expr<Arg> and (index < column_extent_of<Arg>::value), int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    return reduce_columns(std::forward<Arg>(arg));
  }


/// Return column <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    return column(make_native_matrix(std::forward<Arg>(arg)), index);
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, typename Arg> requires (eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and (index < column_extent_of_v<Arg>)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<(eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and (index < column_extent_of<Arg>::value), int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    return column<index>(make_native_matrix(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<fixed_coefficients Coefficients, typename Arg>
  requires (eigen_zero_expr<Arg> or eigen_constant_expr<Arg>) and
    (dynamic_rows<Arg> or Coefficients::dimensions == row_extent_of_v<Arg>)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<fixed_coefficients<Coefficients> and
    (eigen_zero_expr<Arg> or eigen_constant_expr<Arg>) and
    (dynamic_rows<Arg> or Coefficients::dimensions == row_extent_of<Arg>::value), int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg) noexcept
  {
    if constexpr (dynamic_rows<Arg>) if (row_count(arg) != Coefficients::dimensions)
      throw std::out_of_range {"Number of rows (" + std::to_string(row_count(arg)) +
        ") does not match Coefficient dimensions (" + std::to_string(Coefficients::dimensions) + ")"};

    if constexpr (Coefficients::axes_only)
      return std::forward<Arg>(arg);
    else
      return ToEuclideanExpr<Coefficients, Arg>(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<fixed_coefficients Coefficients, typename Arg>
  requires (eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (dynamic_rows<Arg> or Coefficients::dimensions == row_extent_of_v<Arg>) and
    (dynamic_columns<Arg> or Coefficients::dimensions == column_extent_of_v<Arg>)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<fixed_coefficients<Coefficients> and
    (eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (dynamic_rows<Arg> or Coefficients::dimensions == row_extent_of<Arg>::value) and
    (dynamic_columns<Arg> or Coefficients::dimensions == column_extent_of<Arg>::value), int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg) noexcept
  {
    if constexpr (dynamic_rows<Arg>) if (row_count(arg) != Coefficients::dimensions)
      throw std::out_of_range {"Number of rows (" + std::to_string(row_count(arg)) +
        ") does not match Coefficient dimensions (" + std::to_string(Coefficients::dimensions) + ")"};

    if constexpr (Coefficients::axes_only)
      return std::forward<Arg>(arg);
    else
      return ToEuclideanExpr<Coefficients, equivalent_dense_writable_matrix_t<Arg>>(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<fixed_coefficients Coefficients, typename Arg>
  requires (eigen_zero_expr<Arg> or eigen_constant_expr<Arg>) and
    (dynamic_rows<Arg> or Coefficients::euclidean_dimensions == row_extent_of_v<Arg>)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<fixed_coefficients<Coefficients> and
    (eigen_zero_expr<Arg> or eigen_constant_expr<Arg>) and
    (dynamic_rows<Arg> or Coefficients::euclidean_dimensions == row_extent_of<Arg>::value), int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg) noexcept
  {
    if constexpr (dynamic_rows<Arg>) if (row_count(arg) != Coefficients::euclidean_dimensions)
      throw std::out_of_range {"Number of rows (" + std::to_string(row_count(arg)) +
        ") does not match Coefficient euclidean dimensions " + std::to_string(Coefficients::euclidean_dimensions)};

    if constexpr (Coefficients::axes_only)
      return std::forward<Arg>(arg);
    else
      return FromEuclideanExpr<Coefficients, Arg>(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<fixed_coefficients Coefficients, typename Arg>
  requires (eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (dynamic_rows<Arg> or Coefficients::euclidean_dimensions == row_extent_of_v<Arg>) and
    (dynamic_columns<Arg> or Coefficients::euclidean_dimensions == column_extent_of_v<Arg>)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<fixed_coefficients<Coefficients> and
    (eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (dynamic_rows<Arg> or Coefficients::euclidean_dimensions == row_extent_of<Arg>::value) and
    (dynamic_columns<Arg> or Coefficients::euclidean_dimensions == column_extent_of<Arg>::value), int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg) noexcept
  {
    if constexpr (dynamic_rows<Arg>) if (row_count(arg) != Coefficients::euclidean_dimensions)
      throw std::out_of_range {"Number of rows (" + std::to_string(row_count(arg)) +
        ") does not match Coefficient euclidean dimensions " + std::to_string(Coefficients::euclidean_dimensions)};

    if constexpr (Coefficients::axes_only)
      return std::forward<Arg>(arg);
    else
      return FromEuclideanExpr<Coefficients, equivalent_dense_writable_matrix_t<Arg>>(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<fixed_coefficients Coefficients, typename Arg>
  requires (eigen_zero_expr<Arg> or eigen_constant_expr<Arg> or eigen_diagonal_expr<Arg> or
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (dynamic_rows<Arg> or Coefficients::dimensions == row_extent_of_v<Arg>)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<fixed_coefficients<Coefficients> and
    (eigen_zero_expr<Arg> or eigen_constant_expr<Arg> or eigen_diagonal_expr<Arg> or
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (dynamic_rows<Arg> or Coefficients::dimensions == row_extent_of<Arg>::value), int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg) noexcept
  {
    if constexpr (dynamic_rows<Arg>) if (row_count(arg) != Coefficients::dimensions)
      throw std::out_of_range {"Number of rows (" + std::to_string(row_count(arg)) +
        ") does not match Coefficient dimensions (" + std::to_string(Coefficients::dimensions) + ")"};

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
  template<eigen_zero_expr Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_zero_expr<Arg> and (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    using Scalar = scalar_type_of_t<Arg>;

    constexpr std::size_t dim = dynamic_rows<Arg> ? column_extent_of_v<Arg> : row_extent_of_v<Arg>;

    if constexpr (dim == dynamic_extent)
    {
      auto rows = row_count(arg);
      assert(rows == column_count(arg));
      return ZeroMatrix<Scalar, dynamic_extent, 1> {rows};
    }
    else
    {
      if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
      return ZeroMatrix<Scalar, dim, 1> {};
    }
  }


#ifdef __cpp_concepts
  template<eigen_constant_expr Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_constant_expr<Arg> and (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    using Scalar = scalar_type_of_t<Arg>;

    constexpr std::size_t dim = dynamic_rows<Arg> ? column_extent_of_v<Arg> : row_extent_of_v<Arg>;

    constexpr auto constant = constant_coefficient_v<Arg>;

    if constexpr (dim == dynamic_extent)
    {
      auto rows = row_count(arg);
      assert(rows == column_count(arg));
      return ConstantMatrix<Scalar, constant, dynamic_extent, 1> {row_count(arg)};
    }
    else
    {
      if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
      return ConstantMatrix<Scalar, constant, dim, 1> {};
    }
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline decltype(auto)
  diagonal_of(Arg&& arg) noexcept
  {
    return nested_matrix(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>, int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    return diagonal_of(nested_matrix(std::forward<Arg>(arg)));
  }


} // namespace OpenKalman::Eigen3


namespace OpenKalman::interface
{

#ifdef __cpp_concepts
  template<typename T, typename...I> requires eigen_zero_expr<T> or eigen_constant_expr<T>
  struct GetElement<T, I...>
#else
  template<typename T, typename...I>
  struct GetElement<T, std::enable_if_t<eigen_zero_expr<T> or eigen_constant_expr<T>>, I...>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, I...) { return constant_coefficient_v<Arg>; }
  };


#ifdef __cpp_concepts
  template<eigen_diagonal_expr T, std::convertible_to<const std::size_t&>...I> requires (sizeof...(I) <= 2) and
    (element_gettable<nested_matrix_of<T>, std::size_t> or
      element_gettable<nested_matrix_of<T>, std::size_t, std::size_t>)
  struct GetElement<T, I...>
#else
  template<typename T, typename...I>
  struct GetElement<T, std::enable_if_t<eigen_diagonal_expr<T> and
    ((sizeof...(I) <= 2) and ... and std::is_convertible_v<I, const std::size_t&>) and
    (element_gettable<nested_matrix_of<T>, std::size_t> or
      element_gettable<nested_matrix_of<T>, std::size_t, std::size_t>)>, I...>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, const std::size_t row, const std::size_t col)
    {
      if (row == col)
      {
        if constexpr (element_gettable<nested_matrix_of<Arg>, std::size_t>)
          return get_element(nested_matrix(std::forward<Arg>(arg)), row);
        else
          return get_element(nested_matrix(std::forward<Arg>(arg)), row, 1);
      }
      else
      {
        return scalar_type_of_t<Arg>(0);
      }
    }

    template<typename Arg>
    constexpr auto get(Arg&& arg, const std::size_t i)
    {
      if constexpr (element_gettable<nested_matrix_of<Arg>, std::size_t>)
        return get_element(nested_matrix(std::forward<Arg>(arg)), i);
      else
        return get_element(nested_matrix(std::forward<Arg>(arg)), i, 1);
    }
  };


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr T, std::convertible_to<const std::size_t&>...I> requires (sizeof...(I) == 2) and
    (not diagonal_matrix<T>) and element_gettable<nested_matrix_of<T>, std::size_t, std::size_t>
  struct GetElement<T, I...>
#else
  template<typename T, typename...I>
  struct GetElement<T, std::enable_if_t<eigen_self_adjoint_expr<T> and
    ((sizeof...(I) <= 2) and ... and std::is_convertible_v<I, const std::size_t&>) and
    (not diagonal_matrix<T>) and element_gettable<nested_matrix_of<T>, std::size_t, std::size_t>>, I...>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, const std::size_t i, const std::size_t j)
    {
      decltype(auto) n = nested_matrix(std::forward<Arg>(arg));
      using N = decltype(n);
      using Scalar = scalar_type_of_t<Arg>;

      if (lower_self_adjoint_matrix<Arg> ? i >= j : i <= j)
      {
        if constexpr (complex_number<Scalar>)
        {
          if (i == j) return std::real(get_element(std::forward<N>(n), i, j));
        }
        return get_element(std::forward<N>(n), i, j);
      }
      else
      {
        if constexpr (complex_number<Scalar>)
          return std::conj(get_element(std::forward<N>(n), j, i));
        else
          return get_element(std::forward<N>(n), j, i);
      }
    }
  };


#ifdef __cpp_concepts
  template<eigen_triangular_expr T, std::convertible_to<const std::size_t&>...I> requires (sizeof...(I) == 2) and
    (not diagonal_matrix<T>) and element_gettable<nested_matrix_of<T>, std::size_t, std::size_t>
  struct GetElement<T, I...>
#else
  template<typename T, typename...I>
  struct GetElement<T, std::enable_if_t<eigen_triangular_expr<T> and
    ((sizeof...(I) <= 2) and ... and std::is_convertible_v<I, const std::size_t&>) and
    (not diagonal_matrix<T>) and element_gettable<nested_matrix_of<T>, std::size_t, std::size_t>>, I...>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, const std::size_t i, const std::size_t j)
    {
      if (lower_triangular_matrix<Arg> ? i >= j : i <= j)
        return get_element(nested_matrix(std::forward<Arg>(arg)), i, j);
      else
        return scalar_type_of_t<Arg>(0);
    }
  };


#ifdef __cpp_concepts
  template<diagonal_matrix T, std::convertible_to<const std::size_t&>...I> requires (sizeof...(I) <= 2) and
    (not diagonal_matrix<T>) and (eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>) and
    (element_gettable<nested_matrix_of<T>, std::size_t, std::size_t> or
      element_gettable<nested_matrix_of<T>, std::size_t>)
  struct GetElement<T, I...>
#else
  template<typename T, typename...I>
  struct GetElement<T, std::enable_if_t<diagonal_matrix<T> and
    ((sizeof...(I) <= 2) and ... and std::is_convertible_v<I, const std::size_t&>) and
    (not diagonal_matrix<T>) and (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<T>) and
    (element_gettable<nested_matrix_of<T>, std::size_t, std::size_t> or
      element_gettable<nested_matrix_of<T>, std::size_t>)>, I...>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, const std::size_t i, const std::size_t j)
    {
      if (i == j)
      {
        if constexpr(element_gettable<nested_matrix_of<Arg>, std::size_t>)
          return get_element(nested_matrix(std::forward<Arg>(arg)), i);
        else
          return get_element(nested_matrix(std::forward<Arg>(arg)), i, i);
      }
      else return scalar_type_of_t<Arg>(0);
    }


    template<typename Arg>
    constexpr auto get(Arg&& arg, const std::size_t i)
    {
      using NestedMatrix = nested_matrix_of<Arg>;
      if constexpr(element_gettable<NestedMatrix, std::size_t>)
      {
        return get(nested_matrix(std::forward<Arg>(arg)), i);
      }
      else
      {
        return get(nested_matrix(std::forward<Arg>(arg)), i, i);
      }
    }
  };


#ifdef __cpp_concepts
  template<eigen_diagonal_expr T, std::convertible_to<const std::size_t&>...I> requires (sizeof...(I) <= 2) and
    (element_settable<nested_matrix_of<T>, std::size_t> or
      element_settable<nested_matrix_of<T>, std::size_t, std::size_t>)
  struct SetElement<T, I...>
#else
  template<typename T, typename...I>
  struct SetElement<T, std::enable_if_t<eigen_diagonal_expr<T> and
    ((sizeof...(I) <= 2) and ... and std::is_convertible_v<I, const std::size_t&>) and
    (element_settable<nested_matrix_of<T>, std::size_t> or
      element_settable<nested_matrix_of<T>, std::size_t, std::size_t>)>, I...>
#endif
  {
    template<typename Arg, typename Scalar>
    static void set(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
    {
      if (i == j)
      {
        if constexpr (element_settable<nested_matrix_of<Arg>, std::size_t>)
          set_element(nested_matrix(arg), s, i);
        else
          set_element(nested_matrix(arg), s, i, 1);
      }
      else if (s != 0)
        throw std::out_of_range("Cannot set non-diagonal element of a diagonal matrix to a non-zero value.");
    }


    template<typename Arg, typename Scalar>
    static void set(Arg& arg, const Scalar s, const std::size_t i)
    {
      if constexpr (element_settable<nested_matrix_of<Arg>, std::size_t>)
        set_element(nested_matrix(arg), s, i);
      else
        set_element(nested_matrix(arg), s, i, 1);
    }
  };


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr T, std::convertible_to<const std::size_t&>...I> requires (sizeof...(I) == 2) and
    (not diagonal_matrix<T>) and element_settable<nested_matrix_of<T>, std::size_t, std::size_t>
  struct SetElement<T, I...>
#else
  template<typename T, typename...I>
  struct SetElement<T, std::enable_if_t<eigen_self_adjoint_expr<T> and
    ((sizeof...(I) == 2) and ... and std::is_convertible_v<I, const std::size_t&>) and
    (not diagonal_matrix<T>) and element_settable<nested_matrix_of<T>, std::size_t, std::size_t>>, I...>
#endif
  {
    template<typename Arg, typename Scalar>
    static void set(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
    {
      if (lower_self_adjoint_matrix<Arg> ? i >= j : i <= j)
      {
        set_element(nested_matrix(arg), s, i, j);
      }
      else
      {
        if constexpr (complex_number<Scalar>)
          set_element(nested_matrix(arg), std::conj(s), j, i);
        else
          set_element(nested_matrix(arg), s, j, i);
      }
    }
  };


#ifdef __cpp_concepts
  template<eigen_triangular_expr T, std::convertible_to<const std::size_t&>...I> requires (sizeof...(I) == 2) and
    (not diagonal_matrix<T>) and element_settable<nested_matrix_of<T>, std::size_t, std::size_t>
  struct SetElement<T, I...>
#else
  template<typename T, typename...I>
  struct SetElement<T, std::enable_if_t<eigen_triangular_expr<T> and
    ((sizeof...(I) == 2) and ... and std::is_convertible_v<I, const std::size_t&>) and
    (not diagonal_matrix<T>) and element_settable<nested_matrix_of<T>, std::size_t, std::size_t>>, I...>
#endif
  {
    template<typename Arg, typename Scalar>
    static void set(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
    {
      if (lower_triangular_matrix<Arg> ? i >= j : i <= j)
        set_element(nested_matrix(arg), s, i, j);
      else if (s != 0)
        throw std::out_of_range("Cannot set elements of a triangular matrix to non-zero values outside the triangle.");
    }
  };


#ifdef __cpp_concepts
  template<diagonal_matrix T, std::convertible_to<const std::size_t&>...I> requires (sizeof...(I) <= 2) and
    (eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>) and
    (element_settable<nested_matrix_of<T>, std::size_t> or
      element_settable<nested_matrix_of<T>, std::size_t, std::size_t>)
  struct SetElement<T, I...>
#else
  template<typename T, typename...I>
  struct SetElement<T, std::enable_if_t<diagonal_matrix<T> and
    ((sizeof...(I) <= 2) and ... and std::is_convertible_v<I, const std::size_t&>) and
    (eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>) and
    (element_settable<nested_matrix_of<T>, std::size_t> or
      element_settable<nested_matrix_of<T>, std::size_t, std::size_t>)>, I...>
#endif
  {
    template<typename Arg, typename Scalar>
    static void set(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
    {
      if (i == j)
      {
        if constexpr(element_settable<nested_matrix_of<Arg>, std::size_t>)
          set_element(nested_matrix(arg), s, i);
        else
          set_element(nested_matrix(arg), s, i, i);
      }
      else if (s != 0)
        throw std::out_of_range("Cannot set non-diagonal element of a diagonal matrix to a non-zero value.");
    }


    template<typename Arg, typename Scalar>
    static void set(Arg& arg, const Scalar s, const std::size_t i)
    {
      using NestedMatrix = nested_matrix_of<Arg>;
      if constexpr(element_settable<NestedMatrix, std::size_t>)
        set_element(nested_matrix(arg), s, i);
      else
        set_element(nested_matrix(arg), s, i, i);
    }
  };


#ifdef __cpp_concepts
  template<typename T> requires eigen_zero_expr<T> or eigen_constant_expr<T> or eigen_diagonal_expr<T> or
    eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>
  struct ElementWiseOperations<T>
#else
  template<typename T>
  struct ElementWiseOperations<T, std::enable_if_t<eigen_zero_expr<T> or eigen_constant_expr<T> or
    eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>>>
#endif
  {

    template<ElementOrder order, typename BinaryFunction, typename Accum, typename Arg>
    static constexpr auto fold(const BinaryFunction& b, Accum&& accum, Arg&& arg)
    {
      return OpenKalman::fold<order>(b, std::forward<Accum>(accum), make_native_matrix(std::forward<Arg>(arg)));
    }

  };


#ifdef __cpp_concepts
  template<typename T> requires eigen_zero_expr<T> or eigen_constant_expr<T> or eigen_diagonal_expr<T> or
    eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>
  struct Conversions<T>
#else
  template<typename T>
  struct Conversions<T, std::enable_if_t<eigen_zero_expr<T> or eigen_constant_expr<T> or eigen_diagonal_expr<T> or
    eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>>>
#endif
  {
  };


#ifdef __cpp_concepts
  template<typename T> requires eigen_zero_expr<T> or eigen_constant_expr<T> or eigen_diagonal_expr<T> or
    eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>
  struct LinearAlgebra<T>
#else
  template<typename T>
  struct linearAlgebra<T, std::enable_if_t<eigen_zero_expr<T> or eigen_constant_expr<T> or eigen_diagonal_expr<T> or
    eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>>>
#endif
  {

    template<typename Arg>
    static constexpr decltype(auto) conjugate(Arg&& arg) noexcept
    {
      if constexpr (eigen_constant_expr<Arg>)
      {
        using Scalar = scalar_type_of_t<Arg>;
        constexpr auto constant = constant_coefficient_v<Arg>;
        constexpr auto rows = row_extent_of_v<Arg>;
        constexpr auto cols = column_extent_of_v<Arg>;

#     ifdef __cpp_lib_constexpr_complex
        constexpr auto adj = std::conj(constant);
#     else
        constexpr auto adj = std::complex(std::real(constant), -std::imag(constant));
#     endif

        if constexpr (dynamic_rows<Arg>)
        {
          if constexpr (dynamic_columns<Arg>)
            return ConstantMatrix<Scalar, adj, dynamic_extent, dynamic_extent> {row_count(arg), column_count(arg)};
          else
            return ConstantMatrix<Scalar, adj, dynamic_extent, cols> {row_count(arg)};
        }
        else
        {
          if constexpr (dynamic_columns<Arg>)
            return ConstantMatrix<Scalar, adj, rows, dynamic_extent> {column_count(arg)};
          else
            return ConstantMatrix<Scalar, adj, rows, cols> {};
        }
      }
      else if constexpr (eigen_diagonal_expr<Arg>)
      {
        return make_self_contained<Arg>(to_diagonal(OpenKalman::conjugate(diagonal_of(std::forward<Arg>(arg)))));
      }
      else if constexpr (eigen_self_adjoint_expr<Arg>)
      {
        auto n = make_self_contained<Arg>(OpenKalman::conjugate(nested_matrix(std::forward<Arg>(arg))));
        return MatrixTraits<Arg>::template make<self_adjoint_triangle_type_of_v<Arg>>(std::move(n));
      }
      else
      {
        static_assert(eigen_triangular_expr<Arg>);
        auto n = make_self_contained<Arg>(OpenKalman::conjugate(nested_matrix(std::forward<Arg>(arg))));
        return MatrixTraits<Arg>::template make<triangle_type_of_v<Arg>>(std::move(n));
      }
    }


  private:

    template<auto constant, typename Arg>
    static constexpr decltype(auto) eigen_constant_transpose_impl(Arg&& arg) noexcept
    {
      using Scalar = scalar_type_of_t<Arg>;
      constexpr auto rows = row_extent_of_v<Arg>;
      constexpr auto cols = column_extent_of_v<Arg>;

      if constexpr (dynamic_rows<Arg>)
      {
        if constexpr (dynamic_columns<Arg>)
          return ConstantMatrix<Scalar, constant, dynamic_extent, dynamic_extent> {column_count(arg), row_count(arg)};
        else
          return ConstantMatrix<Scalar, constant, cols, dynamic_extent> {row_count(arg)};
      }
      else
      {
        if constexpr (dynamic_columns<Arg>)
          return ConstantMatrix<Scalar, constant, dynamic_extent, rows> {column_count(arg)};
        else if constexpr (rows == cols)
          return std::forward<Arg>(arg);
        else
          return ConstantMatrix<Scalar, constant, cols, rows> {};
      }
    }

  public:

    template<typename Arg>
    static constexpr decltype(auto) transpose(Arg&& arg) noexcept
    {
      if constexpr (eigen_zero_expr<Arg>)
      {
        using Scalar = scalar_type_of_t<Arg>;
        constexpr auto rows = row_extent_of_v<Arg>;
        constexpr auto cols = column_extent_of_v<Arg>;

        if constexpr (dynamic_rows<Arg>)
        {
          if constexpr (dynamic_columns<Arg>)
            return ZeroMatrix<Scalar, dynamic_extent, dynamic_extent> {column_count(arg), row_count(arg)};
          else
            return ZeroMatrix<Scalar, cols, dynamic_extent> {row_count(arg)};
        }
        else
        {
          if constexpr (dynamic_columns<Arg>)
            return ZeroMatrix<Scalar, dynamic_extent, rows> {column_count(arg)};
          else
            return ZeroMatrix<Scalar, cols, rows> {};
        }
      }
      else if constexpr (eigen_constant_expr<Arg>)
      {
        return eigen_constant_transpose_impl<constant_coefficient_v<Arg>>(std::forward<Arg>(arg));
      }
      else if constexpr (eigen_self_adjoint_expr<Arg>)
      {
        if constexpr (self_adjoint_matrix<nested_matrix_of<Arg>>)
        {
          return OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg)));
        }
        else
        {
          constexpr auto t = (lower_self_adjoint_matrix<Arg> ? TriangleType::upper : TriangleType::lower);
          return MatrixTraits<Arg>::template make<t>(
            make_self_contained<Arg>(OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg)))));
        }
      }
      else if constexpr (eigen_triangular_expr<Arg>)
      {
        if constexpr (triangular_matrix<nested_matrix_of<Arg>>)
        {
          return OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg)));
        }
        else
        {
          constexpr auto t = lower_triangular_matrix<Arg> ? TriangleType::upper : TriangleType::lower;
          return MatrixTraits<Arg>::template make<t>(
            make_self_contained<Arg>(OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg)))));
        }
      }
    }


    template<typename Arg>
    static constexpr decltype(auto) adjoint(Arg&& arg) noexcept
    {
      if constexpr (constant_matrix<Arg>)
      {
        constexpr auto constant = constant_coefficient_v<Arg>;

#     ifdef __cpp_lib_constexpr_complex
        constexpr auto adj = std::conj(constant);
#     else
        constexpr auto adj = std::complex(std::real(constant), -std::imag(constant));
#     endif

        return eigen_constant_transpose_impl<adj>(std::forward<Arg>(arg));
      }
      else if constexpr (eigen_diagonal_expr<Arg>)
      {
        return make_self_contained<Arg>(to_diagonal(OpenKalman::conjugate(diagonal_of(std::forward<Arg>(arg)))));
      }
      else
      {
        static_assert(eigen_triangular_expr<Arg>);

        if constexpr (diagonal_matrix<nested_matrix_of<Arg>>)
        {
          return make_self_contained<Arg>(
            to_diagonal(OpenKalman::conjugate(diagonal_of(nested_matrix(std::forward<Arg>(arg))))));
        }
        else
        {
          constexpr auto t = lower_self_adjoint_matrix<Arg> or lower_triangular_matrix<Arg> ?
            TriangleType::upper : TriangleType::lower;
          return MatrixTraits<Arg>::template make<t>(
            make_self_contained<Arg>(OpenKalman::adjoint(nested_matrix(std::forward<Arg>(arg)))));
        }
      }
    }


    template<typename Arg>
    static constexpr auto determinant(Arg&& arg) noexcept
    {
      // The determinant function handles eigen_zero_expr<T> or eigen_constant_expr<T> cases.
      if (diagonal_matrix<Arg>)
      {
        return fold<ElementOrder::column_major>(std::multiplies{}, 1, diagonal_of(std::forward<Arg>(arg)));
      }
      else
      {
        static_assert(eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>);
        return OpenKalman::determinant(make_native_matrix(std::forward<Arg>(arg)));
      }
    }


    template<typename Arg>
    static constexpr auto trace(Arg&& arg) noexcept
    {
      // The trace function handles eigen_zero_expr<T> or eigen_constant_expr<T> cases.
      if constexpr (eigen_diagonal_expr<Arg>)
      {
        return fold<ElementOrder::column_major>(std::plus{}, 0, diagonal_of(std::forward<Arg>(arg)));
      }
      else
      {
        static_assert(eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>);
        return OpenKalman::trace(nested_matrix(std::forward<Arg>(arg)));
      }
    }


#ifdef __cpp_concepts
    template<TriangleType t, typename A, typename U, typename Alpha> requires (not eigen_triangular_expr<A>)
#else
    template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
      not eigen_triangular_expr<A>, int> = 0>
#endif
    static decltype(auto) rank_update_self_adjoint(A&& a, U&& u, const Alpha alpha = 1)
    {
      if constexpr (zero_matrix<A>)
      {
        if constexpr (diagonal_matrix<U>)
        {
          auto du = diagonal_of(std::forward<U>(u));
          return to_diagonal(alpha * du * adjoint(du));
        }
        else
        {
          auto res = alpha * u * adjoint(std::forward<U>(u)); // \todo This is probably not efficient
          return SelfAdjointMatrix<std::decay_t<decltype(res)>, t> {std::move(res)};
        }
      }
      // \todo Add diagonal case
      else
      {
        static_assert(eigen_self_adjoint_expr<A>);
        return rank_update_self_adjoint<t>(nested_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
    }


#ifdef __cpp_concepts
    template<TriangleType t, typename A, typename U, typename Alpha> requires (not eigen_self_adjoint_expr<A>)
#else
    template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
      not eigen_self_adjoint_expr<A>, int> = 0>
#endif
    static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha = 1)
    {
      if constexpr (zero_matrix<A>)
      {
        if constexpr (diagonal_matrix<U>)
        {
          return to_diagonal(std::sqrt(alpha) * diagonal_of(std::forward<U>(u)));
        }
        else if constexpr (t == TriangleType::upper)
        {
          return QR_decomposition(std::sqrt(alpha) * adjoint(std::forward<U>(u)));
        }
        else
        {
          return LQ_decomposition(std::sqrt(alpha) * std::forward<U>(u));
        }
      }
      else if constexpr (diagonal_matrix<A>)
      {
        if constexpr (diagonal_matrix<U>)
        {
          auto a2 = (nested_matrix(a).array().square() + alpha * diagonal_of(u).array().square()).sqrt().matrix();
          if constexpr (std::is_lvalue_reference_v<A> and not std::is_const_v<std::remove_reference_t<A>>)
          {
            a.nested_matrix() = std::move(a2);
          }
          else
          {
            return make_self_contained(to_diagonal(std::move(a2)));
          }
        }
        else
        {
          auto m = make_native_matrix(std::forward<A>(a));
          TriangularMatrix<std::remove_const_t<decltype(m)>> sa {std::move(m)};
          rank_update(sa, u, alpha);
          return sa;
        }
      }
      else
      {
        static_assert(eigen_triangular_expr<A>);
        return rank_update_triangular<t>(nested_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
    }

  };


} // namespace OpenKalman::interface


namespace OpenKalman::Eigen3
{

  /**
   * \brief Solve the equation AX = B for X. A is an invertible square zero matrix.
   * \note These solutions, if they exist, are non-unique.
   * \returns A zero matrix.
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr A, eigen_matrix B>
  requires (dynamic_shape<A> or square_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<B> or row_extent_of_v<A> == row_extent_of_v<B>)
#else
  template<typename A, typename B, std::enable_if_t<eigen_zero_expr<A> and eigen_matrix<B> and
      (dynamic_shape<A> or square_matrix<A>) and
      (dynamic_rows<A> or dynamic_rows<B> or row_extent_of<A>::value == row_extent_of<B>::value), int> = 0>
#endif
  constexpr auto
  solve(A&& a, B&& b)
  {
    using Scalar = scalar_type_of_t<A>;

    if constexpr (dynamic_shape<A>) assert(row_count(a) == column_count(a));
    if constexpr (dynamic_rows<A> or dynamic_rows<B>) assert(row_count(a) == row_count(b));

    constexpr std::size_t dim = dynamic_rows<A> ?
      (dynamic_rows<B> ? column_extent_of_v<A>: row_extent_of_v<B>) : row_extent_of_v<A>;

    if constexpr (zero_matrix<B>)
    {
      return std::forward<B>(b);
    }
    // For the remainder of the cases, there is no actual solution unless b is zero, so we pick zero.
    else if constexpr (dim == dynamic_extent)
    {
      if constexpr (dynamic_columns<B>)
        return ZeroMatrix<Scalar, dynamic_extent, dynamic_extent> {row_count(b), column_count(b)};
      else
        return ZeroMatrix<Scalar, dynamic_extent, column_extent_of_v<B>> {row_count(b)};
    }
    else
    {
      if constexpr (dynamic_columns<B>)
        return ZeroMatrix<Scalar, dim, dynamic_extent> {column_count(b)};
      else
        return ZeroMatrix<Scalar, dim, column_extent_of_v<B>> {};
    }
  }


  /// Solve the equation AX = B for X. A is an invertible square matrix.
#ifdef __cpp_concepts
  template<eigen_constant_expr A, eigen_matrix B>
  requires (dynamic_shape<A> or square_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<B> or row_extent_of_v<A> == row_extent_of_v<B>)
#else
  template<typename A, typename B, std::enable_if_t<eigen_constant_expr<A> and eigen_matrix<B> and
    (dynamic_shape<A> or square_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<B> or row_extent_of<A>::value == row_extent_of<B>::value), int> = 0>
#endif
  constexpr auto
  solve(A&& a, B&& b)
  {
    using Scalar = scalar_type_of_t<A>;

    if constexpr (dynamic_shape<A>) assert(row_count(a) == column_count(a));
    if constexpr (dynamic_rows<A> or dynamic_rows<B>) assert(row_count(a) == row_count(b));

    constexpr std::size_t dim = dynamic_rows<A> ?
      (dynamic_rows<B> ? column_extent_of_v<A>: row_extent_of_v<B>) : row_extent_of_v<A>;

    if constexpr (zero_matrix<B>)
    {
      return std::forward<B>(b);
    }
    else if constexpr (zero_matrix<A>)
    {
      return solve(ZeroMatrix {std::forward<A>(a)}, std::forward<B>(b));
    }
    else if constexpr (dim == 1)
    {
      return make_self_contained(std::forward<B>(b) / constant_coefficient_v<A>);
    }
    else if constexpr (constant_matrix<B>)
    {
      if constexpr (dim == dynamic_extent)
      {
        auto c = static_cast<Scalar>(constant_coefficient_v<B>) / (row_count(b) * constant_coefficient_v<A>);
        return eigen_matrix_t<Scalar, dim, column_extent_of_v<B>> ::Constant(row_count(b), column_count(b), c);
      }
      else
      {
#if __cpp_nontype_template_args >= 201911L
        constexpr auto c = static_cast<Scalar>(constant_coefficient_v<B>) / (dim * constant_coefficient_v<A>);

        if constexpr (dynamic_columns<B>)
          return ConstantMatrix<Scalar, c, dim, dynamic_extent> {column_count(b)};
        else
          return ConstantMatrix<Scalar, c, dim, column_extent_of_v<B>> {};
#else
        if constexpr(constant_coefficient_v<B> % (dim * constant_coefficient_v<A>) == 0)
        {
          constexpr auto c = constant_coefficient_v<B> / (dim * constant_coefficient_v<A>);

          if constexpr (dynamic_columns<B>)
            return ConstantMatrix<Scalar, c, dim, dynamic_extent> {column_count(b)};
          else
            return ConstantMatrix<Scalar, c, dim, column_extent_of_v<B>> {};
        }
        else
        {
          auto c = static_cast<Scalar>(constant_coefficient_v<B>) / (row_count(b) * constant_coefficient_v<A>);

          if constexpr (dynamic_columns<B>)
            return eigen_matrix_t<Scalar, dim, dynamic_extent>::Constant(row_count(b), column_count(b), c);
          else
            return eigen_matrix_t<Scalar, dim, column_extent_of_v<B>>::Constant(row_count(b), column_count(b), c);
        }
#endif
      }
    }
    else
    {
      // In this general case, there is only an exact solution if all coefficients in each column are the same.
      // We select a good approximate solution.

      if constexpr (dim == dynamic_extent or not dynamic_rows<B>)
      {
        return make_self_contained(b / (row_count(b) * constant_coefficient_v<A>));
      }
      else if constexpr (dynamic_columns<B>)
      {
        using M = eigen_matrix_t<Scalar, dim, dynamic_extent>;
        return M {b / (row_count(b) * constant_coefficient_v<A>)};
      }
      else
      {
        using M = eigen_matrix_t<Scalar, dim, column_extent_of_v<B>>;
        return M {b / (row_count(b) * constant_coefficient_v<A>)};
      }
    }
  }


  /// Solve the equation AX = B for X. A is a diagonal matrix.
#ifdef __cpp_concepts
  template<eigen_diagonal_expr A, eigen_matrix B> requires (row_extent_of_v<A> == row_extent_of_v<B>)
#else
  template<typename A, typename B, std::enable_if_t<eigen_diagonal_expr<A> and eigen_matrix<B> and
    (row_extent_of<A>::value == row_extent_of<B>::value), int> = 0>
#endif
  inline auto
  solve(const A& a, const B& b)
  {
    if constexpr (dynamic_shape<A>) assert(row_count(a) == column_count(a));
    if constexpr (dynamic_rows<A> or dynamic_rows<B>) assert(row_count(a) == row_count(b));

    return (b.array().colwise() / nested_matrix(a).array()).matrix();
  }


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr A, eigen_matrix B>
#else
  template<typename A, typename B, std::enable_if_t<eigen_self_adjoint_expr<A> and eigen_matrix<B>, int> = 0>
#endif
  constexpr auto
  solve(A&& a, B&& b)
  {
    using Scalar = scalar_type_of_t<A>;
    static_assert(std::is_same_v<Scalar, scalar_type_of_t<B>>);

    if constexpr (dynamic_rows<A> or dynamic_rows<B>) assert(row_count(a) == row_count(b));

    auto v {std::forward<A>(a).view()};
    using M = Eigen3::eigen_matrix_t<Scalar, row_extent_of_v<A>, column_extent_of_v<B>>;
    auto llt {v.llt()};

    M ret;
    if (llt.info() == Eigen::Success)
    {
      ret = Eigen::Solve {llt, std::forward<B>(b)};
    }
    else [[unlikely]]
    {
      // A is semidefinite. Use LDLT decomposition instead.
      auto ldlt {v.ldlt()};
      if ((not ldlt.isPositive() and not ldlt.isNegative()) or ldlt.info() != Eigen::Success)
      {
        throw (std::runtime_error("SelfAdjointMatrix solve: A is indefinite"));
      }
      ret = Eigen::Solve {ldlt, std::forward<B>(b)};
    }
    return ret;
  }


#ifdef __cpp_concepts
  template<eigen_triangular_expr A, eigen_matrix B>
#else
  template<typename A, typename B, std::enable_if_t<eigen_triangular_expr<A> and eigen_matrix<B>, int> = 0>
#endif
  constexpr auto
  solve(A&& a, B&& b)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<B>) assert(row_count(a) == row_count(b));

    return make_self_contained<A, B>(Eigen::Solve {std::forward<A>(a).view(), std::forward<B>(b)});
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    using Scalar = scalar_type_of_t<Arg>;

    if constexpr (column_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (dynamic_rows<Arg>)
    {
      return ZeroMatrix<Scalar, dynamic_extent, 1> {row_count(arg)};
    }
    else
    {
      constexpr auto rows = row_extent_of_v<Arg>;
      return ZeroMatrix<Scalar, rows, 1> {};
    }
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    using Scalar = scalar_type_of_t<Arg>;
    constexpr auto constant = constant_coefficient_v<Arg>;

    if constexpr (column_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (dynamic_rows<Arg>)
    {
      return ConstantMatrix<Scalar, constant, dynamic_extent, 1> {row_count(arg)};
    }
    else
    {
      return ConstantMatrix<Scalar, constant, row_extent_of_v<Arg>, 1> {};
    }
  }


  /// Create a column vector from a diagonal matrix. (Same as nested_matrix()).
#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    return nested_matrix(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg,
    std::enable_if_t<eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_columns(Arg&& arg)
  {
    return make_native_matrix(make_native_matrix(std::forward<Arg>(arg)).rowwise().sum() / row_extent_of_v<Arg>);
  }


  /// Create a row vector by taking the mean of each column in a set of row vectors.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_rows(Arg&& arg) noexcept
  {
    using Scalar = scalar_type_of_t<Arg>;

    if constexpr (row_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (dynamic_columns<Arg>)
    {
      return ZeroMatrix<Scalar, 1, dynamic_extent> {column_count(arg)};
    }
    else
    {
      return ZeroMatrix<Scalar, 1, column_extent_of_v<Arg>> {};
    }
  }


  /// Create a row vector by taking the mean of each column in a set of row vectors.
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_rows(Arg&& arg) noexcept
  {
    using Scalar = scalar_type_of_t<Arg>;
    constexpr auto constant = constant_coefficient_v<Arg>;

    if constexpr (row_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (dynamic_columns<Arg>)
    {
      return ConstantMatrix<Scalar, constant, 1, dynamic_extent> {column_count(arg)};
    }
    else
    {
      return ConstantMatrix<Scalar, constant, 1, column_extent_of_v<Arg>> {};
    }
  }


  /// Create a row vector from a diagonal matrix. (Same as nested_matrix()).
#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_rows(Arg&& arg) noexcept
  {
    return nested_matrix(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg,
      std::enable_if_t<eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_rows(Arg&& arg)
  {
    return make_native_matrix(make_native_matrix(std::forward<Arg>(arg)).colwise().sum() / column_count(arg));
  }


  /**
   * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * Returns L as a lower-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr A>
#else
  template<typename A, std::enable_if_t<eigen_zero_expr<A>, int> = 0>
#endif
  inline auto
  LQ_decomposition(A&& a)
  {
    using Scalar = scalar_type_of_t<A>;
    if constexpr (dynamic_rows<A>)
    {
      auto dim = row_count(a);
      return ZeroMatrix<Scalar, dynamic_extent, dynamic_extent> {dim, dim};
    }
    else
    {
      constexpr auto dim = row_extent_of_v<A>;
      return ZeroMatrix<Scalar, dim, dim> {};
    }
  }


/**
 * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
 * Returns L as a lower-triangular matrix.
 */
#ifdef __cpp_concepts
  template<eigen_constant_expr A>
#else
  template<typename A, std::enable_if_t<eigen_constant_expr<A>, int> = 0>
#endif
  inline auto
  LQ_decomposition(A&& a)
  {
    using Scalar = scalar_type_of_t<A>;
    constexpr auto constant = constant_coefficient_v<A>;

    const Scalar elem = constant * (
      dynamic_columns<A> ?
      std::sqrt((Scalar) column_count(a)) :
      OpenKalman::internal::constexpr_sqrt((Scalar) column_extent_of_v<A>));

    if constexpr (dynamic_rows<A>)
    {
      auto dim = row_count(a);
      auto col1 = Eigen3::eigen_matrix_t<Scalar, dynamic_extent, 1>::Constant(dim, elem);

      eigen_matrix_t<Scalar, dynamic_extent, dynamic_extent> ret {dim, dim};

      if (dim == 1)
        ret = std::move(col1);
      else
        ret = concatenate_horizontal(std::move(col1), ZeroMatrix<Scalar, dynamic_extent, dynamic_extent> {dim, dim - 1});
      return ret;
    }
    else
    {
      constexpr auto dim = row_extent_of_v<A>;
      auto col1 = Eigen3::eigen_matrix_t<Scalar, dim, 1>::Constant(elem);

      if constexpr (dim > 1)
        return concatenate_horizontal(col1, ZeroMatrix<Scalar, dim, dim - 1> {});
      else
        return col1;
    }
  }


  /**
   * \brief Perform an LQ decomposition of matrix A=[L,0]Q, where L is a lower-triangular matrix, and Q is orthogonal.
   * \return L as a lower-triangular matrix.
   */
#ifdef __cpp_concepts
  template<typename A> requires eigen_diagonal_expr<A> or eigen_self_adjoint_expr<A> or eigen_triangular_expr<A>
#else
  template<typename A, std::enable_if_t<
    eigen_diagonal_expr<A> or eigen_self_adjoint_expr<A> or eigen_triangular_expr<A>, int> = 0>
#endif
  constexpr decltype(auto)
  LQ_decomposition(A&& a)
  {
    if constexpr(lower_triangular_matrix<A>) return std::forward<A>(a);
    else return LQ_decomposition(make_native_matrix(std::forward<A>(a)));
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr A>
#else
  template<typename A, std::enable_if_t<eigen_zero_expr<A>, int> = 0>
#endif
  inline auto
  QR_decomposition(A&& a)
  {
    using Scalar = scalar_type_of_t<A>;
    if constexpr (dynamic_columns<A>)
    {
      auto dim = column_count(a);
      return ZeroMatrix<Scalar, dynamic_extent, dynamic_extent> {dim, dim};
    }
    else
    {
      constexpr auto dim = column_extent_of_v<A>;
      return ZeroMatrix<Scalar, dim, dim> {};
    }
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr A>
#else
  template<typename A, std::enable_if_t<eigen_constant_expr<A>, int> = 0>
#endif
  inline auto
  QR_decomposition(A&& a)
  {
    using Scalar = scalar_type_of_t<A>;
    constexpr auto constant = constant_coefficient_v<A>;

    const Scalar elem = constant * (
      dynamic_rows<A> ?
      std::sqrt((Scalar) row_count(a)) :
      OpenKalman::internal::constexpr_sqrt((Scalar) row_extent_of_v<A>));

    if constexpr (dynamic_columns<A>)
    {
      auto dim = column_count(a);
      auto row1 = Eigen3::eigen_matrix_t<Scalar, 1, dynamic_extent>::Constant(dim, elem);

      eigen_matrix_t<Scalar, dynamic_extent, dynamic_extent> ret {dim, dim};

      if (dim == 1)
      {
        ret = std::move(row1);
      }
      else
      {
        ret = concatenate_vertical(std::move(row1), ZeroMatrix<Scalar, dynamic_extent, dynamic_extent> {dim - 1, dim});
      }
      return ret;
    }
    else
    {
      constexpr auto dim = column_extent_of_v<A>;
      auto row1 = Eigen3::eigen_matrix_t<Scalar, 1, dim>::Constant(elem);
      if constexpr (dim > 1)
      {
        return concatenate_vertical(row1, ZeroMatrix<Scalar, dim - 1, dim> {});
      }
      else
      {
        return row1;
      }
    }
  }


  /**
   * \brief Perform a QR decomposition of matrix A=Q[U,0], where U is an upper-triangular matrix, and Q is orthogonal.
   * \return U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<typename A> requires eigen_diagonal_expr<A> or eigen_self_adjoint_expr<A> or eigen_triangular_expr<A>
#else
  template<typename A, std::enable_if_t<
    eigen_diagonal_expr<A> or eigen_self_adjoint_expr<A> or eigen_triangular_expr<A>, int> = 0>
#endif
  constexpr decltype(auto)
  QR_decomposition(A&& a)
  {
    if constexpr(upper_triangular_matrix<A>) return std::forward<A>(a);
    else return QR_decomposition(make_native_matrix(std::forward<A>(a)));
  }


  /// Concatenate diagonally.
#ifdef __cpp_concepts
  template<diagonal_matrix V, diagonal_matrix ... Vs>
  requires
    ((eigen_matrix<V> or eigen_self_adjoint_expr<V> or eigen_triangular_expr<V> or
      eigen_diagonal_expr<V> or from_euclidean_expr<V>) and ... and
    (eigen_matrix<Vs> or eigen_self_adjoint_expr<Vs> or eigen_triangular_expr<Vs> or
      eigen_diagonal_expr<Vs> or from_euclidean_expr<Vs>))
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (diagonal_matrix<V> and ... and diagonal_matrix<Vs>) and
    ((eigen_matrix<V> or eigen_self_adjoint_expr<V> or eigen_triangular_expr<V> or
      eigen_diagonal_expr<V> or from_euclidean_expr<V>) and ... and
    (eigen_matrix<Vs> or eigen_self_adjoint_expr<Vs> or eigen_triangular_expr<Vs> or
      eigen_diagonal_expr<Vs> or from_euclidean_expr<Vs>)), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs)
  {
    using Scalar = std::common_type_t<scalar_type_of_t<V>, scalar_type_of_t<Vs>...>;

    if constexpr(sizeof...(Vs) > 0)
    {
      if constexpr ((zero_matrix<V> and ... and zero_matrix<Vs>))
      {
        if constexpr ((dynamic_shape<V> or ... or dynamic_shape<Vs>))
        {
          auto dim = (row_count(v) + ... + row_count(vs));
          return DiagonalMatrix {ZeroMatrix<Scalar, dynamic_extent, 1> {dim}};
        }
        else
        {
          constexpr auto dim = (row_extent_of_v<V> + ... + row_extent_of_v<Vs>);
          static_assert(dim == (column_extent_of_v<V> + ... + column_extent_of_v<Vs>));
          return ZeroMatrix<Scalar, dim, dim> {};
        }
      }
      else if constexpr ((identity_matrix<V> and ... and identity_matrix<Vs>))
      {
        if constexpr ((dynamic_shape<V> or ... or dynamic_shape<Vs>))
        {
          auto dim = (row_count(v) + ... + row_count(vs));
          return MatrixTraits<equivalent_dense_writable_matrix_t<Scalar, dynamic_extent, dynamic_extent>>::identity(dim);
        }
        else
        {
          constexpr auto dim = (row_extent_of_v<V> + ... + row_extent_of_v<Vs>);
          static_assert(dim == (column_extent_of_v<V> + ... + column_extent_of_v<Vs>));
          return MatrixTraits<equivalent_dense_writable_matrix_t<Scalar, dim, dim>>::identity();
        }
      }
      else
      {
        return DiagonalMatrix {
          concatenate_vertical(diagonal_of(std::forward<V>(v)), diagonal_of(std::forward<Vs>(vs))...)};
      }
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  namespace detail
  {
#ifdef __cpp_concepts
    template<TriangleType t, eigen_self_adjoint_expr M>
#else
    template<TriangleType t, typename M, std::enable_if_t<eigen_self_adjoint_expr<M>, int> = 0>
#endif
    decltype(auto)
    maybe_transpose(M&& m)
    {
      if constexpr(t == self_adjoint_triangle_type_of_v<M>) return nested_matrix(std::forward<M>(m));
      else return transpose(nested_matrix(std::forward<M>(m)));
    }
  }


  /// Concatenate diagonally.
#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr V, eigen_self_adjoint_expr ... Vs>
  requires (not (diagonal_matrix<V> and ... and diagonal_matrix<Vs>))
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (eigen_self_adjoint_expr<V> and ... and eigen_self_adjoint_expr<Vs>) and
    (not (diagonal_matrix<V> and ... and diagonal_matrix<Vs>)), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs)
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      constexpr auto t = self_adjoint_triangle_type_of_v<V>;
      return MatrixTraits<V>::make(
        concatenate_diagonal(nested_matrix(std::forward<V>(v)), detail::maybe_transpose<t>(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  };


    /// Concatenate diagonally.
#ifdef __cpp_concepts
  template<eigen_triangular_expr V, eigen_triangular_expr ... Vs>
  requires (not (diagonal_matrix<V> and ... and diagonal_matrix<Vs>))
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (eigen_triangular_expr<V> and ... and eigen_triangular_expr<Vs>) and
    (not (diagonal_matrix<V> and ... and diagonal_matrix<Vs>)), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs)
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      if constexpr (((upper_triangular_matrix<V> == upper_triangular_matrix<Vs>) and ...))
      {
        return MatrixTraits<V>::make(
          concatenate_diagonal(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
      }
      else // There is a mixture of upper and lower triangles.
      {
        return concatenate_diagonal(
          make_native_matrix(std::forward<V>(v)), make_native_matrix(std::forward<Vs>(vs))...);
      }
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  namespace internal
  {
    template<typename G, typename Expr>
    struct SplitSpecF
    {
      template<typename RC, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        return G::template call<RC, CC>(MatrixTraits<Expr>::template make(std::forward<Arg>(arg)));
      }
    };
  }


  /// Split a diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<typename F, coefficients ... Cs, eigen_diagonal_expr Arg> requires (not coefficients<F>)
#else
  template<typename F, typename ... Cs, typename Arg,
    std::enable_if_t<eigen_diagonal_expr<Arg> and not coefficients<F> and (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= row_extent_of_v<Arg>);
    return split_vertical<internal::SplitSpecF<F, Arg>, Cs...>(nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split a self-adjoint or triangular matrix diagonally.
#ifdef __cpp_concepts
  template<typename F, coefficients ... Cs, typename Arg> requires
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and (not coefficients<F>)
#else
  template<typename F, typename ... Cs, typename Arg,
    std::enable_if_t<(eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    not coefficients<F> and (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= row_extent_of_v<Arg>);
    return split_diagonal<internal::SplitSpecF<F, Arg>, Cs...>(nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<coefficients ... Cs, typename Arg> requires
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>)
#else
  template<typename ... Cs, typename Arg, std::enable_if_t<(coefficients<Cs> and ...) and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= row_extent_of_v<Arg>);
    return split_diagonal<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }


  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, typename Arg>
  requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((cut + ... + cuts) <= row_extent_of_v<Arg>);
    return split_diagonal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Split a self-adjoint, triangular, or diagonal matrix vertically, returning a regular matrix.
#ifdef __cpp_concepts
  template<typename F, coefficients ... Cs, typename Arg> requires (not coefficients<F>) and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>)
#else
  template<typename F, typename ... Cs, typename Arg, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
      not coefficients<F> and (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= row_extent_of_v<Arg>);
    return split_vertical<internal::SplitSpecF<F, equivalent_dense_writable_matrix_t<Arg>>, Cs...>(make_native_matrix(std::forward<Arg>(arg)));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<coefficients ... Cs, typename Arg>
  requires (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (dynamic_rows<Arg> or (0 + ... + Cs::dimensions) <= row_extent_of_v<Arg>)
#else
  template<typename ... Cs, typename Arg, std::enable_if_t<(coefficients<Cs> and ...) and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (dynamic_rows<Arg> or (0 + ... + Cs::dimensions) <= row_extent_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg)
  {
    return split_vertical<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, typename Arg>
  requires (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (dynamic_rows<Arg> or (cut + ... + cuts) <= row_extent_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (dynamic_rows<Arg> or (cut + ... + cuts) <= row_extent_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg)
  {
    return split_vertical<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Split a self-adjoint, triangular, or diagonal matrix horizontally, returning a regular matrix.
#ifdef __cpp_concepts
  template<typename F, coefficients ... Cs, typename Arg> requires (not coefficients<F>) and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>)
#else
  template<typename F, typename ... Cs, typename Arg, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
      not coefficients<F> and (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= row_extent_of_v<Arg>);
    return split_horizontal<internal::SplitSpecF<F, equivalent_dense_writable_matrix_t<Arg>>, Cs...>(make_native_matrix(std::forward<Arg>(arg)));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix horizontally.
#ifdef __cpp_concepts
  template<coefficients ... Cs, typename Arg> requires
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<typename ... Cs, typename Arg, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::dimensions) <= row_extent_of_v<Arg>);
    return split_horizontal<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix horizontally.
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, typename Arg> requires
  eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg)
  {
    static_assert((cut + ... + cuts) <= row_extent_of_v<Arg>);
    return split_horizontal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<typename Function, typename Arg> requires
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (requires(Arg&& arg, const Function& f) { {f(column(arg, 0))} -> column_vector; } or
      requires(Arg&& arg, const Function& f, std::size_t i) { {f(column(arg, 0), i)} -> column_vector; })
#else
  template<typename Function, typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, Arg&& arg)
  {
    return apply_columnwise(f, make_native_matrix(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<typename Function, typename Arg> requires
  (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (requires(Arg&& arg, const Function& f) { {f(row(arg, 0))} -> row_vector; } or
      requires(Arg&& arg, const Function& f, std::size_t i) { {f(row(arg, 0), i)} -> row_vector; })
#else
  template<typename Function, typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> or
      eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  apply_rowwise(const Function& f, Arg&& arg)
  {
    return apply_rowwise(f, make_native_matrix(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<typename Function, typename Arg> requires
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (requires(Function& f, scalar_type_of_t<Arg>& s) {
      {f(s)} -> std::convertible_to<const scalar_type_of_t<Arg>>;
    } or
    requires(Function& f, scalar_type_of_t<Arg>& s, std::size_t& i, std::size_t& j) {
      {f(s, i, j)} -> std::convertible_to<const scalar_type_of_t<Arg>>;
    })
#else
  template<typename Function, typename Arg, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    std::is_convertible_v<
      std::invoke_result_t<Function&, typename scalar_type_of<Arg>::type&>,
      const typename scalar_type_of<Arg>::type>, int> = 0>
  inline auto
  apply_coefficientwise(const Function& f, Arg&& arg)
  {
    return apply_coefficientwise(f, make_native_matrix(std::forward<Arg>(arg)));
  }


  template<typename Function, typename Arg, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    std::is_convertible_v<
      std::invoke_result_t<Function&, typename scalar_type_of<Arg>::type&, std::size_t&, std::size_t&>,
      const typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f, Arg&& arg)
  {
    return apply_coefficientwise(f, make_native_matrix(std::forward<Arg>(arg)));
  }


  /**
   * \brief Fill a fixed diagonal matrix with random values selected from one or more random distributions.
   * \details The following example constructs 2-by-2 diagonal matrices in which each diagonal element is
   * a random value selected as indicated:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using D2 = DiagonalMatrix<eigen_matrix_t<double, 2, 1>>;
   *     D2 m = randomize<D2>(N {1.0, 0.3})); // Both diagonal elements have mean 1.0, s.d. 0.3
   *     D2 n = randomize<D2>(N {1.0, 0.3}, N {2.0, 0.2})); // Second diagonal element has mean 2.0, s.d. 0.2.
   *     D2 p = randomize<D2>(N {1.0, 0.3}, 2.0)); // Second diagonal element is exactly 2.0
   *   \endcode
   * \tparam ReturnType The type of the matrix to be filled.
   * \tparam random_number_engine The random number engine (e.g., std::mt19937).
   * \tparam Dists A set of distributions (e.g., std::normal_distribution<double>) or, alternatively,
   * means (a definite, non-stochastic value).
   **/
#ifdef __cpp_concepts
  template<eigen_diagonal_expr ReturnType,
    std::uniform_random_bit_generator random_number_engine = std::mt19937, typename...Dists>
  requires (not dynamic_shape<ReturnType>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename...Dists,
    std::enable_if_t<eigen_diagonal_expr<ReturnType> and (not dynamic_shape<ReturnType>), int> = 0>
#endif
  inline auto
  randomize(Dists&&...dists)
  {
    using B = nested_matrix_of<ReturnType>;
    return MatrixTraits<ReturnType>::make(randomize<B, random_number_engine>(std::forward<Dists>(dists)...));
  }


  /**
   * \overload
   * \brief Fill a dynamic-shape Eigen matrix with random values selected from a single random distribution.
   * \details The following example constructs matrices (m, and n) in which each element is a
   * random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using D0 = DiagonalMatrix<eigen_matrix_t<double, dynamic_extent, 1>>;
   *     auto m = randomize(D0, 2, 2, std::normal_distribution<double> {1.0, 0.3})); // constructs a 2-by-2 matrix
   *     auto n = randomize(D0, 3, 3, std::normal_distribution<double> {1.0, 0.3}); // constructs a 3-by-2 matrix
   *   \endcode
   * \tparam ReturnType The type of the matrix to be filled.
   * \tparam random_number_engine The random number engine (e.g., std::mt19937).
   * \param rows Number of rows, decided at runtime.
   * \param columns Number of columns, decided at runtime. Columns must equal rows.
   * \tparam Dist A distribution (type distribution_type).
   **/
#ifdef __cpp_concepts
  template<eigen_diagonal_expr ReturnType,
    std::uniform_random_bit_generator random_number_engine = std::mt19937, typename Dist>
  requires
    dynamic_shape<ReturnType> and
    requires { typename std::decay_t<Dist>::result_type; typename std::decay_t<Dist>::param_type; } and
    (not std::is_const_v<std::remove_reference_t<Dist>>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename Dist, std::enable_if_t<
    eigen_diagonal_expr<ReturnType> and dynamic_shape<ReturnType> and
    (not std::is_const_v<std::remove_reference_t<Dist>>), int> = 0>
#endif
  inline auto
  randomize(const std::size_t rows, const std::size_t columns, Dist&& dist)
  {
    assert(rows == columns);
    using B = nested_matrix_of<ReturnType>;
    return MatrixTraits<ReturnType>::make(randomize<B, random_number_engine>(rows, 1, std::forward<Dist>(dist)));
  }


} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_SPECIAL_MATRIX_OVERLOADS_HPP
