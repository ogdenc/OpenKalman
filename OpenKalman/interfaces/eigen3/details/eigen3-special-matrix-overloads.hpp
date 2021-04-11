/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_SPECIAL_MATRIX_OVERLOADS_HPP
#define OPENKALMAN_EIGEN3_SPECIAL_MATRIX_OVERLOADS_HPP

namespace OpenKalman::Eigen3
{
#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  static constexpr decltype(auto)
  nested_matrix(Arg&& arg) { return std::forward<Arg>(arg).nested_matrix(); }


  /// Convert to self-contained version of the special matrix.
#ifdef __cpp_concepts
  template<typename...Ts, typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename...Ts, typename Arg, std::enable_if_t<
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr(self_contained<nested_matrix_t<Arg>> or
      ((sizeof...(Ts) > 0) and ... and std::is_lvalue_reference_v<Ts>))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return MatrixTraits<Arg>::make(make_self_contained(nested_matrix(std::forward<Arg>(arg))));
    }
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  inline auto
  determinant(Arg&& arg) noexcept
  {
    return make_native_matrix(std::forward<Arg>(arg)).determinant();
  }


#ifdef __cpp_concepts
  template<typename Arg> requires eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  inline auto
  trace(Arg&& arg) noexcept
  {
    return nested_matrix(std::forward<Arg>(arg)).trace();
  }


#ifdef __cpp_concepts
  template<eigen_native Arg, typename U> requires one_by_one_matrix<Arg> and (MatrixTraits<U>::rows == 1) and
    (eigen_matrix<U> or eigen_triangular_expr<U> or eigen_self_adjoint_expr<U> or eigen_diagonal_expr<U>) and
    (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U, std::enable_if_t<
    eigen_native<Arg> and one_by_one_matrix<Arg> and (MatrixTraits<U>::rows == 1) and
    (eigen_matrix<U> or eigen_triangular_expr<U> or eigen_self_adjoint_expr<U> or eigen_diagonal_expr<U>) and
    not std::is_const_v<std::remove_reference_t<Arg>>, int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    arg(0, 0) = std::sqrt(trace(arg) * trace(arg) + alpha * trace(u * adjoint(u)));
    return arg;
  }


#ifdef __cpp_concepts
  template<eigen_native Arg, typename U> requires one_by_one_matrix<Arg> and (MatrixTraits<U>::rows == 1) and
    (eigen_matrix<U> or eigen_triangular_expr<U> or eigen_self_adjoint_expr<U> or eigen_diagonal_expr<U>)
#else
  template<typename Arg, typename U, std::enable_if_t<
    eigen_native<Arg> and one_by_one_matrix<Arg> and (MatrixTraits<U>::rows == 1) and
    (eigen_matrix<U> or eigen_triangular_expr<U> or eigen_self_adjoint_expr<U> or eigen_diagonal_expr<U>), int> = 0>
#endif
  inline auto
  rank_update(const Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    auto b = std::sqrt(trace(arg) * trace(arg) + alpha * trace(u * adjoint(u)));
    return Eigen::Matrix<typename MatrixTraits<Arg>::Scalar, 1, 1> {b};
  }


#ifdef __cpp_concepts
  template<typename A, eigen_matrix B> requires
    eigen_self_adjoint_expr<A> or eigen_triangular_expr<A>
#else
  template<
    typename A, typename B, std::enable_if_t<eigen_matrix<B> and
      (eigen_self_adjoint_expr<A> or eigen_triangular_expr<A>), int> = 0>
#endif
  constexpr decltype(auto)
  solve(A&& a, B&& b)
  {
    return std::forward<A>(a).solve(std::forward<B>(b));
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
    return make_native_matrix(make_native_matrix(std::forward<Arg>(arg)).rowwise().sum() / MatrixTraits<Arg>::rows);
  }


  /// Concatenate diagonally.
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires
    (eigen_self_adjoint_expr<V> and ... and eigen_self_adjoint_expr<Vs>) or
    (eigen_triangular_expr<V> and ... and eigen_triangular_expr<Vs>)
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (eigen_self_adjoint_expr<V> and ... and eigen_self_adjoint_expr<Vs>) or
    (eigen_triangular_expr<V> and ... and eigen_triangular_expr<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs)
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      /// \todo Add diagonal case
      if constexpr (
        (eigen_self_adjoint_expr<V> and
          ((upper_triangular_storage<V> == upper_triangular_storage<Vs>) and ...)) or
        (eigen_triangular_expr<V> and ((upper_triangular_matrix<V> == upper_triangular_matrix<Vs>) and ...)))
      {
        return MatrixTraits<V>::make(
          concatenate_diagonal(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
      }
      else if constexpr (eigen_self_adjoint_expr<V>)
      {
        constexpr auto t = MatrixTraits<V>::storage_triangle;
        return concatenate_diagonal(std::forward<V>(v), make_EigenSelfAdjointMatrix<t>(std::forward<Vs>(vs))...);
      }
      else // eigen_triangular_expr<V> and there is a mixture of upper and lower triangles.
      {
        return concatenate_diagonal(make_native_matrix(std::forward<V>(v)), make_native_matrix(std::forward<Vs>(vs))...);
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
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::rows);
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
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::rows);
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
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::rows);
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
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::rows);
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
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::rows);
    return split_vertical<internal::SplitSpecF<F, native_matrix_t<Arg>>, Cs...>(make_native_matrix(std::forward<Arg>(arg)));
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
  split_vertical(Arg&& arg)
  {
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::rows);
    return split_vertical<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, typename Arg> requires
  eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg)
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::rows);
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
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::rows);
    return split_horizontal<internal::SplitSpecF<F, native_matrix_t<Arg>>, Cs...>(make_native_matrix(std::forward<Arg>(arg)));
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
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::rows);
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
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::rows);
    return split_horizontal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Get element (i, j) of self-adjoint matrix arg.
#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr Arg> requires (not diagonal_matrix<Arg>) and
    element_gettable<nested_matrix_t<Arg>, 2>
#else
  template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> and not diagonal_matrix<Arg> and
    element_gettable<nested_matrix_t<Arg>, 2>, int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if constexpr(lower_triangular_storage<Arg>)
    {
      if (i >= j) return get_element(nested_matrix(arg), i, j);
      else return get_element(nested_matrix(std::forward<Arg>(arg)), j, i);
    }
    else
    {
      if (i <= j) return get_element(nested_matrix(arg), i, j);
      else return get_element(nested_matrix(std::forward<Arg>(arg)), j, i);
    }
  }


  /// Get element (i, j) of triangular matrix arg.
#ifdef __cpp_concepts
  template<eigen_triangular_expr Arg> requires (not diagonal_matrix<Arg>) and
    element_gettable<nested_matrix_t<Arg>, 2>
#else
  template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg> and not diagonal_matrix<Arg> and
    element_gettable<nested_matrix_t<Arg>, 2>, int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if constexpr(lower_triangular_matrix<Arg>)
    {
      if (i >= j) return get_element(nested_matrix(std::forward<Arg>(arg)), i, j);
      else return typename MatrixTraits<Arg>::Scalar(0);
    }
    else
    {
      if (i <= j) return get_element(nested_matrix(std::forward<Arg>(arg)), i, j);
      else return typename MatrixTraits<Arg>::Scalar(0);
    }
  }


  /// Get element (i, j) of a self-adjoint or triangular matrix that is also diagonal.
#ifdef __cpp_concepts
  template<typename Arg> requires
    diagonal_matrix<Arg> and (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (element_gettable<nested_matrix_t<Arg>, 2> or
      element_gettable<nested_matrix_t<Arg>, 1>)
#else
  template<typename Arg, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and diagonal_matrix<Arg> and
    (element_gettable<nested_matrix_t<Arg>, 2> or
      element_gettable<nested_matrix_t<Arg>, 1>), int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if (i == j)
    {
      if constexpr(element_gettable<nested_matrix_t<Arg>, 1>)
        return get_element(nested_matrix(std::forward<Arg>(arg)), i);
      else
        return get_element(nested_matrix(std::forward<Arg>(arg)), i, i);
    }
    else return typename MatrixTraits<Arg>::Scalar(0);
  }


  /// Get element (i) of diagonal self-adjoint or triangular matrix.
#ifdef __cpp_concepts
  template<typename Arg> requires
    diagonal_matrix<Arg> and (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (element_gettable<nested_matrix_t<Arg>, 1> or
      element_gettable<nested_matrix_t<Arg>, 2>)
#else
  template<typename Arg, std::enable_if_t<
    diagonal_matrix<Arg> and (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (element_gettable<nested_matrix_t<Arg>, 1> or
      element_gettable<nested_matrix_t<Arg>, 2>), int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    using NestedMatrix = nested_matrix_t<Arg>;
    if constexpr(element_gettable<NestedMatrix, 1>)
    {
      return get_element(nested_matrix(std::forward<Arg>(arg)), i);
    }
    else
    {
      return get_element(nested_matrix(std::forward<Arg>(arg)), i, i);
    }
  }


  /// Set element (i, j) of self-adjoint matrix arg to s.
#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and (not diagonal_matrix<Arg>) and
    element_settable<nested_matrix_t<Arg>, 2>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<eigen_self_adjoint_expr<Arg> and
    not std::is_const_v<std::remove_reference_t<Arg>> and not diagonal_matrix<Arg> and
    element_settable<nested_matrix_t<Arg>, 2>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if constexpr(lower_triangular_storage<Arg>)
    {
      if (i >= j) set_element(nested_matrix(arg), s, i, j);
      else set_element(nested_matrix(arg), s, j, i);
    }
    else
    {
      if (i <= j) set_element(nested_matrix(arg), s, i, j);
      else set_element(nested_matrix(arg), s, j, i);
    }
  }


  /// Set element (i, j) of triangular matrix arg to s.
#ifdef __cpp_concepts
  template<eigen_triangular_expr Arg, typename Scalar> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and (not diagonal_matrix<Arg>) and
    element_settable<nested_matrix_t<Arg>, 2>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<eigen_triangular_expr<Arg> and
    not std::is_const_v<std::remove_reference_t<Arg>> and not diagonal_matrix<Arg> and
    element_settable<nested_matrix_t<Arg>, 2>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if constexpr(lower_triangular_matrix<Arg>)
    {
      if (i >= j) set_element(nested_matrix(arg), s, i, j);
      else throw std::out_of_range("Only lower-triangle elements of a lower-triangular matrix may be set.");
    }
    else
    {
      if (i <= j) set_element(nested_matrix(arg), s, i, j);
      else throw std::out_of_range("Only upper-triangle elements of an upper-triangular matrix may be set.");
    }
  }


  /// Set element (i, j) of a self-adjoint or triangular matrix that is also diagonal.
#ifdef __cpp_concepts
  template<typename Arg, typename Scalar> requires
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and diagonal_matrix<Arg> and
    (element_settable<nested_matrix_t<Arg>, 2> or
      element_settable<nested_matrix_t<Arg>, 1>)
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    not std::is_const_v<std::remove_reference_t<Arg>> and diagonal_matrix<Arg> and
    (element_settable<nested_matrix_t<Arg>, 2> or
      element_settable<nested_matrix_t<Arg>, 1>), int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if (i == j)
    {
      if constexpr(element_settable<nested_matrix_t<Arg>, 1>)
        set_element(nested_matrix(arg), s, i);
      else
        set_element(nested_matrix(arg), s, i, i);
    }
    else throw std::out_of_range("Only diagonal elements of a diagonal matrix may be set.");
  }


  /// Set element (i) of diagonal self-adjoint or triangular matrix.
#ifdef __cpp_concepts
  template<typename Arg, typename Scalar> requires diagonal_matrix<Arg> and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (element_settable<nested_matrix_t<Arg>, 1> or
      element_settable<nested_matrix_t<Arg>, 2>)
#else
  template<typename Arg, typename Scalar, std::enable_if_t<diagonal_matrix<Arg> and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    (element_settable<nested_matrix_t<Arg>, 1> or
      element_settable<nested_matrix_t<Arg>, 2>), int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    using NestedMatrix = nested_matrix_t<Arg>;
    if constexpr(element_settable<NestedMatrix, 1>)
    {
      set_element(nested_matrix(arg), s, i);
    }
    else
    {
      set_element(nested_matrix(arg), s, i, i);
    }
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
    return make_self_contained(column(make_native_matrix(std::forward<Arg>(arg)), index));
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, typename Arg> requires eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<std::size_t index, typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    static_assert(index < MatrixTraits<Arg>::columns);
    return make_self_contained(column<index>(make_native_matrix(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<typename Arg, typename Function> requires eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  apply_columnwise(Arg&& arg, const Function& f)
  {
    return make_self_contained(apply_columnwise(make_native_matrix(std::forward<Arg>(arg)), f));
  }


#ifdef __cpp_concepts
  template<typename Arg, typename Function> requires eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(Arg&& arg, const Function& f)
  {
    return make_self_contained(apply_coefficientwise(make_native_matrix(std::forward<Arg>(arg)), f));
  }


} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_SPECIAL_MATRIX_OVERLOADS_HPP
