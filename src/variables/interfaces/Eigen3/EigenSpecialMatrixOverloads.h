/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENSPECIALMATRIXOVERLOADS_H
#define OPENKALMAN_EIGENSPECIALMATRIXOVERLOADS_H

namespace OpenKalman
{
  template<typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>, int> = 0>
  static constexpr decltype(auto)
  base_matrix(Arg&& arg) { return std::forward<Arg>(arg).base_matrix(); }


  /// Convert to strict version of the special matrix.
  template<typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>, int> = 0>
  constexpr decltype(auto)
  strict(Arg&& arg)
  {
    if constexpr(is_strict_v<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return MatrixTraits<Arg>::make(strict(base_matrix(std::forward<Arg>(arg))));
    }
  }


  template<typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>, int> = 0>
  inline auto
  determinant(Arg&& arg) noexcept
  {
    return strict_matrix(std::forward<Arg>(arg)).determinant();
  }


  template<typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>, int> = 0>
  inline auto
  trace(Arg&& arg) noexcept
  {
    return base_matrix(std::forward<Arg>(arg)).trace();
  }


#ifdef __cpp_concepts
  template<native_Eigen_type Arg, typename U>
  requires is_1by1_v<Arg> and
    (Eigen_matrix<U> or is_EigenTriangularMatrix_v<U> or is_EigenSelfAdjointMatrix_v<U> or is_EigenDiagonal_v<U>)
    and (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
  template<typename Arg, typename U,
    std::enable_if_t<is_native_Eigen_type_v<Arg> and is_1by1_v<Arg> and
      (is_Eigen_matrix_v<U> or is_EigenTriangularMatrix_v<U> or is_EigenSelfAdjointMatrix_v<U> or is_EigenDiagonal_v<U>) and
      not std::is_const_v<std::remove_reference_t<Arg>>, int> = 0>
#endif
  inline Arg&
  rank_update(Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    arg(0, 0) = std::sqrt(trace(arg) * trace(arg) + alpha * trace(u) * trace(u));
    return arg;
  }


#ifdef __cpp_concepts
  template<native_Eigen_type Arg, typename U>
  requires is_1by1_v<Arg> and
    (Eigen_matrix<U> or is_EigenTriangularMatrix_v<U> or is_EigenSelfAdjointMatrix_v<U> or is_EigenDiagonal_v<U>)
#else
  template<typename Arg, typename U,
    std::enable_if_t<is_native_Eigen_type_v<Arg> and is_1by1_v<Arg> and
    (is_Eigen_matrix_v<U> or is_EigenTriangularMatrix_v<U> or is_EigenSelfAdjointMatrix_v<U> or is_EigenDiagonal_v<U>), int> = 0>
#endif
  inline auto
  rank_update(const Arg& arg, const U& u, const typename MatrixTraits<Arg>::Scalar alpha = 1)
  {
    static_assert(MatrixTraits<U>::dimension == MatrixTraits<Arg>::dimension);
    auto b = std::sqrt(trace(arg) * trace(arg) + alpha * trace(u) * trace(u));
    return Eigen::Matrix<typename MatrixTraits<Arg>::Scalar, 1, 1>(b);
  }


#ifdef __cpp_concepts
  template<typename A, Eigen_matrix B> requires is_EigenSelfAdjointMatrix_v<A> or is_EigenTriangularMatrix_v<A>
#else
  template<
    typename A, typename B,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<A> or is_EigenTriangularMatrix_v<A>, int> = 0,
    std::enable_if_t<is_Eigen_matrix_v<B>, int> = 0>
#endif
  constexpr decltype(auto)
  solve(A&& a, B&& b)
  {
    return std::forward<A>(a).solve(std::forward<B>(b));
  }


#ifdef __cpp_concepts
  template<typename Arg> requires is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>
#else
  template<typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_columns(Arg&& arg)
  {
    return strict_matrix(strict_matrix(std::forward<Arg>(arg)).rowwise().sum() / MatrixTraits<Arg>::dimension);
  }


  /// Concatenate diagonally if the arguments are all self-adjoint or all triangular, and of the same type.
#ifdef __cpp_concepts
  template<typename V, typename ... Vs>
  requires (std::conjunction_v<is_EigenSelfAdjointMatrix<V>, is_EigenSelfAdjointMatrix<Vs>...> and
        ((is_Eigen_upper_storage_triangle_v<V> == is_Eigen_upper_storage_triangle_v<Vs>) and ... and true)) or
      (std::conjunction_v<is_EigenTriangularMatrix<V>, is_EigenTriangularMatrix<Vs>...> and
        ((is_upper_triangular_v<V> == is_upper_triangular_v<Vs>) and ... and true))
#else
  template<typename V, typename ... Vs,
    std::enable_if_t<
      (std::conjunction_v<is_EigenSelfAdjointMatrix<V>, is_EigenSelfAdjointMatrix<Vs>...> and
        ((is_Eigen_upper_storage_triangle_v<V> == is_Eigen_upper_storage_triangle_v<Vs>) and ... and true)) or
      (std::conjunction_v<is_EigenTriangularMatrix<V>, is_EigenTriangularMatrix<Vs>...> and
        ((is_upper_triangular_v<V> == is_upper_triangular_v<Vs>) and ... and true)), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate(V&& v, Vs&& ... vs)
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      return MatrixTraits<V>::make(concatenate_diagonal(base_matrix(std::forward<V>(v)), base_matrix(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  /// Concatenate diagonally if arguments are a heterogeneous mix upper and lower-storage self-adjoint.
  /// The result will have the storage type of the first matrix.
#ifdef __cpp_concepts
  template<typename V, typename ... Vs>
  requires std::conjunction_v<is_EigenSelfAdjointMatrix<V>, is_EigenSelfAdjointMatrix<Vs>...> and
    (not ((is_Eigen_upper_storage_triangle_v<V> == is_Eigen_upper_storage_triangle_v<Vs>) and ...))
#else
  template<typename V, typename ... Vs,
    std::enable_if_t<
      std::conjunction_v<is_EigenSelfAdjointMatrix<V>, is_EigenSelfAdjointMatrix<Vs>...> and
      not ((is_Eigen_upper_storage_triangle_v<V> == is_Eigen_upper_storage_triangle_v<Vs>) and ...), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate(V&& v, Vs&& ... vs)
  {
    constexpr auto t = MatrixTraits<V>::storage_type;
    return concatenate(std::forward<V>(v), make_EigenSelfAdjointMatrix<t>(std::forward<Vs>(vs))...);
  };


  /// Concatenate diagonally.
#ifdef __cpp_concepts
  template<typename V, typename ... Vs>
  requires std::conjunction_v<is_EigenSelfAdjointMatrix<V>, is_EigenSelfAdjointMatrix<Vs>...> or
      (std::conjunction_v<is_EigenTriangularMatrix<V>, is_EigenTriangularMatrix<Vs>...> and
        ((is_upper_triangular_v<V> == is_upper_triangular_v<Vs>) and ...))
#else
  template<typename V, typename ... Vs,
    std::enable_if_t<
      std::conjunction_v<is_EigenSelfAdjointMatrix<V>, is_EigenSelfAdjointMatrix<Vs>...> or
      (std::conjunction_v<is_EigenTriangularMatrix<V>, is_EigenTriangularMatrix<Vs>...> and
        ((is_upper_triangular_v<V> == is_upper_triangular_v<Vs>) and ...)), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs)
  {
    return concatenate(std::forward<V>(v), std::forward<Vs>(vs)...);
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
  template<typename F, typename ... Cs, typename Arg>
  requires is_EigenDiagonal_v<Arg> and (not is_coefficient_v<F>) and
    (not is_coefficient_v<F>) and std::conjunction_v<is_coefficient<Cs>...>
#else
  template<typename F, typename ... Cs, typename Arg,
    std::enable_if_t<is_EigenDiagonal_v<Arg> and not is_coefficient_v<F> and
    not is_coefficient_v<F> and std::conjunction_v<is_coefficient<Cs>...>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::dimension);
    return split_vertical<internal::SplitSpecF<F, Arg>, Cs...>(base_matrix(std::forward<Arg>(arg)));
  }

  /// Split a self-adjoint or triangular matrix diagonally.
#ifdef __cpp_concepts
  template<typename F, typename ... Cs, typename Arg>
  requires (is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>) and
    (not is_coefficient_v<F>) and std::conjunction_v<is_coefficient<Cs>...>
#else
  template<typename F, typename ... Cs, typename Arg,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>) and
    not is_coefficient_v<F> and std::conjunction_v<is_coefficient<Cs>...>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::dimension);
    return split_diagonal<internal::SplitSpecF<F, Arg>, Cs...>(base_matrix(std::forward<Arg>(arg)));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<typename ... Cs, typename Arg>
  requires (is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>) and
    std::conjunction_v<is_coefficient<Cs>...>
#else
  template<typename ... Cs, typename Arg,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>) and
      std::conjunction_v<is_coefficient<Cs>...>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::dimension);
    return split_diagonal<internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, typename Arg>
  requires is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::dimension);
    return split_diagonal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Split a self-adjoint, triangular, or diagonal matrix vertically, returning a regular matrix.
  template<typename F, typename ... Cs, typename Arg,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>) and
      not is_coefficient_v<F> and std::conjunction_v<is_coefficient<Cs>...>, int> = 0>
  inline auto
  split_vertical(Arg&& arg)
  {
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::dimension);
    return split_vertical<internal::SplitSpecF<F, strict_matrix_t<Arg>>, Cs...>(strict_matrix(std::forward<Arg>(arg)));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
  template<typename ... Cs, typename Arg,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>) and
      std::conjunction_v<is_coefficient<Cs>...>, int> = 0>
  inline auto
  split_vertical(Arg&& arg)
  {
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::dimension);
    return split_vertical<internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
  template<std::size_t cut, std::size_t ... cuts, typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  split_vertical(Arg&& arg)
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::dimension);
    return split_vertical<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Split a self-adjoint, triangular, or diagonal matrix horizontally, returning a regular matrix.
  template<typename F, typename ... Cs, typename Arg,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>) and
      not is_coefficient_v<F> and std::conjunction_v<is_coefficient<Cs>...>, int> = 0>
  inline auto
  split_horizontal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::dimension);
    return split_horizontal<internal::SplitSpecF<F, strict_matrix_t<Arg>>, Cs...>(strict_matrix(std::forward<Arg>(arg)));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix horizontally.
  template<typename ... Cs, typename Arg,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>) and
      std::conjunction_v<is_coefficient<Cs>...>, int> = 0>
  inline auto
  split_horizontal(Arg&& arg)
  {
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::dimension);
    return split_horizontal<internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix horizontally.
  template<std::size_t cut, std::size_t ... cuts, typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  split_horizontal(Arg&& arg)
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::dimension);
    return split_horizontal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Get element (i, j) of self-adjoint matrix arg.
  template<typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> and
    not is_diagonal_v<Arg> and
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>, int> = 0>
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if constexpr(is_Eigen_lower_storage_triangle_v<Arg>)
    {
      if (i >= j) return get_element(base_matrix(arg), i, j);
      else return get_element(base_matrix(std::forward<Arg>(arg)), j, i);
    }
    else
    {
      if (i <= j) return get_element(base_matrix(arg), i, j);
      else return get_element(base_matrix(std::forward<Arg>(arg)), j, i);
    }
  }


  /// Get element (i, j) of triangular matrix arg.
  template<typename Arg, std::enable_if_t<is_EigenTriangularMatrix_v<Arg> and
    not is_diagonal_v<Arg> and
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>, int> = 0>
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if constexpr(is_lower_triangular_v<Arg>)
    {
      if (i >= j) return get_element(base_matrix(std::forward<Arg>(arg)), i, j);
      else return typename MatrixTraits<Arg>::Scalar(0);
    }
    else
    {
      if (i <= j) return get_element(base_matrix(std::forward<Arg>(arg)), i, j);
      else return typename MatrixTraits<Arg>::Scalar(0);
    }
  }


  /// Get element (i, j) of a self-adjoint or triangular matrix that is also diagonal.
  template<typename Arg, std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>) and
    is_diagonal_v<Arg> and
    (is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 2> or
      is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>), int> = 0>
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if (i == j)
    {
      if constexpr(is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>)
        return get_element(base_matrix(std::forward<Arg>(arg)), i);
      else
        return get_element(base_matrix(std::forward<Arg>(arg)), i, i);
    }
    else return typename MatrixTraits<Arg>::Scalar(0);
  }


  /// Get element (i) of diagonal self-adjoint or triangular matrix.
  template<typename Arg, std::enable_if_t<is_diagonal_v<Arg> and
    (is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>) and
    (is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1> or
      is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>), int> = 0>
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    using BaseMatrix = typename MatrixTraits<Arg>::BaseMatrix;
    if constexpr(is_element_gettable_v<BaseMatrix, 1>)
    {
      return get_element(base_matrix(std::forward<Arg>(arg)), i);
    }
    else
    {
      return get_element(base_matrix(std::forward<Arg>(arg)), i, i);
    }
  }


  /// Set element (i, j) of self-adjoint matrix arg to s.
  template<typename Arg, typename Scalar, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> and
    not std::is_const_v<std::remove_reference_t<Arg>> and not is_diagonal_v<Arg> and
    is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>, int> = 0>
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if constexpr(is_Eigen_lower_storage_triangle_v<Arg>)
    {
      if (i >= j) set_element(base_matrix(arg), s, i, j);
      else set_element(base_matrix(arg), s, j, i);
    }
    else
    {
      if (i <= j) set_element(base_matrix(arg), s, i, j);
      else set_element(base_matrix(arg), s, j, i);
    }
  }


  /// Set element (i, j) of triangular matrix arg to s.
  template<typename Arg, typename Scalar, std::enable_if_t<is_EigenTriangularMatrix_v<Arg> and
    not std::is_const_v<std::remove_reference_t<Arg>> and not is_diagonal_v<Arg> and
    is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>, int> = 0>
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if constexpr(is_lower_triangular_v<Arg>)
    {
      if (i >= j) set_element(base_matrix(arg), s, i, j);
      else throw std::out_of_range("Only lower-triangle elements of a lower-triangular matrix may be set.");
    }
    else
    {
      if (i <= j) set_element(base_matrix(arg), s, i, j);
      else throw std::out_of_range("Only upper-triangle elements of an upper-triangular matrix may be set.");
    }
  }


  /// Set element (i, j) of a self-adjoint or triangular matrix that is also diagonal.
  template<typename Arg, typename Scalar,
    std::enable_if_t<(is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>) and
    not std::is_const_v<std::remove_reference_t<Arg>> and is_diagonal_v<Arg> and
    (is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 2> or
      is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>), int> = 0>
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if (i == j)
    {
      if constexpr(is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>)
        set_element(base_matrix(arg), s, i);
      else
        set_element(base_matrix(arg), s, i, i);
    }
    else throw std::out_of_range("Only diagonal elements of a diagonal matrix may be set.");
  }


  /// Set element (i) of diagonal self-adjoint or triangular matrix.
  template<typename Arg, typename Scalar, std::enable_if_t<is_diagonal_v<Arg> and
    (is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>) and
    (is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1> or
      is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>), int> = 0>
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    using BaseMatrix = typename MatrixTraits<Arg>::BaseMatrix;
    if constexpr(is_element_settable_v<BaseMatrix, 1>)
    {
      set_element(base_matrix(arg), s, i);
    }
    else
    {
      set_element(base_matrix(arg), s, i, i);
    }
  }


  /// Return column <code>index</code> of Arg.
  template<typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or
    is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    return strict(column(strict_matrix(std::forward<Arg>(arg)), index));
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
  template<std::size_t index, typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or
    is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline decltype(auto)
  column(Arg&& arg)
  {
    static_assert(index < MatrixTraits<Arg>::columns);
    return strict(column<index>(strict_matrix(std::forward<Arg>(arg))));
  }


  template<typename Arg, typename Function, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or
    is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  apply_columnwise(Arg&& arg, const Function& f)
  {
    return strict(apply_columnwise(strict_matrix(std::forward<Arg>(arg)), f));
  }


  template<typename Arg, typename Function, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or
    is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  apply_coefficientwise(Arg&& arg, const Function& f)
  {
    return strict(apply_coefficientwise(strict_matrix(std::forward<Arg>(arg)), f));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_EIGENSPECIALMATRIXOVERLOADS_H
