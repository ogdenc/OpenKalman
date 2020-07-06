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
    if constexpr(is_strict_v<typename MatrixTraits<Arg>::BaseMatrix>)
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


  template<
    typename A, typename B,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<A> or is_EigenTriangularMatrix_v<A>, int> = 0,
    std::enable_if_t<is_Eigen_matrix_v<B>, int> = 0>
  constexpr decltype(auto)
  solve(A&& a, B&& b)
  {
    return std::forward<A>(a).solve(std::forward<B>(b));
  }


  template<typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>, int> = 0>
  constexpr auto
  reduce_columns(Arg&& arg)
  {
    return strict_matrix(strict_matrix(std::forward<Arg>(arg)).rowwise().sum() / MatrixTraits<Arg>::dimension);
  }


  /// Concatenate diagonally.
  template<typename V, typename ... Vs,
    std::enable_if_t<std::conjunction_v<is_EigenSelfAdjointMatrix<V>, is_EigenSelfAdjointMatrix<Vs>...> or
      std::conjunction_v<is_EigenTriangularMatrix<V>, is_EigenTriangularMatrix<Vs>...>, int> = 0>
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


  /// Concatenate diagonally.
  template<typename V, typename ... Vs,
    std::enable_if_t<std::conjunction_v<is_EigenSelfAdjointMatrix<V>, is_EigenSelfAdjointMatrix<Vs>...> or
      std::conjunction_v<is_EigenTriangularMatrix<V>, is_EigenTriangularMatrix<Vs>...>, int> = 0>
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs)
  {
    return concatenate(std::forward<V>(v), std::forward<Vs>(vs)...);
  };


  /// Split a matrix diagonally.
  template<std::size_t ... cuts, typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg>, int> = 0>
  inline auto
  split(Arg&& arg)
  {
    static_assert((0 + ... + cuts) <= MatrixTraits<Arg>::dimension);
    if constexpr(sizeof...(cuts) == 0)
    {
      return std::tuple {};
    }
    else if constexpr(sizeof...(cuts) == 1 and (0 + ... + cuts) == MatrixTraits<Arg>::dimension)
    {
      return std::tuple {std::forward<Arg>(arg)};
    }
    else
    {
      return std::apply([](const auto&...bs){ return std::tuple {MatrixTraits<Arg>::make(bs)...}; },
        split_diagonal<cuts...>(base_matrix(std::forward<Arg>(arg))));
    }
  }


  /// Split a special matrix diagonally, returning a regular matrix.
  template<std::size_t ... cuts, typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  split_diagonal(Arg&& arg)
  {
    static_assert((0 + ... + cuts) <= MatrixTraits<Arg>::dimension);
    return split<cuts...>(std::forward<Arg>(arg));
  }


  /// Split a special matrix vertically, returning a regular matrix.
  template<std::size_t ... cuts, typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  split_vertical(Arg&& arg)
  {
    static_assert((0 + ... + cuts) <= MatrixTraits<Arg>::dimension);
    if constexpr(sizeof...(cuts) == 0)
    {
      return std::tuple {};
    }
    else if constexpr(sizeof...(cuts) == 1 and (0 + ... + cuts) == MatrixTraits<Arg>::dimension)
    {
      return std::tuple {std::forward<Arg>(arg)};
    }
    else
    {
      return std::apply([](const auto&...bs){ return std::tuple {strict(bs)...}; },
        split_vertical<cuts...>(strict_matrix(std::forward<Arg>(arg))));
    }
  }


  /// Split a special matrix horizontally, returning a regular matrix.
  template<std::size_t ... cuts, typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  split_horizontal(Arg&& arg)
  {
    static_assert((0 + ... + cuts) <= MatrixTraits<Arg>::dimension);
    if constexpr(sizeof...(cuts) == 0)
    {
      return std::tuple {};
    }
    else if constexpr(sizeof...(cuts) == 1 and (0 + ... + cuts) == MatrixTraits<Arg>::dimension)
    {
      return std::tuple {std::forward<Arg>(arg)};
    }
    else
    {
      return std::apply([](const auto&...bs){ return std::tuple {strict(bs)...}; },
        split_horizontal<cuts...>(strict_matrix(std::forward<Arg>(arg))));
    }
  }


  /// Return column <code>index</code> of Arg.
  template<typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    return strict(column(strict_matrix(std::forward<Arg>(arg)), index));
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
  template<std::size_t index, typename Arg,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline decltype(auto)
  column(Arg&& arg)
  {
    static_assert(index < MatrixTraits<Arg>::columns);
    return strict(column<index>(strict_matrix(std::forward<Arg>(arg))));
  }


  template<typename Arg, typename Function,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  apply_columnwise(Arg&& arg, const Function& f)
  {
    return strict(apply_columnwise(strict_matrix(std::forward<Arg>(arg)), f));
  }


  template<typename Arg, typename Function,
    std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg> or is_EigenTriangularMatrix_v<Arg> or is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  apply_coefficientwise(Arg&& arg, const Function& f)
  {
    return strict(apply_coefficientwise(strict_matrix(std::forward<Arg>(arg)), f));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_EIGENSPECIALMATRIXOVERLOADS_H
