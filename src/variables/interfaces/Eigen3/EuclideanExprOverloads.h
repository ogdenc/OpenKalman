/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EUCLIDEANEXPROVERLOADS_H
#define OPENKALMAN_EUCLIDEANEXPROVERLOADS_H

namespace OpenKalman
{
  //////////////////////////////
  //        Overloads         //
  //////////////////////////////

  template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  constexpr decltype(auto)
  base_matrix(Arg&& arg) noexcept { return std::forward<Arg>(arg).base_matrix(); }


  /// Convert to strict version of the matrix.
  template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  constexpr decltype(auto)
  strict_matrix(Arg&& arg)
  {
    if constexpr(MatrixTraits<Arg>::Coefficients::axes_only)
    {
      return strict_matrix(base_matrix(std::forward<Arg>(arg)));
    }
    else
    {
      using S = typename MatrixTraits<Arg>::Scalar;
      constexpr Eigen::Index rows = MatrixTraits<Arg>::dimension;
      constexpr Eigen::Index cols = MatrixTraits<Arg>::columns;
      return static_cast<Eigen::Matrix<S, rows, cols>>(std::forward<Arg>(arg));
    }
  }


  /// Convert to strict version of the matrix.
  template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  constexpr decltype(auto)
  strict(Arg&& arg)
  {
    if constexpr(MatrixTraits<Arg>::Coefficients::axes_only)
    {
      return strict(base_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr(is_strict_v<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return MatrixTraits<Arg>::make(strict(base_matrix(std::forward<Arg>(arg))));
    }
  }


  /// Special case for converting a matrix to Euclidean form. This is a shortcut.
  /// Returns the base matrix of the argument, because ToEuclideanExpr<FromEuclideanExpr<M>> reduces to M.
  template<typename Arg, std::enable_if_t<is_FromEuclideanExpr_v<Arg>, int> = 0>
  constexpr decltype(auto)
  to_Euclidean(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg).base_matrix();
  }


  template<typename Coefficients, typename Arg, std::enable_if_t<is_FromEuclideanExpr_v<Arg>, int> = 0>
  constexpr decltype(auto)
  to_Euclidean(Arg&& arg) noexcept
  {
    static_assert(OpenKalman::is_equivalent_v<Coefficients, typename MatrixTraits<Arg>::Coefficients>);
    return to_Euclidean(std::forward<Arg>(arg));
  }


  template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg>, int> = 0>
  constexpr decltype(auto)
  from_Euclidean(Arg&& arg) noexcept
  {
    using Coefficients = typename MatrixTraits<Arg>::Coefficients;
    if constexpr(Coefficients::axes_only)
    {
      return std::forward<Arg>(arg).base_matrix();
    }
    else
    {
      return FromEuclideanExpr<Coefficients, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }
  }


  template<typename Coefficients, typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg>, int> = 0>
  constexpr decltype(auto)
  from_Euclidean(Arg&& arg) noexcept
  {
    static_assert(OpenKalman::is_equivalent_v<Coefficients, typename MatrixTraits<Arg>::Coefficients>);
    return from_Euclidean(std::forward<Arg>(arg));
  }


  template<typename Coefficients, typename Arg, std::enable_if_t<is_FromEuclideanExpr_v<Arg>, int> = 0>
  constexpr decltype(auto)
  wrap_angles(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


  template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  inline auto
  to_diagonal(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::columns == 1);
    return EigenDiagonal(strict_matrix(std::forward<Arg>(arg)));
  }


  template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    return strict_matrix(strict_matrix(std::forward<Arg>(arg)).transpose());
  }


  template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  constexpr decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    return strict_matrix(strict_matrix(std::forward<Arg>(arg)).adjoint());
  }


  template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  inline auto
  determinant(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return strict_matrix(std::forward<Arg>(arg)).determinant();
  }


  template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  inline auto
  trace(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return strict_matrix(std::forward<Arg>(arg)).trace();
  }


  /// Solves AX = B for X (A is a regular matrix type, and B is a Euclidean expression).
  /// A must be invertible. (Does not check.)
  template<
    typename A, typename B,
    std::enable_if_t<is_ToEuclideanExpr_v<A> or is_FromEuclideanExpr_v<A>, int> = 0,
    std::enable_if_t<is_Eigen_matrix_v<B>, int> = 0>
  inline auto solve(A&& a, B&& b) noexcept
  {
    static_assert(MatrixTraits<A>::dimension == MatrixTraits<A>::columns);
    static_assert(MatrixTraits<A>::dimension == MatrixTraits<B>::dimension);
    return solve(strict_matrix(std::forward<A>(a)), std::forward<B>(b));
  }


  template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  constexpr auto
  reduce_columns(Arg&& arg) noexcept { return strict_matrix(reduce_columns(strict_matrix(std::forward<Arg>(arg)))); }

  /**
   * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * Returns L as a triangular matrix.
   */
  template<typename A, std::enable_if_t<is_ToEuclideanExpr_v<A> or is_FromEuclideanExpr_v<A>, int> = 0>
  inline auto
  LQ_decomposition(A&& a)
  {
    return LQ_decomposition(strict_matrix(std::forward<A>(a)));
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
  template<typename A, std::enable_if_t<is_ToEuclideanExpr_v<A> or is_FromEuclideanExpr_v<A>, int> = 0>
  inline auto
  QR_decomposition(A&& a)
  {
    return QR_decomposition(strict_matrix(std::forward<A>(a)));
  }


  /// Concatenate one or more EuclideanExpr objects vertically.
  template<typename V, typename ... Vs,
    std::enable_if_t<
      std::conjunction_v<is_ToEuclideanExpr<V>, is_ToEuclideanExpr<Vs>...> or
      std::conjunction_v<is_FromEuclideanExpr<V>, is_FromEuclideanExpr<Vs>...>, int> = 0>
  constexpr decltype(auto)
  concatenate_vertical(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      constexpr auto cols = MatrixTraits<V>::columns;
      static_assert(((cols == MatrixTraits<Vs>::columns) and ...));
      using C = Concatenate<typename MatrixTraits<V>::Coefficients, typename MatrixTraits<Vs>::Coefficients...>;
      return MatrixTraits<V>::template make<C>(
        concatenate_vertical(base_matrix(std::forward<V>(v)), base_matrix(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  /// Concatenate one or more EuclideanExpr objects horizontally.
  template<typename V, typename ... Vs,
    std::enable_if_t<
      std::conjunction_v<is_ToEuclideanExpr<V>, is_ToEuclideanExpr<Vs>...> or
      std::conjunction_v<is_FromEuclideanExpr<V>, is_FromEuclideanExpr<Vs>...>, int> = 0>
  constexpr decltype(auto)
  concatenate_horizontal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      using C = typename MatrixTraits<V>::Coefficients;
      static_assert(std::conjunction_v<std::is_same<C, typename MatrixTraits<Vs>::Coefficients>...>);
      return MatrixTraits<V>::template make<C>(
        concatenate_horizontal(base_matrix(std::forward<V>(v)), base_matrix(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  namespace internal
  {
    template<typename G, typename Expr, typename CC>
    struct SplitEuclideanVertF
    {
      template<typename RC, typename, typename Arg>
      static auto call(Arg&& arg)
      {
        return G::template call<RC, CC>(MatrixTraits<Expr>::template make<RC>(std::forward<Arg>(arg)));
      }
    };

    template<typename G, typename Expr, typename RC>
    struct SplitEuclideanHorizF
    {
      template<typename, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        return G::template call<RC, CC>(MatrixTraits<Expr>::template make<RC>(std::forward<Arg>(arg)));
      }
    };

    template<typename G, typename Expr>
    struct SplitEuclideanDiagF
    {
      template<typename RC, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        return G::template call<RC, CC>(MatrixTraits<Expr>::template make<RC>(std::forward<Arg>(arg)));
      }
    };
  }

  /// Split into one or more Euclidean expressions vertically.
  template<typename F, typename...Cs, typename Arg,
    std::enable_if_t<(is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>) and not is_coefficient_v<F>, int> = 0>
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    static_assert(is_prefix_v<Concatenate<Cs...>, typename MatrixTraits<Arg>::Coefficients>);
    using CC = Axes<MatrixTraits<Arg>::columns>;
    constexpr auto euclidean = is_FromEuclideanExpr_v<Arg>;
    return split_vertical<internal::SplitEuclideanVertF<F, Arg, CC>, euclidean, Cs...>(
      base_matrix(std::forward<Arg>(arg)));
  }

  /// Split into one or more Euclidean expressions vertically.
  template<typename F, bool, typename...Cs, typename Arg,
    std::enable_if_t<(is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>) and not is_coefficient_v<F>, int> = 0>
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<F, Cs...>(std::forward<Arg>(arg));
  }

  /// Split into one or more Euclidean expressions vertically.
  template<typename...Cs, typename Arg,
    std::enable_if_t<(is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>) and
      std::conjunction_v<is_coefficient<Cs>...>, int> = 0>
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    static_assert(is_prefix_v<Concatenate<Cs...>, typename MatrixTraits<Arg>::Coefficients>);
    return split_vertical<internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }

  /// Split into one or more Euclidean expressions vertically. The expression is evaluated to a strict matrix first.
  /// @tparam cut Number of rows in the first cut.
  /// @tparam cuts Number of rows in the second and subsequent cuts.
  template<std::size_t cut, std::size_t ... cuts,
    typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::dimension);
    if constexpr(cut == MatrixTraits<Arg>::dimension and sizeof...(cuts) == 0)
    {
      return std::tuple {std::forward<Arg>(arg)};
    }
    else
    {
      return split_vertical<cut, cuts...>(strict_matrix(std::forward<Arg>(arg)));
    }
  }


  /// Split into one or more Euclidean expressions horizontally.
  template<typename F, typename...Cs, typename Arg,
    std::enable_if_t<(is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>) and not is_coefficient_v<F>, int> = 0>
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    static_assert((0 + ... + Cs::size) <= MatrixTraits<Arg>::columns);
    using RC = typename MatrixTraits<Arg>::Coefficients;
    return split_horizontal<internal::SplitEuclideanHorizF<F, Arg, RC>, Cs...>(
      base_matrix(std::forward<Arg>(arg)));
  }

  /// Split into one or more Euclidean expressions horizontally.
  template<typename...Cs, typename Arg,
    std::enable_if_t<(is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>) and
      std::conjunction_v<is_coefficient<Cs>...>, int> = 0>
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }

  /// Split into one or more Euclidean expressions horizontally.
  /// @tparam cut Number of columns in the first cut.
  /// @tparam cuts Number of columns in the second and subsequent cuts.
  template<std::size_t cut, std::size_t ... cuts,
    typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::columns);
    return split_horizontal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
  template<typename F, typename...Cs, typename Arg,
    std::enable_if_t<(is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>) and not is_coefficient_v<F>, int> = 0>
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    static_assert(is_prefix_v<Concatenate<Cs...>, typename MatrixTraits<Arg>::Coefficients>);
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    constexpr auto euclidean = is_FromEuclideanExpr_v<Arg>;
    return split_diagonal<internal::SplitEuclideanDiagF<F, Arg>, euclidean, Cs...>(
      base_matrix(std::forward<Arg>(arg)));
  }

  /// Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
  template<typename F, bool, typename...Cs, typename Arg,
    std::enable_if_t<(is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>) and not is_coefficient_v<F>, int> = 0>
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    static_assert(is_prefix_v<Concatenate<Cs...>, typename MatrixTraits<Arg>::Coefficients>);
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return split_diagonal<F, Cs...>(std::forward<Arg>(arg));
  }

  /// Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
  template<typename...Cs, typename Arg,
    std::enable_if_t<(is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>) and
      std::conjunction_v<is_coefficient<Cs>...>, int> = 0>
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    static_assert(is_prefix_v<Concatenate<Cs...>, typename MatrixTraits<Arg>::Coefficients>);
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return split_diagonal<internal::default_split_function, false, Cs...>(std::forward<Arg>(arg));
  }

  /// Split into one or more Euclidean expressions diagonally.
  /// The expression (which must be square) is evaluated to a strict matrix first.
  /// @tparam cut Number of rows in the first cut.
  /// @tparam cuts Number of rows in the second and subsequent cuts.
  template<std::size_t cut, std::size_t ... cuts,
    typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::dimension);
    if constexpr(cut == MatrixTraits<Arg>::dimension and sizeof...(cuts) == 0)
    {
      return std::tuple {std::forward<Arg>(arg)};
    }
    else
    {
      return split_diagonal<cut, cuts...>(strict_matrix(std::forward<Arg>(arg)));
    }
  }


  /// Get element (i, j) of ToEuclideanExpr or FromEuclideanExpr matrix arg.
  template<typename Arg, std::enable_if_t<
    ((is_FromEuclideanExpr_v<Arg> and not is_ToEuclideanExpr_v<typename MatrixTraits<Arg>::BaseMatrix>) or
      is_ToEuclideanExpr_v<Arg>) and
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>, int> = 0>
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if constexpr (MatrixTraits<Arg>::Coefficients::axes_only)
    {
      return get_element(base_matrix(std::forward<Arg>(arg)), i, j);
    }
    else
    {
      using Coeffs = typename MatrixTraits<Arg>::Coefficients;
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      const auto get_coeff = [j, &arg] (const std::size_t row)
        {
          return get_element(base_matrix(std::forward<Arg>(arg)), row, j);
        };
      if constexpr(is_ToEuclideanExpr_v<Arg>)
      {
        return to_Euclidean<Coeffs, Scalar>(i, get_coeff);
      }
      else
      {
        return from_Euclidean<Coeffs, Scalar>(i, get_coeff);
      }
    }
  }


  /// Get element (i) of one-column ToEuclideanExpr or FromEuclideanExpr matrix arg.
  template<typename Arg, std::enable_if_t<
    ((is_FromEuclideanExpr_v<Arg> and not is_ToEuclideanExpr_v<typename MatrixTraits<Arg>::BaseMatrix>) or
      is_ToEuclideanExpr_v<Arg>) and
    is_element_gettable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>, int> = 0>
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    if constexpr (MatrixTraits<Arg>::Coefficients::axes_only)
    {
      return get_element(base_matrix(std::forward<Arg>(arg)), i);
    }
    else
    {
      using Coeffs = typename MatrixTraits<Arg>::Coefficients;
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      const auto get_coeff = [&arg] (const std::size_t row)
        {
          return get_element(base_matrix(std::forward<Arg>(arg)), row);
        };
      if constexpr(is_ToEuclideanExpr_v<Arg>)
      {
        return to_Euclidean<Coeffs, Scalar>((std::size_t) i, get_coeff);
      }
      else
      {
        return from_Euclidean<Coeffs, Scalar>((std::size_t) i, get_coeff);
      }
    }
  }


  /// Get element (i, j) of FromEuclideanExpr(ToEuclideanExpr) matrix.
  template<typename Arg,
    std::enable_if_t<is_FromEuclideanExpr_v<Arg> and is_ToEuclideanExpr_v<typename MatrixTraits<Arg>::BaseMatrix> and
      is_element_gettable_v<typename MatrixTraits<typename MatrixTraits<Arg>::BaseMatrix>::BaseMatrix, 2>, int> = 0>
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if constexpr (MatrixTraits<Arg>::Coefficients::axes_only)
    {
      return get_element(base_matrix(base_matrix(std::forward<Arg>(arg))), i, j);
    }
    else
    {
      using Coeffs = typename MatrixTraits<Arg>::Coefficients;
      const auto get_coeff = [j, &arg] (const std::size_t row) {
        return get_element(base_matrix(base_matrix(std::forward<Arg>(arg))), row, j);
      };
      return wrap_get<Coeffs>(i, get_coeff);
    }
  }


  /// Get element (i) of FromEuclideanExpr(ToEuclideanExpr) matrix.
  template<typename Arg,
    std::enable_if_t<is_FromEuclideanExpr_v<Arg> and is_ToEuclideanExpr_v<typename MatrixTraits<Arg>::BaseMatrix> and
      is_element_gettable_v<typename MatrixTraits<typename MatrixTraits<Arg>::BaseMatrix>::BaseMatrix, 1>, int> = 0>
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    if constexpr (MatrixTraits<Arg>::Coefficients::axes_only)
    {
      return get_element(base_matrix(base_matrix(std::forward<Arg>(arg))), i);
    }
    else
    {
      using Coeffs = typename MatrixTraits<Arg>::Coefficients;
      const auto get_coeff = [&arg] (const std::size_t row) {
        return get_element(base_matrix(base_matrix(std::forward<Arg>(arg))), row);
      };
      return wrap_get<Coeffs>(i, get_coeff);
    }
  }


  /// Set element (i, j) of ToEuclideanExpr or FromEuclideanExpr matrix arg if coefficients are only axes.
  template<typename Arg, typename Scalar, std::enable_if_t<
    ((is_FromEuclideanExpr_v<Arg> and not is_ToEuclideanExpr_v<typename MatrixTraits<Arg>::BaseMatrix>) or
      is_ToEuclideanExpr_v<Arg>) and
    not std::is_const_v<std::remove_reference_t<Arg>> and
    MatrixTraits<Arg>::Coefficients::axes_only and
    is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 2>, int> = 0>
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    set_element(base_matrix(arg), s, i, j);
  }


  /// Set element (i) of ToEuclideanExpr or FromEuclideanExpr matrix arg if coefficients are only axes.
  template<typename Arg, typename Scalar, std::enable_if_t<
    ((is_FromEuclideanExpr_v<Arg> and not is_ToEuclideanExpr_v<typename MatrixTraits<Arg>::BaseMatrix>) or
      is_ToEuclideanExpr_v<Arg>) and
    not std::is_const_v<std::remove_reference_t<Arg>> and
    MatrixTraits<Arg>::Coefficients::axes_only and
    is_element_settable_v<typename MatrixTraits<Arg>::BaseMatrix, 1>, int> = 0>
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    set_element(base_matrix(arg), s, i);
  }


  /// Set element (i, j) of arg in FromEuclideanExpr(ToEuclideanExpr(arg)) to s.
  /// This function sets the base matrix, not the wrapped resulting matrix.
  /// For example, if the coefficient is Polar<Distance, Angle> and the initial value of a
  /// single-column vector is {-1., pi/2}, then set_element(arg, pi/4, 1, 0) will replace p/2 with pi/4 to
  /// yield {-1., pi/4} in the base matrix. The resulting wrapped expression will yield {1., -3*pi/4}.
  template<typename Arg, typename Scalar,
    std::enable_if_t<is_FromEuclideanExpr_v<Arg> and is_ToEuclideanExpr_v<typename MatrixTraits<Arg>::BaseMatrix> and
      not std::is_const_v<std::remove_reference_t<Arg>> and
      is_element_settable_v<typename MatrixTraits<typename MatrixTraits<Arg>::BaseMatrix>::BaseMatrix, 2>, int> = 0>
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if constexpr (MatrixTraits<Arg>::Coefficients::axes_only)
    {
      set_element(base_matrix(base_matrix(arg)), s, i, j);
    }
    else
    {
      using Coeffs = typename MatrixTraits<Arg>::Coefficients;
      const auto get_coeff = [&arg, j] (const std::size_t row) {
        return get_element(base_matrix(base_matrix(arg)), row, j);
      };
      const auto set_coeff = [&arg, j] (const Scalar value, const std::size_t row) {
        set_element(base_matrix(base_matrix(arg)), value, row, j);
      };
      wrap_set<Coeffs>(s, i, set_coeff, get_coeff);
    }
  }


  /// Set element (i) of arg in FromEuclideanExpr(ToEuclideanExpr(arg)) to s, where arg is a single-column vector.
  /// This function sets the base matrix, not the wrapped resulting matrix.
  /// For example, if the coefficient is Polar<Distance, Angle> and the initial value of a
  /// single-column vector is {-1., pi/2}, then set_element(arg, pi/4, 1) will replace p/2 with pi/4 to
  /// yield {-1., pi/4} in the base matrix. The resulting wrapped expression will yield {1., -3*pi/4}.
  template<typename Arg, typename Scalar,
    std::enable_if_t<is_FromEuclideanExpr_v<Arg> and is_ToEuclideanExpr_v<typename MatrixTraits<Arg>::BaseMatrix> and
      not std::is_const_v<std::remove_reference_t<Arg>> and
      is_element_settable_v<typename MatrixTraits<typename MatrixTraits<Arg>::BaseMatrix>::BaseMatrix, 1>, int> = 0>
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    if constexpr (MatrixTraits<Arg>::Coefficients::axes_only)
    {
      set_element(base_matrix(base_matrix(arg)), s, i);
    }
    else
    {
      using Coeffs = typename MatrixTraits<Arg>::Coefficients;
      const auto get_coeff = [&arg] (const std::size_t row) {
        return get_element(base_matrix(base_matrix(arg)), row);
      };
      const auto set_coeff = [&arg] (const Scalar value, const std::size_t row) {
        set_element(base_matrix(base_matrix(arg)), value, row);
      };
      wrap_set<Coeffs>(s, i, set_coeff, get_coeff);
    }
  }


  template<typename Arg, std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  inline decltype(auto)
  column(Arg&& arg, const std::size_t index)
  {
    return MatrixTraits<Arg>::make(column(base_matrix(std::forward<Arg>(arg)), index));
  }


  template<size_t index, typename Arg,
    std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0>
  inline decltype(auto)
  column(Arg&& arg)
  {
    return MatrixTraits<Arg>::make(column<index>(base_matrix(std::forward<Arg>(arg))));
  }


  template<
    typename Arg, typename Function,
    std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0,
    std::enable_if_t<std::is_void_v<std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0))&>>, int> = 0>
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    using Coefficients = typename MatrixTraits<Arg>::Coefficients;
    const auto f_base = [&f](auto& col)
    {
      auto mc = MatrixTraits<Arg>::template make<Coefficients>(col);
      f(mc);
    };
    auto& c = base_matrix(arg);
    apply_columnwise(c, f_base);
    return arg;
  }


  template<
    typename Arg, typename Function,
    std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0,
    std::enable_if_t<std::is_void_v<std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0))&, std::size_t>>, int> = 0>
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    using Coefficients = typename MatrixTraits<Arg>::Coefficients;
    const auto f_base = [&f](auto& col, std::size_t i)
    {
      auto mc = MatrixTraits<Arg>::template make<Coefficients>(col);
      f(mc, i);
    };
    auto& c = base_matrix(arg);
    apply_columnwise(c, f_base);
    return arg;
  }


  template<
    typename Arg, typename Function,
    std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0,
    std::enable_if_t<not std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&>>, int> = 0>
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    using Coefficients = typename MatrixTraits<Arg>::Coefficients;
    const auto f_base = [&f](const auto& col) {
      return f(MatrixTraits<Arg>::template make<Coefficients>(col));
    };
    return apply_columnwise(base_matrix(arg), f_base);
  }


  template<
    typename Arg, typename Function,
    std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0,
    std::enable_if_t<not std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&, std::size_t>>, int> = 0>
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    using Coefficients = typename MatrixTraits<Arg>::Coefficients;
    const auto f_base = [&f](const auto& col, std::size_t i) {
      return f(MatrixTraits<Arg>::template make<Coefficients>(col), i);
    };
    return apply_columnwise(base_matrix(arg), f_base);
  }


  template<
    std::size_t count, typename Function,
    std::enable_if_t<is_ToEuclideanExpr_v<std::invoke_result_t<Function>> or
      is_FromEuclideanExpr_v<std::invoke_result_t<Function>>, int> = 0>
  inline auto
  apply_columnwise(const Function& f)
  {
    using ResultType = std::invoke_result_t<Function>;
    const auto f_base = [&f] { return base_matrix(f()); };
    return MatrixTraits<ResultType>::make(apply_columnwise<count>(f_base));
  }


  template<
    std::size_t count, typename Function,
    std::enable_if_t<is_ToEuclideanExpr_v<std::invoke_result_t<Function, std::size_t>> or
      is_FromEuclideanExpr_v<std::invoke_result_t<Function, std::size_t>>, int> = 0>
  inline auto
  apply_columnwise(const Function& f)
  {
    using ResultType = std::invoke_result_t<Function, std::size_t>;
    const auto f_base = [&f](std::size_t i) { return base_matrix(f(i)); };
    return MatrixTraits<ResultType>::make(apply_columnwise<count>(f_base));
  }


  ////

  template<typename Arg, typename Function,
    std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0,
    std::enable_if_t<std::is_arithmetic_v<std::invoke_result_t<Function,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>&& >>, int> = 0>
  inline auto
  apply_coefficientwise(const Arg& arg, const Function& f)
  {
    return strict_matrix(apply_coefficientwise(strict_matrix(arg), f));
  }


  template<typename Arg, typename Function,
    std::enable_if_t<is_ToEuclideanExpr_v<Arg> or is_FromEuclideanExpr_v<Arg>, int> = 0,
    std::enable_if_t<std::is_arithmetic_v<std::invoke_result_t<Function,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>&&, std::size_t, std::size_t>>, int> = 0>
  inline auto
  apply_coefficientwise(const Arg& arg, const Function& f)
  {
    return apply_coefficientwise(strict_matrix(arg), f);
  }


  /**
   * Fill a matrix of to-Euclidean- or from_Euclidean-transformed values selected from a random distribution.
   * The Gaussian distribution has zero mean and standard deviation sigma (1, if not specified).
   **/
  template<
    typename ReturnType,
    template<typename Scalar> typename distribution_type = std::normal_distribution,
    typename random_number_engine = std::mt19937,
    typename...Params,
    std::enable_if_t<is_ToEuclideanExpr_v<ReturnType> or is_FromEuclideanExpr_v<ReturnType>, int> = 0>
  static auto
  randomize(Params...params)
  {
    using Scalar = typename MatrixTraits<ReturnType>::Scalar;
    using B = typename MatrixTraits<ReturnType>::BaseMatrix;
    constexpr auto rows = MatrixTraits<B>::dimension;
    constexpr auto cols = MatrixTraits<B>::columns;
    using Ps = typename distribution_type<Scalar>::param_type;
    static_assert(std::is_constructible_v<Ps, Params...> or sizeof...(Params) == rows or sizeof...(Params) == rows * cols,
      "Params... must be (1) a parameter set or list of parameter sets, "
      "(2) a list of parameter sets, one for each row, or (3) a list of parameter sets, one for each coefficient.");
    return MatrixTraits<ReturnType>::make(randomize<B, distribution_type, random_number_engine>(params...));
  }


  /////////////////////////////////////////
  //        Arithmetic Operators         //
  /////////////////////////////////////////

  /// Shortcut rule for adding FromEuclideanExpr<ToEuclideanExpr<Arg>> to matrix value, possible because of symmetry.
  template<typename Arg1, typename Arg2,
    std::enable_if_t<is_FromEuclideanExpr_v<Arg1>, int> = 0,
    std::enable_if_t<is_FromEuclideanExpr_v<Arg2>, int> = 0,
    std::enable_if_t<is_ToEuclideanExpr_v<typename MatrixTraits<Arg1>::BaseMatrix>, int> = 0,
    std::enable_if_t<is_ToEuclideanExpr_v<typename MatrixTraits<Arg2>::BaseMatrix>, int> = 0>
  constexpr decltype(auto) operator+(Arg1&& v1, Arg2&& v2)
  {
    static_assert(is_equivalent_v<typename MatrixTraits<Arg1>::Coefficients, typename MatrixTraits<Arg2>::Coefficients>);
    using C = typename MatrixTraits<Arg1>::Coefficients;
    auto m1 = strict_matrix(std::forward<Arg1>(v1));
    auto m2 = strict_matrix(std::forward<Arg2>(v2));
    return strict(wrap_angles<C>(m1 + m2));
  }


  /// Shortcut rule for subtracting FromEuclideanExpr<ToEuclideanExpr<Arg>> to matrix value.
  template<typename Arg1, typename Arg2,
    std::enable_if_t<is_FromEuclideanExpr_v<Arg1>, int> = 0,
    std::enable_if_t<is_FromEuclideanExpr_v<Arg2>, int> = 0,
    std::enable_if_t<is_ToEuclideanExpr_v<typename MatrixTraits<Arg1>::BaseMatrix>, int> = 0,
    std::enable_if_t<is_ToEuclideanExpr_v<typename MatrixTraits<Arg2>::BaseMatrix>, int> = 0>
  constexpr decltype(auto) operator-(Arg1&& v1, Arg2&& v2)
  {
    static_assert(is_equivalent_v<typename MatrixTraits<Arg1>::Coefficients, typename MatrixTraits<Arg2>::Coefficients>);
    using C = typename MatrixTraits<Arg1>::Coefficients;
    auto m1 = strict_matrix(std::forward<Arg1>(v1));
    auto m2 = strict_matrix(std::forward<Arg2>(v2));
    return strict(wrap_angles<C>(m1 - m2));
  }


  /// Shortcut rule for negating FromEuclideanExpr<ToEuclideanExpr<Arg>>, possible because of symmetry.
  template<typename Arg,
    std::enable_if_t<is_FromEuclideanExpr_v<Arg>, int> = 0,
    std::enable_if_t<is_ToEuclideanExpr_v<typename MatrixTraits<Arg>::BaseMatrix>, int> = 0>
  constexpr decltype(auto) operator-(Arg&& v)
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    auto m = strict_matrix(std::forward<Arg>(v));
    return strict(wrap_angles<C>(-m));
  }


}

#endif //OPENKALMAN_EUCLIDEANEXPROVERLOADS_H
