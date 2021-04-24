/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded functions relating to Eigen3::euclidean_expr types
 */

#ifndef OPENKALMAN_EIGEN3_EUCLIDEAN_OVERLOADS_HPP
#define OPENKALMAN_EIGEN3_EUCLIDEAN_OVERLOADS_HPP

namespace OpenKalman::Eigen3
{
  //////////////////////////////
  //        Overloads         //
  //////////////////////////////

#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  nested_matrix(Arg&& arg) noexcept { return std::forward<Arg>(arg).nested_matrix(); }


  /**
   * Convert to a self-contained Eigen3 matrix.
   */
#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_native_matrix(Arg&& arg) noexcept
  {
    if constexpr(MatrixTraits<Arg>::RowCoefficients::axes_only)
    {
      return make_native_matrix(nested_matrix(std::forward<Arg>(arg)));
    }
    else
    {
      using S = typename MatrixTraits<Arg>::Scalar;
      constexpr Eigen::Index rows = MatrixTraits<Arg>::rows;
      constexpr Eigen::Index cols = MatrixTraits<Arg>::columns;
      return static_cast<Eigen::Matrix<S, rows, cols>>(std::forward<Arg>(arg));
    }
  }


  /// Convert to self-contained version of the matrix.
#ifdef __cpp_concepts
  template<typename...Ts, euclidean_expr Arg>
#else
  template<typename...Ts, typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr(MatrixTraits<Arg>::RowCoefficients::axes_only)
    {
      return make_self_contained<Ts...>(nested_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr(self_contained<nested_matrix_t<Arg>> or
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
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t row_count(Arg&& arg)
  {
    if constexpr (dynamic_rows<Arg>)
    {
      if constexpr (from_euclidean_expr<Arg>)
        return std::forward<Arg>(arg).row_coefficients.dimensions;
      else
        return std::forward<Arg>(arg).row_coefficients.euclidean_dimensions;
    }
    else
    {
      return MatrixTraits<Arg>::rows;
    }
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  constexpr std::size_t column_count(Arg&& arg)
  {
    if constexpr (dynamic_columns<Arg>)
    {
      return column_count(nested_matrix(std::forward<Arg>(arg)));
    }
    else
    {
      return MatrixTraits<Arg>::columns;
    }
  }


  /// Special case for converting a matrix to Euclidean form. This is a shortcut.
  /// Returns the nested matrix of the argument, because ToEuclideanExpr<FromEuclideanExpr<M>> reduces to M.
#ifdef __cpp_concepts
  template<from_euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg).nested_matrix();
  }


#ifdef __cpp_concepts
  template<coefficients Coefficients, from_euclidean_expr Arg> requires
    equivalent_to<Coefficients, typename MatrixTraits<Arg>::RowCoefficients>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<coefficients<Coefficients> and
    from_euclidean_expr<Arg> and equivalent_to<Coefficients, typename MatrixTraits<Arg>::RowCoefficients>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg) noexcept
  {
    return to_euclidean(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<to_euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg) noexcept
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(Coefficients::axes_only)
    {
      return std::forward<Arg>(arg).nested_matrix();
    }
    else
    {
      return FromEuclideanExpr<Coefficients, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<coefficients Coefficients, to_euclidean_expr Arg> requires
    equivalent_to<Coefficients, typename MatrixTraits<Arg>::RowCoefficients>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<coefficients<Coefficients> and
    to_euclidean_expr<Arg> and equivalent_to<Coefficients, typename MatrixTraits<Arg>::RowCoefficients>, int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg) noexcept
  {
    return from_euclidean(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<coefficients Coefficients, from_euclidean_expr Arg>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<coefficients<Coefficients> and
    from_euclidean_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg> and column_vector<Arg>, int> = 0>
#endif
  inline auto
  to_diagonal(Arg&& arg) noexcept
  {
    return DiagonalMatrix(make_native_matrix(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    return make_self_contained(make_native_matrix(std::forward<Arg>(arg)).diagonal());
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    return make_native_matrix(make_native_matrix(std::forward<Arg>(arg)).transpose());
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    return make_native_matrix(make_native_matrix(std::forward<Arg>(arg)).adjoint());
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  inline auto
  determinant(Arg&& arg) noexcept
  {
    return determinant(make_native_matrix(std::forward<Arg>(arg)));
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  trace(Arg&& arg) noexcept
  {
    return make_native_matrix(std::forward<Arg>(arg)).trace();
  }


  /// Solves AX = B for X (A is a regular matrix type, and B is a Euclidean expression).
  /// A must be invertible. (Does not check.)
#ifdef __cpp_concepts
  template<euclidean_expr A, eigen_matrix B> requires square_matrix<A>
#else
  template<typename A, typename B, std::enable_if_t<euclidean_expr<A> and square_matrix<A> and
    eigen_matrix<B>, int> = 0>
#endif
  inline auto solve(A&& a, B&& b) noexcept
  {
    static_assert(MatrixTraits<A>::rows == MatrixTraits<B>::rows);
    return solve(make_native_matrix(std::forward<A>(a)), std::forward<B>(b));
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_columns(Arg&& arg) noexcept { return make_native_matrix(reduce_columns(make_native_matrix(std::forward<Arg>(arg)))); }

  /**
   * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * Returns L as a triangular matrix.
   */
#ifdef __cpp_concepts
  template<euclidean_expr A>
#else
  template<typename A, std::enable_if_t<euclidean_expr<A>, int> = 0>
#endif
  inline auto
  LQ_decomposition(A&& a)
  {
    return LQ_decomposition(make_native_matrix(std::forward<A>(a)));
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  inline auto
  QR_decomposition(Arg&& arg)
  {
    return QR_decomposition(make_native_matrix(std::forward<Arg>(arg)));
  }


  /// Concatenate one or more EuclideanExpr objects vertically.
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires (to_euclidean_expr<V> and ... and to_euclidean_expr<Vs>) or
    (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (to_euclidean_expr<V> and ... and to_euclidean_expr<Vs>) or
    (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_vertical(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      constexpr auto cols = MatrixTraits<V>::columns;
      static_assert(((cols == MatrixTraits<Vs>::columns) and ...));
      using C = Concatenate<typename MatrixTraits<V>::RowCoefficients, typename MatrixTraits<Vs>::RowCoefficients...>;
      return MatrixTraits<V>::template make<C>(
        concatenate_vertical(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  /// Concatenate one or more EuclideanExpr objects horizontally.
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires (to_euclidean_expr<V> and ... and to_euclidean_expr<Vs>) or
    (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (to_euclidean_expr<V> and ... and to_euclidean_expr<Vs>) or
    (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_horizontal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      using C = typename MatrixTraits<V>::RowCoefficients;
      static_assert(std::conjunction_v<std::is_same<C, typename MatrixTraits<Vs>::RowCoefficients>...>);
      return MatrixTraits<V>::template make<C>(
        concatenate_horizontal(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
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
  } // internal


  /// Split into one or more Euclidean expressions vertically.
#ifdef __cpp_concepts
  template<typename F, typename...Cs, euclidean_expr Arg> requires (not coefficients<F>)
#else
  template<typename F, typename...Cs, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and not coefficients<F>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    static_assert(prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>);
    using CC = Axes<MatrixTraits<Arg>::columns>;
    constexpr auto euclidean = from_euclidean_expr<Arg>;
    return split_vertical<internal::SplitEuclideanVertF<F, Arg, CC>, euclidean, Cs...>(
      nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split into one or more Euclidean expressions vertically.
#ifdef __cpp_concepts
  template<typename F, bool, typename...Cs, euclidean_expr Arg> requires (not coefficients<F>)
#else
  template<typename F, bool, typename...Cs, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and not coefficients<F>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<F, Cs...>(std::forward<Arg>(arg));
  }


  /// Split into one or more Euclidean expressions vertically.
#ifdef __cpp_concepts
  template<coefficients...Cs, euclidean_expr Arg>
#else
  template<typename...Cs, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    static_assert(prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>);
    return split_vertical<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split into one or more Euclidean expressions vertically.
   * \details The expression is evaluated to a self_contained matrix first.
   * \tparam cut Number of rows in the first cut.
   * \tparam cuts Number of rows in the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, euclidean_expr Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::rows);
    if constexpr(cut == MatrixTraits<Arg>::rows and sizeof...(cuts) == 0)
    {
      return std::tuple {std::forward<Arg>(arg)};
    }
    else
    {
      return split_vertical<cut, cuts...>(make_native_matrix(std::forward<Arg>(arg)));
    }
  }


  /// Split into one or more Euclidean expressions horizontally.
#ifdef __cpp_concepts
  template<typename F, typename...Cs, euclidean_expr Arg> requires (not coefficients<F>)
#else
  template<typename F, typename...Cs, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and not coefficients<F>, int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    static_assert((0 + ... + Cs::dimensions) <= MatrixTraits<Arg>::columns);
    using RC = typename MatrixTraits<Arg>::RowCoefficients;
    return split_horizontal<internal::SplitEuclideanHorizF<F, Arg, RC>, Cs...>(
      nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split into one or more Euclidean expressions horizontally.
#ifdef __cpp_concepts
  template<coefficients...Cs, euclidean_expr Arg>
#else
  template<typename...Cs, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split into one or more Euclidean expressions horizontally.
   * \tparam cut Number of columns in the first cut.
   * \tparam cuts Number of columns in the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, euclidean_expr Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::columns);
    return split_horizontal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
   */
#ifdef __cpp_concepts
  template<typename F, typename...Cs, euclidean_expr Arg> requires square_matrix<Arg> and (not coefficients<F>)
#else
  template<typename F, typename...Cs, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and square_matrix<Arg> and (not coefficients<F>), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    static_assert(prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>);
    constexpr auto euclidean = from_euclidean_expr<Arg>;
    return split_diagonal<internal::SplitEuclideanDiagF<F, Arg>, euclidean, Cs...>(
      nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
#ifdef __cpp_concepts
  template<typename F, bool, typename...Cs, euclidean_expr Arg> requires square_matrix<Arg> and (not coefficients<F>)
#else
  template<typename F, bool, typename...Cs, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and square_matrix<Arg> and (not coefficients<F>), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    static_assert(prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>);
    return split_diagonal<F, Cs...>(std::forward<Arg>(arg));
  }


  /// Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
#ifdef __cpp_concepts
  template<coefficients...Cs, euclidean_expr Arg> requires square_matrix<Arg>
#else
  template<typename...Cs, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and square_matrix<Arg> and (coefficients<Cs> and ...), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    static_assert(prefix_of<Concatenate<Cs...>, typename MatrixTraits<Arg>::RowCoefficients>);
    return split_diagonal<OpenKalman::internal::default_split_function, false, Cs...>(std::forward<Arg>(arg));
  }

  /// Split into one or more Euclidean expressions diagonally.
  /// The expression (which must be square) is evaluated to a self_contained matrix first.
  /// \tparam cut Number of rows in the first cut.
  /// \tparam cuts Number of rows in the second and subsequent cuts.
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, euclidean_expr Arg> requires square_matrix<Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::rows);
    if constexpr(cut == MatrixTraits<Arg>::rows and sizeof...(cuts) == 0)
    {
      return std::tuple {std::forward<Arg>(arg)};
    }
    else
    {
      return split_diagonal<cut, cuts...>(make_native_matrix(std::forward<Arg>(arg)));
    }
  }


  /// Get element (i, j) of ToEuclideanExpr or FromEuclideanExpr matrix arg.
#ifdef __cpp_concepts
  template<euclidean_expr Arg> requires element_gettable<nested_matrix_t<Arg>, 2> and
    (not to_euclidean_expr<nested_matrix_t<Arg>>)
#else
  template<typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and element_gettable<nested_matrix_t<Arg>, 2> and
      (not to_euclidean_expr<nested_matrix_t<Arg>>), int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if constexpr (MatrixTraits<Arg>::RowCoefficients::axes_only)
    {
      return get_element(nested_matrix(std::forward<Arg>(arg)), i, j);
    }
    else
    {
      using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
      const auto get_coeff = [j, &arg] (const std::size_t row)
        {
          return get_element(nested_matrix(std::forward<Arg>(arg)), row, j);
        };
      if constexpr(to_euclidean_expr<Arg>)
      {
        return OpenKalman::internal::to_euclidean_coeff<Coeffs>(i, get_coeff);
      }
      else
      {
        return OpenKalman::internal::from_euclidean_coeff<Coeffs>(i, get_coeff);
      }
    }
  }


  /// Get element (i) of one-column ToEuclideanExpr or FromEuclideanExpr matrix arg.
#ifdef __cpp_concepts
  template<euclidean_expr Arg> requires element_gettable<nested_matrix_t<Arg>, 1> and
    (not to_euclidean_expr<nested_matrix_t<Arg>>)
#else
  template<typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and element_gettable<nested_matrix_t<Arg>, 1> and
      (not to_euclidean_expr<nested_matrix_t<Arg>>), int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    if constexpr (MatrixTraits<Arg>::RowCoefficients::axes_only)
    {
      return get_element(nested_matrix(std::forward<Arg>(arg)), i);
    }
    else
    {
      using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
      const auto get_coeff = [&arg] (const std::size_t row)
        {
          return get_element(nested_matrix(std::forward<Arg>(arg)), row);
        };
      if constexpr(to_euclidean_expr<Arg>)
      {
        return OpenKalman::internal::to_euclidean_coeff<Coeffs>((std::size_t) i, get_coeff);
      }
      else
      {
        return OpenKalman::internal::from_euclidean_coeff<Coeffs>((std::size_t) i, get_coeff);
      }
    }
  }


  /// Get element (i, j) of FromEuclideanExpr(ToEuclideanExpr) matrix.
#ifdef __cpp_concepts
  template<from_euclidean_expr Arg> requires to_euclidean_expr<nested_matrix_t<Arg>> and
    element_gettable<nested_matrix_t<nested_matrix_t<Arg>>, 2>
#else
  template<typename Arg, std::enable_if_t<
    from_euclidean_expr<Arg> and to_euclidean_expr<nested_matrix_t<Arg>> and
      element_gettable<nested_matrix_t<nested_matrix_t<Arg>>, 2>, int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i, const std::size_t j)
  {
    if constexpr (MatrixTraits<Arg>::RowCoefficients::axes_only)
    {
      return get_element(nested_matrix(nested_matrix(std::forward<Arg>(arg))), i, j);
    }
    else
    {
      using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
      const auto get_coeff = [j, &arg] (const std::size_t row) {
        return get_element(nested_matrix(nested_matrix(std::forward<Arg>(arg))), row, j);
      };
      return OpenKalman::internal::wrap_get<Coeffs>(i, get_coeff);
    }
  }


  /// Get element (i) of FromEuclideanExpr(ToEuclideanExpr) matrix.
#ifdef __cpp_concepts
  template<from_euclidean_expr Arg> requires to_euclidean_expr<nested_matrix_t<Arg>> and
    element_gettable<nested_matrix_t<nested_matrix_t<Arg>>, 1>
#else
  template<typename Arg, std::enable_if_t<
    from_euclidean_expr<Arg> and to_euclidean_expr<nested_matrix_t<Arg>> and
      element_gettable<nested_matrix_t<nested_matrix_t<Arg>>, 1>, int> = 0>
#endif
  inline auto
  get_element(Arg&& arg, const std::size_t i)
  {
    if constexpr (MatrixTraits<Arg>::RowCoefficients::axes_only)
    {
      return get_element(nested_matrix(nested_matrix(std::forward<Arg>(arg))), i);
    }
    else
    {
      using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
      const auto get_coeff = [&arg] (const std::size_t row) {
        return get_element(nested_matrix(nested_matrix(std::forward<Arg>(arg))), row);
      };
      return OpenKalman::internal::wrap_get<Coeffs>(i, get_coeff);
    }
  }


  /**
   * \brief Set element (i, j) of ToEuclideanExpr or FromEuclideanExpr matrix arg if coefficients are only axes.
   * \param arg The matrix whose element is to be set.
   * \param s A scalar value.
   * \param i An index.
   * \param j An index.
   */
#ifdef __cpp_concepts
  template<euclidean_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> Scalar> requires
    (not to_euclidean_expr<nested_matrix_t<Arg>>) and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    MatrixTraits<Arg>::RowCoefficients::axes_only and
    element_settable<nested_matrix_t<Arg>, 2>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    euclidean_expr<Arg> and
    std::is_convertible_v<Scalar, typename MatrixTraits<Arg>::Scalar> and
    not to_euclidean_expr<nested_matrix_t<Arg>> and
    not std::is_const_v<std::remove_reference_t<Arg>> and
    MatrixTraits<Arg>::RowCoefficients::axes_only and
    element_settable<nested_matrix_t<Arg>, 2>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    set_element(nested_matrix(arg), s, i, j);
  }


  /**
   * \brief Set an element of ToEuclideanExpr or FromEuclideanExpr matrix arg if coefficients are only axes.
   * \param arg The matrix whose element is to be set.
   * \param s A scalar value.
   * \param i Index of the element.
   */
#ifdef __cpp_concepts
  template<euclidean_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> Scalar> requires
    (not to_euclidean_expr<nested_matrix_t<Arg>>) and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    MatrixTraits<Arg>::RowCoefficients::axes_only and
    element_settable<nested_matrix_t<Arg>, 1>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    euclidean_expr<Arg> and
    std::is_convertible_v<Scalar, typename MatrixTraits<Arg>::Scalar> and
    not to_euclidean_expr<nested_matrix_t<Arg>> and
    not std::is_const_v<std::remove_reference_t<Arg>> and
    MatrixTraits<Arg>::RowCoefficients::axes_only and
    element_settable<nested_matrix_t<Arg>, 1>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    set_element(nested_matrix(arg), s, i);
  }


  /**
   * \brief Set element (i, j) of arg in FromEuclideanExpr(ToEuclideanExpr(arg)) to s.
   * \details This function sets the nested matrix, not the wrapped resulting matrix.
   * For example, if the coefficient is Polar<Distance, angle::Radians> and the initial value of a
   * single-column vector is {-1., pi/2}, then set_element(arg, pi/4, 1, 0) will replace p/2 with pi/4 to
   * yield {-1., pi/4} in the nested matrix. The resulting wrapped expression will yield {1., -3*pi/4}.
   * \tparam Arg The matrix to set.
   * \tparam Scalar The value to set the coefficient to.
   * \param i The row of the coefficient.
   * \param j The column of the coefficient.
   */
#ifdef __cpp_concepts
  template<from_euclidean_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> Scalar> requires
    to_euclidean_expr<nested_matrix_t<Arg>> and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    element_settable<nested_matrix_t<nested_matrix_t<Arg>>, 2>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    std::is_convertible_v<Scalar, typename MatrixTraits<Arg>::Scalar> and
    from_euclidean_expr<Arg> and to_euclidean_expr<nested_matrix_t<Arg>> and
    not std::is_const_v<std::remove_reference_t<Arg>> and
    element_settable<nested_matrix_t<nested_matrix_t<Arg>>, 2>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    if constexpr (MatrixTraits<Arg>::RowCoefficients::axes_only)
    {
      set_element(nested_matrix(nested_matrix(arg)), s, i, j);
    }
    else
    {
      using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
      const auto get_coeff = [&arg, j] (const std::size_t row) {
        return get_element(nested_matrix(nested_matrix(arg)), row, j);
      };
      const auto set_coeff = [&arg, j] (const std::size_t row, const Scalar value) {
        set_element(nested_matrix(nested_matrix(arg)), value, row, j);
      };
      OpenKalman::internal::wrap_set<Coeffs>(i, s, set_coeff, get_coeff);
    }
  }


  /**
   * \brief Set element (i) of arg in FromEuclideanExpr(ToEuclideanExpr(arg)) to s, where arg is a single-column vector.
   * \details This function sets the nested matrix, not the wrapped resulting matrix.
   * For example, if the coefficient is Polar<Distance, angle::Radians> and the initial value of a
   * single-column vector is {-1., pi/2}, then set_element(arg, pi/4, 1) will replace p/2 with pi/4 to
   * yield {-1., pi/4} in the nested matrix. The resulting wrapped expression will yield {1., -3*pi/4}.
   * \tparam Arg The matrix to set.
   * \tparam Scalar The value to set the coefficient to.
   * \param i The row of the coefficient.
   */
#ifdef __cpp_concepts
  template<from_euclidean_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> Scalar> requires
    to_euclidean_expr<nested_matrix_t<Arg>> and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and
    element_settable<nested_matrix_t<nested_matrix_t<Arg>>, 1>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<
    std::is_convertible_v<Scalar, typename MatrixTraits<Arg>::Scalar> and
    from_euclidean_expr<Arg> and to_euclidean_expr<nested_matrix_t<Arg>> and
    not std::is_const_v<std::remove_reference_t<Arg>> and
    element_settable<nested_matrix_t<nested_matrix_t<Arg>>, 1>, int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    if constexpr (MatrixTraits<Arg>::RowCoefficients::axes_only)
    {
      set_element(nested_matrix(nested_matrix(arg)), s, i);
    }
    else
    {
      using Coeffs = typename MatrixTraits<Arg>::RowCoefficients;
      const auto get_coeff = [&arg] (const std::size_t row) {
        return get_element(nested_matrix(nested_matrix(arg)), row);
      };
      const auto set_coeff = [&arg] (const std::size_t row, const Scalar value) {
        set_element(nested_matrix(nested_matrix(arg)), value, row);
      };
      OpenKalman::internal::wrap_set<Coeffs>(i, s, set_coeff, get_coeff);
    }
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg, const std::size_t index)
  {
    return MatrixTraits<Arg>::make(column(nested_matrix(std::forward<Arg>(arg)), index));
  }


#ifdef __cpp_concepts
  template<std::size_t index, euclidean_expr Arg>
#else
  template<std::size_t index, typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    return MatrixTraits<Arg>::make(column<index>(nested_matrix(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg, typename Function> requires
    std::is_void_v<std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0))&>> and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_t<Arg>>>) and
    modifiable<nested_matrix_t<Arg>, nested_matrix_t<Arg>>
#else
  template<typename Arg, typename Function, std::enable_if_t<euclidean_expr<Arg> and
    std::is_void_v<std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0))&>> and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_t<Arg>>>) and
    modifiable<nested_matrix_t<Arg>, nested_matrix_t<Arg>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto& col)
    {
      auto mc = MatrixTraits<Arg>::template make<Coefficients>(make_self_contained(std::move(col)));
      // note: mc needs to be self-contained to be successfully returned by f.
      f(mc);
      col = std::move(nested_matrix(mc));
    };
    auto& c = nested_matrix(arg);
    apply_columnwise(c, f_nested);
    return arg;
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg, typename Function> requires
    std::is_void_v<std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0))&, std::size_t>> and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_t<Arg>>>) and
    modifiable<nested_matrix_t<Arg>, nested_matrix_t<Arg>>
#else
  template<typename Arg, typename Function, std::enable_if_t<euclidean_expr<Arg> and
    std::is_void_v<std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0))&, std::size_t>> and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_t<Arg>>>) and
    modifiable<nested_matrix_t<Arg>, nested_matrix_t<Arg>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](auto& col, std::size_t i)
    {
      auto mc = MatrixTraits<Arg>::template make<Coefficients>(make_self_contained(std::move(col)));
      // note: mc needs to be self-contained to be successfully returned by f.
      f(mc, i);
      col = std::move(nested_matrix(mc));
    };
    auto& c = nested_matrix(arg);
    apply_columnwise(c, f_nested);
    return arg;
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg, typename Function> requires
    (not std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&>>)
#else
  template<typename Arg, typename Function, std::enable_if_t<euclidean_expr<Arg> and
    not std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](const auto& col) {
      return f(MatrixTraits<Arg>::template make<Coefficients>(col));
    };
    return apply_columnwise(nested_matrix(arg), f_nested);
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg, typename Function> requires
    (not std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&, std::size_t>>)
#else
  template<typename Arg, typename Function, std::enable_if_t<euclidean_expr<Arg> and
    not std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    using Coefficients = typename MatrixTraits<Arg>::RowCoefficients;
    const auto f_nested = [&f](const auto& col, std::size_t i) {
      return f(MatrixTraits<Arg>::template make<Coefficients>(col), i);
    };
    return apply_columnwise(nested_matrix(arg), f_nested);
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires euclidean_expr<std::invoke_result_t<Function>>
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    euclidean_expr<std::invoke_result_t<Function>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f)
  {
    using ResultType = std::invoke_result_t<Function>;
    const auto f_nested = [&f] { return nested_matrix(f()); };
    return MatrixTraits<ResultType>::make(apply_columnwise<count>(f_nested));
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires euclidean_expr<std::invoke_result_t<Function, std::size_t>>
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    euclidean_expr<std::invoke_result_t<Function, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f)
  {
    using ResultType = std::invoke_result_t<Function, std::size_t>;
    const auto f_nested = [&f](std::size_t i) { return nested_matrix(f(i)); };
    return MatrixTraits<ResultType>::make(apply_columnwise<count>(f_nested));
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg, typename Function> requires
    std::is_arithmetic_v<std::invoke_result_t<Function, std::decay_t<typename MatrixTraits<Arg>::Scalar>&& >>
#else
  template<typename Arg, typename Function, std::enable_if_t<euclidean_expr<Arg> and
    std::is_arithmetic_v<std::invoke_result_t<Function, std::decay_t<typename MatrixTraits<Arg>::Scalar>&& >>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Arg& arg, const Function& f)
  {
    return make_native_matrix(apply_coefficientwise(make_native_matrix(arg), f));
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg, typename Function> requires std::is_arithmetic_v<std::invoke_result_t<Function,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>&&, std::size_t, std::size_t>>
#else
  template<typename Arg, typename Function, std::enable_if_t<euclidean_expr<Arg> and
    std::is_arithmetic_v<std::invoke_result_t<Function,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>&&, std::size_t, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Arg& arg, const Function& f)
  {
    return apply_coefficientwise(make_native_matrix(arg), f);
  }


  /**
   * Fill a matrix of to-Euclidean- or from_euclidean-transformed values selected from a random distribution.
   * The Gaussian distribution has zero mean and standard deviation sigma (1, if not specified).
   **/
#ifdef __cpp_concepts
  template<
    euclidean_expr ReturnType,
    template<typename Scalar> typename distribution_type = std::normal_distribution,
    std::uniform_random_bit_generator random_number_engine = std::mt19937,
    typename...Params>
#else
  template<
    typename ReturnType,
    template<typename Scalar> typename distribution_type = std::normal_distribution,
    typename random_number_engine = std::mt19937,
    typename...Params,
    std::enable_if_t<euclidean_expr<ReturnType>, int> = 0>
#endif
  inline auto
  randomize(Params...params)
  {
    using Scalar = typename MatrixTraits<ReturnType>::Scalar;
    using B = nested_matrix_t<ReturnType>;
    constexpr auto rows = MatrixTraits<B>::rows;
    constexpr auto cols = MatrixTraits<B>::columns;
    using Ps = typename distribution_type<Scalar>::param_type;
    static_assert(std::is_constructible_v<Ps, Params...> or sizeof...(Params) == rows or sizeof...(Params) == rows * cols,
      "Params... must be (1) a parameter set or list of parameter sets, "
      "(2) a list of parameter sets, one for each row, or (3) a list of parameter sets, one for each coefficient.");
    return MatrixTraits<ReturnType>::make(randomize<B, distribution_type, random_number_engine>(params...));
  }

} // namespace OpenKalman::eigen3

#endif //OPENKALMAN_EIGEN3_EUCLIDEAN_OVERLOADS_HPP
