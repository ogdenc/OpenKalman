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

namespace OpenKalman::interface
{

#ifdef __cpp_concepts
  template<typename T, typename...I> requires
    eigen_zero_expr<T> or eigen_constant_expr<T>
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
  template<diagonal_matrix T, std::convertible_to<const std::size_t&> I> requires
    (eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>) and
    (element_gettable<nested_matrix_of_t<T>, I> or element_gettable<nested_matrix_of_t<T>, I, I>)
  struct GetElement<T, I>
#else
  template<typename T, typename I>
  struct GetElement<T, std::enable_if_t<diagonal_matrix<T> and std::is_convertible_v<I, const std::size_t&> and
    (eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>) and
    (element_gettable<nested_matrix_of_t<T>, I> or element_gettable<nested_matrix_of_t<T>, I, I>)>, I>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, I i)
    {
      if constexpr (element_gettable<nested_matrix_of_t<Arg>, I>)
        return get_element(nested_matrix(std::forward<Arg>(arg)), i);
      else if constexpr (eigen_diagonal_expr<T>)
        return get_element(nested_matrix(std::forward<Arg>(arg)), i, static_cast<I>(1));
      else
        return get_element(nested_matrix(std::forward<Arg>(arg)), i, i);
    }
  };


#ifdef __cpp_concepts
  template<typename T, std::convertible_to<const std::size_t&> I, std::convertible_to<const std::size_t&> J>
  requires (eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>) and
    (element_gettable<nested_matrix_of_t<T>, I, J> or (diagonal_matrix<T> and element_gettable<nested_matrix_of_t<T>, I>))
    struct GetElement<T, I, J>
#else
  template<typename T, typename I, typename J>
  struct GetElement<T, std::enable_if_t<
    (eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>) and
    std::is_convertible_v<I, const std::size_t&> and std::is_convertible_v<J, const std::size_t&> and
    (element_gettable<nested_matrix_of_t<T>, I, J> or (diagonal_matrix<T> and element_gettable<nested_matrix_of_t<T>, I>))>, I, J>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, I i, J j)
    {
      if constexpr (diagonal_matrix<T>)
      {
        if (i == static_cast<I>(j))
        {
          if constexpr (element_gettable<nested_matrix_of_t<Arg>, I>)
            return get_element(nested_matrix(std::forward<Arg>(arg)), i);
          else if constexpr (eigen_diagonal_expr<T>)
            return get_element(nested_matrix(std::forward<Arg>(arg)), i, static_cast<J>(1));
          else
            return get_element(nested_matrix(std::forward<Arg>(arg)), i, j);
        }
        else
        {
          return static_cast<scalar_type_of_t<Arg>>(0);
        }
      }
      else if constexpr (eigen_triangular_expr<T>)
      {
        if (lower_triangular_matrix<Arg> ? i >= static_cast<I>(j) : i <= static_cast<I>(j))
          return get_element(nested_matrix(std::forward<Arg>(arg)), i, j);
        else
          return static_cast<scalar_type_of_t<Arg>>(0);
      }
      else
      {
        static_assert(eigen_self_adjoint_expr<T>);

        if (lower_self_adjoint_matrix<Arg> ? i >= static_cast<I>(j) : i <= static_cast<I>(j))
        {
          if constexpr (complex_number<scalar_type_of_t<Arg>>)
          {
            if (i == j) return std::real(get_element(nested_matrix(std::forward<Arg>(arg)), i, j));
          }
          return get_element(nested_matrix(std::forward<Arg>(arg)), i, j);
        }
        else
        {
          if constexpr (complex_number<scalar_type_of_t<Arg>>)
            return std::conj(get_element(nested_matrix(std::forward<Arg>(arg)), j, i));
          else
            return get_element(nested_matrix(std::forward<Arg>(arg)), j, i);
        }
      }
    }
  };


#ifdef __cpp_concepts
  template<diagonal_matrix T, std::convertible_to<const std::size_t&> I> requires
    (eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>) and
    (element_settable<nested_matrix_of_t<T>, I> or element_settable<nested_matrix_of_t<T>, I, I>)
  struct SetElement<T, I>
#else
  template<typename T, typename I>
  struct SetElement<T, std::enable_if_t<diagonal_matrix<T> and std::is_convertible_v<I, const std::size_t&> and
    (eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>) and
    (element_settable<nested_matrix_of_t<T>, I> or element_settable<nested_matrix_of_t<T>, I, I>)>, I>
#endif
  {
    template<typename Arg, typename Scalar>
    static void set(Arg& arg, const Scalar s, I i)
    {
      if constexpr (element_settable<nested_matrix_of_t<Arg>, I>)
        set_element(nested_matrix(arg), s, i);
      else
        set_element(nested_matrix(arg), s, i, static_cast<I>(1));
    }
  };


#ifdef __cpp_concepts
  template<typename T, std::convertible_to<const std::size_t&> I, std::convertible_to<const std::size_t&> J>
  requires (eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>) and
    (element_settable<nested_matrix_of_t<T>, I, J> or (diagonal_matrix<T> and element_settable<nested_matrix_of_t<T>, I>))
    struct SetElement<T, I, J>
#else
  template<typename T, typename I, typename J>
  struct SetElement<T, std::enable_if_t<
    (eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>) and
    std::is_convertible_v<I, const std::size_t&> and std::is_convertible_v<J, const std::size_t&> and
    (element_settable<nested_matrix_of_t<T>, I, J> or (diagonal_matrix<T> and element_settable<nested_matrix_of_t<T>, I>))>, I, J>
#endif
  {
    template<typename Arg, typename Scalar>
    static void set(Arg& arg, const Scalar s, I i, J j)
    {
      if constexpr (diagonal_matrix<T>)
      {
        if (i == static_cast<I>(j))
        {
          if constexpr (element_settable<nested_matrix_of_t<Arg>, I>)
            set_element(nested_matrix(arg), s, i);
          else if constexpr (eigen_diagonal_expr<T>)
            set_element(nested_matrix(arg), s, i, static_cast<I>(1));
          else
            set_element(nested_matrix(arg), s, i, j);
        }
        else if (s != 0)
          throw std::out_of_range("Cannot set non-diagonal element of a diagonal matrix to a non-zero value.");
      }
      else if constexpr (eigen_triangular_expr<T>)
      {
        if (lower_triangular_matrix<Arg> ? i >= static_cast<I>(j) : i <= static_cast<I>(j))
          set_element(nested_matrix(arg), s, i, j);
        else if (s != 0)
          throw std::out_of_range("Cannot set elements of a triangular matrix to non-zero values outside the triangle.");
      }
      else
      {
        static_assert(eigen_self_adjoint_expr<T>);

        if (lower_self_adjoint_matrix<Arg> ? i >= static_cast<I>(j) : i <= static_cast<I>(j))
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
    }
  };


#ifdef __cpp_concepts
  template<untyped_adapter T>
  struct Subsets<T>
#else
  template<typename T>
  struct Subsets<T, std::enable_if_t<untyped_adapter<T>>>
#endif
  {
    template<std::size_t...compile_time_index, typename Arg, typename...runtime_index_t>
    static constexpr decltype(auto)
    column(Arg&& arg, runtime_index_t...i)
    {
      return column<compile_time_index...>(make_dense_writable_matrix_from(std::forward<Arg>(arg)), i...);
    }


    template<std::size_t...compile_time_index, typename Arg, typename...runtime_index_t>
    static constexpr decltype(auto)
    row(Arg&& arg, runtime_index_t...i)
    {
      return row<compile_time_index...>(make_dense_writable_matrix_from(std::forward<Arg>(arg)), i...);
    }
  };


#ifdef __cpp_concepts
  template<typename T> requires eigen_zero_expr<T> or eigen_constant_expr<T> or eigen_diagonal_expr<T> or
    eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>
  struct ArrayOperations<T>
#else
  template<typename T>
  struct ArrayOperations<T, std::enable_if_t<eigen_zero_expr<T> or eigen_constant_expr<T> or
    eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>>>
#endif
  {

    template<ElementOrder order, typename BinaryFunction, typename Accum, typename Arg>
    static constexpr auto fold(const BinaryFunction& b, Accum&& accum, Arg&& arg)
    {
      return OpenKalman::fold<order>(b, std::forward<Accum>(accum), make_dense_writable_matrix_from(std::forward<Arg>(arg)));
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

    template<typename Arg>
    static decltype(auto)
    to_diagonal(Arg&& arg) noexcept
    {
      // Note: the interface only needs to handle eigen_constant_expr or a dynamic-sized eigen_zero_expr.
      using P = pattern_matrix_of_t<T>;
      return Conversions<P>::to_diagonal(to_native_matrix<P>(std::forward<Arg>(arg)));
    }


    template<typename Arg>
    static decltype(auto)
    diagonal_of(Arg&& arg) noexcept
    {
      // Note: the global diagonal_of function already handles all eigen_zero_expr and eigen_constant_expr cases.

      if constexpr (eigen_diagonal_expr<Arg>)
      {
        return nested_matrix(std::forward<Arg>(arg));
      }
      else
      {
        return OpenKalman::diagonal_of(nested_matrix(std::forward<Arg>(arg)));
      }
    }

  };


#ifdef __cpp_concepts
  template<untyped_adapter T>
  struct ModularTransformationTraits<T>
#else
  template<typename T>
  struct ModularTransformationTraits<T, std::enable_if_t<untyped_adapter<T>>>
#endif
  {

    template<typename...FC, typename Arg, typename...DC>
    constexpr decltype(auto)
    to_euclidean(Arg&& arg, DC&&...dc) noexcept
    {
      if constexpr (eigen_zero_expr<Arg> or eigen_constant_expr<Arg>)
      {
        return ToEuclideanExpr<FC..., DC..., Arg> {std::forward<Arg>(arg), std::forward<DC>(dc)...};
      }
      else
      {
        return ToEuclideanExpr<FC..., DC..., Arg> {
          make_dense_writable_matrix_from(std::forward<Arg>(arg)), std::forward<DC>(dc)...};
      }
    }


    template<typename...FC, typename Arg, typename...DC>
    constexpr decltype(auto)
    from_euclidean(Arg&& arg, DC&&...dc) noexcept
    {
      if constexpr (eigen_zero_expr<Arg> or eigen_constant_expr<Arg>)
      {
        return FromEuclideanExpr<FC..., DC..., Arg> {std::forward<Arg>(arg), std::forward<DC>(dc)...};
      }
      else
      {
        return FromEuclideanExpr<FC..., DC..., Arg> {
          make_dense_writable_matrix_from(std::forward<Arg>(arg)), std::forward<DC>(dc)...};
      }
    }


    template<typename...FC, typename Arg, typename...DC>
    constexpr decltype(auto)
    wrap_angles(Arg&& arg, DC&&...dc) noexcept
    {
      return FromEuclideanExpr<FC..., DC..., Arg>
        {to_euclidean<Coefficients>(std::forward<Arg>(arg), std::forward<DC>(dc)...)};
    }

  };


#ifdef __cpp_concepts
  template<typename T> requires eigen_zero_expr<T> or eigen_constant_expr<T> or eigen_diagonal_expr<T> or
    eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>
  struct LinearAlgebra<T>
#else
  template<typename T>
  struct LinearAlgebra<T, std::enable_if_t<eigen_zero_expr<T> or eigen_constant_expr<T> or eigen_diagonal_expr<T> or
    eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>>>
#endif
  {

    template<typename Arg>
    static constexpr decltype(auto) conjugate(Arg&& arg) noexcept
    {
      if constexpr (eigen_constant_expr<Arg>)
      {
        constexpr auto constant = constant_coefficient_v<Arg>;
#     ifdef __cpp_lib_constexpr_complex
        constexpr auto adj = std::conj(constant);
#     else
        constexpr auto adj = std::complex(std::real(constant), -std::imag(constant));
#     endif
        return make_constant_matrix_like<adj>(std::forward<Arg>(arg));
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

      if constexpr (not any_dynamic_dimension<Arg> and index_dimension_of_v<Arg, 0> == index_dimension_of_v<Arg, 1>)
        return std::forward<Arg>(arg);
      else
        return make_constant_matrix_like<Arg, constant, Scalar>(get_dimensions_of<1>(arg), get_dimensions_of<0>(arg));
    }

  public:

    template<typename Arg>
    static constexpr decltype(auto) transpose(Arg&& arg) noexcept
    {
      if constexpr (eigen_zero_expr<Arg>)
      {
        constexpr auto rows = index_dimension_of_v<Arg, 0>;
        constexpr auto columns = index_dimension_of_v<Arg, 1>;

        if constexpr (rows == dynamic_size and columns == dynamic_size)
          return make_zero_matrix_like<Arg, columns, rows>(runtime_dimension_of<1>(arg), runtime_dimension_of<0>(arg));
        else if constexpr (rows == dynamic_size)
          return make_zero_matrix_like<Arg, columns, rows>(runtime_dimension_of<0>(arg));
        else if constexpr (columns == dynamic_size)
          return make_zero_matrix_like<Arg, columns, rows>(runtime_dimension_of<1>(arg));
        else
          return make_zero_matrix_like<Arg, columns, rows>();
      }
      else if constexpr (eigen_constant_expr<Arg>)
      {
        return eigen_constant_transpose_impl<constant_coefficient_v<Arg>>(std::forward<Arg>(arg));
      }
      else if constexpr (eigen_self_adjoint_expr<Arg>)
      {
        if constexpr (self_adjoint_matrix<nested_matrix_of_t<Arg>>)
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
        if constexpr (triangular_matrix<nested_matrix_of_t<Arg>>)
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

        if constexpr (diagonal_matrix<nested_matrix_of_t<Arg>>)
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
        return OpenKalman::determinant(make_dense_writable_matrix_from(std::forward<Arg>(arg)));
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


    template<TriangleType t, typename A, typename U, typename Alpha>
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


    template<TriangleType t, typename A, typename U, typename Alpha>
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
          auto m = make_dense_writable_matrix_from(std::forward<A>(a));
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


  /// Solve the equation AX = B for X. A is a diagonal matrix.
   template<bool must_be_unique, bool must_be_exact, typename A, typename B>
    static constexpr decltype(auto)
    solve(A&& a, B&& b)
    {
      using N = decltype(nested_matrix(a));
      return LinearAlgebra<N>::template solve<must_be_unique, must_be_exact>(std::forward<A>(a), std::forward<B>(b));
    }

  };


} // namespace OpenKalman::interface


namespace OpenKalman::Eigen3
{

  /// Create a column vector by taking the mean of each row in a set of column vectors.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    if constexpr (column_vector<Arg>)
      return std::forward<Arg>(arg);
    else
      return make_zero_matrix_like<index_dimension_of_v<Arg, 0>, 1>(arg);
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
    if constexpr (column_vector<Arg>)
      return std::forward<Arg>(arg);
    else
      return make_constant_matrix_like<Arg, constant_coefficient_v<Arg>>(get_dimensions_of<0>(arg), Dimensions<1>{});
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
    return make_dense_writable_matrix_from(make_dense_writable_matrix_from(std::forward<Arg>(arg)).rowwise().sum() / row_dimension_of_v<Arg>);
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
    if constexpr (row_vector<Arg>)
      return std::forward<Arg>(arg);
    else
      return make_zero_matrix_like<1, column_dimension_of_v<Arg>>(arg);
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
    if constexpr (row_vector<Arg>)
      return std::forward<Arg>(arg);
    else
      return make_constant_matrix_like<Arg, constant_coefficient_v<Arg>>(Dimensions<1>{}, get_dimensions_of<1>(arg));
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
    return make_dense_writable_matrix_from(make_dense_writable_matrix_from(std::forward<Arg>(arg)).colwise().sum() /
      runtime_dimension_of<1>(arg));
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
    constexpr auto dim = index_dimension_of_v<A, 0>;
    return make_zero_matrix_like<dim, dim>(a);
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
      std::sqrt((Scalar) runtime_dimension_of<1>(a)) :
      OpenKalman::internal::constexpr_sqrt((Scalar) column_dimension_of_v<A>));

    if constexpr (dynamic_rows<A>)
    {
      auto dim = runtime_dimension_of<0>(a);
      auto col1 = Eigen3::eigen_matrix_t<Scalar, dynamic_size, 1>::Constant(dim, elem);

      eigen_matrix_t<Scalar, dynamic_size, dynamic_size> ret {dim, dim};

      if (dim == 1)
        ret = std::move(col1);
      else
        ret = concatenate_horizontal(std::move(col1), make_zero_matrix_like<A, dynamic_size, dynamic_size>(dim, dim - 1));
      return ret;
    }
    else
    {
      constexpr auto dim = row_dimension_of_v<A>;
      auto col1 = Eigen3::eigen_matrix_t<Scalar, dim, 1>::Constant(elem);

      if constexpr (dim != dynamic_size)
        return concatenate_horizontal(col1, make_zero_matrix_like<A, dim, dim - 1>());
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
    else return LQ_decomposition(make_dense_writable_matrix_from(std::forward<A>(a)));
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
    constexpr auto dim = index_dimension_of_v<A, 1>;
    return make_zero_matrix_like<dim, dim>(a);
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
      std::sqrt((Scalar) runtime_dimension_of<0>(a)) :
      OpenKalman::internal::constexpr_sqrt((Scalar) row_dimension_of_v<A>));

    if constexpr (dynamic_columns<A>)
    {
      auto dim = runtime_dimension_of<1>(a);
      auto row1 = Eigen3::eigen_matrix_t<Scalar, 1, dynamic_size>::Constant(dim, elem);

      eigen_matrix_t<Scalar, dynamic_size, dynamic_size> ret {dim, dim};

      if (dim == 1)
      {
        ret = std::move(row1);
      }
      else
      {
        ret = concatenate_vertical(std::move(row1), make_zero_matrix_like<A, dynamic_size, dynamic_size>(dim - 1, dim));
      }
      return ret;
    }
    else
    {
      constexpr auto dim = column_dimension_of_v<A>;
      auto row1 = Eigen3::eigen_matrix_t<Scalar, 1, dim>::Constant(elem);
      if constexpr (dim > 1)
      {
        return concatenate_vertical(row1, make_zero_matrix_like<A, dim - 1, dim>());
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
    else return QR_decomposition(make_dense_writable_matrix_from(std::forward<A>(a)));
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
    if constexpr(sizeof...(Vs) > 0)
    {
      if constexpr ((zero_matrix<V> and ... and zero_matrix<Vs>))
      {
        if constexpr ((any_dynamic_dimension<V> or ... or any_dynamic_dimension<Vs>))
        {
          auto dim = (runtime_dimension_of<0>(v) + ... + runtime_dimension_of<0>(vs));
          return DiagonalMatrix {make_zero_matrix_like<V, dynamic_size, 1>(dim)};
        }
        else
        {
          constexpr auto dim = (row_dimension_of_v<V> + ... + row_dimension_of_v<Vs>);
          static_assert(dim == (column_dimension_of_v<V> + ... + column_dimension_of_v<Vs>));
          return make_zero_matrix_like<V, dim, dim>();
        }
      }
      else if constexpr ((identity_matrix<V> and ... and identity_matrix<Vs>))
      {
        if constexpr ((any_dynamic_dimension<V> or ... or any_dynamic_dimension<Vs>))
        {
          auto dim = (runtime_dimension_of<0>(v) + ... + runtime_dimension_of<0>(vs));
          return make_identity_matrix_like<V, dynamic_size, dynamic_size>(dim);
        }
        else
        {
          constexpr auto dim = (row_dimension_of_v<V> + ... + row_dimension_of_v<Vs>);
          static_assert(dim == (column_dimension_of_v<V> + ... + column_dimension_of_v<Vs>));
          return make_identity_matrix_like<V, dim, dim>();
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
          make_dense_writable_matrix_from(std::forward<V>(v)), make_dense_writable_matrix_from(std::forward<Vs>(vs))...);
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
    static_assert((0 + ... + Cs::dimension) <= row_dimension_of_v<Arg>);
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
    static_assert((0 + ... + Cs::dimension) <= row_dimension_of_v<Arg>);
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
    static_assert((0 + ... + Cs::dimension) <= row_dimension_of_v<Arg>);
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
    static_assert((cut + ... + cuts) <= row_dimension_of_v<Arg>);
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
    static_assert((0 + ... + Cs::dimension) <= row_dimension_of_v<Arg>);
    return split_vertical<internal::SplitSpecF<F, dense_writable_matrix_t<Arg>>, Cs...>(make_dense_writable_matrix_from(std::forward<Arg>(arg)));
  }

  /// Split a self-adjoint, triangular, or diagonal matrix diagonally.
#ifdef __cpp_concepts
  template<coefficients ... Cs, typename Arg>
  requires (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (dynamic_rows<Arg> or (0 + ... + Cs::dimension) <= row_dimension_of_v<Arg>)
#else
  template<typename ... Cs, typename Arg, std::enable_if_t<(coefficients<Cs> and ...) and
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (dynamic_rows<Arg> or (0 + ... + Cs::dimension) <= row_dimension_of<Arg>::value), int> = 0>
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
    (dynamic_rows<Arg> or (cut + ... + cuts) <= row_dimension_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    (eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg> or eigen_diagonal_expr<Arg>) and
    (dynamic_rows<Arg> or (cut + ... + cuts) <= row_dimension_of<Arg>::value), int> = 0>
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
    static_assert((0 + ... + Cs::dimension) <= row_dimension_of_v<Arg>);
    return split_horizontal<internal::SplitSpecF<F, dense_writable_matrix_t<Arg>>, Cs...>(make_dense_writable_matrix_from(std::forward<Arg>(arg)));
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
    static_assert((0 + ... + Cs::dimension) <= row_dimension_of_v<Arg>);
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
    static_assert((cut + ... + cuts) <= row_dimension_of_v<Arg>);
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
    return apply_columnwise(f, make_dense_writable_matrix_from(std::forward<Arg>(arg)));
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
    return apply_rowwise(f, make_dense_writable_matrix_from(std::forward<Arg>(arg)));
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
    return apply_coefficientwise(f, make_dense_writable_matrix_from(std::forward<Arg>(arg)));
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
    return apply_coefficientwise(f, make_dense_writable_matrix_from(std::forward<Arg>(arg)));
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
  requires (not any_dynamic_dimension<ReturnType>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename...Dists,
    std::enable_if_t<eigen_diagonal_expr<ReturnType> and (not any_dynamic_dimension<ReturnType>), int> = 0>
#endif
  inline auto
  randomize(Dists&&...dists)
  {
    using B = nested_matrix_of_t<ReturnType>;
    return MatrixTraits<ReturnType>::make(randomize<B, random_number_engine>(std::forward<Dists>(dists)...));
  }


  /**
   * \overload
   * \brief Fill a dynamic-shape Eigen matrix with random values selected from a single random distribution.
   * \details The following example constructs matrices (m, and n) in which each element is a
   * random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using D0 = DiagonalMatrix<eigen_matrix_t<double, dynamic_size, 1>>;
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
    any_dynamic_dimension<ReturnType> and
    requires { typename std::decay_t<Dist>::result_type; typename std::decay_t<Dist>::param_type; } and
    (not std::is_const_v<std::remove_reference_t<Dist>>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename Dist, std::enable_if_t<
    eigen_diagonal_expr<ReturnType> and any_dynamic_dimension<ReturnType> and
    (not std::is_const_v<std::remove_reference_t<Dist>>), int> = 0>
#endif
  inline auto
  randomize(const std::size_t rows, const std::size_t columns, Dist&& dist)
  {
    assert(rows == columns);
    using B = nested_matrix_of_t<ReturnType>;
    return MatrixTraits<ReturnType>::make(randomize<B, random_number_engine>(rows, 1, std::forward<Dist>(dist)));
  }


} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_SPECIAL_MATRIX_OVERLOADS_HPP
