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

#ifndef OPENKALMAN_SPECIAL_MATRIX_OVERLOADS_HPP
#define OPENKALMAN_SPECIAL_MATRIX_OVERLOADS_HPP

#include <complex>


namespace OpenKalman::interface
{

#ifdef __cpp_concepts
  template<untyped_adapter T>
  struct Subsets<T>
#else
  template<typename T>
  struct Subsets<T, std::enable_if_t<untyped_adapter<T>>>
#endif
  {
    template<typename Arg, typename...Begin, typename...Size>
    static auto get_block(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
    {
      return get_block(make_dense_writable_matrix_from(std::forward<Arg>(arg)), begin, size);
    }


    // set_block is undefined


    template<TriangleType t, typename A, typename B>
    static A& set_triangle(A& a, B&& b)
    {
      if constexpr (eigen_triangular_expr<A>)
      {
        return OpenKalman::set_triangle<t>(nested_matrix(a), std::forward<B>(b));
      }
      else if constexpr (eigen_self_adjoint_expr<A>)
      {
        if constexpr ((t == TriangleType::upper and not hermitian_adapter<A, HermitianAdapterType::upper>) or
          (t == TriangleType::lower and not hermitian_adapter<A, HermitianAdapterType::lower>))
          OpenKalman::set_triangle<t>(nested_matrix(a), adjoint(std::forward<B>(b)));
        else
          OpenKalman::set_triangle<t>(nested_matrix(a), std::forward<B>(b));
      }
      else
      {
        static_assert(eigen_diagonal_expr<A>);
        static_assert(diagonal_matrix<B>);
        nested_matrix(a) = diagonal_of(std::forward<B>(b));
      }
      return a;
    }

  };


#ifdef __cpp_concepts
  template<typename T> requires eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>
  struct ArrayOperations<T>
#else
  template<typename T>
  struct ArrayOperations<T, std::enable_if_t<eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>>>
#endif
  {
    template<typename...Ds, typename Operation, typename...Args>
    static constexpr decltype(auto)
    n_ary_operation(const std::tuple<Ds...>& tup, Operation&& op, Args&&...args)
    {
      using P = pattern_matrix_of_t<T>;
      return ArrayOperations<P>::template n_ary_operation(tup, std::forward<Operation>(op), std::forward<Args>(args)...);
    }


    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr decltype(auto)
    reduce(BinaryFunction&& b, Arg&& arg)
    {
      using P = pattern_matrix_of_t<T>;
      return ArrayOperations<P>::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
    }

  };


#ifdef __cpp_concepts
  template<typename T> requires eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>
  struct Conversions<T>
#else
  template<typename T>
  struct Conversions<T, std::enable_if_t<eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>>>
#endif
  {
    template<typename Arg>
    static decltype(auto)
    to_diagonal(Arg&& arg) noexcept
    {
      // Note: the interface only needs to handle constant and dynamic-sized zero matrices.
      using P = pattern_matrix_of_t<T>;
      return Conversions<P>::to_diagonal(std::forward<Arg>(arg));
    }


    template<typename Arg>
    static decltype(auto)
    diagonal_of(Arg&& arg) noexcept
    {
      // Note: the global diagonal_of function already handles all zero and constant cases.

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


  // ModularTransformationTraits for untyped_adapter are not specially defined. Relies on the default definition.


#ifdef __cpp_concepts
  template<typename T> requires eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>
  struct LinearAlgebra<T>
#else
  template<typename T>
  struct LinearAlgebra<T, std::enable_if_t<eigen_diagonal_expr<T> or eigen_triangular_expr<T> or eigen_self_adjoint_expr<T>>>
#endif
  {

    template<typename Arg>
    static constexpr decltype(auto) conjugate(Arg&& arg) noexcept
    {
      // Global conjugate function already handles DiagonalMatrix
      if constexpr (eigen_self_adjoint_expr<Arg>)
      {
        auto n = OpenKalman::conjugate(nested_matrix(std::forward<Arg>(arg)));
        return MatrixTraits<std::decay_t<Arg>>::template make<hermitian_adapter_type_of_v<Arg>>(std::move(n));
      }
      else
      {
        static_assert(eigen_triangular_expr<Arg>);
        auto n = OpenKalman::conjugate(nested_matrix(std::forward<Arg>(arg)));
        return MatrixTraits<std::decay_t<Arg>>::template make<triangle_type_of_v<Arg>>(std::move(n));
      }
    }


    template<typename Arg>
    static constexpr decltype(auto) transpose(Arg&& arg) noexcept
    {
      // Global transpose function already handles DiagonalMatrix
      if constexpr (eigen_self_adjoint_expr<Arg>)
      {
        if constexpr (hermitian_matrix<nested_matrix_of_t<Arg>>)
        {
          return OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg)));
        }
        else
        {
          constexpr auto t = (hermitian_adapter<Arg, HermitianAdapterType::lower> ? TriangleType::upper : TriangleType::lower);
          return MatrixTraits<std::decay_t<Arg>>::template make<t>(
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
          constexpr auto t = triangular_matrix<Arg, TriangleType::lower> ? TriangleType::upper : TriangleType::lower;
          return MatrixTraits<std::decay_t<Arg>>::template make<t>(
            make_self_contained<Arg>(OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg)))));
        }
      }
    }


    template<typename Arg>
    static constexpr decltype(auto) adjoint(Arg&& arg) noexcept
    {
      // Global conjugate function already handles SelfAdjointMatrix and DiagonalMatrix
      static_assert(eigen_triangular_expr<Arg>);

      constexpr auto t = triangular_matrix<Arg, TriangleType::lower> ? TriangleType::upper : TriangleType::lower;
      return make_triangular_matrix<t>(OpenKalman::adjoint(nested_matrix(std::forward<Arg>(arg))));
    }


    template<typename Arg>
    static constexpr auto determinant(Arg&& arg) noexcept
    {
      // The general determinant function already handles TriangularMatrix and DiagonalMatrix.
      static_assert(eigen_self_adjoint_expr<Arg>);
      return OpenKalman::determinant(make_dense_writable_matrix_from(std::forward<Arg>(arg)));
    }


    template<typename A, typename B>
    static constexpr auto sum(A&& a, B&& b)
    {
      return LinearAlgebra<std::decay_t<nested_matrix_of_t<T>>>::sum(std::forward<A>(a), std::forward<B>(b));
    }


    template<typename A, typename B>
    static constexpr auto contract(A&& a, B&& b)
    {
      return LinearAlgebra<std::decay_t<nested_matrix_of_t<T>>>::contract(std::forward<A>(a), std::forward<B>(b));
    }


    template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
    static decltype(auto) rank_update_self_adjoint(A&& a, U&& u, const Alpha alpha)
    {
      decltype(auto) n = nested_matrix(std::forward<A>(a));
      using Trait = interface::LinearAlgebra<std::decay_t<decltype(n)>>;
      if constexpr (eigen_self_adjoint_expr<A>)
      {
        decltype(auto) m = Trait::template rank_update_self_adjoint<significant_triangle>(std::forward<decltype(n)>(n), std::forward<U>(u), alpha);
        return make_hermitian_matrix<significant_triangle>(std::forward<decltype(m)>(m));
      }
      else
      {
        static_assert(eigen_diagonal_expr<A>);
        return Trait::template rank_update_self_adjoint<significant_triangle>(to_native_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
    }


    template<TriangleType triangle, typename A, typename U, typename Alpha>
    static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha)
    {
      decltype(auto) n = nested_matrix(std::forward<A>(a));
      using Trait = interface::LinearAlgebra<std::decay_t<decltype(n)>>;
      if constexpr (eigen_triangular_expr<A>)
      {
        decltype(auto) m = Trait::template rank_update_triangular<triangle>(std::forward<decltype(n)>(n), std::forward<U>(u), alpha);
        return make_triangular_matrix<triangle>(std::forward<decltype(m)>(m));
      }
      else
      {
        static_assert(eigen_diagonal_expr<A>);
        return Trait::template rank_update_triangular<triangle>(to_native_matrix(std::forward<A>(a)), std::forward<U>(u), alpha);
      }
    }


    template<bool must_be_unique, bool must_be_exact, typename A, typename B>
    static constexpr decltype(auto)
    solve(A&& a, B&& b)
    {
      using N = std::decay_t<decltype(nested_matrix(a))>;
      return LinearAlgebra<N>::template solve<must_be_unique, must_be_exact>(to_native_matrix(std::forward<A>(a)), std::forward<B>(b));
    }


    template<typename A>
    static constexpr decltype(auto)
    LQ_decomposition(A&& a)
    {
      using N = std::decay_t<decltype(nested_matrix(a))>;
      return LinearAlgebra<N>::LQ_decomposition(to_native_matrix(std::forward<A>(a)));
    }


    template<typename A>
    static constexpr decltype(auto)
    QR_decomposition(A&& a)
    {
      using N = std::decay_t<decltype(nested_matrix(a))>;
      return LinearAlgebra<N>::QR_decomposition(to_native_matrix(std::forward<A>(a)));
    }

  };


} // namespace OpenKalman::interface


#endif //OPENKALMAN_SPECIAL_MATRIX_OVERLOADS_HPP
