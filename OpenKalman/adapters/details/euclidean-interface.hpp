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

#ifndef OPENKALMAN_EUCLIDEAN_INTERFACE_HPP
#define OPENKALMAN_EUCLIDEAN_INTERFACE_HPP

namespace OpenKalman::interface
{

#ifdef __cpp_concepts
  template<euclidean_expr T>
  struct library_interface<T>
#else
  template<typename T>
  struct library_interface<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {
    template<typename Derived>
    using LibraryBase = internal::library_base_t<Derived, pattern_matrix_of_t<T>>;


    template<typename Arg>
    static decltype(auto) to_native_matrix(Arg&& arg)
    {
      return OpenKalman::to_native_matrix<nested_object_of_t<Arg>>(std::forward<Arg>(arg));
    }


    template<Layout layout, typename Scalar, typename...D>
    static auto make_default(D&&...d)
    {
      return make_dense_object<nested_object_of_t<T>, layout, Scalar>(std::forward<D>(d)...);
    }


    // fill_components not necessary because T is not a dense writable matrix.


    template<typename C, typename...D>
    static constexpr auto make_constant(C&& c, D&&...d)
    {
      return make_constant<nested_object_of_t<T>>(std::forward<C>(c), std::forward<D>(d)...);
    }


    template<typename Scalar, typename...D>
    static constexpr auto make_identity_matrix(D&&...d)
    {
      return make_identity_matrix_like<nested_object_of_t<T>, Scalar>(std::forward<D>(d)...);
    }


    // get_block


    // set_block


    template<typename Arg>
    static auto
    to_diagonal(Arg&& arg)
    {
      if constexpr( has_untyped_index<Arg, 0>)
      {
        return to_diagonal(nested_object(std::forward<Arg>(arg)));
      }
      else
      {
        using P = pattern_matrix_of_t<T>;
        return library_interface<P>::to_diagonal(to_native_matrix<P>(std::forward<Arg>(arg)));
      }
    }


    template<typename Arg>
    static auto
    diagonal_of(Arg&& arg)
    {
      if constexpr(has_untyped_index<Arg, 0>)
      {
        return diagonal_of(nested_object(std::forward<Arg>(arg)));
      }
      else
      {
        using P = pattern_matrix_of_t<T>;
        return library_interface<P>::diagonal_of(to_native_matrix<P>(std::forward<Arg>(arg)));
      }
    }


    template<typename Arg, typename...Factors>
    static auto
    broadcast(Arg&& arg, const Factors&...factors)
    {
      return library_interface<std::decay_t<nested_object_of_t<Arg>>>::broadcast(std::forward<Arg>(arg), factors...);
    }


    template<typename...Ds, typename Operation, typename...Args>
    static constexpr decltype(auto)
    n_ary_operation(const std::tuple<Ds...>& tup, Operation&& op, Args&&...args)
    {
      using P = pattern_matrix_of_t<T>;
      return library_interface<P>::template n_ary_operation(tup, std::forward<Operation>(op), std::forward<Args>(args)...);
    }


    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr decltype(auto)
    reduce(BinaryFunction&& b, Arg&& arg)
    {
      using P = pattern_matrix_of_t<T>;
      return library_interface<P>::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<from_euclidean_expr Arg, vector_space_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<from_euclidean_expr<Arg> and vector_space_descriptor<C>, int> = 0>
#endif
    constexpr decltype(auto)
    to_euclidean(Arg&& arg, const C&)
    {
      return nested_object(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<to_euclidean_expr Arg, vector_space_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<to_euclidean_expr<Arg> and vector_space_descriptor<C>, int> = 0>
#endif
    constexpr decltype(auto)
    from_euclidean(Arg&& arg, const C& c)
    {
      return FromEuclideanExpr<C, Arg> {std::forward<Arg>(arg), c};
    }


#ifdef __cpp_concepts
    template<from_euclidean_expr Arg, vector_space_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<from_euclidean_expr<Arg> and vector_space_descriptor<C>, int> = 0>
#endif
    constexpr decltype(auto)
    wrap_angles(Arg&& arg, const C&)
    {
      return std::forward<Arg>(arg);
    }


    template<typename Arg>
    static constexpr decltype(auto)
    conjugate(Arg&& arg)
    {
      if constexpr(has_untyped_index<Arg, 0>)
      {
        return OpenKalman::conjugate(nested_object(std::forward<Arg>(arg)));
      }
      else
      {
        return std::forward<Arg>(arg).conjugate(); //< \todo Generalize this.
      }
    }


    template<typename Arg>
    static constexpr decltype(auto)
    transpose(Arg&& arg)
    {
      if constexpr(has_untyped_index<Arg, 0>)
      {
        return OpenKalman::transpose(nested_object(std::forward<Arg>(arg)));
      }
      else
      {
        return std::forward<Arg>(arg).transpose(); //< \todo Generalize this.
      }
    }


    template<typename Arg>
    static constexpr decltype(auto)
    adjoint(Arg&& arg)
    {
      if constexpr(has_untyped_index<Arg, 0>)
      {
        return OpenKalman::adjoint(nested_object(std::forward<Arg>(arg)));
      }
      else
      {
        return std::forward<Arg>(arg).adjoint(); //< \todo Generalize this.
      }
    }


    template<typename Arg>
    static constexpr auto
    determinant(Arg&& arg)
    {
      if constexpr(has_untyped_index<Arg, 0>)
      {
        return OpenKalman::determinant(nested_object(std::forward<Arg>(arg)));
      }
      else
      {
        return arg.determinant(); //< \todo Generalize this.
      }
    }


    template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_hermitian(A&& a, U&& u, const Alpha alpha)
    {
      return OpenKalman::rank_update_hermitian<significant_triangle>(make_hermitian_matrix(to_dense_object(std::forward<A>(a))), std::forward<U>(u), alpha);
    }


    template<TriangleType triangle, typename A, typename U, typename Alpha>
    static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha)
    {
      return OpenKalman::rank_update_triangular(make_triangular_matrix<triangle>(to_dense_object(std::forward<A>(a))), std::forward<U>(u), alpha);
    }


    template<bool must_be_unique, bool must_be_exact, typename A, typename B>
    static constexpr decltype(auto)
    solve(A&& a, B&& b)
    {
      return OpenKalman::solve<must_be_unique, must_be_exact>(
        to_native_matrix<T>(std::forward<A>(a)), std::forward<B>(b));
    }


    template<typename A>
    static inline auto
    LQ_decomposition(A&& a)
    {
      return LQ_decomposition(to_dense_object(std::forward<A>(a)));
    }


    template<typename A>
    static inline auto
    QR_decomposition(A&& a)
    {
      return QR_decomposition(to_dense_object(std::forward<A>(a)));
    }

  };


} // namespace OpenKalman::interface


#endif //OPENKALMAN_EUCLIDEAN_INTERFACE_HPP
