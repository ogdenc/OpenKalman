/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for rank_update_hermitian function.
 */

#ifndef OPENKALMAN_RANK_UPDATE_HERMITIAN_HPP
#define OPENKALMAN_RANK_UPDATE_HERMITIAN_HPP


namespace OpenKalman
{
  /**
   * \brief Do a rank update on a hermitian matrix.
   * \note This may (or may not) be performed as an in-place operation if argument A is writable and hermitian.
   * \details The update is A += αUU<sup>*</sup>, returning the updated hermitian A.
   * If A is an lvalue reference, hermitian, and writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam A The hermitian matrix to be rank updated.
   * \tparam U The update vector or matrix.
   * \returns an updated native, writable matrix in hermitian form.
   */
#ifdef __cpp_concepts
  template<hermitian_matrix<Likelihood::maybe> A, indexible U> requires
    dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 0>, Likelihood::maybe> and
    dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 1>, Likelihood::maybe> and
    std::convertible_to<scalar_type_of_t<U>, const scalar_type_of_t<A>>
  inline hermitian_matrix decltype(auto)
#else
  template<typename A, typename U, std::enable_if_t<indexible<U> and hermitian_matrix<A, Likelihood::maybe> and
    dimension_size_of_index_is<U, 0, index_dimension_of<A, 0>::value, Likelihood::maybe> and
    dimension_size_of_index_is<U, 0, index_dimension_of<A, 1>::value, Likelihood::maybe> and
    std::is_convertible_v<typename scalar_type_of<U>::type, const typename scalar_type_of<A>::type>, int> = 0>
  inline decltype(auto)
#endif
  rank_update_hermitian(A&& a, U&& u, scalar_type_of_t<A> alpha = 1)
  {
    constexpr auto t = hermitian_adapter<A> ? hermitian_adapter_type_of_v<A> : HermitianAdapterType::lower;

    if constexpr (zero<U> or dimension_size_of_index_is<A, 0, 1> or dimension_size_of_index_is<A, 1, 1> or dimension_size_of_index_is<U, 0, 1>)
    {
      if constexpr ((dynamic_dimension<A, 0> and dynamic_dimension<A, 1>) or dynamic_dimension<U, 0>)
        if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
          throw std::invalid_argument {"In rank_update_hermitian, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
            ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

      if constexpr (zero<U>)
      {
        return make_hermitian_matrix<t>(std::forward<A>(a));
      }
      else
      {
        auto e = get_component(a) + alpha * get_component(internal::FixedSizeAdapter {contract(u, adjoint(u))});

        if constexpr (element_settable<A&&, 0>)
        {
          set_component(a, e);
          return make_hermitian_matrix<t>(std::forward<A>(a));
        }
        else
        {
          auto ret {make_dense_object_from<A>(std::tuple{}, e)};
          if constexpr (std::is_assignable_v<A, decltype(std::move(ret))>)
          {
            a = std::move(ret);
            return make_hermitian_matrix<t>(std::forward<A>(a));
          }
          else return ret;
        }
      }
    }
    else if constexpr (zero<A> and diagonal_matrix<U>)
    {
      if constexpr (has_dynamic_dimensions<A>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<1>(a))
        throw std::invalid_argument {
          "In rank_update_hermitian, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
          ") do not match columns of a (" + std::to_string(get_index_dimension_of<1>(a)) + ")"};

      return alpha * cholesky_square(std::forward<U>(u));
    }
    else if constexpr (diagonal_matrix<A> and diagonal_matrix<U>)
    {
      auto d = sum(std::forward<A>(a), alpha * cholesky_square(std::forward<U>(u)));
      if constexpr (std::is_assignable_v<A, decltype(std::move(d))>) return a = std::move(d);
      else return d;
    }
    else if constexpr (hermitian_adapter<A>)
    {
      auto&& aw = internal::make_writable_square_matrix<U>(nested_object(std::forward<A>(a)));
      using Trait = interface::library_interface<std::decay_t<decltype(aw)>>;
      auto&& ret = Trait::template rank_update_hermitian<t>(std::forward<decltype(aw)>(aw), std::forward<U>(u), alpha);
      return make_hermitian_matrix<t>(std::forward<decltype(ret)>(ret));
    }
    else // hermitian_matrix but not hermitian_adapter
    {
      auto&& aw = internal::make_writable_square_matrix<U>(std::forward<A>(a));
      using Trait = interface::library_interface<std::decay_t<decltype(aw)>>;
      auto&& ret = Trait::template rank_update_hermitian<t>(std::forward<decltype(aw)>(aw), std::forward<U>(u), alpha);
      return make_hermitian_matrix<t>(std::forward<decltype(ret)>(ret));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_RANK_UPDATE_HERMITIAN_HPP
