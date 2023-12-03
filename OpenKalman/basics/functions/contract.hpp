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
 * \brief The contract function.
 */

#ifndef OPENKALMAN_CONTRACT_HPP
#define OPENKALMAN_CONTRACT_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename C, typename A, typename B, std::size_t...Is>
    static constexpr auto contract_constant(C&& c, A&& a, B&& b, std::index_sequence<Is...>) noexcept
    {
      return make_constant_matrix_like<A>(std::forward<C>(c),
        get_vector_space_descriptor<0>(a), get_vector_space_descriptor<1>(b), get_vector_space_descriptor<Is + 2>(a)...);
    }


    /* // Only for use with alternate code below
    template<typename A, typename B, std::size_t...Is>
    static constexpr auto contract_dimensions(A&& a, B&& b, std::index_sequence<Is...>) noexcept
    {
      return std::tuple {get_vector_space_descriptor<0>(a), get_vector_space_descriptor<1>(b), get_vector_space_descriptor<Is + 2>(a)...};
    }
    */
  } // namespace detail


  /**
   * \brief Matrix multiplication of A * B.
   */
#ifdef __cpp_concepts
  template<indexible A, indexible B> requires dimension_size_of_index_is<A, 1, index_dimension_of_v<B, 0>, Likelihood::maybe> and
    (index_count_v<A> == dynamic_size or index_count_v<A> <= 2) and (index_count_v<B> == dynamic_size or index_count_v<B> <= 2)
#else
  template<typename A, typename B, std::enable_if_t<indexible<A> and indexible<B> and
    (dimension_size_of_index_is<A, 1, index_dimension_of<B, 0>::value, Likelihood::maybe>) and
    (index_count<A>::value == dynamic_size or index_count<A>::value <= 2) and (index_count<B>::value == dynamic_size or index_count<B>::value <= 2), int> = 0>
#endif
  constexpr decltype(auto)
  contract(A&& a, B&& b)
  {
    if constexpr (dynamic_dimension<A, 1> or dynamic_dimension<B, 0>) if (get_vector_space_descriptor<1>(a) != get_vector_space_descriptor<0>(b))
      throw std::domain_error {"In contract, columns of a (" + std::to_string(get_index_dimension_of<1>(a)) +
        ") do not match rows of b (" + std::to_string(get_index_dimension_of<0>(b)) + ")"};

    constexpr std::size_t dims = std::max({index_count_v<A>, index_count_v<B>, 2_uz});
    constexpr std::make_index_sequence<dims - 2> seq;

    if constexpr (identity_matrix<B>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (identity_matrix<A>)
    {
      return std::forward<B>(b);
    }
    else if constexpr (zero_matrix<A> or zero_matrix<B>)
    {
      using Scalar = std::decay_t<decltype(std::declval<scalar_type_of_t<A>>() * std::declval<scalar_type_of_t<B>>())>;
      return detail::contract_constant(internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{}, std::forward<A>(a), std::forward<B>(b), seq);
    }
    else if constexpr (constant_matrix<A> and constant_matrix<B>)
    {
      auto dim_const = [](const auto& a, const auto& b) {
        if constexpr (dynamic_dimension<A, 1>) return internal::index_dimension_scalar_constant_of<0>(b);
        else return internal::index_dimension_scalar_constant_of<1>(a);
      }(a, b);

      auto abd = constant_coefficient{a} * constant_coefficient{b} * std::move(dim_const);
      return detail::contract_constant(std::move(abd), std::forward<A>(a), std::forward<B>(b), seq);
    }
    else if constexpr (diagonal_matrix<A> and constant_matrix<B>)
    {
      auto col = make_self_contained(diagonal_of(std::forward<A>(a)) * constant_coefficient{b}());
      return chipwise_operation<1>([&]{ return col; }, get_index_dimension_of<1>(b));
      //Another way to do this:
      //auto tup = detail::contract_dimensions(std::forward<A>(a), std::forward<B>(b), seq);
      //auto op = [](auto&& x){ return std::forward<decltype(x)>(x); };
      //return n_ary_operation(std::move(tup), op, make_self_contained(std::move(col)));
    }
    else if constexpr (constant_matrix<A> and diagonal_matrix<B>)
    {
      auto row = make_self_contained(transpose(diagonal_of(std::forward<B>(b))) * constant_coefficient{a}());
      return chipwise_operation<0>([&]{ return row; }, get_index_dimension_of<0>(a));
      //Another way to do this:
      //auto tup = detail::contract_dimensions(std::forward<A>(a), std::forward<B>(b), seq);
      //auto op = [](auto&& x){ return std::forward<decltype(x)>(x); };
      //return n_ary_operation(std::move(tup), op, make_self_contained(std::move(row)));
    }
    else if constexpr (diagonal_matrix<A> and diagonal_matrix<B>)
    {
      auto ret {to_diagonal(n_ary_operation(std::multiplies<>{}, diagonal_of(std::forward<A>(a)), diagonal_of(std::forward<B>(b))))};
      return ret;
    }
    else
    {
      return interface::library_interface<std::decay_t<A>>::contract(std::forward<A>(a), std::forward<B>(b));
    }
  }

} // namespace OpenKalman


#endif //OPENKALMAN_CONTRACT_HPP
