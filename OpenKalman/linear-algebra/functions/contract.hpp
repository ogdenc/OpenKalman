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
    static constexpr auto contract_constant(C&& c, A&& a, B&& b, std::index_sequence<Is...>)
    {
      return make_constant<A>(std::forward<C>(c),
        get_vector_space_descriptor<0>(a), get_vector_space_descriptor<1>(b), get_vector_space_descriptor<Is + 2>(a)...);
    }


    /* // Only for use with alternate code below
    template<typename A, typename B, std::size_t...Is>
    static constexpr auto contract_dimensions(A&& a, B&& b, std::index_sequence<Is...>)
    {
      return std::tuple {get_vector_space_descriptor<0>(a), get_vector_space_descriptor<1>(b), get_vector_space_descriptor<Is + 2>(a)...};
    }
    */
  } // namespace detail


  /**
   * \brief Matrix multiplication of A * B.
   */
#ifdef __cpp_concepts
  template<indexible A, indexible B> requires dimension_size_of_index_is<A, 1, index_dimension_of_v<B, 0>, Qualification::depends_on_dynamic_shape> and
    (index_count_v<A> == dynamic_size or index_count_v<A> <= 2) and (index_count_v<B> == dynamic_size or index_count_v<B> <= 2)
  constexpr compatible_with_vector_space_descriptor_collection<std::tuple<vector_space_descriptor_of_t<A, 0>, vector_space_descriptor_of_t<B, 1>>> decltype(auto)
#else
  template<typename A, typename B, std::enable_if_t<indexible<A> and indexible<B> and
    (dimension_size_of_index_is<A, 1, index_dimension_of<B, 0>::value, Qualification::depends_on_dynamic_shape>) and
    (index_count<A>::value == dynamic_size or index_count<A>::value <= 2) and (index_count<B>::value == dynamic_size or index_count<B>::value <= 2), int> = 0>
  constexpr decltype(auto)
#endif
  contract(A&& a, B&& b)
  {
    if constexpr (dynamic_dimension<A, 1> or dynamic_dimension<B, 0>) if (get_vector_space_descriptor<1>(a) != get_vector_space_descriptor<0>(b))
      throw std::domain_error {"In contract, columns of a (" + std::to_string(get_index_dimension_of<1>(a)) +
        ") do not match rows of b (" + std::to_string(get_index_dimension_of<0>(b)) + ")"};

    constexpr std::size_t dims = std::max({index_count_v<A>, index_count_v<B>, 2_uz});
    constexpr std::make_index_sequence<dims - 2> seq;

    using Scalar = std::decay_t<decltype(std::declval<scalar_type_of_t<A>>() * std::declval<scalar_type_of_t<B>>())>;

    if constexpr (identity_matrix<B> and square_shaped<B>)
    {
      if constexpr (dynamic_dimension<A, 1> and not dynamic_dimension<B, 1>)
        return internal::make_fixed_size_adapter<vector_space_descriptor_of_t<A, 0>, vector_space_descriptor_of_t<B, 1>>(std::forward<A>(a));
      else
        return std::forward<A>(a);
    }
    else if constexpr (identity_matrix<A> and square_shaped<A>)
    {
      if constexpr (dynamic_dimension<B, 0> and not dynamic_dimension<A, 0>)
        return internal::make_fixed_size_adapter<vector_space_descriptor_of_t<A, 0>, vector_space_descriptor_of_t<B, 1>>(std::forward<B>(b));
      else
        return std::forward<B>(b);
    }
    else if constexpr (zero<A> or zero<B>)
    {
      return detail::contract_constant(value::Fixed<Scalar, 0>{}, std::forward<A>(a), std::forward<B>(b), seq);
    }
    else if constexpr (constant_matrix<A> and constant_matrix<B>)
    {
      auto dim_const = [](const auto& a, const auto& b) {
        if constexpr (dynamic_dimension<A, 1>) return value::cast_to<Scalar>(get_index_dimension_of<0>(b));
        else return value::cast_to<Scalar>(get_index_dimension_of<1>(a));
      }(a, b);

      auto abd = constant_coefficient{a} * constant_coefficient{b} * std::move(dim_const);
      return detail::contract_constant(std::move(abd), std::forward<A>(a), std::forward<B>(b), seq);
    }
    else if constexpr (diagonal_matrix<A> and constant_matrix<B>)
    {
      auto col = diagonal_of(std::forward<A>(a)) * constant_coefficient{b}();
      return chipwise_operation<1>([&]{ return col; }, get_index_dimension_of<1>(b));
      //Another way to do this:
      //auto tup = detail::contract_dimensions(std::forward<A>(a), std::forward<B>(b), seq);
      //auto op = [](auto&& x){ return std::forward<decltype(x)>(x); };
      //return n_ary_operation(std::move(tup), op, std::move(col));
    }
    else if constexpr (constant_matrix<A> and diagonal_matrix<B>)
    {
      auto row = transpose(diagonal_of(std::forward<B>(b))) * constant_coefficient{a}();
      return chipwise_operation<0>([&]{ return row; }, get_index_dimension_of<0>(a));
      //Another way to do this:
      //auto tup = detail::contract_dimensions(std::forward<A>(a), std::forward<B>(b), seq);
      //auto op = [](auto&& x){ return std::forward<decltype(x)>(x); };
      //return n_ary_operation(std::move(tup), op, std::move(row));
    }
    else if constexpr (diagonal_matrix<A> and diagonal_matrix<B>)
    {
      auto ret {to_diagonal(n_ary_operation(std::multiplies<Scalar>{}, diagonal_of(std::forward<A>(a)), diagonal_of(std::forward<B>(b))))};
      return ret;
    }
    else if constexpr (interface::contract_defined_for<A, A, B>)
    {
      auto x = interface::library_interface<std::decay_t<A>>::contract(std::forward<A>(a), std::forward<B>(b));
      auto ret = internal::make_fixed_size_adapter<vector_space_descriptor_of_t<A, 0>, vector_space_descriptor_of_t<B, 1>>(std::move(x));

      constexpr TriangleType tri = triangle_type_of_v<A, B>;
      if constexpr (tri != TriangleType::any and not triangular_matrix<decltype(ret), tri>)
        return make_triangular_matrix<tri>(std::move(ret));
      else
        return ret;
    }
    else if constexpr ((hermitian_matrix<A> or hermitian_matrix<B>) and
      interface::contract_defined_for<B, decltype(adjoint(std::declval<B>())), decltype(adjoint(std::declval<A>()))>)
    {
      return adjoint(interface::library_interface<std::decay_t<B>>::contract(adjoint(std::forward<B>(b)), adjoint(std::forward<A>(a))));
    }
    else if constexpr (interface::contract_defined_for<B, decltype(transpose(std::declval<B>())), decltype(transpose(std::declval<A>()))>)
    {
      return transpose(interface::library_interface<std::decay_t<B>>::contract(transpose(std::forward<B>(b)), transpose(std::forward<A>(a))));
    }
    else
    {
      return interface::library_interface<std::decay_t<A>>::contract(std::forward<A>(a), to_native_matrix<A>(std::forward<B>(b)));
    }
  }

} // namespace OpenKalman


#endif //OPENKALMAN_CONTRACT_HPP
