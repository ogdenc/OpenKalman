/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref average_reduce function.
 */

#ifndef OPENKALMAN_AVERAGE_REDUCE_HPP
#define OPENKALMAN_AVERAGE_REDUCE_HPP

namespace OpenKalman
{
  namespace detail
  {
    template<typename Arg, std::size_t...indices, std::size_t...Ix>
    constexpr decltype(auto)
    average_reduce_impl(Arg&& arg, std::index_sequence<indices...> indices_seq, std::index_sequence<Ix...> seq)
    {
      if constexpr (constant_matrix<Arg>)
      {
        return make_constant<Arg>(constant_coefficient{arg}, internal::get_reduced_vector_space_descriptor<Ix, indices...>(std::forward<Arg>(arg))...);
      }
      else
      {
        return scalar_quotient(
          reduce<indices...>(std::plus<scalar_type_of_t<Arg>> {}, std::forward<Arg>(arg)),
          internal::count_reduced_dimensions(arg, indices_seq, seq));
      }
    }

  } // namespace detail


  /**
   * \brief Perform a partial reduction by taking the average along one or more indices.
   * \tparam index an index to be reduced. For example, if the index is 0, the result will have only one row.
   * If the index is 1, the result will have only one column.
   * \tparam indices Other indices to be reduced. Because the binary function is associative, the order
   * of the indices does not matter.
   * \returns A vector or tensor with reduced dimensions.
   */
#ifdef __cpp_concepts
  template<std::size_t index, std::size_t...indices, internal::has_uniform_static_vector_space_descriptors<index, indices...> Arg> requires
    (not empty_object<Arg>)
  constexpr indexible decltype(auto)
#else
  template<std::size_t index, std::size_t...indices, typename Arg, std::enable_if_t<
    internal::has_uniform_static_vector_space_descriptors<Arg, index, indices...> and (not empty_object<Arg>), int> = 0>
  constexpr decltype(auto)
#endif
  average_reduce(Arg&& arg)
  {
    if constexpr (dimension_size_of_index_is<Arg, index, 1>)
    {
      // Check if Arg is already reduced along index.
      if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
      else return average_reduce<indices...>(std::forward<Arg>(arg));
    }
    else
    {
      return detail::average_reduce_impl(std::forward<Arg>(arg), std::index_sequence<index, indices...>{}, std::make_index_sequence<index_count_v<Arg>>{});
    }
  }


  /**
   * \overload
   * \brief Perform a complete reduction by taking the average along all indices and returning a scalar value.
   * \returns A scalar representing the average of all components.
   */
#ifdef __cpp_concepts
  template<internal::has_uniform_static_vector_space_descriptors Arg>
#else
  template<typename Arg, std::enable_if_t<internal::has_uniform_static_vector_space_descriptors<Arg>, int> = 0>
#endif
  constexpr scalar_type_of_t<Arg>
  average_reduce(Arg&& arg)
  {
    if constexpr (zero<Arg>)
    {
      return 0;
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return constant_coefficient{arg}();
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      // Arg cannot be a zero matrix, so the denominator should never be zero.
      return values::scalar_constant_operation {
        std::divides<scalar_type_of_t<Arg>>{},
        constant_diagonal_coefficient{arg},
        internal::largest_vector_space_descriptor<scalar_type_of_t<Arg>>(get_vector_space_descriptor<0>(arg), get_vector_space_descriptor<1>(arg))};
    }
    else
    {
      auto r = reduce(std::plus<scalar_type_of_t<Arg>>{}, std::forward<Arg>(arg));
      // Arg cannot be a zero matrix, so the denominator should never be zero.
      if constexpr (index_count_v<Arg> == dynamic_size)
      {
        std::size_t denom = 1;
        for (std::size_t i = 0; i < count_indices(arg); ++i) denom *= get_index_dimension_of(arg, i);
        return r / denom;
      }
      else
      {
        auto denom = (std::apply([](const auto&...d){ return (get_dimension_size_of(d) * ...); }, all_vector_space_descriptors(arg)));
        return r / denom;
      }
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_AVERAGE_REDUCE_HPP
