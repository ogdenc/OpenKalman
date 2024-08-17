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
 * \brief Definition of \ref average_reduce function.
 */

#ifndef OPENKALMAN_AVERAGE_REDUCE_HPP
#define OPENKALMAN_AVERAGE_REDUCE_HPP

namespace OpenKalman
{
  namespace detail
  {
    template<typename Arg, std::size_t I, std::size_t...Is>
    constexpr auto const_diagonal_matrix_dim(const Arg& arg, std::index_sequence<I, Is...>)
    {
      if constexpr(not dynamic_dimension<Arg, I>) return index_dimension_of<Arg, I>{};
      else if constexpr (sizeof...(Is) == 0) return get_index_dimension_of<I>(arg);
      else return const_diagonal_matrix_dim(arg, std::index_sequence<Is...>{});
    }
  }


  /**
   * \brief Perform a partial reduction by taking the average along one or more indices.
   * \tparam index an index to be reduced. For example, if the index is 0, the result will have only one row.
   * If the index is 1, the result will have only one column.
   * \tparam indices Other indicesto be reduced. Because the binary function is associative, the order
   * of the indices does not matter.
   * \returns A vector or tensor with reduced dimensions.
   */
#ifdef __cpp_concepts
  template<std::size_t index, std::size_t...indices, indexible Arg> requires
    (internal::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {})) and (not empty_object<Arg>)
  constexpr indexible decltype(auto)
#else
  template<std::size_t index, std::size_t...indices, typename Arg, std::enable_if_t<indexible<Arg> and
    (internal::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {})) and (not empty_object<Arg>), int> = 0>
  constexpr decltype(auto)
#endif
  average_reduce(Arg&& arg)
  {
    // \todo Check if Arg is already in reduced and, if so, return the argument.
    if constexpr (covariance<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<vector_space_descriptor_of_t<Arg, 0>>, vector_space_descriptor_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<vector_space_descriptor_of_t<Arg, 1>>, vector_space_descriptor_of_t<Arg, 1>>;
      auto m = average_reduce<index, indices...>(to_covariance_nestable(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr(mean<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<vector_space_descriptor_of_t<Arg, 0>>, vector_space_descriptor_of_t<Arg, 0>>;
      auto m = from_euclidean<C>(average_reduce<index, indices...>(nested_object(to_euclidean(std::forward<Arg>(arg)))));
      return Mean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (euclidean_transformed<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<vector_space_descriptor_of_t<Arg, 0>>, vector_space_descriptor_of_t<Arg, 0>>;
      auto m = average_reduce<index, indices...>(nested_object(std::forward<Arg>(arg)));
      return EuclideanMean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (typed_matrix<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<vector_space_descriptor_of_t<Arg, 0>>, vector_space_descriptor_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<vector_space_descriptor_of_t<Arg, 1>>, vector_space_descriptor_of_t<Arg, 1>>;
      auto m = average_reduce<index, indices...>(nested_object(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr (dimension_size_of_index_is<Arg, index, 1>)
    {
      if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
      else return average_reduce<indices...>(std::forward<Arg>(arg));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr std::make_index_sequence< index_count_v<Arg>> seq;
      return internal::make_constant_matrix_reduction<index, indices...>(constant_coefficient{arg}, std::forward<Arg>(arg), seq);
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      // \todo Handle diagonal tensors of order greater than 2 ?
      constexpr std::make_index_sequence< index_count_v<Arg>> seq;
      auto c = constant_diagonal_coefficient{arg} / detail::const_diagonal_matrix_dim(arg, seq);
      auto ret {internal::make_constant_matrix_reduction<index>(std::move(c), std::forward<Arg>(arg), seq)};
      if constexpr (sizeof...(indices) > 0) return average_reduce<indices...>(std::move(ret));
      else return ret;
    }
    else
    {
      using Scalar = scalar_type_of_t<Arg>;
      return scalar_quotient(reduce<index, indices...>(std::plus<Scalar> {}, std::forward<Arg>(arg)),
        (get_index_dimension_of<index>(arg) * ... * get_index_dimension_of<indices>(arg)));
    }
  }


  /**
   * \overload
   * \brief Perform a complete reduction by taking the average along all indices and returning a scalar value.
   * \returns A scalar representing the average of all components.
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires
    (internal::has_uniform_reduction_indices<Arg>(std::make_index_sequence<index_count_v<Arg>> {})) and (not empty_object<Arg>)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and
    internal::has_uniform_reduction_indices<Arg>(std::make_index_sequence<index_count_v<Arg>> {}) and (not empty_object<Arg>), int> = 0>
#endif
  constexpr scalar_type_of_t<Arg>
  average_reduce(Arg&& arg)
  {
    if constexpr (zero<Arg>)
      return 0;
    else if constexpr (constant_matrix<Arg>)
      return constant_coefficient{arg}();
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return constant_diagonal_coefficient{arg} / detail::const_diagonal_matrix_dim(arg, std::make_index_sequence<index_count_v<Arg>>{});
    }
    else
    {
      // \todo Check for divide by zero?
      auto r = reduce(std::plus<scalar_type_of_t<Arg>> {}, std::forward<Arg>(arg));
      if constexpr (index_count_v<Arg> == dynamic_size)
      {
        std::size_t prod = 1;
        for (std::size_t i = 0; i < count_indices(arg); ++i) prod *= get_index_dimension_of(arg, i);
        return r / prod;
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
