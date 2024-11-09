/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Broadcast function.
 */

#ifndef OPENKALMAN_BROADCAST_HPP
#define OPENKALMAN_BROADCAST_HPP

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace details
  {
    template<typename Factor, typename = void>
    struct FactorIs1 : std::false_type {};

    template<typename Factor>
    struct FactorIs1<Factor, std::enable_if_t<Factor::value == 1>> : std::true_type {};
  } // namespace details
#endif

  /**
   * \brief Broadcast an object by replicating it by factors specified for each index.
   * \details The operation may increase the order of the object by specifying factors greater than 1 for higher indices.
   * Any such higher indices will have a \ref vector_space_descriptor of <code>Dimensions&lt;n&gt;<code> where <code>n</code> is the factor.
   * \tparam Arg The object.
   * \tparam Factors A set of factors, each an \ref index_value, indicating the increase in size of each index.
   * Any omitted trailing factors are treated as factor 1 (no broadcasting along that index).
   */
#ifdef __cpp_concepts
  template<indexible Arg, index_value...Factors> 
  constexpr indexible decltype(auto)
#else
  template<typename Arg, typename...Factors, std::enable_if_t<indexible<Arg> and (... and index_value<Factors>), int> = 0>
  constexpr decltype(auto)
#endif
  broadcast(Arg&& arg, const Factors&...factors)
  {
    if constexpr (sizeof...(Factors) == 0)
    {
      return std::forward<Arg>(arg);
    }
#ifdef __cpp_concepts
    else if constexpr (requires { requires std::tuple_element_t<sizeof...(Factors) - 1, std::tuple<Factors...>>::value == 1; })
#else
    else if constexpr (details::FactorIs1<std::tuple_element_t<sizeof...(Factors) - 1, std::tuple<Factors...>>>::value)
#endif
    {
      // Recursively remove any trailing 1D vector space descriptors
      return std::apply(
        [](Arg&& arg, const auto&...fs) { return broadcast(std::forward<Arg>(arg), fs...); },
        std::tuple_cat(
          std::forward_as_tuple(std::forward<Arg>(arg)),
          internal::tuple_slice<0, sizeof...(Factors) - 1>(std::forward_as_tuple(factors...))));
    }
    else
    {
      if constexpr ((... or dynamic_index_value<Factors>))
      {
        if ((... or (factors <= 0))) throw std::invalid_argument {"In broadcast, all factors must be positive"};
      }

      using Trait = interface::library_interface<std::decay_t<Arg>>;
      return Trait::broadcast(std::forward<Arg>(arg), factors...);
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_BROADCAST_HPP
