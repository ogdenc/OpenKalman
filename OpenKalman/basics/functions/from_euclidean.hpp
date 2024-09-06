/* This file is part of OpenKalman, a header-only V++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (v) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref from_euclidean function.
 */

#ifndef OPENKALMAN_FROM_EUCLIDEAN_HPP
#define OPENKALMAN_FROM_EUCLIDEAN_HPP

namespace OpenKalman
{
  /**
   * \brief Project a Euclidean space associated with index 0 to a (potentially wrapped) vector space after applying directional statistics
   * \tparam Arg A matrix or tensor.
   * \tparam V The new vector space descriptor of index 0.
   */
#ifdef __cpp_concepts
  template<wrappable Arg, vector_space_descriptor V>
  requires (dynamic_vector_space_descriptor<V> or dynamic_dimension<Arg, 0> or has_untyped_index<Arg, 0> or
    equivalent_to<V, vector_space_descriptor_of_t<Arg, 0>>)
#else
  template<typename Arg, typename V, std::enable_if_t<wrappable<Arg> and vector_space_descriptor<V> and
    (dynamic_vector_space_descriptor<V> or dynamic_dimension<Arg, 0> or has_untyped_index<Arg, 0> or
      equivalent_to<V, vector_space_descriptor_of_t<Arg, 0>>), int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg, V&& v)
  {
    if constexpr (dynamic_dimension<Arg, 0> and not euclidean_vector_space_descriptor<vector_space_descriptor_of_t<Arg, 0>>)
      if (not get_vector_space_descriptor_is_euclidean(get_vector_space_descriptor<0>(arg)) and v != get_vector_space_descriptor<0>(arg))
        throw std::domain_error {"In from_euclidean, specified vector space descriptor does not match that of the object's index 0"};
    using Interface = interface::library_interface<std::decay_t<Arg>>;

    if constexpr (euclidean_vector_space_descriptor<V>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::from_euclidean_defined_for<Arg, Arg&&, V&&>)
    {
      return Interface::from_euclidean(std::forward<Arg>(arg), std::forward<V>(v));
    }
    else
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_wrappable(arg))
        throw std::domain_error {"Argument of from_euclidean is not wrappable"};

      return FromEuclideanExpr<V, Arg>(std::forward<Arg>(arg), std::forward<V>(v));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_FROM_EUCLIDEAN_HPP
