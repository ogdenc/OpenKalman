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
 * \brief definitions for \ref to_euclidean function.
 */

#ifndef OPENKALMAN_TO_EUCLIDEAN_HPP
#define OPENKALMAN_TO_EUCLIDEAN_HPP

namespace OpenKalman
{
  /**
   * \brief Project the (potentially wrapped)vector space associated with index 0 to a Euclidean space for applying directional statistics.
   * \tparam Arg A matrix or tensor.
   */
#ifdef __cpp_concepts
  template<wrappable Arg>
#else
  template<typename Arg, std::enable_if_t<wrappable<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg)
  {
    if constexpr (dynamic_dimension<Arg, 0> and not euclidean_vector_space_descriptor<vector_space_descriptor_of_t<Arg, 0>>)
      if (not get_vector_space_descriptor_is_euclidean(get_vector_space_descriptor<0>(arg)))
        throw std::domain_error {"In to_euclidean, specified vector space descriptor does not match that of the object's index 0"};
    using Interface = interface::library_interface<std::decay_t<Arg>>;

    if constexpr (euclidean_vector_space_descriptor<vector_space_descriptor_of_t<Arg, 0>>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::to_euclidean_defined_for<Arg, Arg&&>)
    {
      return Interface::to_euclidean(std::forward<Arg>(arg), get_vector_space_descriptor<0>(arg));
    }
    else
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_wrappable(arg))
        throw std::domain_error {"Argument of to_euclidean is not wrappable"};

      return ToEuclideanExpr<Arg>(std::forward<Arg>(arg), get_vector_space_descriptor<0>(arg));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_TO_EUCLIDEAN_HPP
