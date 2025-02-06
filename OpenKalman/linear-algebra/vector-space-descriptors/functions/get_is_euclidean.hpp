/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref get_is_euclidean.
 */

#ifndef OPENKALMAN_GET_IS_EUCLIDEAN_HPP
#define OPENKALMAN_GET_IS_EUCLIDEAN_HPP

#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/get_component_collection.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief Determine, whether \ref descriptor::vector_space_descriptor T is euclidean.
   */
#ifdef __cpp_concepts
  template<descriptor::vector_space_descriptor T>
#else
  template<typename T, std::enable_if_t<descriptor::vector_space_descriptor<T>, int> = 0>
#endif
  constexpr auto
  get_is_euclidean(const T& t)
  {
    if constexpr (descriptor::atomic_vector_space_descriptor<T>)
    {
      return interface::vector_space_traits<T>::is_euclidean(t);
    }
    else if constexpr (descriptor::static_vector_space_descriptor<T>)
    {
      return std::apply([](const auto&...cs){ return std::bool_constant<(... and get_is_euclidean(cs))>{}; },
        descriptor::internal::get_component_collection(t));
    }
    else
    {
      for (const auto& i : descriptor::internal::get_component_collection(t)) if (not get_is_euclidean(i)) return false;
      return true;
    }
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_GET_IS_EUCLIDEAN_HPP
