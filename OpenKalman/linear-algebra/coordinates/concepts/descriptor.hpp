/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref coordinates::descriptor.
 */

#ifndef OPENKALMAN_COORDINATES_GROUP_HPP
#define OPENKALMAN_COORDINATES_GROUP_HPP

#include <functional>
#include "values/concepts/index.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "fixed_pattern.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief T is an atomic (non-separable or non-composite) grouping of \ref coordinates::pattern objects.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept descriptor =
    interface::coordinate_descriptor_traits<std::decay_t<T>>::is_specialized and
    requires(const T& t, std::vector<double>& r, double x, std::size_t i) {
      {std::invoke(interface::coordinate_descriptor_traits<std::decay_t<T>>::dimension, t)} -> values::index;
      {std::invoke(interface::coordinate_descriptor_traits<std::decay_t<T>>::stat_dimension, t)} -> values::index;
      {std::invoke(interface::coordinate_descriptor_traits<std::decay_t<T>>::to_stat_space, t, r)} -> collections::collection;
      {std::invoke(interface::coordinate_descriptor_traits<std::decay_t<T>>::from_stat_space, t, r)} -> collections::collection;
      {std::invoke(interface::coordinate_descriptor_traits<std::decay_t<T>>::get_wrapped_component, t, r, i)} -> std::convertible_to<double>;
      std::invoke(interface::coordinate_descriptor_traits<std::decay_t<T>>::set_wrapped_component, t, r, x, i);
    } and
    std::is_invocable_r_v<bool, decltype(interface::coordinate_descriptor_traits<std::decay_t<T>>::is_euclidean), const T&> and
    std::is_invocable_r_v<std::size_t, decltype(interface::coordinate_descriptor_traits<std::decay_t<T>>::hash_code), const T&>;
#else
  constexpr bool descriptor = interface::coordinate_descriptor_traits<std::decay_t<T>>::is_specialized;
#endif


} // namespace OpenKalman::coordinates

#endif //OPENKALMAN_COORDINATES_GROUP_HPP
