/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref concatenate_fixed_vector_space_descriptor.
 */

#ifndef OPENKALMAN_CONCATENATE_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_CONCATENATE_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>


namespace OpenKalman
{
  template<>
  struct concatenate_fixed_vector_space_descriptor<>
  {
    using type = TypedIndex<>;
  };

  template<typename C, typename...Cs>
  struct concatenate_fixed_vector_space_descriptor<C, Cs...>
  {
    using type = typename concatenate_fixed_vector_space_descriptor<Cs...>::type::template Prepend<C>;
  };

  template<typename...C, typename...Cs>
  struct concatenate_fixed_vector_space_descriptor<TypedIndex<C...>, Cs...>
  {
    using type = typename concatenate_fixed_vector_space_descriptor<Cs...>::type::template Prepend<C...>;
  };


} // namespace OpenKalman

#endif //OPENKALMAN_CONCATENATE_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
