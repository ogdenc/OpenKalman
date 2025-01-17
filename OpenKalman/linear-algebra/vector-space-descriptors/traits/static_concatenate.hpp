/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref static_concatenate.
 */

#ifndef OPENKALMAN_STATIC_CONCATENATE_HPP
#define OPENKALMAN_STATIC_CONCATENATE_HPP

#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"


namespace OpenKalman::descriptor
{
  /**
   * \brief Concatenate any number of StaticDescriptor<...> types.
   * \details
   * Example:
   * - \code static_concatenate_t<StaticDescriptor<angle::Radians>,
   * StaticDescriptor<Axis, Distance>> == StaticDescriptor<angle::Radians, Axis, Distance> \endcode.
   */
#ifdef __cpp_concepts
  template<typename...Cs>
  struct static_concatenate {};
#else
  template<typename...Cs>
  struct static_concatenate;
#endif


  /**
   * \brief Helper template for \ref static_concatenate.
   */
  template<typename...Cs>
  using static_concatenate_t = typename static_concatenate<Cs...>::type;


#ifdef __cpp_concepts
  template<static_vector_space_descriptor...Cs>
  struct static_concatenate<Cs...>
#else
  template<typename...Cs>
  struct static_concatenate
#endif
  {
    using type = StaticDescriptor<Cs...>;
  };


#ifdef __cpp_concepts
  template<static_vector_space_descriptor...Cs, static_vector_space_descriptor...Ds, static_vector_space_descriptor...Es>
#else
  template<typename...Cs, typename...Ds, typename...Es>
#endif
  struct static_concatenate<StaticDescriptor<Cs...>, StaticDescriptor<Ds...>, Es...>
  {
    using type = static_concatenate_t<StaticDescriptor<Cs..., Ds...>, Es...>;
  };


#ifdef __cpp_concepts
  template<static_vector_space_descriptor...Cs, static_vector_space_descriptor...Ds>
#else
  template<typename...Cs, typename...Ds>
#endif
  struct static_concatenate<StaticDescriptor<Cs...>, Ds...>
  {
    using type = static_concatenate_t<Cs..., Ds...>;
  };


#ifdef __cpp_concepts
  template<static_vector_space_descriptor C, static_vector_space_descriptor...Ds, static_vector_space_descriptor...Es>
#else
  template<typename C, typename...Ds, typename...Es>
#endif
  struct static_concatenate<C, StaticDescriptor<Ds...>, Es...>
  {
    using type = static_concatenate_t<StaticDescriptor<C, Ds...>, Es...>;
  };


} // namespace OpenKalman

#endif //OPENKALMAN_STATIC_CONCATENATE_HPP
