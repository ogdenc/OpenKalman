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
 * \internal
 * \brief Definition for \ref tail_of.
 */

#ifndef OPENKALMAN_TAIL_OF_HPP
#define OPENKALMAN_TAIL_OF_HPP


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Type trait extracting the tail of a \ref fixed_vector_space_descriptor.
   */
  #ifdef __cpp_concepts
  template<typename T>
  #else
  template<typename T, typename = void>
  #endif
  struct tail_of;


#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
  struct tail_of<T>
#else
  template<typename T>
  struct tail_of<T, std::enable_if_t<fixed_vector_space_descriptor<T>>>
#endif
    { using type = typename detail::head_tail_id_split<canonical_fixed_vector_space_descriptor_t<T>>::tail; };


  /**
   * \internal
   * \brief Helper for \ref tail_of.
   */
  template<typename T>
  using tail_of_t = typename tail_of<T>::type;


} // namespace OpenKalman::internal

#endif //OPENKALMAN_TAIL_OF_HPP
