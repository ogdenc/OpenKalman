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
 * \brief Definition for \ref head_of.
 */

#ifndef OPENKALMAN_HEAD_OF_HPP
#define OPENKALMAN_HEAD_OF_HPP


namespace OpenKalman
{
  namespace detail
  {
  #ifdef __cpp_concepts
    template<typename T>
  #else
    template<typename T, typename = void>
  #endif
    struct head_tail_id_split;


    template<>
    struct head_tail_id_split<FixedDescriptor<>>  { using head = FixedDescriptor<>; using tail = FixedDescriptor<>; };


    template<typename C>
    struct head_tail_id_split<FixedDescriptor<C>> { using head = C; using tail = FixedDescriptor<>; };


    template<typename C0, typename...Cs>
    struct head_tail_id_split<FixedDescriptor<C0, Cs...>> { using head = C0; using tail = FixedDescriptor<Cs...>; };


  #ifdef __cpp_concepts
    template<atomic_fixed_vector_space_descriptor C>
    struct head_tail_id_split<C>
  #else
    template<typename C>
    struct head_tail_id_split<C, std::enable_if_t<atomic_fixed_vector_space_descriptor<C>>>
  #endif
    { using head = C; using tail = FixedDescriptor<>; };

  } // namespace detail


  /**
   * \brief Type trait extracting the head of a \ref fixed_vector_space_descriptor.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct head_of;


#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
  struct head_of<T>
#else
  template<typename T>
  struct head_of<T, std::enable_if_t<fixed_vector_space_descriptor<T>>>
#endif
    { using type = typename detail::head_tail_id_split<canonical_fixed_vector_space_descriptor_t<T>>::head; };


  /**
   * \brief Helper for \ref head_of.
   */
  template<typename T>
  using head_of_t = typename head_of<T>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_HEAD_OF_HPP
