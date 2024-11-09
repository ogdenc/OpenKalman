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
 * \brief Definition for \ref nested_object_of.
 */

#ifndef OPENKALMAN_NESTED_OBJECT_OF_HPP
#define OPENKALMAN_NESTED_OBJECT_OF_HPP


namespace OpenKalman
{
  /**
   * \brief A wrapper type's nested object type, if it exists.
   * \details For example, for OpenKalman::TriangularAdapter<M, TriangleType::lower>, the nested object type is M.
   * \tparam T A wrapper type that has a nested object.
   * \internal \sa interface::indexible_object_traits
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct nested_object_of;


#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename>
#endif
  struct nested_object_of {};


#ifdef __cpp_concepts
  template<has_nested_object T>
  struct nested_object_of<T>
#else
  template<typename T>
  struct nested_object_of<T, std::enable_if_t<has_nested_object<T>>>
#endif
  {
    using type = decltype(interface::indexible_object_traits<std::decay_t<T>>::nested_object(std::declval<T>()));
  };


  /**
   * \brief Helper type for \ref nested_object_of.
   * \tparam T A wrapper type that has a nested matrix.
   * \tparam i Index of the dependency (0 by default)
   */
#ifdef __cpp_concepts
  template<has_nested_object T>
#else
  template<typename T>
#endif
  using nested_object_of_t = typename nested_object_of<T>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_NESTED_OBJECT_OF_HPP
