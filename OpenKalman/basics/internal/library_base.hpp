/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of library_base.
 */

#ifndef OPENKALMAN_LIBRARY_BASE_HPP
#define OPENKALMAN_LIBRARY_BASE_HPP


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief The base class of an object within a particular linear-algebra library.
   * \details By default the library base is std::monostate.
   * \tparam Derived A derived class, such as an adapter.
   * \tparam LibraryObject A class within a particular library
   */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
  template<typename Derived, typename LibraryObject>
#else
  template<typename Derived, typename LibraryObject, typename = void>
#endif
  struct library_base { using type = std::monostate; };


  /**
   * \overload
   */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS // GCC compiler issue leads to the requires clause being false
  template<indexible Derived, indexible LibraryObject> requires
    requires { typename interface::library_interface<std::decay_t<LibraryObject>>::template LibraryBase<std::decay_t<Derived>>; }
  struct library_base<Derived, LibraryObject>
#else
  template<typename Derived, typename LibraryObject>
  struct library_base<Derived, LibraryObject,
    std::void_t<typename interface::library_interface<std::decay_t<LibraryObject>>::template LibraryBase<std::decay_t<Derived>>>>
#endif
  {
    using type = typename interface::library_interface<std::decay_t<LibraryObject>>::template LibraryBase<std::decay_t<Derived>>;
  };


  /**
   * \brief Helper template for library_base.
   */
  template<typename Derived, typename LibraryObject>
  using library_base_t = typename library_base<Derived, LibraryObject>::type;

} // namespace OpenKalman::internal


#endif //OPENKALMAN_LIBRARY_BASE_HPP
